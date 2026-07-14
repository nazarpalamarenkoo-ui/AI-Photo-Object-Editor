import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
from PIL import Image
import numpy as np

from app.ml.inpainter import LaMaInpainter, InpaintMode


@pytest.fixture
def inpainter():
    with patch('app.ml.model_manager.ModelManager') as mock_mm:
        mock_manager = MagicMock()
        mock_mm.return_value = mock_manager

        mock_model = MagicMock()
        mock_manager.load_model.return_value = mock_model

        fake_result = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_model.return_value = fake_result

        inpainter = LaMaInpainter()

        yield inpainter


@pytest.fixture
def test_image_bytes():
    img = Image.new('RGB', (640, 480), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_mask_bytes():
    mask = Image.new('L', (640, 480), color=0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([200, 150, 400, 350], fill=255)
    buffer = BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def test_replacement_bytes():
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


# --- існуючі тести ---

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_remove_with_mask(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=True
    )
    assert 'result_bytes' in result
    assert 'metrics' in result
    assert isinstance(result['result_bytes'], bytes)
    assert len(result['result_bytes']) > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_remove_with_bbox(inpainter, test_image_bytes):
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REMOVE
    )
    assert 'result_bytes' in result
    assert isinstance(result['result_bytes'], bytes)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_with_bbox(inpainter, test_image_bytes, test_replacement_bytes):
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_preserves_background(inpainter, test_image_bytes, test_replacement_bytes):
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes
    )
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_without_replacement_image(inpainter, test_image_bytes):
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    with pytest.raises(ValueError, match="replacement_image_bytes"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            bbox=bbox,
            mode=InpaintMode.REPLACE
        )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_no_mask_no_bbox(inpainter, test_image_bytes):
    with pytest.raises(ValueError, match="mask_bytes or bbox"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            mode=InpaintMode.REMOVE
        )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_calculate_metrics(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=True
    )
    metrics = result['metrics']
    assert 'processing_time_ms' in metrics
    assert 'mask_size_pixels' in metrics
    assert 'image_size' in metrics
    assert metrics['processing_time_ms'] > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_small_bbox(inpainter, test_image_bytes):
    small_bbox = {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 150}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=small_bbox,
        mode=InpaintMode.REMOVE
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_large_bbox(inpainter, test_image_bytes):
    large_bbox = {'x1': 10, 'y1': 10, 'x2': 630, 'y2': 470}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=large_bbox,
        mode=InpaintMode.REMOVE
    )
    assert 'result_bytes' in result

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_fast_preset(inpainter, test_image_bytes, test_mask_bytes):
    """Fast preset — 10 steps."""
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        ldm_steps=10,
        ldm_sampler='plms',
        hd_strategy='CROP',
        track_metrics=False
    )
    assert 'result_bytes' in result
    assert isinstance(result['result_bytes'], bytes)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_quality_preset(inpainter, test_image_bytes, test_mask_bytes):
    """Quality preset — 25 steps."""
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
        track_metrics=False
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_ddim_sampler(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        ldm_steps=20,
        ldm_sampler='ddim',
        hd_strategy='CROP',
        track_metrics=False
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_hd_strategy_resize(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='RESIZE',
        track_metrics=False
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_hd_strategy_original(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='ORIGINAL',
        track_metrics=False
    )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_with_ldm_params(inpainter, test_image_bytes, test_replacement_bytes):
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes,
        ldm_steps=10,
        ldm_sampler='ddim',
        hd_strategy='CROP',
        track_metrics=False
    )
    assert 'result_bytes' in result
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_metrics_contain_mode(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=True
    )
    assert result['metrics']['mode'] == 'remove'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_metrics_image_size(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )
    assert result['metrics']['image_size'] == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_metrics_mask_size_pixels(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )
    assert result['metrics']['mask_size_pixels'] > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_metrics_mask_size_from_bbox(inpainter, test_image_bytes):
    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )
    assert result['metrics']['mask_size_pixels'] == 10000


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_result_is_valid_jpeg(inpainter, test_image_bytes, test_mask_bytes):
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )
    assert result['result_bytes'][:2] == b'\xff\xd8'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_track_metrics_false_skips_tracking(inpainter, test_image_bytes, test_mask_bytes):
    inpainter.tracker = MagicMock()
    inpainter.tracker.log_inpaint_metrics = MagicMock()

    await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )

    inpainter.tracker.log_inpaint_metrics.assert_not_called()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_custom_ldm_steps_range(inpainter, test_image_bytes, test_mask_bytes):
    for steps in [5, 10, 25, 50]:
        result = await inpainter.inpaint(
            image_bytes=test_image_bytes,
            mask_bytes=test_mask_bytes,
            mode=InpaintMode.REMOVE,
            ldm_steps=steps,
            track_metrics=False
        )
        assert 'result_bytes' in result, f"Failed for ldm_steps={steps}"