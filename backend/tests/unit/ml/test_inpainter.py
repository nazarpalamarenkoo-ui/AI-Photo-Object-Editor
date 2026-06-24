import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import numpy as np
from PIL import Image
from io import BytesIO
import sys

with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.inpainter import LaMaInpainter, InpaintMode

mock_lama = MagicMock()
mock_lama.model_manager.ModelManager = MagicMock()
mock_lama.schema.Config = MagicMock()
mock_lama.schema.HDStrategy = MagicMock()
sys.modules['lama_cleaner'] = mock_lama
sys.modules['lama_cleaner.model_manager'] = mock_lama.model_manager
sys.modules['lama_cleaner.schema'] = mock_lama.schema

from app.ml.inpainter import LaMaInpainter, InpaintMode, get_inpainter


def _make_image_bytes(width=640, height=480, color='white'):
    img = Image.new('RGB', (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def _make_mask_bytes(width=640, height=480):
    mask = Image.new('L', (width, height), color=0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([200, 150, 400, 350], fill=255)
    buf = BytesIO()
    mask.save(buf, format='PNG')
    return buf.getvalue()


def _make_inpainter():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    fake_result = np.ones((480, 640, 3), dtype=np.uint8) * 128
    inpainter.model_manager = MagicMock(return_value=fake_result)
    inpainter.default_config = MagicMock()
    inpainter.tracker = MagicMock()
    inpainter.tracker.log_inpaint_metrics = MagicMock()
    return inpainter


# --- існуючі тести ---

@pytest.mark.unit
def test_inpainter_init():
    mock_tracker = MagicMock()
    inpainter = LaMaInpainter(device='cpu', tracker=mock_tracker)
    assert inpainter.device == 'cpu'
    assert inpainter.tracker is mock_tracker
    assert inpainter.model_manager is not None


@pytest.mark.unit
def test_create_mask_from_bbox():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert np.all(mask[100:200, 100:200] == 255)
    assert np.all(mask[0:100, 0:100] == 0)


@pytest.mark.unit
def test_get_bbox_from_mask():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 150:250] = 255
    bbox = inpainter._get_bbox_from_mask(mask)
    assert bbox['x1'] == 150
    assert bbox['x2'] == 249
    assert bbox['y1'] == 100
    assert bbox['y2'] == 199


@pytest.mark.unit
def test_get_bbox_from_empty_mask():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    mask = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="Empty mask"):
        inpainter._get_bbox_from_mask(mask)


@pytest.mark.unit
def test_inpainter_singleton():
    import app.ml.inpainter
    app.ml.inpainter._inpainter_instance = None
    mock_tracker = MagicMock()
    inpainter1 = get_inpainter(device='cpu', tracker=mock_tracker)
    inpainter2 = get_inpainter(device='cpu', tracker=mock_tracker)
    assert inpainter1 is inpainter2
    app.ml.inpainter._inpainter_instance = None


@pytest.mark.unit
def test_create_mask_edge_cases():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    bbox = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)
    assert mask.shape == (480, 640)
    assert np.all(mask[0:50, 0:50] == 255)


@pytest.mark.unit
def test_create_mask_full_image():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    bbox = {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 480}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)
    assert np.all(mask == 255)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_passes_ldm_steps_to_config():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()
    mock_hd.CROP = 'CROP'

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            ldm_steps=10,
            ldm_sampler='plms',
            hd_strategy='CROP',
            track_metrics=False
        )

        mock_config_cls.assert_called_once()
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs['ldm_steps'] == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_passes_ldm_sampler_to_config():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            ldm_steps=25,
            ldm_sampler='ddim',
            hd_strategy='CROP',
            track_metrics=False
        )

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs['ldm_sampler'] == 'ddim'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_passes_hd_strategy_to_config():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()
    mock_hd.RESIZE = 'RESIZE_VALUE'

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='RESIZE',
            track_metrics=False
        )

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs['hd_strategy'] == mock_hd.RESIZE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_default_ldm_params():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()
    mock_hd.CROP = 'CROP_VALUE'

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            track_metrics=False
        )

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs['ldm_steps'] == 25
        assert call_kwargs['ldm_sampler'] == 'plms'
        assert call_kwargs['hd_strategy'] == mock_hd.CROP


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_config_has_correct_fixed_params():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            track_metrics=False
        )

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs['hd_strategy_crop_margin'] == 32
        assert call_kwargs['hd_strategy_crop_trigger_size'] == 800
        assert call_kwargs['hd_strategy_resize_limit'] == 2048


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_raises_without_mask_or_bbox():
    inpainter = _make_inpainter()
    with pytest.raises(ValueError, match="mask_bytes or bbox"):
        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mode=InpaintMode.REMOVE
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_raises_replace_without_replacement():
    inpainter = _make_inpainter()
    with pytest.raises(ValueError, match="replacement_image_bytes"):
        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REPLACE
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_remove_calls_inpaint_remove():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._inpaint_replace', new_callable=AsyncMock) as mock_replace, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            track_metrics=False
        )

        mock_remove.assert_called_once()
        mock_replace.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_replace_calls_inpaint_replace():
    inpainter = _make_inpainter()

    mock_config_cls = MagicMock()
    mock_hd = MagicMock()

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._inpaint_replace', new_callable=AsyncMock) as mock_replace, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_replace.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'replace'}

        replacement = _make_image_bytes(color='red')

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REPLACE,
            replacement_image_bytes=replacement,
            track_metrics=False
        )

        mock_replace.assert_called_once()
        mock_remove.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inpaint_config_passed_to_inpaint_remove():
    inpainter = _make_inpainter()

    mock_config_instance = MagicMock()
    mock_config_cls = MagicMock(return_value=mock_config_instance)
    mock_hd = MagicMock()

    with patch('app.ml.inpainter.LaMaInpainter._inpaint_remove', new_callable=AsyncMock) as mock_remove, \
         patch('app.ml.inpainter.LaMaInpainter._calculate_metrics', new_callable=AsyncMock) as mock_metrics, \
         patch('app.ml.inpainter.LaMaInpainter._track_metrics', new_callable=AsyncMock), \
         patch('lama_cleaner.schema.Config', mock_config_cls), \
         patch('lama_cleaner.schema.HDStrategy', mock_hd):

        mock_remove.return_value = _make_image_bytes()
        mock_metrics.return_value = {'processing_time_ms': 1.0, 'mask_size_pixels': 0, 'image_size': (640, 480), 'mode': 'remove'}

        await inpainter.inpaint(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            mode=InpaintMode.REMOVE,
            track_metrics=False
        )

        call_args = mock_remove.call_args
        passed_config = call_args.args[3] if len(call_args.args) > 3 else call_args.kwargs.get('config')
        assert passed_config is mock_config_instance


@pytest.mark.unit
def test_create_remove_mask_with_neighbor_right():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    neighbor = {'x1': 210, 'y1': 100, 'x2': 300, 'y2': 200}

    mask = inpainter.create_remove_mask(
        (480, 640), bbox,
        expand_pixels=20,
        other_bboxes=[neighbor]
    )

    assert np.all(mask[100:200, 210:300] == 0)


@pytest.mark.unit
def test_create_remove_mask_with_neighbor_left():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 200, 'y1': 100, 'x2': 300, 'y2': 200}
    neighbor = {'x1': 100, 'y1': 100, 'x2': 190, 'y2': 200}

    mask = inpainter.create_remove_mask(
        (480, 640), bbox,
        expand_pixels=20,
        other_bboxes=[neighbor]
    )

    assert np.all(mask[100:200, 100:190] == 0)


@pytest.mark.unit
def test_create_remove_mask_no_other_bboxes():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    expand = 10

    mask = inpainter.create_remove_mask(
        (480, 640), bbox,
        expand_pixels=expand,
        other_bboxes=None
    )

    assert mask[100 - expand, 150] == 255
    assert mask[200 + expand - 1, 150] == 255
    assert mask[150, 100 - expand] == 255
    assert mask[150, 200 + expand - 1] == 255


@pytest.mark.unit
def test_create_remove_mask_neighbor_no_overlap():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    neighbor = {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400}

    mask_with = inpainter.create_remove_mask(
        (480, 640), bbox, expand_pixels=10, other_bboxes=[neighbor]
    )
    mask_without = inpainter.create_remove_mask(
        (480, 640), bbox, expand_pixels=10, other_bboxes=None
    )

    np.testing.assert_array_equal(mask_with, mask_without)

@pytest.mark.unit
def test_create_replace_mask_exact_bbox():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 50, 'y1': 60, 'x2': 150, 'y2': 160}
    mask = inpainter.create_replace_mask((480, 640), bbox)

    assert np.all(mask[60:160, 50:150] == 255)
    assert np.all(mask[0:60, :] == 0)
    assert np.all(mask[160:, :] == 0)


@pytest.mark.unit
def test_create_replace_mask_shape():
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    bbox = {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100}
    mask = inpainter.create_replace_mask((200, 300), bbox)
    assert mask.shape == (200, 300)
    assert mask.dtype == np.uint8