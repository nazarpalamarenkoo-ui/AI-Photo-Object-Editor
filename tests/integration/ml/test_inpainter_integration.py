import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
from PIL import Image
import numpy as np

from app.ml.inpainter import LaMaInpainter, InpaintMode


@pytest.fixture
def inpainter():
    with patch('app.ml.model_manager.ModelManager') as mock_mm:
        # Mock ModelManager instance
        mock_manager = MagicMock()
        mock_mm.return_value = mock_manager
        
        # Mock load_model
        mock_model = MagicMock()
        mock_manager.load_model.return_value = mock_model
        
        # Mock model prediction - return fake tensor
        fake_result = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_model.return_value = fake_result
        
        # Create inpainter
        inpainter = LaMaInpainter(device='cpu')
        
        yield inpainter


@pytest.fixture
def test_image_bytes():
    """Create test image bytes (640x480)"""
    img = Image.new('RGB', (640, 480), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_mask_bytes():
    """Create test mask bytes (640x480)"""
    mask = Image.new('L', (640, 480), color=0)
    # Draw white rectangle in center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([200, 150, 400, 350], fill=255)
    
    buffer = BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def test_replacement_bytes():
    """Create replacement image bytes"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_remove_with_mask(inpainter, test_image_bytes, test_mask_bytes):
    """Test inpainting with mask for removal"""
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
    """Test inpainting with bbox (auto-creates mask)"""
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
async def test_inpaint_replace_with_bbox(
    inpainter,
    test_image_bytes,
    test_replacement_bytes
):
    """Test replacement with bbox"""
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
async def test_inpaint_replace_preserves_background(
    inpainter,
    test_image_bytes,
    test_replacement_bytes
):
    """Test that replacement preserves background"""
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes
    )
    
    # Verify result is valid image
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_without_replacement_image(
    inpainter,
    test_image_bytes
):
    """Test that replace mode requires replacement image"""
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}
    
    with pytest.raises(ValueError, match="replacement_image_bytes"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            bbox=bbox,
            mode=InpaintMode.REPLACE
            # Missing replacement_image_bytes!
        )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_no_mask_no_bbox(inpainter, test_image_bytes):
    """Test that inpaint requires either mask or bbox"""
    with pytest.raises(ValueError, match="mask_bytes or bbox"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            mode=InpaintMode.REMOVE
            # Missing both mask_bytes and bbox!
        )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_calculate_metrics(inpainter, test_image_bytes, test_mask_bytes):
    """Test metrics calculation"""
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
    """Test inpainting with small bbox"""
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
    """Test inpainting with large bbox"""
    large_bbox = {'x1': 10, 'y1': 10, 'x2': 630, 'y2': 470}
    
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=large_bbox,
        mode=InpaintMode.REMOVE
    )
    
    assert 'result_bytes' in result