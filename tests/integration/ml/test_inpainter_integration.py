"""
Integration Tests for LaMaInpainter - ВИПРАВЛЕНО

Location: tests/integration/ml/test_inpainter_integration.py
"""
import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO
import numpy as np
from PIL import Image

from app.ml.inpainter import LaMaInpainter, InpaintMode


# ==================== FIXTURES ====================

@pytest.fixture
def mock_model_manager():
    """Mock LaMa model manager - ВИПРАВЛЕНО!"""
    manager = MagicMock()
    
    # ВАЖЛИВО: Повертаємо РЕАЛЬНИЙ numpy array!
    def mock_inpaint(image, mask, config):
        # image - це numpy array з shape (H, W, 3)
        return np.random.randint(0, 255, image.shape, dtype=np.uint8)
    
    # Призначаємо як callable
    manager.side_effect = mock_inpaint
    
    return manager


@pytest.fixture
def mock_tracker():
    """Mock experiment tracker"""
    tracker = MagicMock()
    tracker.log_inpaint_metrics = MagicMock()
    return tracker


@pytest.fixture
def inpainter(mock_tracker, mock_model_manager):
    """Inpainter з mock компонентами"""
    with patch('app.ml.inpainter.ModelManager') as mock_mm:
        mock_mm.return_value = mock_model_manager
        
        inpainter = LaMaInpainter(
            device='cpu',
            tracker=mock_tracker
        )
        
        yield inpainter


@pytest.fixture
def test_image_bytes():
    """Create test image bytes"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_mask_bytes():
    """Create test mask bytes"""
    mask = Image.new('L', (640, 480), color=0)
    mask_array = np.array(mask)
    mask_array[200:280, 200:280] = 255
    mask_img = Image.fromarray(mask_array)
    
    buffer = BytesIO()
    mask_img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def test_replacement_bytes():
    """Create replacement image bytes"""
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_bbox():
    """Test bbox"""
    return {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300}


# ==================== INTEGRATION TESTS - REMOVE ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_remove_with_mask(inpainter, test_image_bytes, test_mask_bytes):
    """Test REMOVE mode with mask"""
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        mode=InpaintMode.REMOVE,
        track_metrics=True
    )
    
    assert 'result_bytes' in result
    assert 'metrics' in result
    
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)
    
    assert result['metrics']['mode'] == 'remove'
    assert 'processing_time_ms' in result['metrics']


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_remove_with_bbox(inpainter, test_image_bytes, test_bbox):
    """Test REMOVE mode with bbox"""
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=test_bbox,
        mode=InpaintMode.REMOVE,
        track_metrics=False
    )
    
    assert 'result_bytes' in result
    assert result['metrics']['mode'] == 'remove'


# ==================== INTEGRATION TESTS - REPLACE ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_with_bbox(
    inpainter,
    test_image_bytes,
    test_bbox,
    test_replacement_bytes
):
    """Test REPLACE mode with bbox"""
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=test_bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes,
        track_metrics=True
    )
    
    assert 'result_bytes' in result
    assert result['metrics']['mode'] == 'replace'
    
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_preserves_background(
    inpainter,
    test_image_bytes,
    test_bbox,
    test_replacement_bytes
):
    """Test that REPLACE mode preserves background"""
    orig_img = Image.open(BytesIO(test_image_bytes))
    orig_array = np.array(orig_img)
    
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=test_bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes,
        track_metrics=False
    )
    
    result_img = Image.open(BytesIO(result['result_bytes']))
    result_array = np.array(result_img)
    
    # Background outside bbox should be unchanged
    orig_corner = orig_array[0:50, 0:50]
    result_corner = result_array[0:50, 0:50]
    
    diff = np.abs(orig_corner.astype(float) - result_corner.astype(float))
    assert np.mean(diff) < 10


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_replace_without_replacement_image(
    inpainter,
    test_image_bytes,
    test_bbox
):
    """Test REPLACE mode fails without replacement image"""
    with pytest.raises(ValueError, match="replacement_image_bytes required"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            bbox=test_bbox,
            mode=InpaintMode.REPLACE,
            track_metrics=False
        )


# ==================== VALIDATION TESTS ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_no_mask_no_bbox(inpainter, test_image_bytes):
    """Test inpaint fails without mask or bbox"""
    with pytest.raises(ValueError, match="Either mask_bytes or bbox must be provided"):
        await inpainter.inpaint(
            image_bytes=test_image_bytes,
            mode=InpaintMode.REMOVE,
            track_metrics=False
        )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_calculate_metrics(inpainter, test_image_bytes, test_bbox):
    """Test metrics calculation"""
    metrics = await inpainter._calculate_metrics(
        test_image_bytes,
        None,
        test_bbox,
        100.5,
        InpaintMode.REMOVE
    )
    
    assert metrics['processing_time_ms'] == 100.5
    assert metrics['image_size'] == (640, 480)
    expected_size = (test_bbox['x2'] - test_bbox['x1']) * (test_bbox['y2'] - test_bbox['y1'])
    assert metrics['mask_size_pixels'] == expected_size
    assert metrics['mode'] == 'remove'


# ==================== EDGE CASES ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_small_bbox(inpainter, test_image_bytes, test_replacement_bytes):
    """Test with very small bbox"""
    small_bbox = {'x1': 10, 'y1': 10, 'x2': 20, 'y2': 20}
    
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=small_bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes,
        track_metrics=False
    )
    
    assert 'result_bytes' in result
    result_img = Image.open(BytesIO(result['result_bytes']))
    assert result_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_inpaint_large_bbox(inpainter, test_image_bytes, test_replacement_bytes):
    """Test with bbox covering entire image"""
    large_bbox = {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 480}
    
    result = await inpainter.inpaint(
        image_bytes=test_image_bytes,
        bbox=large_bbox,
        mode=InpaintMode.REPLACE,
        replacement_image_bytes=test_replacement_bytes,
        track_metrics=False
    )
    
    assert 'result_bytes' in result