"""
Unit Tests for BackgroundRemover

Location: tests/unit/test_background_remover.py
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from rembg import remove
from app.ml.processors.background_remover import BackgroundRemover


# ==================== FIXTURES ====================

@pytest.fixture
def background_remover():
    """BackgroundRemover instance (without rembg)"""
    return BackgroundRemover(rembg_available=False)


@pytest.fixture
def test_image_bytes():
    """Create test image 640x480"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_mask_bytes():
    """Create test mask with white object in center"""
    mask = Image.new('L', (640, 480), color=0)
    mask_array = np.array(mask)
    # White circle in center
    mask_array[140:340, 220:420] = 255
    mask_img = Image.fromarray(mask_array)
    
    buffer = BytesIO()
    mask_img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def test_bbox():
    """Test bbox"""
    return {'x1': 220, 'y1': 140, 'x2': 420, 'y2': 340}


# ==================== UNIT TESTS ====================

@pytest.mark.unit
def test_background_remover_init(background_remover):
    """Test BackgroundRemover initialization"""
    assert background_remover is not None
    assert background_remover.rembg_available == False


@pytest.mark.unit
def test_rgba_to_rgb_white_bg(background_remover):
    """Test _rgba_to_rgb_white_bg method"""
    # Create RGBA image with transparency
    rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    
    rgb_img = background_remover._rgba_to_rgb_white_bg(rgba_img)
    
    assert rgb_img.mode == 'RGB'
    assert rgb_img.size == rgba_img.size


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_with_mask_png(background_remover, test_image_bytes, test_mask_bytes):
    """Test remove_with_mask method with PNG output"""
    result = await background_remover.remove_with_mask(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        return_format='png'
    )
    
    assert isinstance(result, bytes)
    
    # Check result is valid PNG
    result_img = Image.open(BytesIO(result))
    assert result_img.format == 'PNG'
    assert result_img.mode == 'RGBA'
    assert result_img.size == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_with_mask_jpeg(background_remover, test_image_bytes, test_mask_bytes):
    """Test remove_with_mask method with JPEG output"""
    result = await background_remover.remove_with_mask(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        return_format='jpeg'
    )
    
    assert isinstance(result, bytes)
    
    # Check result is valid JPEG
    result_img = Image.open(BytesIO(result))
    assert result_img.format == 'JPEG'
    assert result_img.mode == 'RGB'
    assert result_img.size == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_with_mask_inverted(background_remover, test_image_bytes, test_mask_bytes):
    """Test remove_with_mask with inverted mask"""
    result = await background_remover.remove_with_mask(
        image_bytes=test_image_bytes,
        mask_bytes=test_mask_bytes,
        return_format='png',
        invert_mask=True
    )
    
    assert isinstance(result, bytes)
    result_img = Image.open(BytesIO(result))
    assert result_img.mode == 'RGBA'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_background_threshold(background_remover, test_image_bytes):
    """Test remove_background with threshold method"""
    result = await background_remover.remove_background(
        image_bytes=test_image_bytes,
        method='threshold',
        return_format='png'
    )
    
    assert isinstance(result, bytes)
    result_img = Image.open(BytesIO(result))
    assert result_img.mode == 'RGBA'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_background_grabcut(background_remover, test_image_bytes, test_bbox):
    """Test remove_background with grabcut method"""
    result = await background_remover.remove_background(
        image_bytes=test_image_bytes,
        method='grabcut',
        return_format='png',
        bbox=test_bbox
    )
    
    assert isinstance(result, bytes)
    result_img = Image.open(BytesIO(result))
    assert result_img.mode == 'RGBA'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_background_grabcut_no_bbox(background_remover, test_image_bytes):
    """Test remove_background with grabcut method without bbox"""
    result = await background_remover.remove_background(
        image_bytes=test_image_bytes,
        method='grabcut',
        return_format='png',
        bbox=None
    )
    
    assert isinstance(result, bytes)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_background_rembg_not_available(background_remover, test_image_bytes):
    """Test remove_background with rembg when not available"""
    with pytest.raises(ValueError, match="rembg not available"):
        await background_remover.remove_background(
            image_bytes=test_image_bytes,
            method='rembg'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_background_invalid_method(background_remover, test_image_bytes):
    """Test remove_background with invalid method"""
    with pytest.raises(ValueError, match="Unknown method"):
        await background_remover.remove_background(
            image_bytes=test_image_bytes,
            method='invalid_method'
        )


@pytest.mark.unit
def test_background_remover_singleton():
    """Test singleton pattern"""
    from app.ml.processors.background_remover import get_background_remover
    
    # Reset singleton
    import app.ml.processors.background_remover
    app.ml.processors.background_remover._background_remover_instance = None
    
    remover1 = get_background_remover(rembg_available=False)
    remover2 = get_background_remover(rembg_available=False)
    
    assert remover1 is remover2
    
    # Cleanup
    app.ml.processors.background_remover._background_remover_instance = None