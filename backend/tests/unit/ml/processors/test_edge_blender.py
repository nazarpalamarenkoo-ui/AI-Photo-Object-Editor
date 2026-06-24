import pytest
import numpy as np
from PIL import Image
from io import BytesIO
 
from app.ml.processors.edge_blender import EdgeBlender
 
 
@pytest.fixture
def edge_blender():
    """EdgeBlender instance"""
    return EdgeBlender()
 
 
@pytest.fixture
def test_image_bytes():
    """Create test image 640x480"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()
 
 
@pytest.fixture
def test_mask_bytes():
    """Create test mask with white square in center"""
    mask = Image.new('L', (640, 480), color=0)
    mask_array = np.array(mask)
    # White square 200x200 in center
    mask_array[140:340, 220:420] = 255
    mask_img = Image.fromarray(mask_array)
    
    buffer = BytesIO()
    mask_img.save(buffer, format='PNG')
    return buffer.getvalue()
 
 
@pytest.fixture
def modified_image_bytes():
    """Create modified image (different color in center)"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    img_array = np.array(img)
    # Red square in center
    img_array[140:340, 220:420] = [255, 0, 0]
    modified_img = Image.fromarray(img_array)
    
    buffer = BytesIO()
    modified_img.save(buffer, format='JPEG')
    return buffer.getvalue()
 

@pytest.mark.unit
def test_edge_blender_init(edge_blender):
    """Test EdgeBlender initialization"""
    assert edge_blender is not None
 
 
@pytest.mark.unit
def test_feather_mask(edge_blender):
    """Test _feather_mask method"""
    # Create binary mask
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    
    # Feather with gaussian
    feathered = edge_blender._feather_mask(mask, feather_radius=5, blur_method='gaussian')
    
    assert feathered.shape == mask.shape
    assert feathered.dtype == np.float32
    assert 0.0 <= feathered.min() <= feathered.max() <= 1.0
    
    # Center should still be ~1.0
    assert feathered[50, 50] > 0.9
    
    # Edges should be blurred (not 0 or 1)
    assert 0.1 < feathered[25, 50] < 0.9  # Top edge
 
 
@pytest.mark.unit
def test_feather_mask_box_blur(edge_blender):
    """Test _feather_mask with box blur"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    
    feathered = edge_blender._feather_mask(mask, feather_radius=5, blur_method='box')
    
    assert feathered.shape == mask.shape
    assert 0.0 <= feathered.min() <= feathered.max() <= 1.0
 
 
@pytest.mark.unit
def test_feather_mask_invalid_method(edge_blender: EdgeBlender):
    """Test _feather_mask with invalid method"""
    mask = np.zeros((100, 100), dtype=np.float32)
    
    with pytest.raises(ValueError, match="Unknown blur_method"):
        edge_blender._feather_mask(mask, feather_radius=5, blur_method='invalid')
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_blend_edges(edge_blender, test_image_bytes, modified_image_bytes, test_mask_bytes):
    """Test blend_edges method"""
    result = await edge_blender.blend_edges(
        original_image_bytes=test_image_bytes,
        processed_image_bytes=modified_image_bytes,
        mask_bytes=test_mask_bytes,
        feather_radius=10
    )
    
    assert isinstance(result, bytes)
    
    # Check result is valid image
    result_img = Image.open(BytesIO(result))
    assert result_img.size == (640, 480)
    assert result_img.mode == 'RGB'
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_blend_edges_preserves_size(edge_blender, test_image_bytes, modified_image_bytes, test_mask_bytes):
    """Test that blend_edges preserves image size"""
    result = await edge_blender.blend_edges(
        original_image_bytes=test_image_bytes,
        processed_image_bytes=modified_image_bytes,
        mask_bytes=test_mask_bytes,
        feather_radius=5
    )
    
    original_img = Image.open(BytesIO(test_image_bytes))
    result_img = Image.open(BytesIO(result))
    
    assert result_img.size == original_img.size
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_blend(edge_blender, test_image_bytes, modified_image_bytes, test_mask_bytes):
    """Test auto_blend method"""
    # Test with different expand_mask_pixels values
    for expand_pixels in [0, 5, 10]:
        result = await edge_blender.auto_blend(
            original_image_bytes=test_image_bytes,
            processed_image_bytes=modified_image_bytes,
            mask_bytes=test_mask_bytes,
            expand_mask_pixels=expand_pixels
        )
        
        assert isinstance(result, bytes)
        result_img = Image.open(BytesIO(result))
        assert result_img.size == (640, 480)
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_blend_edges_different_feather_radius(edge_blender, test_image_bytes, modified_image_bytes, test_mask_bytes):
    """Test blend_edges with different feather radius values"""
    for radius in [5, 10, 20]:
        result = await edge_blender.blend_edges(
            original_image_bytes=test_image_bytes,
            processed_image_bytes=modified_image_bytes,
            mask_bytes=test_mask_bytes,
            feather_radius=radius
        )
        
        assert isinstance(result, bytes)
 
 
@pytest.mark.unit
def test_edge_blender_singleton():
    """Test singleton pattern"""
    from app.ml.processors.edge_blender import get_edge_blender
    
    blender1 = get_edge_blender()
    blender2 = get_edge_blender()
    
    assert blender1 is blender2