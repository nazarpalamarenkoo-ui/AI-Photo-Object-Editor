import pytest
import numpy as np
from PIL import Image
from io import BytesIO
 
from app.ml.processors.color_matcher import ColorMatcher
 
 
# ==================== FIXTURES ====================
 
@pytest.fixture
def color_matcher():
    """ColorMatcher instance"""
    return ColorMatcher()
 
 
@pytest.fixture
def test_image_with_bbox():
    """
    Create image with different colors:
    - Background (context): blue
    - Object in bbox: red
    """
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))  # Blue background
    img_array = np.array(img)
    
    # Red object in center
    bbox = {'x1': 220, 'y1': 140, 'x2': 420, 'y2': 340}
    img_array[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']] = [255, 50, 50]  # Red
    
    result_img = Image.fromarray(img_array)
    buffer = BytesIO()
    result_img.save(buffer, format='JPEG')
    
    return buffer.getvalue(), bbox
 
 
# ==================== UNIT TESTS ====================
 
@pytest.mark.unit
def test_color_matcher_init(color_matcher):
    """Test ColorMatcher initialization"""
    assert color_matcher is not None
 
 
@pytest.mark.unit
def test_extract_context(color_matcher):
    """Test _extract_context method"""
    # Create test image
    img_array = np.ones((480, 640, 3), dtype=np.float32) * 100
    
    bbox = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 300}
    
    context = color_matcher._extract_context(
        img_array,
        bbox,
        margin=20,
        width=640,
        height=480
    )
    
    # Context should be (N, 3) where N > 0
    assert context.ndim == 2
    assert context.shape[1] == 3
    assert context.shape[0] > 0
 
 
@pytest.mark.unit
def test_match_mean_std(color_matcher):
    """Test _match_mean_std method"""
    # Source: high mean, low std
    source = np.ones((100, 100, 3), dtype=np.float32) * 200
    
    # Target context: low mean, high std
    target_context = np.random.randint(50, 100, (1000, 3)).astype(np.float32)
    
    matched = color_matcher._match_mean_std(source, target_context)
    
    assert matched.shape == source.shape
    
    # Matched should have similar mean to target
    matched_mean = np.mean(matched, axis=(0, 1))
    target_mean = np.mean(target_context, axis=0)
    
    # Should be close (within 10%)
    assert np.allclose(matched_mean, target_mean, rtol=0.1)
 
 
@pytest.mark.unit
def test_match_histogram(color_matcher):
    """Test _match_histogram method"""
    source = np.ones((100, 100, 3), dtype=np.float32) * 150
    target_context = np.ones((1000, 3), dtype=np.float32) * 100
    
    matched = color_matcher._match_histogram(source, target_context)
    
    assert matched.shape == source.shape
    assert matched.dtype == np.float32
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_colors_mean_std(color_matcher, test_image_with_bbox):
    """Test match_colors with mean_std method"""
    image_bytes, bbox = test_image_with_bbox
    
    result = await color_matcher.match_colors(
        image_with_replacement_bytes=image_bytes,
        bbox=bbox,
        method='mean_std',
        context_margin=20
    )
    
    assert isinstance(result, bytes)
    
    # Check result is valid image
    result_img = Image.open(BytesIO(result))
    assert result_img.size == (640, 480)
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_colors_histogram(color_matcher, test_image_with_bbox):
    """Test match_colors with histogram method"""
    image_bytes, bbox = test_image_with_bbox
    
    result = await color_matcher.match_colors(
        image_with_replacement_bytes=image_bytes,
        bbox=bbox,
        method='histogram',
        context_margin=20
    )
    
    assert isinstance(result, bytes)
    result_img = Image.open(BytesIO(result))
    assert result_img.size == (640, 480)
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_colors_color_transfer(color_matcher, test_image_with_bbox):
    """Test match_colors with color_transfer method"""
    image_bytes, bbox = test_image_with_bbox
    
    result = await color_matcher.match_colors(
        image_with_replacement_bytes=image_bytes,
        bbox=bbox,
        method='color_transfer',
        context_margin=20
    )
    
    assert isinstance(result, bytes)
    result_img = Image.open(BytesIO(result))
    assert result_img.size == (640, 480)
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_colors_invalid_method(color_matcher, test_image_with_bbox):
    """Test match_colors with invalid method"""
    image_bytes, bbox = test_image_with_bbox
    
    with pytest.raises(ValueError, match="Unknown method"):
        await color_matcher.match_colors(
            image_with_replacement_bytes=image_bytes,
            bbox=bbox,
            method='invalid_method'
        )
 
 
@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_colors_different_margins(color_matcher, test_image_with_bbox):
    """Test match_colors with different context margins"""
    image_bytes, bbox = test_image_with_bbox
    
    for margin in [10, 20, 30]:
        result = await color_matcher.match_colors(
            image_with_replacement_bytes=image_bytes,
            bbox=bbox,
            method='mean_std',
            context_margin=margin
        )
        
        assert isinstance(result, bytes)
 
 
@pytest.mark.unit
def test_color_matcher_singleton():
    """Test singleton pattern"""
    from app.ml.processors.color_matcher import get_color_matcher
    
    matcher1 = get_color_matcher()
    matcher2 = get_color_matcher()
    
    assert matcher1 is matcher2
 