import pytest
import numpy as np
from PIL import Image
from io import BytesIO

from app.ml.processors.polygon_mask import PolygonMasker


@pytest.fixture
def polygon_masker():
    """PolygonMasker instance"""
    return PolygonMasker()


@pytest.fixture
def square_points():
    """Simple square polygon, 4 points, ordered clockwise"""
    return [(100, 100), (300, 100), (300, 300), (100, 300)]


@pytest.fixture
def triangle_points():
    """Simple triangle polygon, 3 points"""
    return [(50, 50), (250, 50), (150, 250)]


@pytest.mark.unit
def test_polygon_masker_init(polygon_masker):
    """Test PolygonMasker initialization"""
    assert polygon_masker is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_returns_bytes(polygon_masker, square_points):
    """Test generate_mask returns valid PNG bytes"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
    )

    assert isinstance(result, bytes)

    mask_img = Image.open(BytesIO(result))
    assert mask_img.mode == 'L'
    assert mask_img.size == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_too_few_points(polygon_masker):
    """Test generate_mask raises ValueError with fewer than 3 points"""
    with pytest.raises(ValueError, match="Need at least 3 points"):
        await polygon_masker.generate_mask(
            image_size=(640, 480),
            points=[(10, 10), (20, 20)],
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_zero_points(polygon_masker):
    """Test generate_mask raises ValueError with empty points list"""
    with pytest.raises(ValueError, match="Need at least 3 points"):
        await polygon_masker.generate_mask(
            image_size=(640, 480),
            points=[],
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_triangle_no_smoothing(polygon_masker, triangle_points):
    """Test generate_mask with exactly 3 points (smoothing disabled internally)"""
    result = await polygon_masker.generate_mask(
        image_size=(300, 300),
        points=triangle_points,
        smooth=True,  # requested, but should fall back to polyline since <4 points
    )

    mask_img = Image.open(BytesIO(result))
    mask_array = np.array(mask_img)

    # Centroid of the triangle should be inside the mask
    assert mask_array[150, 150] == 255

    # Far corner outside the triangle should be outside the mask
    assert mask_array[290, 290] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_smooth_vs_polyline(polygon_masker, square_points):
    """Test generate_mask produces a mask for both smooth and non-smooth paths"""
    smooth_result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=True,
    )
    polyline_result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=False,
    )

    smooth_array = np.array(Image.open(BytesIO(smooth_result)))
    polyline_array = np.array(Image.open(BytesIO(polyline_result)))

    # Both should mark the center of the square as inside
    assert smooth_array[200, 200] == 255
    assert polyline_array[200, 200] == 255

    # Both should leave the top-left corner of the image outside
    assert smooth_array[0, 0] == 0
    assert polyline_array[0, 0] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_content_inside_and_outside(polygon_masker, square_points):
    """Test that generate_mask correctly marks inside vs outside pixels"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=False,
    )

    mask_array = np.array(Image.open(BytesIO(result)))

    # Well inside the square (100,100)-(300,300)
    assert mask_array[200, 200] == 255

    # Well outside the square
    assert mask_array[400, 400] == 0
    assert mask_array[0, 0] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_no_feathering_is_binary(polygon_masker, square_points):
    """Test that with feather_px=0 the mask only contains 0 and 255 values"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=False,
        feather_px=0,
    )

    mask_array = np.array(Image.open(BytesIO(result)))
    unique_values = set(np.unique(mask_array).tolist())
    assert unique_values.issubset({0, 255})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_feathering_creates_gradient(polygon_masker, square_points):
    """Test that feather_px > 0 introduces intermediate values near edges"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=False,
        feather_px=10,
    )

    mask_array = np.array(Image.open(BytesIO(result)))
    unique_values = np.unique(mask_array)

    # Feathering should introduce values other than pure 0/255
    assert len(unique_values) > 2

    # Center should remain fully inside
    assert mask_array[200, 200] > 200

    # Far outside should remain fully outside
    assert mask_array[0, 0] < 50


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_clips_out_of_bounds_points(polygon_masker):
    """Test that points outside image bounds are clipped instead of raising"""
    points = [(-50, -50), (700, -50), (700, 500), (-50, 500)]

    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=points,
        smooth=False,
    )

    mask_img = Image.open(BytesIO(result))
    assert mask_img.size == (640, 480)

    mask_array = np.array(mask_img)
    # Since the clipped polygon covers essentially the whole image,
    # the center should be inside
    assert mask_array[240, 320] == 255


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_num_smooth_points_param(polygon_masker, square_points):
    """Test that generate_mask accepts a custom num_smooth_points without error"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=True,
        num_smooth_points=50,
    )

    assert isinstance(result, bytes)
    mask_img = Image.open(BytesIO(result))
    assert mask_img.size == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_smoothing_factor_param(polygon_masker, square_points):
    """Test that generate_mask accepts a nonzero smoothing_factor without error"""
    result = await polygon_masker.generate_mask(
        image_size=(640, 480),
        points=square_points,
        smooth=True,
        smoothing_factor=5.0,
    )

    assert isinstance(result, bytes)
    mask_img = Image.open(BytesIO(result))
    assert mask_img.size == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_mask_different_image_sizes(polygon_masker, square_points):
    """Test generate_mask with various image sizes"""
    for size in [(100, 100), (640, 480), (1920, 1080)]:
        result = await polygon_masker.generate_mask(
            image_size=size,
            points=square_points,
            smooth=False,
        )
        mask_img = Image.open(BytesIO(result))
        assert mask_img.size == size


@pytest.mark.unit
def test_polygon_masker_singleton():
    """Test singleton pattern"""
    from app.ml.processors.polygon_mask import get_polygon_masker

    masker1 = get_polygon_masker()
    masker2 = get_polygon_masker()

    assert masker1 is masker2