import pytest
import numpy as np
from PIL import Image
from io import BytesIO

from app.ml.processors.image_compositor import ImageCompositor


def _rgb(color=(255, 255, 255), size=(100, 100)):
    arr = np.full((*size, 3), color, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _rgba(color=(255, 0, 0, 255), size=(50, 50)):
    arr = np.zeros((*size, 4), dtype=np.uint8)
    arr[:, :] = color
    buf = BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _rgba_with_fringe(size=(50, 50)):
    arr = np.zeros((*size, 4), dtype=np.uint8)
    arr[:, :] = (255, 0, 0, 255)
    arr[0, :] = (100, 100, 200, 30)   # top fringe — semi-transparent with bg color
    arr[-1, :] = (100, 100, 200, 30)  # bottom fringe
    arr[:, 0] = (100, 100, 200, 30)   # left fringe
    arr[:, -1] = (100, 100, 200, 30)  # right fringe
    buf = BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def compositor():
    return ImageCompositor()



@pytest.mark.unit
def test_compose_basic(compositor):
    bg = _rgb()
    obj = _rgba()
    bbox = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}

    out = compositor.compose(bg, obj, bbox)

    assert isinstance(out, bytes)
    img = Image.open(BytesIO(out))
    assert img.mode == "RGB"


@pytest.mark.unit
def test_compose_output_size_matches_background(compositor):
    bg = _rgb(size=(200, 300))
    obj = _rgba()
    bbox = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}

    out = compositor.compose(bg, obj, bbox)

    img = Image.open(BytesIO(out))
    assert img.size == (300, 200)


@pytest.mark.unit
def test_compose_with_fully_transparent_object(compositor):
    bg = _rgb()
    obj = _rgba(color=(255, 0, 0, 0))
    bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}

    out = compositor.compose(bg, obj, bbox)

    assert isinstance(out, bytes)
    bg_img = np.array(Image.open(BytesIO(_rgb())))
    out_img = np.array(Image.open(BytesIO(out)))
    assert np.mean(np.abs(bg_img.astype(int) - out_img.astype(int))) < 10


@pytest.mark.unit
def test_compose_with_opaque_object_changes_pixels(compositor):
    bg = _rgb(color=(255, 255, 255))
    obj = _rgba(color=(255, 0, 0, 255))
    bbox = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}

    out = compositor.compose(bg, obj, bbox)

    out_img = np.array(Image.open(BytesIO(out)))
    region = out_img[10:60, 10:60]
    assert region[:, :, 0].mean() > 150


@pytest.mark.unit
def test_compose_returns_jpeg_bytes(compositor):
    bg = _rgb()
    obj = _rgba()
    bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}

    out = compositor.compose(bg, obj, bbox)

    assert out[:2] == b'\xff\xd8'


@pytest.mark.unit
def test_compose_bbox_at_origin(compositor):
    bg = _rgb()
    obj = _rgba()
    bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}

    out = compositor.compose(bg, obj, bbox)

    assert isinstance(out, bytes)


@pytest.mark.unit
def test_compose_edge_softness_param_ignored(compositor):
    bg = _rgb()
    obj = _rgba()
    bbox = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}

    out1 = compositor.compose(bg, obj, bbox, edge_softness=0)
    out2 = compositor.compose(bg, obj, bbox, edge_softness=10)

    # edge_softness unused — результат однаковий
    img1 = np.array(Image.open(BytesIO(out1)))
    img2 = np.array(Image.open(BytesIO(out2)))
    assert np.array_equal(img1, img2)


@pytest.mark.unit
def test_clean_alpha_fringe_returns_rgba(compositor):
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[:, :] = (255, 0, 0, 255)
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img)

    assert result.mode == "RGBA"


@pytest.mark.unit
def test_clean_alpha_fringe_preserves_size(compositor):
    arr = np.zeros((60, 80, 4), dtype=np.uint8)
    arr[:, :] = (200, 100, 50, 255)
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img)

    assert result.size == (80, 60)


@pytest.mark.unit
def test_clean_alpha_fringe_erodes_alpha(compositor):
    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    arr[5:15, 5:15] = (255, 0, 0, 255)
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img)

    result_arr = np.array(result)
    assert result_arr[5, 5, 3] == 0
    assert result_arr[14, 14, 3] == 0


@pytest.mark.unit
def test_clean_alpha_fringe_fully_transparent_input(compositor):
    arr = np.zeros((30, 30, 4), dtype=np.uint8)
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img)

    assert result.mode == "RGBA"
    result_arr = np.array(result)
    assert result_arr[:, :, 3].max() == 0


@pytest.mark.unit
def test_clean_alpha_fringe_removes_fringe_colors(compositor):
    arr = np.zeros((30, 30, 4), dtype=np.uint8)
    arr[:, :] = (255, 0, 0, 255)       
    arr[0, :] = (0, 0, 255, 50)        
    arr[:, 0] = (0, 0, 255, 50)         
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img)

    result_arr = np.array(result)
    center = result_arr[10:20, 10:20]
    assert center[:, :, 2].mean() < center[:, :, 0].mean()


@pytest.mark.unit
def test_clean_alpha_fringe_custom_threshold(compositor):
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[:, :] = (100, 200, 50, 255)
    img = Image.fromarray(arr, "RGBA")

    result = compositor._clean_alpha_fringe(img, threshold=50)

    assert result.mode == "RGBA"


@pytest.mark.unit
def test_soft_edge_blend_shape(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = clean.copy()
    bbox = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}

    out = compositor._soft_edge_blend(clean, result, bbox)

    assert out.shape == clean.shape


@pytest.mark.unit
def test_soft_edge_blend_returns_ndarray(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = clean.copy()
    bbox = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}

    out = compositor._soft_edge_blend(clean, result, bbox)

    assert isinstance(out, np.ndarray)


@pytest.mark.unit
def test_soft_edge_blend_identical_inputs(compositor):
    arr = np.full((100, 100, 3), 128, dtype=np.float32)
    bbox = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}

    out = compositor._soft_edge_blend(arr.copy(), arr.copy(), bbox)

    np.testing.assert_allclose(out, arr, atol=1e-4)


@pytest.mark.unit
def test_soft_edge_blend_blends_border_region(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = np.full((100, 100, 3), 255, dtype=np.float32)
    bbox = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}
    edge = 4

    out = compositor._soft_edge_blend(clean, result.copy(), bbox, edge=edge)

    center = out[45:55, 45:55]
    assert center.mean() > 200

    border = out[20:24, 20:80]
    assert 0 < border.mean() < 255


@pytest.mark.unit
def test_soft_edge_blend_does_not_affect_outside_bbox(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = np.full((100, 100, 3), 255, dtype=np.float32)
    bbox = {"x1": 30, "y1": 30, "x2": 70, "y2": 70}

    out = compositor._soft_edge_blend(clean, result.copy(), bbox)

    assert out[0, 0, 0] == 255
    assert out[99, 99, 0] == 255


@pytest.mark.unit
def test_soft_edge_blend_custom_edge_radius(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = np.full((100, 100, 3), 200, dtype=np.float32)
    bbox = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}

    out = compositor._soft_edge_blend(clean, result.copy(), bbox, edge=8)

    assert out.shape == clean.shape