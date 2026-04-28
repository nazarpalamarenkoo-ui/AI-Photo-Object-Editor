import pytest
import numpy as np
from PIL import Image
from io import BytesIO

from app.ml.processors.image_compositor import ImageCompositor


def _rgb(color=(255, 255, 255)):
    arr = np.full((100, 100, 3), color, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _rgba(color=(255, 0, 0, 255)):
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[:, :] = color
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


@pytest.mark.unit
def test_compose_with_alpha_crop(compositor):
    bg = _rgb()
    obj = _rgba((255, 0, 0, 0))  # transparent

    bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}

    out = compositor.compose(bg, obj, bbox)

    assert isinstance(out, bytes)


@pytest.mark.unit
def test_soft_edge_blend_shape(compositor):
    clean = np.zeros((100, 100, 3), dtype=np.float32)
    result = clean.copy()

    bbox = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}

    out = compositor._soft_edge_blend(clean, result, bbox)

    assert out.shape == clean.shape