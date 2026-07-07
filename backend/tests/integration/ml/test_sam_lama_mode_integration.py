from io import BytesIO
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="cv2 required for real mask dilation")

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MODULE_PATH = "app.ml.modes.sam_lama_mode"


def _rgb_png(size: Tuple[int, int], color=(100, 100, 100)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(size: Tuple[int, int], box) -> bytes:
    arr = np.zeros(size[::-1], dtype=np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 255
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _rgba_png(size: Tuple[int, int], color=(200, 30, 30, 255)) -> bytes:
    buf = BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _rgba_cutout(size: Tuple[int, int], color=(0, 200, 0, 255)) -> bytes:
    """A pre-cut RGBA asset (e.g. from an asset library) with no transparency to strip."""
    buf = BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def module():
    return pytest.importorskip(MODULE_PATH)


@pytest.fixture
def real_edge_blender(module):
    try:
        return module.get_edge_blender()
    except Exception as exc:
        pytest.skip(f"could not construct real EdgeBlender: {exc}")


@pytest.fixture
def real_color_matcher(module):
    try:
        return module.get_color_matcher()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not construct real ColorMatcher: {exc}")


@pytest.fixture
def real_compositor(module):
    try:
        return module.get_compositor()
    except Exception as exc:
        pytest.skip(f"could not construct real compositor: {exc}")


@pytest.fixture
def stub_segmentor():
    seg = MagicMock()
    seg.segment_auto = AsyncMock(return_value={"segments": [], "metrics": {}})
    seg.segment_with_prompt = AsyncMock(return_value={"segments": [], "metrics": {}})
    return seg


@pytest.fixture
def stub_inpainter():
    inp = MagicMock()

    async def fake_inpaint(image_bytes, mask_bytes, mode, track_metrics=True, **kwargs):
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
        arr = np.array(img)
        mask_arr = np.array(mask)
        arr[mask_arr > 0] = (180, 180, 180)
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        return {"result_bytes": buf.getvalue(), "metrics": {"fake_inpaint": True}}

    inp.inpaint = AsyncMock(side_effect=fake_inpaint)
    return inp


@pytest.fixture
def stub_background_remover():
    br = MagicMock()

    async def fake_remove_and_resize(image_bytes, size):
        img = Image.open(BytesIO(image_bytes)).convert("RGBA").resize(size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    br.remove_and_resize = AsyncMock(side_effect=fake_remove_and_resize)
    return br


@pytest.fixture
def mode(
    module,
    stub_segmentor,
    stub_inpainter,
    stub_background_remover,
    real_edge_blender,
    real_color_matcher,
    real_compositor,
    monkeypatch,
):
    monkeypatch.setattr(module, "get_compositor", lambda: real_compositor)

    return module.SAMLamaMode(
        segmentor=stub_segmentor,
        inpainter=stub_inpainter,
        edge_blender=real_edge_blender,
        color_matcher=real_color_matcher,
        background_remover=stub_background_remover,
        device="cpu",
    )

async def test_remove_object_end_to_end_returns_correct_size(mode):
    image_bytes = _rgb_png((100, 100), color=(50, 60, 70))
    mask_bytes = _mask_png((100, 100), box=(30, 30, 70, 70))

    result = await mode.remove_object(
        image_bytes, mask_bytes, use_edge_blending=True, expand_mask_pixels=10
    )

    decoded = Image.open(BytesIO(result["result_bytes"]))
    assert decoded.size == (100, 100)


async def test_remove_object_real_dilation_expands_inpaint_region(mode):
    image_bytes = _rgb_png((100, 100), color=(10, 10, 10))
    mask_bytes = _mask_png((100, 100), box=(40, 40, 60, 60))  # 20x20 = 400px

    dilated_used = {}
    original_dilate = mode._dilate_mask

    async def spy_dilate(mb, px):
        out = await original_dilate(mb, px)
        dilated_used["mask"] = out
        return out

    mode._dilate_mask = spy_dilate

    await mode.remove_object(image_bytes, mask_bytes, expand_mask_pixels=10, use_edge_blending=False)

    dilated_arr = np.array(Image.open(BytesIO(dilated_used["mask"])).convert("L"))
    assert (dilated_arr > 0).sum() > 400


async def test_remove_object_without_dilation_uses_original_mask_area(mode):
    image_bytes = _rgb_png((100, 100))
    mask_bytes = _mask_png((100, 100), box=(40, 40, 60, 60))

    result = await mode.remove_object(
        image_bytes, mask_bytes, expand_mask_pixels=0, use_edge_blending=False
    )

    decoded = np.array(Image.open(BytesIO(result["result_bytes"])).convert("RGB"))
    # the fake inpainter flat-fills masked pixels to (180,180,180)
    filled = (decoded == np.array([180, 180, 180])).all(axis=-1)
    assert filled.sum() > 0


async def test_replace_object_end_to_end_produces_correct_canvas_size(mode):
    image_bytes = _rgb_png((120, 120), color=(20, 20, 20))
    mask_bytes = _mask_png((120, 120), box=(40, 40, 80, 80))
    replacement_bytes = _rgb_png((50, 50), color=(0, 200, 0))
    bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

    result = await mode.replace_object(
        image_bytes, mask_bytes, bbox, replacement_bytes,
        use_color_matching=True, use_edge_blending=False,
    )

    decoded = Image.open(BytesIO(result["result_bytes"]))
    assert decoded.size == (120, 120)


async def test_replace_object_resized_replacement_matches_bbox_dimensions(mode):
    image_bytes = _rgb_png((120, 120))
    mask_bytes = _mask_png((120, 120), box=(10, 10, 50, 70))
    replacement_bytes = _rgb_png((10, 10))
    bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 70}  # 40 x 60

    captured = {}
    original = mode.background_remover.remove_and_resize

    async def spy(img, size):
        captured["size"] = size
        return await original(img, size)

    mode.background_remover.remove_and_resize = spy

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_bytes, use_color_matching=False)

    assert captured["size"] == (40, 60)

async def test_extract_then_paste_round_trip_preserves_object_silhouette(mode):
    canvas = np.full((100, 100, 3), 15, dtype=np.uint8)

    canvas[20:50, 20:50] = (220, 40, 40)

    buffer = BytesIO()
    Image.fromarray(canvas).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    mask_bytes = _mask_png(
        (100, 100),
        box=(20, 20, 50, 50),
    )

    bbox = {
        "x1": 20,
        "y1": 20,
        "x2": 50,
        "y2": 50,
    }
    extracted = await mode.extract_object(
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        bbox=bbox,
        padding_pixels=0,
    )
    assert extracted["area_pixels"] == 30 * 30
    target_bbox = {
        "x1": 60,
        "y1": 60,
        "x2": 90,
        "y2": 90,
    }

    pasted = await mode.paste_extracted_object(
        image_bytes=image_bytes,
        extracted_bytes=extracted["extracted_bytes"],
        target_bbox=target_bbox,
        use_color_matching=False,
        use_edge_blending=True,
    )
    decoded = np.array(
        Image.open(BytesIO(pasted["result_bytes"])).convert("RGB")
    )
    assert decoded.shape[:2] == (100, 100)
    region = decoded[60:90, 60:90]
    assert not (region == np.array([15, 15, 15])).all()

    assert region[..., 0].max() > 150
    assert region[..., 1].max() < 100
    assert region[..., 2].max() < 100


async def test_extract_object_alpha_then_alpha_to_mask_consistent_area(mode):
    image_bytes = _rgb_png((80, 80))
    mask_bytes = _mask_png((80, 80), box=(10, 10, 30, 30))  # 20x20
    bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}

    extracted = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=0)

    paste_bbox = {"x1": 40, "y1": 40, "x2": 60, "y2": 60}
    mask_out = await mode._alpha_to_mask(
        extracted_bytes=extracted["extracted_bytes"],
        paste_bbox=paste_bbox,
        object_size=(20, 20),
        canvas_size=(80, 80),
    )

    arr = np.array(Image.open(BytesIO(mask_out)).convert("L"))
    assert (arr > 0).sum() == pytest.approx(20 * 20, abs=5)


async def test_segment_objects_reports_real_image_dimensions(mode, stub_segmentor):
    stub_segmentor.segment_auto = AsyncMock(return_value={
        "segments": [{"area": 999, "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}],
        "metrics": {},
    })
    image_bytes = _rgb_png((333, 217))

    result = await mode.segment_objects(image_bytes, min_area=0)

    assert result["image_size"] == (333, 217)


# ---------------------------------------------------------------------------
# replace_object with replacement_is_cutout=True (pre-cut RGBA asset, no rembg)
# ---------------------------------------------------------------------------

async def test_replace_object_with_cutout_skips_background_remover(mode):
    image_bytes = _rgb_png((120, 120), color=(20, 20, 20))
    mask_bytes = _mask_png((120, 120), box=(40, 40, 80, 80))
    cutout_bytes = _rgba_cutout((50, 50))
    bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

    await mode.replace_object(
        image_bytes, mask_bytes, bbox, cutout_bytes,
        use_color_matching=False, use_edge_blending=False,
        replacement_is_cutout=True,
    )

    mode.background_remover.remove_and_resize.assert_not_called()


async def test_replace_object_with_cutout_resizes_to_bbox_dimensions(mode):
    image_bytes = _rgb_png((120, 120), color=(20, 20, 20))
    mask_bytes = _mask_png((120, 120), box=(10, 10, 50, 70))
    cutout_bytes = _rgba_cutout((10, 10))
    bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 70}  # 40 x 60

    captured = {}
    original = mode._resize_rgba_to_bbox

    def spy(img_bytes, size):
        captured["size"] = size
        return original(img_bytes, size)

    mode._resize_rgba_to_bbox = spy

    await mode.replace_object(
        image_bytes, mask_bytes, bbox, cutout_bytes,
        use_color_matching=False, replacement_is_cutout=True,
    )

    assert captured["size"] == (40, 60)


async def test_replace_object_with_cutout_places_color_at_bbox_location(mode):
    image_bytes = _rgb_png((120, 120), color=(20, 20, 20))
    mask_bytes = _mask_png((120, 120), box=(40, 40, 80, 80))
    cutout_bytes = _rgba_cutout((50, 50), color=(0, 220, 0, 255))
    bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

    result = await mode.replace_object(
        image_bytes, mask_bytes, bbox, cutout_bytes,
        use_color_matching=False, use_edge_blending=False,
        replacement_is_cutout=True,
    )

    decoded = np.array(Image.open(BytesIO(result["result_bytes"])).convert("RGB"))
    region = decoded[40:80, 40:80]
    assert region[..., 1].mean() > region[..., 0].mean()
    assert region[..., 1].mean() > region[..., 2].mean()


async def test_replace_object_with_cutout_end_to_end_canvas_size(mode):
    image_bytes = _rgb_png((100, 100), color=(30, 30, 30))
    mask_bytes = _mask_png((100, 100), box=(20, 20, 60, 60))
    cutout_bytes = _rgba_cutout((40, 40), color=(255, 0, 0, 255))
    bbox = {"x1": 20, "y1": 20, "x2": 60, "y2": 60}

    result = await mode.replace_object(
        image_bytes, mask_bytes, bbox, cutout_bytes,
        use_color_matching=True, use_edge_blending=False,
        replacement_is_cutout=True,
    )

    decoded = Image.open(BytesIO(result["result_bytes"]))
    assert decoded.size == (100, 100)


async def test_replace_object_default_still_uses_background_remover(mode):
    """Sanity check: replacement_is_cutout defaults to False, preserving old behavior."""
    image_bytes = _rgb_png((120, 120))
    mask_bytes = _mask_png((120, 120), box=(40, 40, 80, 80))
    replacement_bytes = _rgb_png((50, 50), color=(0, 200, 0))
    bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_bytes)

    mode.background_remover.remove_and_resize.assert_called_once()