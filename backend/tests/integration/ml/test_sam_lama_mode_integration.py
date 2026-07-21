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
def real_polygon_masker(module):
    try:
        return module.get_polygon_masker()
    except Exception as exc:
        pytest.skip(f"could not construct real PolygonMasker: {exc}")


@pytest.fixture
def stub_segmentor():
    seg = MagicMock(name="stub_segmentor")
    seg.segment_auto = AsyncMock(return_value={"segments": [], "metrics": {}})
    seg.segment_with_prompt = AsyncMock(return_value={"segments": [], "metrics": {}})
    seg.segment_with_prompts_batch = AsyncMock(return_value={"segments": [], "metrics": {}})
    return seg


@pytest.fixture
def stub_inpainter():
    inp = MagicMock(name="stub_inpainter")

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
    br = MagicMock(name="stub_background_remover")

    async def fake_remove_and_resize(image_bytes, size):
        img = Image.open(BytesIO(image_bytes)).convert("RGBA").resize(size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    br.remove_and_resize = AsyncMock(side_effect=fake_remove_and_resize)
    return br


@pytest.fixture(autouse=True)
def reset_singleton_and_patch_heavy_deps(
    module,
    monkeypatch,
    stub_segmentor,
    stub_inpainter,
    stub_background_remover,
    real_edge_blender,
    real_color_matcher,
    real_compositor,
    real_polygon_masker,
):
    monkeypatch.setattr(module, "_sam_mode_instance", None)
    monkeypatch.setattr(module.DeviceManager, "get", staticmethod(lambda x: "cpu"))

    monkeypatch.setattr(module, "get_segmentor", MagicMock(return_value=stub_segmentor))
    monkeypatch.setattr(module, "get_inpainter", MagicMock(return_value=stub_inpainter))
    monkeypatch.setattr(
        module, "get_background_remover",
        MagicMock(return_value=stub_background_remover),
    )
    monkeypatch.setattr(module, "get_edge_blender", lambda: real_edge_blender)
    monkeypatch.setattr(module, "get_color_matcher", lambda: real_color_matcher)
    monkeypatch.setattr(module, "get_compositor", lambda: real_compositor)
    monkeypatch.setattr(module, "get_polygon_masker", lambda: real_polygon_masker)


class TestGetSamModeSingletonIntegration:
    def test_returns_same_instance_across_calls(self, module):
        first = module.get_sam_mode()
        second = module.get_sam_mode()

        assert first is second

    def test_builds_heavy_dependencies_exactly_once(self, module):
        module.get_sam_mode()
        module.get_sam_mode()
        module.get_sam_mode()

        assert module.get_segmentor.call_count == 1
        assert module.get_inpainter.call_count == 1

    def test_wires_real_edge_blender_color_matcher_and_compositor(
        self, module, real_edge_blender, real_color_matcher, real_compositor
    ):
        instance = module.get_sam_mode()

        assert instance.edge_blender is real_edge_blender
        assert instance.color_matcher is real_color_matcher
        assert instance.compositor is real_compositor

    async def test_singleton_instance_remove_object_end_to_end(self, module):
        """Exercises the actual singleton-built pipeline: real dilation, real edge blending, on a small synthetic image."""
        instance = module.get_sam_mode()

        image_bytes = _rgb_png((100, 100), color=(50, 60, 70))
        mask_bytes = _mask_png((100, 100), box=(30, 30, 70, 70))

        result = await instance.remove_object(
            image_bytes, mask_bytes, use_edge_blending=True, expand_mask_pixels=10
        )

        decoded = Image.open(BytesIO(result["result_bytes"]))
        assert decoded.size == (100, 100)

    async def test_singleton_instance_replace_object_cutout_end_to_end(self, module):
        instance = module.get_sam_mode()

        image_bytes = _rgb_png((100, 100), color=(30, 30, 30))
        mask_bytes = _mask_png((100, 100), box=(20, 20, 60, 60))
        cutout_bytes = BytesIO()
        Image.new("RGBA", (40, 40), (255, 0, 0, 255)).save(cutout_bytes, format="PNG")
        bbox = {"x1": 20, "y1": 20, "x2": 60, "y2": 60}

        result = await instance.replace_object(
            image_bytes, mask_bytes, bbox, cutout_bytes.getvalue(),
            use_color_matching=True, use_edge_blending=False,
            replacement_is_cutout=True,
        )

        decoded = Image.open(BytesIO(result["result_bytes"]))
        assert decoded.size == (100, 100)
        instance.background_remover.remove_and_resize.assert_not_called()