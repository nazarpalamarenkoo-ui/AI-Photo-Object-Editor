from io import BytesIO
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit

MODULE_PATH = "app.ml.modes.sam_lama_mode"


def _rgb_png(size: Tuple[int, int] = (60, 60), color=(120, 80, 40)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(size: Tuple[int, int] = (60, 60), box=(20, 20, 40, 40)) -> bytes:
    arr = np.zeros(size[::-1], dtype=np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 255
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _rgba_cutout(size: Tuple[int, int] = (20, 20), color=(0, 200, 0, 255)) -> bytes:
    buf = BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()

class TestGetSamModeSingleton:
    @pytest.fixture
    def module(self):
        return pytest.importorskip(MODULE_PATH)

    @pytest.fixture(autouse=True)
    def _reset_singleton_and_patch_factories(self, module, monkeypatch):
        """Every test gets a clean singleton slate and cheap fake
        dependencies so building a SAMLamaMode never touches real ML."""
        monkeypatch.setattr(module, "_sam_mode_instance", None)
        monkeypatch.setattr(module.DeviceManager, "get", staticmethod(lambda x: "cpu"))

        self.fake_segmentor = MagicMock(name="fake_segmentor")
        self.fake_inpainter = MagicMock(name="fake_inpainter")
        self.fake_diffusion_replacer = MagicMock(name="fake_diffusion_replacer")
        self.fake_edge_blender = MagicMock(name="fake_edge_blender")
        self.fake_color_matcher = MagicMock(name="fake_color_matcher")
        self.fake_bg_remover = MagicMock(name="fake_bg_remover")
        self.fake_compositor = MagicMock(name="fake_compositor")
        self.fake_polygon_masker = MagicMock(name="fake_polygon_masker")

        self.get_segmentor_mock = MagicMock(return_value=self.fake_segmentor)
        self.get_inpainter_mock = MagicMock(return_value=self.fake_inpainter)
        self.get_diffusion_replacer_mock = MagicMock(return_value=self.fake_diffusion_replacer)

        monkeypatch.setattr(module, "get_segmentor", self.get_segmentor_mock)
        monkeypatch.setattr(module, "get_inpainter", self.get_inpainter_mock)
        monkeypatch.setattr(module, "get_diffusion_replacer", self.get_diffusion_replacer_mock)
        monkeypatch.setattr(module, "get_edge_blender", lambda: self.fake_edge_blender)
        monkeypatch.setattr(module, "get_color_matcher", lambda: self.fake_color_matcher)
        monkeypatch.setattr(
            module, "get_background_remover", lambda rembg_available=True: self.fake_bg_remover
        )
        monkeypatch.setattr(module, "get_compositor", lambda: self.fake_compositor)
        monkeypatch.setattr(module, "get_polygon_masker", lambda: self.fake_polygon_masker)

    def test_returns_same_instance_across_calls(self, module):
        first = module.get_sam_mode()
        second = module.get_sam_mode()

        assert first is second

    def test_only_builds_dependencies_once(self, module):
        module.get_sam_mode()
        module.get_sam_mode()
        module.get_sam_mode()

        self.get_segmentor_mock.assert_called_once()
        self.get_inpainter_mock.assert_called_once()
        self.get_diffusion_replacer_mock.assert_called_once()

    def test_instance_is_wired_with_the_expected_dependencies(self, module):
        instance = module.get_sam_mode()

        assert instance.segmentor is self.fake_segmentor
        assert instance.inpainter is self.fake_inpainter
        assert instance.diffusion_replacer is self.fake_diffusion_replacer
        assert instance.edge_blender is self.fake_edge_blender
        assert instance.color_matcher is self.fake_color_matcher
        assert instance.background_remover is self.fake_bg_remover
        assert instance.compositor is self.fake_compositor
        assert instance.polygon_masker is self.fake_polygon_masker

    def test_fresh_instance_built_after_singleton_reset(self, module):
        """Sanity check on the reset mechanism itself: clearing
        _sam_mode_instance must force a brand new SAMLamaMode next call."""
        first = module.get_sam_mode()

        module._sam_mode_instance = None
        second = module.get_sam_mode()

        assert first is not second


@pytest.fixture
def image_bytes():
    return _rgb_png((120, 120), color=(20, 20, 20))


@pytest.fixture
def mask_bytes():
    return _mask_png((120, 120), box=(40, 40, 80, 80))


@pytest.fixture
def mock_segmentor():
    seg = MagicMock()
    seg.segment_auto = AsyncMock(return_value={"segments": [], "metrics": {}})
    seg.segment_with_prompt = AsyncMock(return_value={"segments": [], "metrics": {}})
    seg.segment_with_prompts_batch = AsyncMock(return_value={"segments": [], "metrics": {}})
    return seg


@pytest.fixture
def mock_inpainter(image_bytes):
    inp = MagicMock()
    inp.inpaint = AsyncMock(return_value={
        "result_bytes": image_bytes,
        "metrics": {"inpaint_ms": 5.0},
    })
    return inp


@pytest.fixture
def mock_diffusion_replacer(image_bytes):
    dr = MagicMock()
    dr.replace = AsyncMock(return_value={
        "result_bytes": image_bytes,
        "metrics": {"diffusion_ms": 12.0},
    })
    return dr


@pytest.fixture
def mock_edge_blender(image_bytes):
    eb = MagicMock()
    eb.auto_blend = AsyncMock(return_value=image_bytes)
    return eb


@pytest.fixture
def mock_color_matcher(image_bytes):
    cm = MagicMock()
    cm.match_against_original = MagicMock(return_value=image_bytes)
    return cm


@pytest.fixture
def mock_background_remover():
    br = MagicMock()
    br.remove_and_resize = AsyncMock(return_value=_rgba_cutout((20, 20)))
    return br


@pytest.fixture
def mock_compositor(image_bytes):
    comp = MagicMock()
    comp.compose = MagicMock(return_value=image_bytes)
    return comp


@pytest.fixture
def mock_polygon_masker():
    pm = MagicMock()
    return pm


@pytest.fixture
def mode(
    monkeypatch,
    mock_segmentor,
    mock_inpainter,
    mock_diffusion_replacer,
    mock_edge_blender,
    mock_color_matcher,
    mock_background_remover,
    mock_compositor,
    mock_polygon_masker,
):
    module = pytest.importorskip(MODULE_PATH)
    monkeypatch.setattr(module, "get_compositor", lambda: mock_compositor)
    monkeypatch.setattr(module.DeviceManager, "get", staticmethod(lambda x: "cpu"))

    return module.SAMLamaMode(
        segmentor=mock_segmentor,
        inpainter=mock_inpainter,
        diffusion_replacer=mock_diffusion_replacer,
        edge_blender=mock_edge_blender,
        color_matcher=mock_color_matcher,
        background_remover=mock_background_remover,
        polygon_masker=mock_polygon_masker,
    )


class TestReplaceObjectCutout:
    @pytest.mark.asyncio
    async def test_skips_background_remover(self, mode, image_bytes, mask_bytes):
        cutout_bytes = _rgba_cutout((50, 50))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object(
            image_bytes, mask_bytes, bbox, cutout_bytes,
            use_color_matching=False, use_edge_blending=False,
            replacement_is_cutout=True,
        )

        mode.background_remover.remove_and_resize.assert_not_called()

    @pytest.mark.asyncio
    async def test_resizes_cutout_to_bbox_dimensions(self, mode, image_bytes, mask_bytes):
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

    @pytest.mark.asyncio
    async def test_forwards_resized_cutout_to_compositor(self, mode, image_bytes, mask_bytes):
        cutout_bytes = _rgba_cutout((10, 10))
        bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 70}

        await mode.replace_object(
            image_bytes, mask_bytes, bbox, cutout_bytes,
            use_color_matching=False, replacement_is_cutout=True,
        )

        call_kwargs = mode.compositor.compose.call_args.kwargs
        assert call_kwargs["bbox"] == bbox
        decoded = Image.open(BytesIO(call_kwargs["replacement_rgba_bytes"]))
        assert decoded.size == (40, 60)

    @pytest.mark.asyncio
    async def test_still_applies_color_matching_when_requested(
        self, mode, image_bytes, mask_bytes
    ):
        cutout_bytes = _rgba_cutout((50, 50))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object(
            image_bytes, mask_bytes, bbox, cutout_bytes,
            use_color_matching=True, replacement_is_cutout=True,
        )

        mode.color_matcher.match_against_original.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_replacement_is_cutout_false_still_uses_background_remover(
        self, mode, image_bytes, mask_bytes
    ):
        """Sanity check: replacement_is_cutout defaults to False."""
        replacement_bytes = _rgb_png((50, 50), color=(0, 200, 0))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_bytes)

        mode.background_remover.remove_and_resize.assert_called_once()


class TestReplaceObjectDiffusion:
    @pytest.mark.asyncio
    async def test_calls_diffusion_replacer_with_expected_args(self, mode, image_bytes, mask_bytes):
        reference_bytes = _rgb_png((30, 30), color=(0, 0, 200))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object_diffusion(
            image_bytes, mask_bytes, bbox, reference_bytes,
            prompt="a blue vase", use_color_matching=False,
        )

        mode.diffusion_replacer.replace.assert_called_once()
        call_kwargs = mode.diffusion_replacer.replace.call_args.kwargs
        assert call_kwargs["image_bytes"] == image_bytes
        assert call_kwargs["mask_bytes"] == mask_bytes
        assert call_kwargs["reference_image_bytes"] == reference_bytes
        assert call_kwargs["prompt"] == "a blue vase"
        assert call_kwargs["track_metrics"] is True

    @pytest.mark.asyncio
    async def test_applies_color_matching_when_requested(self, mode, image_bytes, mask_bytes):
        reference_bytes = _rgb_png((30, 30), color=(0, 0, 200))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object_diffusion(
            image_bytes, mask_bytes, bbox, reference_bytes,
            use_color_matching=True,
        )

        mode.color_matcher.match_against_original.assert_called_once()
        call_kwargs = mode.color_matcher.match_against_original.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_skips_color_matching_by_default(self, mode, image_bytes, mask_bytes):
        reference_bytes = _rgb_png((30, 30), color=(0, 0, 200))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object_diffusion(
            image_bytes, mask_bytes, bbox, reference_bytes,
        )

        mode.color_matcher.match_against_original.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_metrics_from_diffusion_replacer(self, mode, image_bytes, mask_bytes):
        reference_bytes = _rgb_png((30, 30), color=(0, 0, 200))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        result = await mode.replace_object_diffusion(
            image_bytes, mask_bytes, bbox, reference_bytes,
        )

        assert result["metrics"] == {"diffusion_ms": 12.0}

    @pytest.mark.asyncio
    async def test_forwards_generation_overrides(self, mode, image_bytes, mask_bytes):
        reference_bytes = _rgb_png((30, 30), color=(0, 0, 200))
        bbox = {"x1": 40, "y1": 40, "x2": 80, "y2": 80}

        await mode.replace_object_diffusion(
            image_bytes, mask_bytes, bbox, reference_bytes,
            negative_prompt="blurry, low quality",
            num_inference_steps=30,
            guidance_scale=7.5,
            ip_adapter_scale=0.6,
            strength=0.9,
            seed=42,
        )

        call_kwargs = mode.diffusion_replacer.replace.call_args.kwargs
        assert call_kwargs["negative_prompt"] == "blurry, low quality"
        assert call_kwargs["num_inference_steps"] == 30
        assert call_kwargs["guidance_scale"] == 7.5
        assert call_kwargs["ip_adapter_scale"] == 0.6
        assert call_kwargs["strength"] == 0.9
        assert call_kwargs["seed"] == 42