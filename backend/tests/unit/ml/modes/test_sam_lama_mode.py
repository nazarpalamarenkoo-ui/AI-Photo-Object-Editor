import asyncio
from io import BytesIO
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit

MODULE_PATH = "app.ml.modes.sam_lama_mode"


def _rgb_png(size: Tuple[int, int] = (60, 60), color=(120, 80, 40)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(size: Tuple[int, int] = (60, 60), box=(20, 20, 40, 40)) -> bytes:
    """Binary L-mode mask: white square inside `box`, black elsewhere."""
    arr = np.zeros(size[::-1], dtype=np.uint8)  # (H, W)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 255
    img = Image.fromarray(arr, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _empty_mask_png(size: Tuple[int, int] = (60, 60)) -> bytes:
    arr = np.zeros(size[::-1], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rgba_png(size: Tuple[int, int] = (20, 20), color=(200, 30, 30, 255)) -> bytes:
    img = Image.new("RGBA", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_bytes():
    return _rgb_png((60, 60))


@pytest.fixture
def mask_bytes():
    return _mask_png((60, 60), box=(20, 20, 40, 40))


@pytest.fixture
def replacement_image_bytes():
    return _rgb_png((30, 30), color=(10, 200, 10))


@pytest.fixture
def extracted_bytes():
    return _rgba_png((20, 20))


@pytest.fixture
def polygon_points():
    return [(10, 10), (50, 10), (50, 50), (10, 50)]


@pytest.fixture
def mock_segmentor():
    seg = MagicMock()
    seg.segment_auto = AsyncMock(return_value={
        "segments": [
            {"mask_id": 0, "bbox": {"x1": 0, "y1": 0, "x2": 30, "y2": 30}, "area": 1000},
            {"mask_id": 1, "bbox": {"x1": 5, "y1": 5, "x2": 15, "y2": 15}, "area": 100},
            {"mask_id": 2, "bbox": {"x1": 1, "y1": 1, "x2": 3, "y2": 3}, "area": 4},
        ],
        "metrics": {"num_segments": 3},
    })
    seg.segment_with_prompt = AsyncMock(return_value={
        "segments": [
            {"mask_id": 0, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "stability_score": 0.9},
            {"mask_id": 1, "bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 5}, "stability_score": 0.5},
        ],
        "metrics": {"num_segments": 2},
    })
    seg.segment_with_prompts_batch = AsyncMock(return_value={
        "segments": [
            {"mask_id": 0, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "stability_score": 0.9},
            {"mask_id": 1, "bbox": {"x1": 15, "y1": 15, "x2": 25, "y2": 25}, "stability_score": 0.8},
            {"mask_id": 2, "bbox": {"x1": 30, "y1": 30, "x2": 40, "y2": 40}, "stability_score": 0.7},
        ],
        "metrics": {"num_segments": 3, "encoder_passes": 1},
    })
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
    br.remove_and_resize = AsyncMock(return_value=_rgba_png((20, 20)))
    return br


@pytest.fixture
def mock_compositor(image_bytes):
    comp = MagicMock()
    comp.compose = MagicMock(return_value=image_bytes)
    return comp


@pytest.fixture
def mock_polygon_masker():
    pm = MagicMock()
    pm.generate_mask = AsyncMock(return_value=_mask_png((60, 60), box=(20, 20, 40, 40)))
    return pm


@pytest.fixture
def mode(
    monkeypatch,
    mock_segmentor,
    mock_inpainter,
    mock_edge_blender,
    mock_color_matcher,
    mock_background_remover,
    mock_compositor,
    mock_polygon_masker,
):
    module = pytest.importorskip(MODULE_PATH)
    monkeypatch.setattr(module, "get_compositor", lambda: mock_compositor)

    return module.SAMLamaMode(
        segmentor=mock_segmentor,
        inpainter=mock_inpainter,
        edge_blender=mock_edge_blender,
        color_matcher=mock_color_matcher,
        background_remover=mock_background_remover,
        polygon_masker=mock_polygon_masker,
    )


def test_init_uses_injected_dependencies(
    mode, mock_segmentor, mock_inpainter, mock_compositor, mock_polygon_masker
):
    assert mode.segmentor is mock_segmentor
    assert mode.inpainter is mock_inpainter
    assert mode.compositor is mock_compositor
    assert mode.polygon_masker is mock_polygon_masker
    assert mode.device == "cpu"


def test_init_builds_defaults_when_none_provided(monkeypatch):
    module = pytest.importorskip(MODULE_PATH)

    fake_segmentor = MagicMock(name="default_segmentor")
    fake_inpainter = MagicMock(name="default_inpainter")
    fake_edge_blender = MagicMock(name="default_edge_blender")
    fake_color_matcher = MagicMock(name="default_color_matcher")
    fake_bg_remover = MagicMock(name="default_bg_remover")
    fake_compositor = MagicMock(name="default_compositor")
    fake_polygon_masker = MagicMock(name="default_polygon_masker")

    monkeypatch.setattr(module, "get_segmentor", lambda device="cpu": fake_segmentor)
    monkeypatch.setattr(module, "get_inpainter", lambda device="cpu": fake_inpainter)
    monkeypatch.setattr(module, "get_edge_blender", lambda: fake_edge_blender)
    monkeypatch.setattr(module, "get_color_matcher", lambda: fake_color_matcher)
    monkeypatch.setattr(module, "get_background_remover", lambda rembg_available=True: fake_bg_remover)
    monkeypatch.setattr(module, "get_compositor", lambda: fake_compositor)
    monkeypatch.setattr(module, "get_polygon_masker", lambda: fake_polygon_masker)

    instance = module.SAMLamaMode()

    assert instance.segmentor is fake_segmentor
    assert instance.inpainter is fake_inpainter
    assert instance.edge_blender is fake_edge_blender
    assert instance.color_matcher is fake_color_matcher
    assert instance.background_remover is fake_bg_remover
    assert instance.compositor is fake_compositor
    assert instance.polygon_masker is fake_polygon_masker


@pytest.mark.asyncio
async def test_segment_objects_filters_by_min_area(mode, image_bytes):
    result = await mode.segment_objects(image_bytes, min_area=500)

    areas = [s["area"] for s in result["segments"]]
    assert areas == [1000]


@pytest.mark.asyncio
async def test_segment_objects_respects_max_segments(mode, image_bytes):
    result = await mode.segment_objects(image_bytes, min_area=0, max_segments=2)

    assert len(result["segments"]) == 2
    assert [s["area"] for s in result["segments"]] == [1000, 100]


@pytest.mark.asyncio
async def test_segment_objects_assigns_sequential_bbox_id(mode, image_bytes):
    result = await mode.segment_objects(image_bytes, min_area=0)

    bbox_ids = [s["bbox_id"] for s in result["segments"]]
    assert bbox_ids == list(range(len(bbox_ids)))


@pytest.mark.asyncio
async def test_segment_objects_returns_image_size(mode, image_bytes):
    result = await mode.segment_objects(image_bytes)

    assert result["image_size"] == (60, 60)


@pytest.mark.asyncio
async def test_segment_objects_returns_metrics_passthrough(mode, image_bytes):
    result = await mode.segment_objects(image_bytes)

    assert result["metrics"] == {"num_segments": 3}


@pytest.mark.asyncio
async def test_segment_objects_empty_when_all_below_min_area(mode, image_bytes):
    result = await mode.segment_objects(image_bytes, min_area=10_000)

    assert result["segments"] == []


@pytest.mark.asyncio
async def test_segment_with_prompt_forwards_args_to_segmentor(mode, image_bytes, mock_segmentor):
    bbox = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    await mode.segment_with_prompt(
        image_bytes, point_coords=[(1, 1)], point_labels=[1], bbox=bbox
    )

    mock_segmentor.segment_with_prompt.assert_called_once_with(
        image_bytes, point_coords=[(1, 1)], point_labels=[1], bbox=bbox, multimask_output=None
    )


@pytest.mark.asyncio
async def test_segment_with_prompt_assigns_bbox_id(mode, image_bytes):
    result = await mode.segment_with_prompt(image_bytes, point_coords=[(1, 1)], point_labels=[1])

    assert [s["bbox_id"] for s in result["segments"]] == [0, 1]


@pytest.mark.asyncio
async def test_segment_with_prompt_returns_image_size(mode, image_bytes):
    result = await mode.segment_with_prompt(image_bytes, bbox={"x1": 0, "y1": 0, "x2": 5, "y2": 5})

    assert result["image_size"] == (60, 60)


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_forwards_args_to_segmentor(
    mode, image_bytes, mock_segmentor
):
    bboxes = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 15, "y1": 15, "x2": 25, "y2": 25},
        {"x1": 30, "y1": 30, "x2": 40, "y2": 40},
    ]

    await mode.segment_with_prompts_batch(image_bytes, bboxes)

    mock_segmentor.segment_with_prompts_batch.assert_called_once_with(
        image_bytes=image_bytes, bboxes=bboxes
    )


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_assigns_sequential_bbox_id(mode, image_bytes):
    bboxes = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 15, "y1": 15, "x2": 25, "y2": 25},
        {"x1": 30, "y1": 30, "x2": 40, "y2": 40},
    ]

    result = await mode.segment_with_prompts_batch(image_bytes, bboxes)

    assert [s["bbox_id"] for s in result["segments"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_preserves_input_bbox_order(
    mode, image_bytes, mock_segmentor
):
    """bbox_id must line up with the order segments came back in (== input order)."""
    mock_segmentor.segment_with_prompts_batch = AsyncMock(return_value={
        "segments": [
            {"mask_id": 0, "bbox": {"x1": 30, "y1": 30, "x2": 40, "y2": 40}},
            {"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ],
        "metrics": {},
    })
    bboxes = [
        {"x1": 30, "y1": 30, "x2": 40, "y2": 40},
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]

    result = await mode.segment_with_prompts_batch(image_bytes, bboxes)

    assert result["segments"][0]["bbox"] == bboxes[0]
    assert result["segments"][0]["bbox_id"] == 0
    assert result["segments"][1]["bbox"] == bboxes[1]
    assert result["segments"][1]["bbox_id"] == 1


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_returns_image_size(mode, image_bytes):
    bboxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

    result = await mode.segment_with_prompts_batch(image_bytes, bboxes)

    assert result["image_size"] == (60, 60)


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_uses_real_image_dimensions(mode, mock_segmentor):
    image_bytes = _rgb_png((333, 217))
    bboxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

    result = await mode.segment_with_prompts_batch(image_bytes, bboxes)

    assert result["image_size"] == (333, 217)


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_returns_metrics_passthrough(
    mode, image_bytes, mock_segmentor
):
    result = await mode.segment_with_prompts_batch(
        image_bytes, [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    )

    assert result["metrics"] == {"num_segments": 3, "encoder_passes": 1}


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_empty_bboxes_returns_empty_segments(
    mode, image_bytes, mock_segmentor
):
    mock_segmentor.segment_with_prompts_batch = AsyncMock(return_value={
        "segments": [],
        "metrics": {"num_segments": 0},
    })

    result = await mode.segment_with_prompts_batch(image_bytes, [])

    mock_segmentor.segment_with_prompts_batch.assert_called_once_with(
        image_bytes=image_bytes, bboxes=[]
    )
    assert result["segments"] == []


@pytest.mark.asyncio
async def test_segment_with_prompts_batch_single_bbox(mode, image_bytes, mock_segmentor):
    mock_segmentor.segment_with_prompts_batch = AsyncMock(return_value={
        "segments": [{"mask_id": 0, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}],
        "metrics": {"num_segments": 1},
    })

    result = await mode.segment_with_prompts_batch(
        image_bytes, [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    )

    assert len(result["segments"]) == 1
    assert result["segments"][0]["bbox_id"] == 0


@pytest.mark.asyncio
async def test_segment_by_polygon_calls_polygon_masker_with_params(mode, image_bytes, polygon_points, mock_polygon_masker):
    await mode.segment_by_polygon(
        image_bytes, points=polygon_points, smooth=False, smoothing_factor=1.5, feather_px=3,
    )

    mock_polygon_masker.generate_mask.assert_called_once_with(
        image_size=(60, 60), points=polygon_points, smooth=False, smoothing_factor=1.5, feather_px=3,
    )


@pytest.mark.asyncio
async def test_segment_by_polygon_uses_default_params(mode, image_bytes, polygon_points, mock_polygon_masker):
    await mode.segment_by_polygon(image_bytes, points=polygon_points)

    mock_polygon_masker.generate_mask.assert_called_once_with(
        image_size=(60, 60), points=polygon_points, smooth=True, smoothing_factor=0.0, feather_px=0,
    )


@pytest.mark.asyncio
async def test_segment_by_polygon_returns_single_segment_with_expected_fields(mode, image_bytes, polygon_points):
    result = await mode.segment_by_polygon(image_bytes, points=polygon_points)

    assert len(result["segments"]) == 1
    segment = result["segments"][0]
    assert segment["mask_id"] == 0
    assert segment["bbox_id"] == 0
    assert segment["source"] == "polygon"
    assert isinstance(segment["mask_bytes"], bytes)


@pytest.mark.asyncio
async def test_segment_by_polygon_bbox_and_area_match_mask_white_region(
    mode, image_bytes, polygon_points, mock_polygon_masker
):
    mock_polygon_masker.generate_mask = AsyncMock(return_value=_mask_png((60, 60), box=(20, 20, 40, 40)))

    result = await mode.segment_by_polygon(image_bytes, points=polygon_points)

    segment = result["segments"][0]
    # box=(20, 20, 40, 40) fills rows/cols 20..39 inclusive -> max index 39
    assert segment["bbox"] == {"x1": 20, "y1": 20, "x2": 39, "y2": 39}
    assert segment["area"] == 20 * 20


@pytest.mark.asyncio
async def test_segment_by_polygon_returns_image_size(mode, image_bytes, polygon_points):
    result = await mode.segment_by_polygon(image_bytes, points=polygon_points)

    assert result["image_size"] == (60, 60)


@pytest.mark.asyncio
async def test_segment_by_polygon_returns_metrics(mode, image_bytes, polygon_points, mock_polygon_masker):
    mock_polygon_masker.generate_mask = AsyncMock(return_value=_mask_png((60, 60), box=(20, 20, 40, 40)))

    result = await mode.segment_by_polygon(image_bytes, points=polygon_points)

    assert result["metrics"]["num_segments"] == 1
    assert result["metrics"]["total_area_px"] == 20 * 20


@pytest.mark.asyncio
async def test_segment_by_polygon_raises_on_empty_mask(mode, image_bytes, polygon_points, mock_polygon_masker):
    mock_polygon_masker.generate_mask = AsyncMock(return_value=_empty_mask_png((60, 60)))

    with pytest.raises(ValueError, match="Polygon produced an empty mask"):
        await mode.segment_by_polygon(image_bytes, points=polygon_points)


@pytest.mark.asyncio
async def test_segment_by_polygon_uses_real_image_dimensions(mode, mock_polygon_masker):
    image_bytes = _rgb_png((333, 217))
    mock_polygon_masker.generate_mask = AsyncMock(return_value=_mask_png((333, 217), box=(50, 50, 100, 100)))

    result = await mode.segment_by_polygon(image_bytes, points=[(50, 50), (100, 50), (100, 100), (50, 100)])

    assert result["image_size"] == (333, 217)
    mock_polygon_masker.generate_mask.assert_called_once_with(
        image_size=(333, 217),
        points=[(50, 50), (100, 50), (100, 100), (50, 100)],
        smooth=True, smoothing_factor=0.0, feather_px=0,
    )


@pytest.mark.asyncio
async def test_remove_object_dilates_mask_when_expand_positive(
    mode, image_bytes, mask_bytes, monkeypatch
):
    dilated = b"DILATED"
    mode._dilate_mask = AsyncMock(return_value=dilated)

    await mode.remove_object(image_bytes, mask_bytes, expand_mask_pixels=12)

    mode._dilate_mask.assert_called_once_with(mask_bytes, 12)
    # the (possibly) dilated mask must be the one forwarded to inpaint + blend
    call_kwargs = mode.inpainter.inpaint.call_args.kwargs
    assert call_kwargs["mask_bytes"] == dilated


@pytest.mark.asyncio
async def test_remove_object_skips_dilation_when_expand_zero(mode, image_bytes, mask_bytes):
    mode._dilate_mask = AsyncMock(return_value=b"SHOULD_NOT_BE_USED")

    await mode.remove_object(image_bytes, mask_bytes, expand_mask_pixels=0)

    mode._dilate_mask.assert_not_called()
    call_kwargs = mode.inpainter.inpaint.call_args.kwargs
    assert call_kwargs["mask_bytes"] == mask_bytes


@pytest.mark.asyncio
async def test_remove_object_calls_inpaint_with_remove_mode(mode, image_bytes, mask_bytes):
    module = pytest.importorskip(MODULE_PATH)

    await mode.remove_object(image_bytes, mask_bytes, expand_mask_pixels=0)

    call_kwargs = mode.inpainter.inpaint.call_args.kwargs
    assert call_kwargs["mode"] == module.InpaintMode.REMOVE
    assert call_kwargs["track_metrics"] is True
    assert call_kwargs["image_bytes"] == image_bytes


@pytest.mark.asyncio
async def test_remove_object_applies_edge_blending_when_enabled(mode, image_bytes, mask_bytes):
    await mode.remove_object(image_bytes, mask_bytes, use_edge_blending=True, expand_mask_pixels=0)

    mode.edge_blender.auto_blend.assert_called_once()


@pytest.mark.asyncio
async def test_remove_object_skips_edge_blending_when_disabled(mode, image_bytes, mask_bytes):
    await mode.remove_object(image_bytes, mask_bytes, use_edge_blending=False, expand_mask_pixels=0)

    mode.edge_blender.auto_blend.assert_not_called()


@pytest.mark.asyncio
async def test_remove_object_returns_metrics_from_inpaint(mode, image_bytes, mask_bytes):
    result = await mode.remove_object(image_bytes, mask_bytes, expand_mask_pixels=0)

    assert result["metrics"] == {"inpaint_ms": 5.0}
    assert isinstance(result["result_bytes"], bytes)


@pytest.mark.asyncio
async def test_remove_object_normalizes_size_to_match_input(mode, mask_bytes):
    """If LaMa returns a differently-sized image, the result must be
    resized back to the original input dimensions."""
    original = _rgb_png((60, 60))
    mismatched_output = _rgb_png((64, 64))
    mode.inpainter.inpaint = AsyncMock(return_value={
        "result_bytes": mismatched_output,
        "metrics": {},
    })

    result = await mode.remove_object(original, mask_bytes, expand_mask_pixels=0, use_edge_blending=False)

    decoded = Image.open(BytesIO(result["result_bytes"]))
    assert decoded.size == (60, 60)

@pytest.mark.asyncio
async def test_replace_object_resizes_replacement_to_bbox_dims(
    mode, image_bytes, mask_bytes, replacement_image_bytes
):
    bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 25}

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_image_bytes)

    mode.background_remover.remove_and_resize.assert_called_once_with(
        replacement_image_bytes, (20, 15)
    )


@pytest.mark.asyncio
async def test_replace_object_calls_compositor_with_bbox_and_zero_softness(
    mode, image_bytes, mask_bytes, replacement_image_bytes
):
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_image_bytes)

    call_kwargs = mode.compositor.compose.call_args.kwargs
    assert call_kwargs["bbox"] == bbox
    assert call_kwargs["edge_softness"] == 0


@pytest.mark.asyncio
async def test_replace_object_applies_color_matching_when_enabled(
    mode, image_bytes, mask_bytes, replacement_image_bytes
):
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    await mode.replace_object(
        image_bytes, mask_bytes, bbox, replacement_image_bytes, use_color_matching=True
    )

    mode.color_matcher.match_against_original.assert_called_once()


@pytest.mark.asyncio
async def test_replace_object_skips_color_matching_when_disabled(
    mode, image_bytes, mask_bytes, replacement_image_bytes
):
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    await mode.replace_object(
        image_bytes, mask_bytes, bbox, replacement_image_bytes, use_color_matching=False
    )

    mode.color_matcher.match_against_original.assert_not_called()


@pytest.mark.asyncio
async def test_replace_object_edge_blending_default_is_disabled(
    mode, image_bytes, mask_bytes, replacement_image_bytes
):
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_image_bytes)

    mode.edge_blender.auto_blend.assert_not_called()


@pytest.mark.asyncio
async def test_replace_object_uses_remove_mode_for_inpaint(mode, image_bytes, mask_bytes, replacement_image_bytes):
    module = pytest.importorskip(MODULE_PATH)
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_image_bytes)

    call_kwargs = mode.inpainter.inpaint.call_args.kwargs
    assert call_kwargs["mode"] == module.InpaintMode.REMOVE


@pytest.mark.asyncio
async def test_replace_object_returns_metrics_and_bytes(mode, image_bytes, mask_bytes, replacement_image_bytes):
    bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    result = await mode.replace_object(image_bytes, mask_bytes, bbox, replacement_image_bytes)

    assert isinstance(result["result_bytes"], bytes)
    assert result["metrics"] == {"inpaint_ms": 5.0}


@pytest.mark.asyncio
async def test_extract_object_applies_padding_within_bounds(mode, image_bytes, mask_bytes):
    bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

    result = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=8)

    assert result["cropped_bbox"] == {"x1": 12, "y1": 12, "x2": 48, "y2": 48}


@pytest.mark.asyncio
async def test_extract_object_clamps_padding_at_image_edges(mode, image_bytes, mask_bytes):
    bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

    result = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=20)

    assert result["cropped_bbox"]["x1"] == 0
    assert result["cropped_bbox"]["y1"] == 0


@pytest.mark.asyncio
async def test_extract_object_returns_rgba_with_alpha_from_mask(mode, image_bytes, mask_bytes):
    bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

    result = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=0)

    decoded = Image.open(BytesIO(result["extracted_bytes"]))
    assert decoded.mode == "RGBA"


@pytest.mark.asyncio
async def test_extract_object_area_pixels_matches_mask_white_region(mode, image_bytes, mask_bytes):
    bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

    result = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=0)

    # mask_bytes has a fully-white 20x20 box exactly matching this bbox
    assert result["area_pixels"] == 20 * 20


@pytest.mark.asyncio
async def test_extract_object_original_and_object_size(mode, image_bytes, mask_bytes):
    bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

    result = await mode.extract_object(image_bytes, mask_bytes, bbox, padding_pixels=0)

    assert result["original_size"] == (60, 60)
    assert result["object_size"] == (20, 20)


@pytest.mark.asyncio
async def test_extract_object_respects_output_format_webp(mode, image_bytes, mask_bytes):
    bbox = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}

    result = await mode.extract_object(
        image_bytes, mask_bytes, bbox, padding_pixels=0, output_format="WEBP"
    )

    decoded = Image.open(BytesIO(result["extracted_bytes"]))
    assert decoded.format == "WEBP"


@pytest.mark.asyncio
async def test_paste_extracted_object_scales_to_fit_bbox(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 5, "y1": 5, "x2": 25, "y2": 15}  # 20x10 box, source is 20x20

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=False,
    )

    # fit_ratio = min(20/20, 10/20) = 0.5 -> new size (10, 10)
    assert result["object_size"] == (10, 10)


@pytest.mark.asyncio
async def test_paste_extracted_object_scale_factor_applies_multiplicatively(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox, scale=0.5,
        use_color_matching=False, use_edge_blending=False,
    )

    assert result["object_size"] == (10, 10)


@pytest.mark.asyncio
async def test_paste_extracted_object_centers_within_bbox(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}  # 20x20, matches source exactly

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=False,
    )

    assert result["paste_bbox"] == {"x1": 10, "y1": 10, "x2": 30, "y2": 30}


@pytest.mark.asyncio
async def test_paste_extracted_object_clamps_to_canvas_bounds(mode, image_bytes, extracted_bytes):
    """A target bbox near the edge must not push the pasted object outside the canvas."""
    target_bbox = {"x1": 55, "y1": 55, "x2": 75, "y2": 75}  # near 60x60 canvas edge

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=False,
    )

    w, h = result["object_size"]
    assert 0 <= result["paste_bbox"]["x1"] <= 60 - w
    assert 0 <= result["paste_bbox"]["y1"] <= 60 - h


@pytest.mark.asyncio
async def test_paste_extracted_object_applies_color_matching_with_paste_bbox(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=True, use_edge_blending=False,
    )

    mode.color_matcher.match_against_original.assert_called_once()
    call_kwargs = mode.color_matcher.match_against_original.call_args.kwargs
    assert call_kwargs["bbox"] == result["paste_bbox"]


@pytest.mark.asyncio
async def test_paste_extracted_object_applies_edge_blending_when_enabled(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}

    await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=True,
    )

    mode.edge_blender.auto_blend.assert_called_once()


@pytest.mark.asyncio
async def test_paste_extracted_object_skips_edge_blending_when_disabled(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}

    await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=False,
    )

    mode.edge_blender.auto_blend.assert_not_called()


@pytest.mark.asyncio
async def test_paste_extracted_object_result_bytes_is_valid_image(mode, image_bytes, extracted_bytes):
    target_bbox = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}

    result = await mode.paste_extracted_object(
        image_bytes, extracted_bytes, target_bbox,
        use_color_matching=False, use_edge_blending=False,
    )

    decoded = Image.open(BytesIO(result["result_bytes"]))
    assert decoded.size == (60, 60)


@pytest.mark.asyncio
async def test_alpha_to_mask_places_alpha_at_paste_bbox(mode, extracted_bytes):
    paste_bbox = {"x1": 5, "y1": 5, "x2": 25, "y2": 25}

    mask_bytes_out = await mode._alpha_to_mask(
        extracted_bytes=extracted_bytes,
        paste_bbox=paste_bbox,
        object_size=(20, 20),
        canvas_size=(60, 60),
    )

    decoded = Image.open(BytesIO(mask_bytes_out)).convert("L")
    arr = np.array(decoded)
    assert arr.shape == (60, 60)
    # Region inside the paste bbox should be non-zero (fully opaque source)
    assert arr[5:25, 5:25].min() > 0
    # Outside the bbox should remain zero
    assert arr[0:5, 0:5].max() == 0


# ---------------------------------------------------------------------------
# _resize_rgba_to_bbox
# ---------------------------------------------------------------------------

def test_resize_rgba_to_bbox_returns_target_size(mode):
    source = _rgba_png((20, 20))

    result = mode._resize_rgba_to_bbox(source, size=(40, 15))

    decoded = Image.open(BytesIO(result))
    assert decoded.size == (40, 15)


def test_resize_rgba_to_bbox_returns_valid_png_bytes(mode):
    source = _rgba_png((20, 20))

    result = mode._resize_rgba_to_bbox(source, size=(30, 30))

    assert isinstance(result, bytes)
    decoded = Image.open(BytesIO(result))
    assert decoded.format == "PNG"


def test_resize_rgba_to_bbox_preserves_rgba_mode(mode):
    source = _rgba_png((20, 20))

    result = mode._resize_rgba_to_bbox(source, size=(30, 30))

    decoded = Image.open(BytesIO(result))
    assert decoded.mode == "RGBA"


def test_resize_rgba_to_bbox_converts_non_rgba_input(mode):
    source = _rgb_png((20, 20))  # RGB, no alpha channel

    result = mode._resize_rgba_to_bbox(source, size=(20, 20))

    decoded = Image.open(BytesIO(result))
    assert decoded.mode == "RGBA"


def test_resize_rgba_to_bbox_preserves_opaque_alpha(mode):
    img = Image.new("RGBA", (20, 20), (200, 30, 30, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    source = buf.getvalue()

    result = mode._resize_rgba_to_bbox(source, size=(10, 10))

    decoded = Image.open(BytesIO(result)).convert("RGBA")
    arr = np.array(decoded)
    assert arr[..., 3].min() == 255


def test_resize_rgba_to_bbox_preserves_transparent_alpha(mode):
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    source = buf.getvalue()

    result = mode._resize_rgba_to_bbox(source, size=(10, 10))

    decoded = Image.open(BytesIO(result)).convert("RGBA")
    arr = np.array(decoded)
    assert arr[..., 3].max() == 0


def test_resize_rgba_to_bbox_handles_non_square_target(mode):
    source = _rgba_png((50, 20))

    result = mode._resize_rgba_to_bbox(source, size=(15, 40))

    decoded = Image.open(BytesIO(result))
    assert decoded.size == (15, 40)


@pytest.mark.asyncio
async def test_normalize_size_resizes_mismatched_image():
    module = pytest.importorskip(MODULE_PATH)

    reference = _rgb_png((60, 60))
    processed = _rgb_png((40, 40))

    normalized = await module._normalize_size(processed, reference)

    decoded = Image.open(BytesIO(normalized))
    assert decoded.size == (60, 60)


@pytest.mark.asyncio
async def test_normalize_size_keeps_dimensions_when_already_matching():
    module = pytest.importorskip(MODULE_PATH)

    reference = _rgb_png((60, 60))
    processed = _rgb_png((60, 60))

    normalized = await module._normalize_size(processed, reference)

    decoded = Image.open(BytesIO(normalized))
    assert decoded.size == (60, 60)
    assert decoded.format == "JPEG"
