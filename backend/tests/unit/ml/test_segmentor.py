import numpy as np
import pytest
from PIL import Image as PILImage
from io import BytesIO

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_segment_auto_tracks_metrics_by_default(segmentor, image_bytes, tracker):
    """track_metrics defaults to True when not explicitly passed."""
    await segmentor.segment_auto(image_bytes)

    tracker.log_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_segment_with_prompt_tracks_metrics_by_default(segmentor, image_bytes, tracker):
    await segmentor.segment_with_prompt(image_bytes, point_coords=[(1, 1)], point_labels=[1])

    tracker.log_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_segment_auto_converts_rgba_image_to_rgb(segmentor, fake_sam2_env):
    """RGBA input (e.g. PNG with alpha channel) must be converted to RGB
    before being passed to the auto mask generator."""
    rgba = PILImage.new("RGBA", (20, 20), (10, 20, 30, 255))
    buf = BytesIO()
    rgba.save(buf, format="PNG")

    await segmentor.segment_auto(buf.getvalue())

    img_array = fake_sam2_env["auto_gen_instance"].generate.call_args.args[0]
    assert img_array.shape == (20, 20, 3)
    assert img_array.dtype == np.uint8


@pytest.mark.asyncio
async def test_segment_auto_converts_grayscale_image_to_rgb(segmentor, fake_sam2_env):
    """Grayscale ('L') input must also be converted to 3-channel RGB."""
    gray = PILImage.new("L", (20, 20), 128)
    buf = BytesIO()
    gray.save(buf, format="PNG")

    await segmentor.segment_auto(buf.getvalue())

    img_array = fake_sam2_env["auto_gen_instance"].generate.call_args.args[0]
    assert img_array.shape == (20, 20, 3)


@pytest.mark.asyncio
async def test_segment_with_prompt_no_inputs_passes_all_none(segmentor, image_bytes, fake_sam2_env):
    """Calling without points/labels/bbox should forward None for all
    three predictor kwargs rather than raising."""
    await segmentor.segment_with_prompt(image_bytes)

    call_kwargs = fake_sam2_env["predictor_instance"].predict.call_args.kwargs
    assert call_kwargs["point_coords"] is None
    assert call_kwargs["point_labels"] is None
    assert call_kwargs["box"] is None
    assert call_kwargs["multimask_output"] is True


@pytest.mark.asyncio
async def test_segment_with_prompt_points_without_labels_passes_none_labels(
    segmentor, image_bytes, fake_sam2_env
):
    """point_labels is optional; omitting it while supplying point_coords
    should not crash and should forward labels=None."""
    await segmentor.segment_with_prompt(image_bytes, point_coords=[(2, 3)])

    call_kwargs = fake_sam2_env["predictor_instance"].predict.call_args.kwargs
    assert list(call_kwargs["point_coords"][0]) == [2, 3]
    assert call_kwargs["point_labels"] is None


@pytest.mark.asyncio
async def test_segment_with_prompt_area_matches_decoded_mask_pixel_count(segmentor, image_bytes):
    """The reported 'area' should equal the number of nonzero pixels in
    the PNG-encoded mask_bytes, i.e. area accounting is internally
    consistent with what gets returned to the caller."""
    result = await segmentor.segment_with_prompt(
        image_bytes, point_coords=[(1, 1)], point_labels=[1]
    )

    for seg in result["segments"]:
        decoded = PILImage.open(BytesIO(seg["mask_bytes"]))
        arr = np.array(decoded)
        nonzero_count = int((arr > 0).sum())
        assert seg["area"] == nonzero_count


@pytest.mark.asyncio
async def test_segment_auto_mask_bytes_decode_to_expected_dimensions(segmentor, image_bytes):
    result = await segmentor.segment_auto(image_bytes)

    for seg in result["segments"]:
        decoded = PILImage.open(BytesIO(seg["mask_bytes"]))
        assert decoded.mode == "L"
        assert decoded.size[0] > 0 and decoded.size[1] > 0


@pytest.mark.asyncio
async def test_segment_auto_bbox_x2_y2_consistent_with_width_height(segmentor, image_bytes, fake_sam2_env):
    """SAM returns [x, y, w, h]; ensure x2 = x + w and y2 = y + h exactly,
    guarding against off-by-one regressions in bbox conversion."""
    result = await segmentor.segment_auto(image_bytes)

    for seg in result["segments"]:
        bbox = seg["bbox"]
        assert bbox["x2"] >= bbox["x1"]
        assert bbox["y2"] >= bbox["y1"]


@pytest.mark.asyncio
async def test_segment_auto_called_twice_tracks_metrics_each_time(segmentor, image_bytes, tracker):
    await segmentor.segment_auto(image_bytes, track_metrics=True)
    await segmentor.segment_auto(image_bytes, track_metrics=True)

    assert tracker.log_metrics.call_count == 2


@pytest.mark.asyncio
async def test_segment_auto_results_are_independent_across_calls(segmentor, image_bytes):
    """Mutating the result of one call (e.g. sorting) must not affect a
    subsequent call's output — guards against accidental shared mutable
    state (e.g. caching mask lists by reference)."""
    first = await segmentor.segment_auto(image_bytes)
    first["segments"].clear()

    second = await segmentor.segment_auto(image_bytes)
    assert len(second["segments"]) == 2

def test_get_segmentor_passes_tracker_on_first_construction(fake_sam2_env, monkeypatch, tracker):
    from app.ml import segmentor as segmentor_module

    monkeypatch.setattr(segmentor_module, "_segmentor_instance", None)

    instance = segmentor_module.get_segmentor(
        model_path="weights/fake.pt", device="cpu", tracker=tracker
    )

    assert instance.tracker is tracker


def test_get_segmentor_does_not_reconstruct_on_subsequent_calls_with_different_device(
    fake_sam2_env, monkeypatch
):
    from app.ml import segmentor as segmentor_module

    monkeypatch.setattr(segmentor_module, "_segmentor_instance", None)

    first = segmentor_module.get_segmentor(model_path="weights/fake.pt", device="cpu")
    second = segmentor_module.get_segmentor(model_path="weights/fake.pt", device="cuda")

    assert first is second
    assert first.device == "cpu"
    fake_sam2_env["build_sam2"].assert_called_once()


def test_calculate_metrics_single_segment(segmentor):
    segments = [{"stability_score": 0.42, "area": 7}]

    metrics = segmentor._calculate_metrics(segments, inference_time_ms=1.0)

    assert metrics["num_segments"] == 1
    assert metrics["avg_stability"] == pytest.approx(0.42)
    assert metrics["total_area_px"] == 7


def test_calculate_metrics_zero_area_segments(segmentor):
    segments = [
        {"stability_score": 0.5, "area": 0},
        {"stability_score": 1.0, "area": 0},
    ]

    metrics = segmentor._calculate_metrics(segments, inference_time_ms=0.0)

    assert metrics["total_area_px"] == 0
    assert metrics["avg_stability"] == pytest.approx(0.75)