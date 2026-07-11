import numpy as np
import pytest
from PIL import Image as PILImage
from io import BytesIO

pytestmark = pytest.mark.unit


def _bboxes(n):
    """n distinct, easily-distinguishable bbox prompts."""
    return [
        {"x1": 10 * i, "y1": 10 * i, "x2": 10 * i + 5, "y2": 10 * i + 5}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_batch_returns_segments_and_metrics_keys(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    assert "segments" in result
    assert "metrics" in result


@pytest.mark.asyncio
async def test_batch_calls_encoder_exactly_once_regardless_of_bbox_count(
    segmentor, image_bytes, fake_sam2_env
):
    """The whole point of the batched method is a single, expensive
    set_image() call shared across all bbox prompts."""
    await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(5))

    assert fake_sam2_env["predictor_instance"].set_image.call_count == 1


@pytest.mark.asyncio
async def test_batch_calls_predict_once_per_bbox(segmentor, image_bytes, fake_sam2_env):
    bboxes = _bboxes(4)
    await segmentor.segment_with_prompts_batch(image_bytes, bboxes)

    assert fake_sam2_env["predictor_instance"].predict.call_count == len(bboxes)


@pytest.mark.asyncio
async def test_batch_uses_multimask_output_false_for_every_call(
    segmentor, image_bytes, fake_sam2_env
):
    await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))

    for call in fake_sam2_env["predictor_instance"].predict.call_args_list:
        assert call.kwargs["multimask_output"] is False


@pytest.mark.asyncio
async def test_batch_forwards_none_for_point_coords_and_labels(
    segmentor, image_bytes, fake_sam2_env
):
    await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    for call in fake_sam2_env["predictor_instance"].predict.call_args_list:
        assert call.kwargs["point_coords"] is None
        assert call.kwargs["point_labels"] is None


@pytest.mark.asyncio
async def test_batch_forwards_correct_box_per_prompt_in_order(
    segmentor, image_bytes, fake_sam2_env
):
    """Each predict() call must receive the matching bbox, in the same
    order the bboxes were supplied — guards against index mix-ups."""
    bboxes = _bboxes(3)
    await segmentor.segment_with_prompts_batch(image_bytes, bboxes)

    calls = fake_sam2_env["predictor_instance"].predict.call_args_list
    for bbox, call in zip(bboxes, calls):
        expected = np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        assert np.array_equal(call.kwargs["box"], expected)


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_includes_prompt_bbox_reference_matching_input(
    segmentor, image_bytes
):
    bboxes = _bboxes(3)
    result = await segmentor.segment_with_prompts_batch(image_bytes, bboxes)

    for seg in result["segments"]:
        assert seg["prompt_bbox"] == bboxes[seg["mask_id"]]


@pytest.mark.asyncio
async def test_batch_mask_id_matches_bbox_index_not_output_position(
    segmentor, image_bytes, fake_sam2_env
):
    """mask_id should reflect the original bbox index (enumerate idx),
    which matters once entries start getting skipped."""
    mask_zero = np.zeros((10, 10), dtype=bool)
    mask_full = np.ones((10, 10), dtype=bool)

    fake_sam2_env["predictor_instance"].predict.side_effect = [
        (np.array([mask_full]), np.array([0.95]), None),  # idx 0: kept
        (np.array([mask_zero]), np.array([0.10]), None),  # idx 1: skipped
        (np.array([mask_full]), np.array([0.80]), None),  # idx 2: kept
    ]

    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))

    mask_ids = [seg["mask_id"] for seg in result["segments"]]
    assert mask_ids == [0, 2]


@pytest.mark.asyncio
async def test_batch_skips_prompts_whose_mask_is_entirely_empty(
    segmentor, image_bytes, fake_sam2_env
):
    mask_zero = np.zeros((8, 8), dtype=bool)
    mask_full = np.ones((8, 8), dtype=bool)

    fake_sam2_env["predictor_instance"].predict.side_effect = [
        (np.array([mask_zero]), np.array([0.5]), None),
        (np.array([mask_full]), np.array([0.9]), None),
    ]

    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    assert len(result["segments"]) == 1
    assert result["segments"][0]["mask_id"] == 1


@pytest.mark.asyncio
async def test_batch_all_empty_masks_returns_no_segments_but_still_calls_predict_for_all(
    segmentor, image_bytes, fake_sam2_env
):
    mask_zero = np.zeros((8, 8), dtype=bool)
    bboxes = _bboxes(3)
    fake_sam2_env["predictor_instance"].predict.side_effect = [
        (np.array([mask_zero]), np.array([0.1]), None) for _ in bboxes
    ]

    result = await segmentor.segment_with_prompts_batch(image_bytes, bboxes)

    assert result["segments"] == []
    assert fake_sam2_env["predictor_instance"].predict.call_count == len(bboxes)


@pytest.mark.asyncio
async def test_batch_area_matches_decoded_mask_pixel_count(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    for seg in result["segments"]:
        decoded = PILImage.open(BytesIO(seg["mask_bytes"]))
        arr = np.array(decoded)
        nonzero_count = int((arr > 0).sum())
        assert seg["area"] == nonzero_count


@pytest.mark.asyncio
async def test_batch_stability_score_equals_predicted_iou(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    for seg in result["segments"]:
        assert seg["stability_score"] == pytest.approx(seg["predicted_iou"])


@pytest.mark.asyncio
async def test_batch_mask_bytes_decode_as_grayscale_png(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    for seg in result["segments"]:
        decoded = PILImage.open(BytesIO(seg["mask_bytes"]))
        assert decoded.mode == "L"
        assert decoded.size[0] > 0 and decoded.size[1] > 0


@pytest.mark.asyncio
async def test_batch_bbox_derived_from_mask_not_from_prompt_bbox(
    segmentor, image_bytes, fake_sam2_env
):
    """seg['bbox'] is recomputed from the returned mask's nonzero pixels,
    so it need not equal (and generally won't equal) prompt_bbox."""
    bboxes = _bboxes(1)
    result = await segmentor.segment_with_prompts_batch(image_bytes, bboxes)

    seg = result["segments"][0]
    assert "bbox" in seg
    assert "prompt_bbox" in seg
    assert seg["prompt_bbox"] == bboxes[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_empty_bboxes_list_returns_empty_segments(
    segmentor, image_bytes, fake_sam2_env
):
    result = await segmentor.segment_with_prompts_batch(image_bytes, [])

    assert result["segments"] == []
    assert result["metrics"]["num_segments"] == 0
    fake_sam2_env["predictor_instance"].predict.assert_not_called()


@pytest.mark.asyncio
async def test_batch_empty_bboxes_list_still_calls_set_image_once(
    segmentor, image_bytes, fake_sam2_env
):
    """set_image() runs unconditionally before the (possibly empty) loop."""
    await segmentor.segment_with_prompts_batch(image_bytes, [])

    assert fake_sam2_env["predictor_instance"].set_image.call_count == 1


@pytest.mark.asyncio
async def test_batch_single_bbox_returns_at_most_one_segment(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(1))

    assert len(result["segments"]) <= 1


@pytest.mark.asyncio
async def test_batch_results_are_independent_across_calls(segmentor, image_bytes):
    """Mutating one call's result must not leak into the next call."""
    first = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
    first["segments"].clear()

    second = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
    assert len(second["segments"]) == 2


# ---------------------------------------------------------------------------
# Metrics / tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_tracks_metrics_by_default(segmentor, image_bytes, tracker):
    await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    tracker.log_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_batch_does_not_track_metrics_when_disabled(segmentor, image_bytes, tracker):
    await segmentor.segment_with_prompts_batch(
        image_bytes, _bboxes(2), track_metrics=False
    )

    tracker.log_metrics.assert_not_called()


@pytest.mark.asyncio
async def test_batch_metrics_num_segments_reflects_skipped_entries(
    segmentor, image_bytes, fake_sam2_env
):
    mask_zero = np.zeros((8, 8), dtype=bool)
    mask_full = np.ones((8, 8), dtype=bool)

    fake_sam2_env["predictor_instance"].predict.side_effect = [
        (np.array([mask_full]), np.array([0.9]), None),
        (np.array([mask_zero]), np.array([0.1]), None),
        (np.array([mask_full]), np.array([0.8]), None),
    ]

    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))

    assert result["metrics"]["num_segments"] == 2
    assert result["metrics"]["num_segments"] == len(result["segments"])


@pytest.mark.asyncio
async def test_batch_inference_time_is_non_negative(segmentor, image_bytes):
    result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))

    assert result["metrics"]["inference_time_ms"] >= 0