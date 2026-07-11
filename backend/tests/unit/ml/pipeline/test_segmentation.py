import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def bboxes() -> list:
    """Not provided by conftest.py — defined locally for the batch tests below."""
    return [
        {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
        {"x1": 20, "y1": 20, "x2": 30, "y2": 30},
        {"x1": 40, "y1": 40, "x2": 50, "y2": 50},
    ]


@pytest.fixture
def host(host):
    host.sam_lama_mode.segment_with_prompts_batch = AsyncMock(
        return_value={
            "segments": [
                {
                    "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
                    "prompt_bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
                    "area": 25,
                    "mask_bytes": b"m",
                    "mask_id": 0,
                    "bbox_id": 0,
                    "stability_score": 0.9,
                    "predicted_iou": 0.8,
                },
                {
                    "bbox": {"x1": 20, "y1": 20, "x2": 25, "y2": 25},
                    "prompt_bbox": {"x1": 20, "y1": 20, "x2": 30, "y2": 30},
                    "area": 30,
                    "mask_bytes": b"n",
                    "mask_id": 1,
                    "bbox_id": 1,
                    "stability_score": 0.85,
                    "predicted_iou": 0.75,
                },
            ],
            "image_size": (640, 480),
        }
    )
    return host



async def test_sam_segment_with_prompts_batch_success(host, image_bytes, bboxes):
    result = await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    assert len(result["segments"]) == 2
    assert result["image_size"] == (640, 480)
    assert "timestamp" in result


async def test_sam_segment_with_prompts_batch_validates_image_bytes(host, image_bytes, bboxes):
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)


async def test_sam_segment_with_prompts_batch_validates_each_bbox_in_order(host, image_bytes, bboxes):
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    calls = host.validator.validate_bbox.call_args_list
    assert [c.args[0] for c in calls] == bboxes


async def test_sam_segment_with_prompts_batch_calls_mode_with_params(host, image_bytes, bboxes):
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    host.sam_lama_mode.segment_with_prompts_batch.assert_called_once_with(
        image_bytes=image_bytes, bboxes=bboxes,
    )


async def test_sam_segment_with_prompts_batch_single_bbox_still_works(host, image_bytes):
    single = [{"x1": 1, "y1": 1, "x2": 10, "y2": 10}]

    result = await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=single)

    host.sam_lama_mode.segment_with_prompts_batch.assert_called_once_with(
        image_bytes=image_bytes, bboxes=single,
    )
    assert "timestamp" in result


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

async def test_sam_segment_with_prompts_batch_empty_bboxes_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least one bbox"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=[])

    host.sam_lama_mode.segment_with_prompts_batch.assert_not_called()


async def test_sam_segment_with_prompts_batch_none_bboxes_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least one bbox"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=None)

    host.sam_lama_mode.segment_with_prompts_batch.assert_not_called()


async def test_sam_segment_with_prompts_batch_empty_bboxes_skips_bbox_validation(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least one bbox"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=[])

    host.validator.validate_bbox.assert_not_called()


async def test_sam_segment_with_prompts_batch_raises_on_invalid_bbox(host, image_bytes, bboxes):
    host.validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    host.sam_lama_mode.segment_with_prompts_batch.assert_not_called()


async def test_sam_segment_with_prompts_batch_stops_validating_at_first_invalid_bbox(
    host, image_bytes, bboxes
):
    """Only the bboxes up to and including the invalid one should be
    checked — validation should short-circuit rather than validate the
    remaining bboxes after a failure."""
    host.validator.validate_bbox.side_effect = [None, ValueError("bad bbox"), None]

    with pytest.raises(ValueError, match="bad bbox"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    assert host.validator.validate_bbox.call_count == 2


async def test_sam_segment_with_prompts_batch_invalid_image_bytes_raises_before_bbox_validation(
    host, image_bytes, bboxes
):
    host.validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    host.validator.validate_bbox.assert_not_called()
    host.sam_lama_mode.segment_with_prompts_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Metrics / tracking
# ---------------------------------------------------------------------------

async def test_sam_segment_with_prompts_batch_tracker_called_when_enabled(host, image_bytes, bboxes):
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes, track_metrics=True)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_segment_prompts_batch"
    assert payload["num_bboxes"] == len(bboxes)
    assert payload["num_segments"] == 2


async def test_sam_segment_with_prompts_batch_tracker_not_called_when_disabled(host, image_bytes, bboxes):
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes, track_metrics=False)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_segment_with_prompts_batch_num_bboxes_reflects_input_not_output(
    host, image_bytes, bboxes
):
    """num_bboxes should count the input prompts, independent of how many
    segments actually came back (some prompts can yield empty masks)."""
    await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["num_bboxes"] == 3
    assert payload["num_segments"] == 2
    assert payload["num_bboxes"] != payload["num_segments"]


async def test_sam_segment_with_prompts_batch_tracker_not_called_on_validation_error(
    host, image_bytes
):
    with pytest.raises(ValueError):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=[])

    host.tracker.log_metrics.assert_not_called()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

async def test_sam_segment_with_prompts_batch_propagates_mode_exception(host, image_bytes, bboxes):
    host.sam_lama_mode.segment_with_prompts_batch = AsyncMock(
        side_effect=RuntimeError("batch segmentation failed")
    )

    with pytest.raises(RuntimeError, match="batch segmentation failed"):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)


async def test_sam_segment_with_prompts_batch_tracker_not_called_when_mode_raises(
    host, image_bytes, bboxes
):
    host.sam_lama_mode.segment_with_prompts_batch = AsyncMock(
        side_effect=RuntimeError("batch segmentation failed")
    )

    with pytest.raises(RuntimeError):
        await host.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

    host.tracker.log_metrics.assert_not_called()