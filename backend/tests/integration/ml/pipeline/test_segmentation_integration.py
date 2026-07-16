import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.segmentation import SegmentationMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Segmenter(SegmentationMixin):
    def __init__(self, sam_lama_mode, tracker, validator):
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


def _batch_segments():
    return [
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
    ]


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    segments = [{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "area": 25,
                 "mask_bytes": b"m", "mask_id": 0, "bbox_id": 0,
                 "stability_score": 0.9, "predicted_iou": 0.8}]
    mode.segment_objects = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
    mode.segment_with_prompt = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
    mode.segment_by_polygon = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
    mode.segment_with_prompts_batch = AsyncMock(
        return_value={"segments": _batch_segments(), "image_size": (640, 480)}
    )
    return mode


@pytest.fixture
def tracker() -> MagicMock:
    t = MagicMock(name="ExperimentTracker")
    t.log_metrics = MagicMock()
    return t


@pytest.fixture
def validator() -> MagicMock:
    v = MagicMock(name="Validator")
    v.validate_image_bytes = MagicMock()
    v.validate_bbox = MagicMock()
    return v


@pytest.fixture
def segmenter(sam_lama_mode, tracker, validator) -> _Segmenter:
    return _Segmenter(sam_lama_mode, tracker, validator)


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def bboxes() -> list:
    return [
        {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
        {"x1": 20, "y1": 20, "x2": 30, "y2": 30},
        {"x1": 40, "y1": 40, "x2": 50, "y2": 50},
    ]


class TestSamSegmentWithPromptsBatch:
    async def test_validates_image_bytes(self, segmenter, image_bytes, bboxes, validator):
        await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)

    async def test_validates_every_bbox(self, segmenter, image_bytes, bboxes, validator):
        await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)
        assert validator.validate_bbox.call_count == len(bboxes)
        for bbox, call in zip(bboxes, validator.validate_bbox.call_args_list):
            assert call.args[0] == bbox

    async def test_passes_bboxes_to_mode(self, segmenter, image_bytes, bboxes, sam_lama_mode):
        await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)
        call_kwargs = sam_lama_mode.segment_with_prompts_batch.call_args.kwargs
        assert call_kwargs["image_bytes"] == image_bytes
        assert call_kwargs["bboxes"] == bboxes

    async def test_raises_when_bboxes_empty(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least one bbox"):
            await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=[])

        sam_lama_mode.segment_with_prompts_batch.assert_not_called()

    async def test_raises_when_bboxes_is_none(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least one bbox"):
            await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=None)

        sam_lama_mode.segment_with_prompts_batch.assert_not_called()

    async def test_raises_on_invalid_bbox_and_stops_before_mode_call(
        self, segmenter, image_bytes, bboxes, validator, sam_lama_mode
    ):
        validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

        with pytest.raises(ValueError, match="bbox x1 must be < x2"):
            await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

        sam_lama_mode.segment_with_prompts_batch.assert_not_called()

    async def test_raises_on_invalid_image_bytes(
        self, segmenter, image_bytes, bboxes, validator, sam_lama_mode
    ):
        validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

        with pytest.raises(ValueError, match="Invalid image bytes"):
            await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

        validator.validate_bbox.assert_not_called()
        sam_lama_mode.segment_with_prompts_batch.assert_not_called()

    async def test_track_metrics_false_skips_tracker(self, segmenter, image_bytes, bboxes, tracker):
        await segmenter.sam_segment_with_prompts_batch(
            image_bytes=image_bytes, bboxes=bboxes)
        tracker.log_metrics.assert_not_called()

    async def test_propagates_mode_exception(self, segmenter, image_bytes, bboxes, sam_lama_mode, tracker):
        sam_lama_mode.segment_with_prompts_batch = AsyncMock(
            side_effect=RuntimeError("batch seg crashed")
        )

        with pytest.raises(RuntimeError, match="batch seg crashed"):
            await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)

        tracker.log_metrics.assert_not_called()

    async def test_result_includes_timestamp(self, segmenter, image_bytes, bboxes):
        result = await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=bboxes)
        assert "timestamp" in result
        assert len(result["segments"]) == 2

    async def test_single_bbox_batch(self, segmenter, image_bytes, sam_lama_mode, tracker):
        single = [{"x1": 1, "y1": 1, "x2": 10, "y2": 10}]

        result = await segmenter.sam_segment_with_prompts_batch(image_bytes=image_bytes, bboxes=single)

        sam_lama_mode.segment_with_prompts_batch.assert_called_once_with(
            image_bytes=image_bytes, bboxes=single,
        )
        assert "timestamp" in result