import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.segmentation import SegmentationMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Segmenter(SegmentationMixin):
    def __init__(self, sam_lama_mode, tracker, validator):
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    segments = [{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "area": 25,
                 "mask_bytes": b"m", "mask_id": 0, "bbox_id": 0,
                 "stability_score": 0.9, "predicted_iou": 0.8}]
    mode.segment_objects = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
    mode.segment_with_prompt = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
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
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


class TestSamSegmentObjects:
    async def test_validates_image_bytes(self, segmenter, image_bytes, validator):
        await segmenter.sam_segment_objects(image_bytes=image_bytes)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)

    async def test_passes_min_area_and_max_segments(self, segmenter, image_bytes, sam_lama_mode):
        await segmenter.sam_segment_objects(image_bytes=image_bytes, min_area=1000, max_segments=10)
        call_kwargs = sam_lama_mode.segment_objects.call_args.kwargs
        assert call_kwargs["min_area"] == 1000
        assert call_kwargs["max_segments"] == 10

    async def test_logs_num_segments_metric(self, segmenter, image_bytes, tracker):
        await segmenter.sam_segment_objects(image_bytes=image_bytes)
        payload = tracker.log_metrics.call_args.args[0]
        assert payload["num_segments"] == 1
        assert payload["operation"] == "sam_segment_auto"

    async def test_track_metrics_false_skips_tracker(self, segmenter, image_bytes, tracker):
        await segmenter.sam_segment_objects(image_bytes=image_bytes, track_metrics=False)
        tracker.log_metrics.assert_not_called()

    async def test_propagates_mode_exception(self, segmenter, image_bytes, sam_lama_mode, tracker):
        sam_lama_mode.segment_objects = AsyncMock(side_effect=RuntimeError("sam crashed"))

        with pytest.raises(RuntimeError, match="sam crashed"):
            await segmenter.sam_segment_objects(image_bytes=image_bytes)

        tracker.log_metrics.assert_not_called()


class TestSamSegmentWithPrompt:
    async def test_raises_when_no_prompt_given(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least one of"):
            await segmenter.sam_segment_with_prompt(image_bytes=image_bytes)

        sam_lama_mode.segment_with_prompt.assert_not_called()

    async def test_raises_when_points_and_labels_mismatch(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="must have the same length"):
            await segmenter.sam_segment_with_prompt(
                image_bytes=image_bytes, point_coords=[(1, 1), (2, 2)], point_labels=[1],
            )

        sam_lama_mode.segment_with_prompt.assert_not_called()

    async def test_accepts_points_without_labels(self, segmenter, image_bytes, sam_lama_mode):
        await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, point_coords=[(1, 1)])
        sam_lama_mode.segment_with_prompt.assert_called_once()

    async def test_validates_bbox_when_provided(self, segmenter, image_bytes, bbox, validator):
        await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)
        validator.validate_bbox.assert_called_once_with(bbox)

    async def test_skips_bbox_validation_when_only_points(self, segmenter, image_bytes, validator):
        await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, point_coords=[(1, 1)], point_labels=[1])
        validator.validate_bbox.assert_not_called()

    async def test_raises_on_invalid_bbox(self, segmenter, image_bytes, bbox, validator, sam_lama_mode):
        validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

        with pytest.raises(ValueError, match="bbox x1 must be < x2"):
            await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)

        sam_lama_mode.segment_with_prompt.assert_not_called()

    async def test_logs_has_points_and_has_bbox_flags(self, segmenter, image_bytes, bbox, tracker):
        await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)
        payload = tracker.log_metrics.call_args.args[0]
        assert payload["has_bbox"] is True
        assert payload["has_points"] is False

    async def test_propagates_mode_exception(self, segmenter, image_bytes, bbox, sam_lama_mode, tracker):
        sam_lama_mode.segment_with_prompt = AsyncMock(side_effect=RuntimeError("prompt seg crashed"))

        with pytest.raises(RuntimeError, match="prompt seg crashed"):
            await segmenter.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)

        tracker.log_metrics.assert_not_called()