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
    mode.segment_by_polygon = AsyncMock(return_value={"segments": segments, "image_size": (640, 480)})
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


@pytest.fixture
def points() -> list:
    return [(1, 1), (10, 1), (10, 10), (1, 10)]


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


class TestSamSegmentByPolygon:
    async def test_validates_image_bytes(self, segmenter, image_bytes, points, validator):
        await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)

    async def test_raises_when_too_few_points(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least 3 points"):
            await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=[(1, 1), (2, 2)])

        sam_lama_mode.segment_by_polygon.assert_not_called()

    async def test_raises_when_points_is_none(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least 3 points"):
            await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=None)

        sam_lama_mode.segment_by_polygon.assert_not_called()

    async def test_raises_when_points_is_empty(self, segmenter, image_bytes, sam_lama_mode):
        with pytest.raises(ValueError, match="Provide at least 3 points"):
            await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=[])

        sam_lama_mode.segment_by_polygon.assert_not_called()

    async def test_passes_smooth_smoothing_factor_and_feather_px(self, segmenter, image_bytes, points, sam_lama_mode):
        await segmenter.sam_segment_by_polygon(
            image_bytes=image_bytes, points=points, smooth=False, smoothing_factor=2.5, feather_px=5,
        )
        call_kwargs = sam_lama_mode.segment_by_polygon.call_args.kwargs
        assert call_kwargs["points"] == points
        assert call_kwargs["smooth"] is False
        assert call_kwargs["smoothing_factor"] == 2.5
        assert call_kwargs["feather_px"] == 5

    async def test_uses_default_smooth_params(self, segmenter, image_bytes, points, sam_lama_mode):
        await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)
        call_kwargs = sam_lama_mode.segment_by_polygon.call_args.kwargs
        assert call_kwargs["smooth"] is True
        assert call_kwargs["smoothing_factor"] == 0.0
        assert call_kwargs["feather_px"] == 0

    async def test_logs_num_points_metric(self, segmenter, image_bytes, points, tracker):
        await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)
        payload = tracker.log_metrics.call_args.args[0]
        assert payload["operation"] == "sam_segment_polygon"
        assert payload["num_points"] == len(points)

    async def test_track_metrics_false_skips_tracker(self, segmenter, image_bytes, points, tracker):
        await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points, track_metrics=False)
        tracker.log_metrics.assert_not_called()

    async def test_raises_on_invalid_image_bytes(self, segmenter, image_bytes, points, validator, sam_lama_mode):
        validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

        with pytest.raises(ValueError, match="Invalid image bytes"):
            await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)

        sam_lama_mode.segment_by_polygon.assert_not_called()

    async def test_propagates_mode_exception(self, segmenter, image_bytes, points, sam_lama_mode, tracker):
        sam_lama_mode.segment_by_polygon = AsyncMock(side_effect=RuntimeError("polygon seg crashed"))

        with pytest.raises(RuntimeError, match="polygon seg crashed"):
            await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)

        tracker.log_metrics.assert_not_called()

    async def test_result_includes_timestamp(self, segmenter, image_bytes, points):
        result = await segmenter.sam_segment_by_polygon(image_bytes=image_bytes, points=points)
        assert "timestamp" in result
        assert len(result["segments"]) == 1