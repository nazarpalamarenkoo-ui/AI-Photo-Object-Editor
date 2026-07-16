import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.detection import DetectionMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Detector(DetectionMixin):
    def __init__(self, yolo_lama_mode, tracker, validator):
        self.yolo_lama_mode = yolo_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def yolo_lama_mode() -> MagicMock:
    mode = MagicMock(name="YoloLamaMode")
    mode.detect_objects = AsyncMock(return_value={
        "detections": [
            {"bbox_id": 0, "class": "car", "confidence": 0.9,
             "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
            {"bbox_id": 1, "class": "person", "confidence": 0.7,
             "bbox": {"x1": 60, "y1": 60, "x2": 90, "y2": 90}},
        ],
        "image_size": (640, 480),
        "metrics": {"inference_time": 0.3},
    })
    return mode


@pytest.fixture
def tracker() -> MagicMock:
    t = MagicMock(name="ExperimentTracker")
    t.log_detection_metrics = MagicMock()
    return t


@pytest.fixture
def validator() -> MagicMock:
    v = MagicMock(name="Validator")
    v.validate_image_bytes = MagicMock()
    return v


@pytest.fixture
def detector(yolo_lama_mode, tracker, validator) -> _Detector:
    return _Detector(yolo_lama_mode, tracker, validator)


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


class TestDetectObjects:
    async def test_validates_image_before_inference(self, detector, image_bytes, validator):
        await detector.detect_objects(image_bytes=image_bytes)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)

    async def test_passes_conf_threshold_and_classes_to_mode(self, detector, image_bytes, yolo_lama_mode):
        await detector.detect_objects(image_bytes=image_bytes, conf_threshold=0.8, classes=["car"])

        call_kwargs = yolo_lama_mode.detect_objects.call_args.kwargs
        assert call_kwargs["conf_threshold"] == 0.8
        assert call_kwargs["classes"] == ["car"]

    async def test_track_metrics_false_skips_tracker(self, detector, image_bytes, tracker):
        await detector.detect_objects(image_bytes=image_bytes)
        tracker.log_detection_metrics.assert_not_called()

    async def test_no_metrics_key_skips_tracker(self, detector, image_bytes, yolo_lama_mode, tracker):
        yolo_lama_mode.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (640, 480), "metrics": None,
        })

        await detector.detect_objects(image_bytes=image_bytes)

        tracker.log_detection_metrics.assert_not_called()

    async def test_adds_timestamp_to_result(self, detector, image_bytes):
        result = await detector.detect_objects(image_bytes=image_bytes)
        assert "timestamp" in result

    async def test_raises_on_invalid_image_bytes(self, detector, image_bytes, validator, yolo_lama_mode):
        validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

        with pytest.raises(ValueError, match="Invalid image bytes"):
            await detector.detect_objects(image_bytes=image_bytes)

        yolo_lama_mode.detect_objects.assert_not_called()

    async def test_propagates_mode_exception(self, detector, image_bytes, yolo_lama_mode, tracker):
        yolo_lama_mode.detect_objects = AsyncMock(side_effect=RuntimeError("yolo crashed"))

        with pytest.raises(RuntimeError, match="yolo crashed"):
            await detector.detect_objects(image_bytes=image_bytes)

        tracker.log_detection_metrics.assert_not_called()