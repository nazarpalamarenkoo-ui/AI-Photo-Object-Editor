import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_detect_objects_success(host, image_bytes):
    result = await host.detect_objects(image_bytes=image_bytes)

    assert result["detections"][0]["class"] == "car"
    assert result["image_size"] == (640, 480)
    assert "timestamp" in result


async def test_detect_objects_validates_image_bytes(host, image_bytes):
    await host.detect_objects(image_bytes=image_bytes)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)


async def test_detect_objects_calls_yolo_lama_mode(host, image_bytes):
    await host.detect_objects(image_bytes=image_bytes, conf_threshold=0.7, classes=["car"])

    host.yolo_lama_mode.detect_objects.assert_called_once_with(
        image_bytes=image_bytes, conf_threshold=0.7, classes=["car"],
    )


async def test_detect_objects_adds_timestamp(host, image_bytes):
    result = await host.detect_objects(image_bytes=image_bytes)

    assert isinstance(result["timestamp"], str)
    assert len(result["timestamp"]) > 0


async def test_detect_objects_tracker_called_when_metrics_present(host, image_bytes):
    await host.detect_objects(image_bytes=image_bytes, conf_threshold=0.6)
    host.tracker.log_detection_metrics.assert_not_called()


async def test_detect_objects_tracker_not_called_when_track_metrics_false(host, image_bytes):
    await host.detect_objects(image_bytes=image_bytes)

    host.tracker.log_detection_metrics.assert_not_called()


async def test_detect_objects_tracker_not_called_when_metrics_missing(host, image_bytes):
    host.yolo_lama_mode.detect_objects = AsyncMock(return_value={
        "detections": [],
        "image_size": (640, 480),
    })

    await host.detect_objects(image_bytes=image_bytes)

    host.tracker.log_detection_metrics.assert_not_called()


async def test_detect_objects_empty_detections_metrics_passthrough(host, image_bytes):
    host.yolo_lama_mode.detect_objects = AsyncMock(return_value={
        "detections": [],
        "image_size": (640, 480),
        "metrics": {"inference_time": 0.1},
    })
    result = await host.detect_objects(image_bytes=image_bytes)
    assert result["detections"] == []
    host.tracker.log_detection_metrics.assert_not_called()


async def test_detect_objects_invalid_image_raises_validation_error(host, image_bytes):
    host.validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.detect_objects(image_bytes=image_bytes)

    host.yolo_lama_mode.detect_objects.assert_not_called()


async def test_detect_objects_propagates_mode_exception(host, image_bytes):
    host.yolo_lama_mode.detect_objects = AsyncMock(side_effect=RuntimeError("inference failed"))

    with pytest.raises(RuntimeError, match="inference failed"):
        await host.detect_objects(image_bytes=image_bytes)