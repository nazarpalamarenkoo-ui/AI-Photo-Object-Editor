import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.pipeline import MLPipeline

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def yolo_lama_mode() -> MagicMock:
    mode = MagicMock(name="YoloLamaMode")
    mode.detect_objects = AsyncMock(return_value={
        "detections": [
            {"bbox_id": 0, "class": "car", "confidence": 0.9,
             "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
        ],
        "image_size": (640, 480),
        "metrics": {"inference_time": 0.3},
    })
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"removed"})
    mode.remove_multiple_objects = AsyncMock(return_value={"result_bytes": b"removed_multi"})
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"replaced"})
    mode.get_supported_classes = MagicMock(return_value=["car", "person"])
    return mode


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    mode.segment_objects = AsyncMock(return_value={
        "segments": [{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "area": 25,
                      "mask_bytes": b"m", "mask_id": 0, "bbox_id": 0,
                      "stability_score": 0.9, "predicted_iou": 0.8}],
        "image_size": (640, 480),
    })
    mode.segment_with_prompt = AsyncMock(return_value={
        "segments": [{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "area": 25,
                      "mask_bytes": b"m", "mask_id": 0, "bbox_id": 0,
                      "stability_score": 0.9, "predicted_iou": 0.8}],
        "image_size": (640, 480),
    })
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"sam_removed"})
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"sam_replaced"})
    mode.extract_object = AsyncMock(return_value={
        "extracted_bytes": b"extracted", "cropped_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
        "original_size": (640, 480), "object_size": (5, 5), "area_pixels": 25,
    })
    mode.paste_extracted_object = AsyncMock(return_value={
        "result_bytes": b"pasted", "paste_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
        "object_size": (5, 5),
    })
    return mode


@pytest.fixture
def tracker() -> MagicMock:
    t = MagicMock(name="ExperimentTracker")
    t.log_metrics = MagicMock()
    t.log_detection_metrics = MagicMock()
    return t


@pytest.fixture
def validator() -> MagicMock:
    v = MagicMock(name="Validator")
    v.validate_image_bytes = MagicMock()
    v.validate_mask_bytes = MagicMock()
    v.validate_bbox = MagicMock()
    return v


@pytest.fixture
def pipeline(yolo_lama_mode, sam_lama_mode, tracker, validator) -> MLPipeline:
    return MLPipeline(
        mode=yolo_lama_mode,
        sam_mode=sam_lama_mode,
        tracker=tracker,
        validator=validator,
    )


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def mask_bytes() -> bytes:
    return b"mask"


@pytest.fixture
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


async def test_detect_objects_orchestration(pipeline, image_bytes, validator, yolo_lama_mode, tracker):
    result = await pipeline.detect_objects(image_bytes=image_bytes)

    validator.validate_image_bytes.assert_called_once_with(image_bytes)
    yolo_lama_mode.detect_objects.assert_called_once()
    tracker.log_detection_metrics.assert_called_once()
    assert result["detections"][0]["class"] == "car"
    assert "timestamp" in result


async def test_remove_object_orchestration(pipeline, image_bytes, bbox, validator, yolo_lama_mode, tracker):
    result = await pipeline.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    validator.validate_image_bytes.assert_called_once_with(image_bytes)
    validator.validate_bbox.assert_called_once_with(bbox)
    yolo_lama_mode.remove_object.assert_called_once()
    tracker.log_metrics.assert_called_once()
    assert result["result_bytes"] == b"removed"


async def test_remove_multiple_objects_orchestration(pipeline, image_bytes, bbox, yolo_lama_mode, tracker):
    result = await pipeline.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox, bbox])

    yolo_lama_mode.remove_multiple_objects.assert_called_once()
    payload = tracker.log_metrics.call_args.args[0]
    assert payload["num_objects"] == 2
    assert result["result_bytes"] == b"removed_multi"


async def test_replace_object_orchestration(pipeline, image_bytes, bbox, validator, yolo_lama_mode, tracker):
    result = await pipeline.replace_object(
        image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
    )

    assert validator.validate_image_bytes.call_count == 2
    yolo_lama_mode.replace_object.assert_called_once()
    tracker.log_metrics.assert_called_once()
    assert result["result_bytes"] == b"replaced"


async def test_sam_segment_objects_orchestration(pipeline, image_bytes, sam_lama_mode, tracker):
    result = await pipeline.sam_segment_objects(image_bytes=image_bytes)

    sam_lama_mode.segment_objects.assert_called_once()
    tracker.log_metrics.assert_called_once()
    assert len(result["segments"]) == 1


async def test_sam_segment_with_prompt_orchestration(pipeline, image_bytes, bbox, sam_lama_mode, tracker):
    result = await pipeline.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)

    sam_lama_mode.segment_with_prompt.assert_called_once()
    assert len(result["segments"]) == 1


async def test_sam_remove_object_orchestration(pipeline, image_bytes, mask_bytes, sam_lama_mode, validator):
    result = await pipeline.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

    validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
    sam_lama_mode.remove_object.assert_called_once()
    assert result["result_bytes"] == b"sam_removed"


async def test_sam_replace_object_orchestration(pipeline, image_bytes, mask_bytes, bbox, sam_lama_mode):
    result = await pipeline.sam_replace_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        replacement_image_bytes=image_bytes,
    )

    sam_lama_mode.replace_object.assert_called_once()
    assert result["result_bytes"] == b"sam_replaced"


async def test_sam_extract_object_orchestration(pipeline, image_bytes, mask_bytes, bbox, sam_lama_mode):
    result = await pipeline.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

    sam_lama_mode.extract_object.assert_called_once()
    assert result["extracted_bytes"] == b"extracted"


async def test_sam_paste_extracted_object_orchestration(pipeline, image_bytes, bbox, sam_lama_mode):
    result = await pipeline.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
    )

    sam_lama_mode.paste_extracted_object.assert_called_once()
    assert result["result_bytes"] == b"pasted"


async def test_validation_failure_prevents_mode_invocation(pipeline, image_bytes, bbox, validator, yolo_lama_mode):
    validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await pipeline.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    yolo_lama_mode.remove_object.assert_not_called()


async def test_mode_exception_propagates_through_pipeline(pipeline, image_bytes, bbox, yolo_lama_mode, tracker):
    yolo_lama_mode.remove_object = AsyncMock(side_effect=RuntimeError("inference crashed"))

    with pytest.raises(RuntimeError, match="inference crashed"):
        await pipeline.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    tracker.log_metrics.assert_not_called()


async def test_get_supported_classes_orchestration(pipeline, yolo_lama_mode):
    classes = pipeline.get_supported_classes()

    yolo_lama_mode.get_supported_classes.assert_called_once()
    assert classes == ["car", "person"]