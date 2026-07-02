import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.detection import DetectionMixin
from app.ml.pipeline.removal import RemovalMixin
from app.ml.pipeline.replacement import ReplacementMixin
from app.ml.pipeline.segmentation import SegmentationMixin
from app.ml.pipeline.extraction import ExtractionMixin


class _Host(
    DetectionMixin,
    RemovalMixin,
    ReplacementMixin,
    SegmentationMixin,
    ExtractionMixin,
):
    """Test double combining all pipeline mixins with mocked collaborators."""

    def __init__(self, yolo_lama_mode, sam_lama_mode, tracker, validator):
        self.yolo_lama_mode = yolo_lama_mode
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def mask_bytes() -> bytes:
    return b"mask"


@pytest.fixture
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


@pytest.fixture
def yolo_lama_mode() -> MagicMock:
    mode = MagicMock(name="YoloLamaMode")
    mode.detect_objects = AsyncMock(return_value={
        "detections": [
            {
                "bbox_id": 0,
                "class": "car",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            },
        ],
        "image_size": (640, 480),
        "metrics": {"inference_time": 0.1},
    })
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"removed_image"})
    mode.remove_multiple_objects = AsyncMock(return_value={"result_bytes": b"removed_multiple_image"})
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"replaced_image"})
    mode.get_supported_classes = MagicMock(return_value=["car", "person"])
    return mode


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    mode.segment_objects = AsyncMock(return_value={
        "segments": [{
            "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
            "area": 25,
            "mask_bytes": b"m",
            "mask_id": 0,
            "bbox_id": 0,
            "stability_score": 0.9,
            "predicted_iou": 0.8,
        }],
        "image_size": (640, 480),
    })
    mode.segment_with_prompt = AsyncMock(return_value={
        "segments": [{
            "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
            "area": 25,
            "mask_bytes": b"m",
            "mask_id": 0,
            "bbox_id": 0,
            "stability_score": 0.9,
            "predicted_iou": 0.8,
        }],
        "image_size": (640, 480),
    })
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"sam_removed_image"})
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"sam_replaced_image"})
    mode.extract_object = AsyncMock(return_value={
        "extracted_bytes": b"extracted_object",
        "cropped_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
        "original_size": (640, 480),
        "object_size": (5, 5),
        "area_pixels": 350,
    })
    mode.paste_extracted_object = AsyncMock(return_value={
        "result_bytes": b"pasted_image",
        "paste_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
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
def host(yolo_lama_mode, sam_lama_mode, tracker, validator) -> _Host:
    return _Host(
        yolo_lama_mode=yolo_lama_mode,
        sam_lama_mode=sam_lama_mode,
        tracker=tracker,
        validator=validator,
    )