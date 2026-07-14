from typing import List, Optional

from app.config.settings import settings
from app.config.device_manager import DeviceManager
from app.ml.modes.yolo_lama_mode import YoloLamaMode, get_yolo_lama_mode
from app.ml.modes.sam_lama_mode import SAMLamaMode, get_sam_mode
from app.ml.experiment_tracker import ExperimentTracker, get_tracker
from app.ml.pipeline.validator import Validator, get_validator
from app.core.logging import get_logger

from .detection import DetectionMixin
from .removal import RemovalMixin
from .replacement import ReplacementMixin
from .segmentation import SegmentationMixin
from .extraction import ExtractionMixin

logger = get_logger(__name__)


class MLPipeline(
    DetectionMixin,
    RemovalMixin,
    ReplacementMixin,
    SegmentationMixin,
    ExtractionMixin,
):
    """
    Main ML pipeline orchestrator.

    Provides high-level interface for:

    YOLO + Lama mode
    1. Object Detection (YOLO)
    2. Object Removal (YOLO + LaMa + processors)
    3. Object Replacement (YOLO + LaMa + processors)
    4. Multiple Object Removal (YOLO + LaMa + processors)

    SAM + Lama mode
    5. Auto Segmentation (SAM2 — no prompts)
    6. Prompted Segmentation (SAM2 — points / bbox)
    7. Object Removal (SAM mask → LaMa → EdgeBlend)
    8. Object Replacement (SAM mask → LaMa → composite → ColorMatch)
    9. Object Extraction (SAM mask → RGBA crop)
    10. Paste Extracted Object (RGBA → scale → composite → ColorMatch → EdgeBlend)

    Handles:
    1. ML Operations
    2. Metrics Tracking (MLflow)
    3. Error Handling
    4. Input Validation
    """

    def __init__(
        self,
        mode: Optional[YoloLamaMode] = None,
        sam_mode: Optional[SAMLamaMode] = None,
        tracker: Optional[ExperimentTracker] = None,
        validator: Optional[Validator] = None,
    ):
        """
        Initialize ML Pipeline.

        Args:
            mode:       YoloLamaMode instance (default: auto-created)
            sam_mode:   SAMLamaMode instance (default: auto-created)
            tracker:    ExperimentTracker for MLflow (default: auto-created)
            validator:  Validator instance (default: auto-created)
            device:     Device to use ('cuda' or 'cpu')
        """
        self.device = DeviceManager.get(settings.DEFAULT_DEVICE)
        self.yolo_lama_mode = mode or get_yolo_lama_mode()
        self.sam_lama_mode = sam_mode or get_sam_mode()
        self.tracker = tracker or get_tracker()
        self.validator = validator or get_validator()
        logger.info("ml_pipeline_initialized", device=str(self.device))

    def get_supported_classes(self) -> List[str]:
        """Return list of supported YOLO classes (80 COCO classes)."""
        return self.yolo_lama_mode.get_supported_classes()

import threading
_pipeline_instance = None
_pipeline_lock = threading.Lock()

def get_pipeline() -> MLPipeline:
    """
    Singleton getter for MLPipeline.

    Args:
        device: Device to use ('cuda' or 'cpu')

    Returns:
        MLPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = MLPipeline()
    return _pipeline_instance