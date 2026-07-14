import time
from typing import Dict, List, Optional
from datetime import datetime

from app.ml.modes.yolo_lama_mode import YoloLamaMode
from app.ml.experiment_tracker import ExperimentTracker
from app.ml.pipeline.validator import Validator
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class DetectionMixin:
    yolo_lama_mode: YoloLamaMode
    tracker: ExperimentTracker
    validator: Validator

    async def detect_objects(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        track_metrics: bool = True,
    ) -> Dict:
        """
        Detect objects in image using YOLO.

        Args:
            image_bytes:     Input image bytes
            conf_threshold:  Confidence threshold (0.0–1.0, default: 0.5)
            classes:         Optional list of class names to filter
            track_metrics:   Track metrics to MLflow (default: True)

        Returns:
            Dict:
                - detections:  List[Dict] — bbox_id, class, confidence, bbox
                - image_size:  Tuple[int, int] — (width, height)
                - metrics:     Dict
                - timestamp:   str — ISO timestamp
        """
        start_time = time.time()

        with log_execution(
            "pipeline_detect_objects",
            logger=logger,
            conf_threshold=conf_threshold,
            classes=classes,
        ):
            self.validator.validate_image_bytes(image_bytes)

            result = await self.yolo_lama_mode.detect_objects(
                image_bytes=image_bytes,
                conf_threshold=conf_threshold,
                classes=classes,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics and result.get("metrics"):
                inference_time = time.time() - start_time
                detections = result["detections"]
                avg_confidence = (
                    sum(d["confidence"] for d in detections) / len(detections)
                    if detections
                    else None
                )
                self.tracker.log_detection_metrics(
                    num_detections=len(detections),
                    inference_time=inference_time,
                    avg_confidence=avg_confidence,
                    conf_threshold=conf_threshold,
                )

        return result