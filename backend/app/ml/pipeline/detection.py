import time
from typing import Dict, List, Optional
from datetime import datetime

from app.ml.modes.yolo_lama_mode import YoloLamaMode
from app.ml.pipeline.validator import Validator
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class DetectionMixin:
    yolo_lama_mode: YoloLamaMode
    validator: Validator

    async def detect_objects(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Detect objects in image using YOLO.

        Args:
            image_bytes:     Input image bytes
            conf_threshold:  Confidence threshold (0.0–1.0, default: 0.5)
            classes:         Optional list of class names to filter
            

        Returns:
            Dict:
                - detections:  List[Dict] — bbox_id, class, confidence, bbox
                - image_size:  Tuple[int, int] — (width, height)
                - metrics:     Dict
                - timestamp:   str — ISO timestamp
        """
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

        return result