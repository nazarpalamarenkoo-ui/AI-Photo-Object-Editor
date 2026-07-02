from typing import Dict, List, Optional

from app.db.models.detection import Detection
from app.services.ml.base_ml_service import BaseMLService


class DetectorService(BaseMLService):
    """
    Handles YOLO object detection.

    Workflow:
        Upload image -> detect_objects -> persist to DB + Redis cache
    """

    async def detect_objects(
        self,
        image_id: int,
        user_id: int,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Detect objects in the uploaded image using YOLO.

        Args:
            image_id:       ID of image to process
            user_id:        ID of requesting user
            conf_threshold: Confidence threshold (0.0-1.0, default: 0.5)
            classes:        Optional class name filter list

        Returns:
            Dict:
                - detections:  List[Dict] — detected objects with bbox
                - image_size:  Tuple[int, int] — (width, height)
                - metrics:     Dict
                - timestamp:   str ISO

        Raises:
            ValueError: If image not found or unauthorized.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        result = await self.pipeline.detect_objects(
            image_bytes=image_bytes,
            conf_threshold=conf_threshold,
            classes=classes,
            track_metrics=True,
        )

        detections = result["detections"]

        db_detections = [
            Detection(
                image_id=image_id,
                bbox_id=det["bbox_id"],
                detected_class=det["detected_class"],
                confidence=det["confidence"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
            )
            for det in detections
        ]

        await self.detection_repo.delete_by_image(image_id)
        await self.redis_storage.delete(f"image:{image_id}:detections")
        await self.detection_repo.create_many(db_detections)
        await self.redis_storage.cache_detections(
            image_id=image_id, detections=detections, ttl=3600
        )

        return result

    def get_supported_classes(self) -> List[str]:
        """Return list of supported YOLO class names (80 COCO classes)."""
        return self.pipeline.get_supported_classes()