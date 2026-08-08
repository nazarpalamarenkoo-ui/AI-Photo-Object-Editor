from datetime import datetime
from typing import Dict, List, Optional

from app.db.models.detection import Detection
from app.db.models.image import Image
from app.services.ml.base_ml_service import BaseMLService
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)

DETECTION_FLOOR_THRESHOLD = 0.05


class DetectorService(BaseMLService):
    """
    Handles YOLO object detection and persists results.

    Workflow:
        Upload image -> detect_objects -> persist to DB + Redis cache

    """
    DEFAULT_MODEL_NAME = "yolov10m"
    DEFAULT_MODEL_VERSION = "unknown"

    async def detect_objects(
        self,
        image_id: int,
        user_id: int,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        force_rerun: bool = False,
    ) -> Dict:
        """
        Detect (or read cached detections for) the current version's
        content of an image.

        Args:
            image_id:       ID of image to process
            user_id:        ID of requesting user
            conf_threshold: Confidence threshold to filter/return (0.0-1.0)
            classes:        Optional class name filter (applied on read,
                             never restricts what gets persisted)
            force_rerun:    If True, discard cached detections for this
                             content and run YOLO again

        Returns:
            Dict: detections, image_size, metrics, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or has no current version.
        """
        with log_execution(
            "service_detect_objects",
            logger=logger,
            image_id=image_id,
            conf_threshold=conf_threshold,
            force_rerun=force_rerun,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            content_id = version.content_id

            existing = await self.detection_repo.get_by_content(
                content_id, active_only=False
            )

            if existing and force_rerun:
                await self.detection_repo.delete_by_content(content_id)
                await self.redis_storage.delete(f"image_content:{content_id}:detections")
                existing = []

            if existing:
                filtered = self._filter_cached(existing, conf_threshold, classes)
                logger.debug(
                    "detections_served_from_cache",
                    image_version_id=version.id,
                    content_id=content_id,
                    total_cached=len(existing),
                    returned=len(filtered),
                )
                return self._build_response(filtered, image, cache_hit=True)

            # No cache for this content yet — run the model with a low
            # floor threshold so we capture everything worth keeping.
            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
            run_threshold = min(conf_threshold, DETECTION_FLOOR_THRESHOLD)

            result = await self.pipeline.detect_objects(
                image_bytes=image_bytes,
                conf_threshold=run_threshold,
                classes=None,  # always capture all classes; filter on read
            )

            raw_detections = result["detections"]
            metrics = result.get("metrics", {})
            meta = self._extract_model_meta(result, self.DEFAULT_MODEL_NAME, self.DEFAULT_MODEL_VERSION)
            offset = await self.detection_repo.max_bbox_id(content_id) + 1

            db_detections = [
                Detection(
                    content_id=content_id,
                    bbox_id=offset + idx,
                    detected_class=det.get("detected_class", "unknown"),
                    confidence=det["confidence"],
                    x1=det["x1"],
                    y1=det["y1"],
                    x2=det["x2"],
                    y2=det["y2"],
                    model_name=meta.model_name,
                    model_version=meta.model_version,
                    inference_time_ms=meta.inference_time_ms,
                )
                for idx, det in enumerate(raw_detections)
            ]

            persisted = await self.detection_repo.create_many(db_detections)

            logger.info(
                "detections_persisted",
                image_version_id=version.id,
                content_id=content_id,
                num_detections=len(persisted),
                floor_threshold=run_threshold,
            )

            filtered = self._filter_cached(persisted, conf_threshold, classes)

        return self._build_response(filtered, image, cache_hit=False, raw_metrics=metrics)

    def get_supported_classes(self) -> List[str]:
        """Return list of supported YOLO class names (80 COCO classes)."""
        return self.pipeline.get_supported_classes()

    @staticmethod
    def _filter_cached(
        detections: List[Detection],
        conf_threshold: float,
        classes: Optional[List[str]],
    ) -> List[Detection]:
        """
        Threshold/class filter over already-persisted rows. Soft-deleted
        (is_active=False) rows never resurrect here, regardless of threshold —
        that's the whole point of scoping by content + is_active.
        """
        result = [
            d for d in detections
            if d.is_active and d.confidence >= conf_threshold
        ]
        if classes:
            class_set = set(classes)
            result = [d for d in result if d.detected_class in class_set]
        return result

    @staticmethod
    def _build_response(
        detections: List[Detection],
        image: Image,
        cache_hit: bool,
        raw_metrics: Optional[dict] = None,
    ) -> Dict:
        return {
            "detections": [
                {
                    "bbox_id": d.bbox_id,
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2,
                    "detected_class": d.detected_class,
                    "confidence": d.confidence,
                }
                for d in detections
            ],
            "image_size": (image.width, image.height),
            "metrics": raw_metrics if raw_metrics is not None else {"cache_hit": cache_hit},
            "timestamp": datetime.utcnow().isoformat(),
        }