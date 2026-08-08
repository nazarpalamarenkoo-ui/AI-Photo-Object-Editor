from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.storage.redis.redis_storage import RedisStorage
from app.db.models.detection import Detection
from app.db.models.image_version import ImageVersion
from app.core.logging import get_logger

logger = get_logger(__name__)

class DetectionService:

    def __init__(
        self,
        detection_repo: DetectionRepository,
        image_repo: ImageRepository,
        image_version_repo: ImageVersionRepository,
    ):
        self.detection_repo = detection_repo
        self.image_repo = image_repo
        self.image_version_repo = image_version_repo

    async def _get_authorized_image(self, image_id: int, user_id: int):
        image = await self.image_repo.get_by_id(image_id)
        if not image:
            logger.warning('image_not_found', image_id=image_id)
            raise ValueError(f'Image {image_id} not found')
        if image.user_id != user_id:
            logger.warning(
                'image_access_unauthorized',
                image_id=image_id,
                owner_user_id=image.user_id,
                requesting_user_id=user_id,
            )
            raise ValueError('Unauthorized: image belongs to different user')
        return image

    async def _resolve_version(
        self,
        image_id: int,
        user_id: int,
        version_id: Optional[int] = None,
    ) -> ImageVersion:
        """
        Defaults to the image's current version if version_id isn't given —
        lets callers ask about a specific past version without duplicating
        auth logic everywhere. Returns the full ImageVersion (not just its
        id) since callers need both version.id (for display/logging) and
        version.content_id (for actually querying Detection rows).
        """
        image = await self._get_authorized_image(image_id, user_id)

        if version_id is not None:
            version = await self.image_version_repo.get_by_id(version_id)
            if version is None or version.image_id != image.id:
                logger.warning(
                    'version_not_found_or_mismatch',
                    image_id=image_id,
                    version_id=version_id,
                )
                raise ValueError(f'Version {version_id} not found for image {image_id}')
            return version

        version = await self.image_version_repo.get_current(image)
        if version is None:
            logger.error('image_missing_current_version', image_id=image_id)
            raise ValueError(f'Image {image_id} has no current version')
        return version

    async def get_detections(
        self,
        image_id: int,
        user_id: int,
        version_id: Optional[int] = None,
        active_only: bool = True,
    ) -> List[Detection]:
        version = await self._resolve_version(image_id, user_id, version_id)
        detections = await self.detection_repo.get_by_content(
            version.content_id, active_only=active_only
        )
        logger.debug(
            'detections_loaded',
            image_id=image_id,
            image_version_id=version.id,
            content_id=version.content_id,
            count=len(detections),
        )
        return detections

    async def get_detection_by_bbox_id(
        self,
        image_id: int,
        bbox_id: int,
        user_id: int,
        version_id: Optional[int] = None,
    ) -> Detection:
        detections = await self.get_detections(
            image_id, user_id, version_id=version_id, active_only=False
        )
        detection = next((d for d in detections if d.bbox_id == bbox_id), None)
        if not detection:
            logger.warning('detection_not_found', image_id=image_id, bbox_id=bbox_id)
            raise ValueError(
                f'Detection with bbox_id {bbox_id} not found for image {image_id}'
            )
        return detection

    async def soft_delete_detection(
        self,
        image_id: int,
        bbox_id: int,
        user_id: int,
    ) -> Detection:
        """
        Remove a single known object without touching the model — this is
        the 'known single deletion' path from MD's hybrid deletion design.
        """
        detection = await self.get_detection_by_bbox_id(image_id, bbox_id, user_id)
        updated = await self.detection_repo.soft_delete(detection.id)
        logger.info('detection_soft_deleted', image_id=image_id, bbox_id=bbox_id)
        return updated

    async def delete_version_detections(
        self,
        image_id: int,
        user_id: int,
        version_id: Optional[int] = None,
    ) -> int:
        """Hard-delete all detections for a version's content — used before
        an explicit redetect. NOTE: since detections are content-scoped,
        this also removes them for any other version that happens to share
        the same content_id."""
        version = await self._resolve_version(image_id, user_id, version_id)
        count = await self.detection_repo.delete_by_content(version.content_id)
        logger.info(
            'detections_deleted',
            image_id=image_id,
            image_version_id=version.id,
            content_id=version.content_id,
            count=count,
        )
        return count

    async def get_detection_stats(
        self,
        image_id: int,
        user_id: int,
        version_id: Optional[int] = None,
    ) -> dict:
        detections = await self.get_detections(image_id, user_id, version_id=version_id)

        if not detections:
            return {
                'total_detections': 0,
                'classes': [],
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
            }

        confidences = [d.confidence for d in detections]
        classes = list({d.detected_class for d in detections})

        return {
            'total_detections': len(detections),
            'classes': classes,
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
        }