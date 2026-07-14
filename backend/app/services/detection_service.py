from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.storage.redis.redis_storage import RedisStorage
from app.db.models.detection import Detection
from app.core.logging import get_logger

logger = get_logger(__name__)

class DetectionService:
    
    def __init__(
        self,
        db: AsyncSession,
        redis_cache: RedisStorage,
        detection_repo: DetectionRepository,
        image_repo: ImageRepository
    ):
        
        self.db = db
        self.redis = redis_cache
        self.detection_repo = detection_repo
        self.image_repo = image_repo
        
    async def get_image_detections(
        self,
        image_id: int,
        user_id: int,
        use_cache: bool = True
    ) -> List[Detection]:
        
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
        
        # Try cache first
        if use_cache:
            cached_detection = await self.redis.get_cached_detections(image_id)
            if cached_detection:
                logger.debug('detections_cache_hit', image_id=image_id)
                return cached_detection
        
        # Query DB   
        detections = await self.detection_repo.get_by_image(image_id)
        logger.debug('detections_loaded_from_db', image_id=image_id, count=len(detections))
        
        # Cache result
        if use_cache and detections:
            await self.redis.cache_detections(
                image_id = image_id,
                detections = detections,
                ttl = 3600
            )
            
        return detections
    
    async def get_detection_by_bbox_id(
        self,
        image_id: int,
        bbox_id: int,
        user_id: int
    ) -> Detection:
        
        detections = await self.get_image_detections(image_id, user_id)
        
        # Find detection by bbox_id
        detection = next(
            (d for d in detections if d.bbox_id == bbox_id),
            None
        )
        
        if not detection:
            logger.warning(
                'detection_not_found', image_id=image_id, bbox_id=bbox_id
            )
            raise ValueError(
                f'Detection with bbox_id {bbox_id} not found '
                f'for image {image_id}'
            )
        
        return detection
    
    async def delete_image_detections(
        self,
        image_id: int,
        user_id: int
    ) -> int:
        
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
        
        # Delete detections
        count = await self.detection_repo.delete_by_image(image_id)
        
        # Invalidate cache
        await self.redis.invalidate_image(image_id)
        
        logger.info('detections_deleted', image_id=image_id, count=count)
        
        return count
    
    async def get_detection_stats(
        self,
        image_id: int,
        user_id: int
    ) -> dict:
        
        detections = await self.get_image_detections(image_id, user_id)
        
        if not detections:
            return {
                'total_detections': 0,
                'classes': [],
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }
            
        confidences = [d.confidence for d in detections]
        classes = list(set(d.detected_class for d in detections))
        
        return {
            'total_detections': len(detections),
            'classes': classes,
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }