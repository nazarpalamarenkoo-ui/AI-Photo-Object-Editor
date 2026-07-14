from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.detection import DetectionResponse
from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.services.detection_service import DetectionService
from app.storage.redis.redis_storage import RedisStorage
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/detections", tags=["Detections"])


def get_detection_service(db: AsyncSession = Depends(get_db)) -> DetectionService:
    return DetectionService(
        db=db,
        redis_cache=RedisStorage(),
        detection_repo=DetectionRepository(db),
        image_repo=ImageRepository(db)
    )


def _status_for(e: ValueError) -> int:
    msg = str(e).lower()
    return 404 if "not found" in msg else 403 if "unauthorized" in msg else 400


@router.get("/images/{image_id}", response_model=List[DetectionResponse])
async def get_image_detections(
    image_id: int,
    use_cache: bool = True,
    current_user: User = Depends(get_current_user),
    service: DetectionService = Depends(get_detection_service)
):
    """Get all detections for an image. Tries Redis cache first."""
    try:
        return await service.get_image_detections(
            image_id=image_id,
            user_id=current_user.id,
            use_cache=use_cache
        )
    except ValueError as e:
        raise HTTPException(status_code=_status_for(e), detail=str(e))


@router.get("/images/{image_id}/bbox/{bbox_id}", response_model=DetectionResponse)
async def get_detection_by_bbox(
    image_id: int,
    bbox_id: int,
    current_user: User = Depends(get_current_user),
    service: DetectionService = Depends(get_detection_service)
):
    """Get a single detection by bbox_id."""
    try:
        return await service.get_detection_by_bbox_id(
            image_id=image_id,
            bbox_id=bbox_id,
            user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=_status_for(e), detail=str(e))


@router.get("/images/{image_id}/stats")
async def get_detection_stats(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: DetectionService = Depends(get_detection_service)
):
    """Get aggregated detection stats for an image (count, classes, confidence)."""
    try:
        return await service.get_detection_stats(
            image_id=image_id,
            user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=_status_for(e), detail=str(e))


@router.delete("/images/{image_id}")
async def delete_image_detections(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: DetectionService = Depends(get_detection_service)
):
    """Delete all detections for an image and invalidate Redis cache."""
    try:
        count = await service.delete_image_detections(
            image_id=image_id,
            user_id=current_user.id
        )
        logger.info("image_detections_deleted", image_id=image_id, deleted=count)
        return {"deleted": count}
    except ValueError as e:
        logger.warning("image_detections_delete_failed", image_id=image_id, error=str(e))
        raise HTTPException(status_code=_status_for(e), detail=str(e))