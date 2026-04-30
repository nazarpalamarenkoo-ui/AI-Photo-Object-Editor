from typing import List, Literal
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    MLResultResponse
)
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.services.ml_service import MLService
from app.storage.s3_storage import S3Storage
from app.storage.redis_storage import RedisImageCache

router = APIRouter(prefix="/ml", tags=["ML"])


def get_ml_service(db: AsyncSession = Depends(get_db)) -> MLService:
    return MLService(
        db=db,
        s3_storage=S3Storage(),
        redis_storage=RedisImageCache(),
        image_repo=ImageRepository(db),
        detection_repo=DetectionRepository(db)
    )


def _http_status(e: ValueError) -> int:
    msg = str(e).lower()
    if "not found" in msg or "no valid detections" in msg:
        return 404
    if "unauthorized" in msg:
        return 403
    return 400


@router.post("/images/{image_id}/detect")
async def detect_objects(
    image_id: int,
    body: DetectRequest = DetectRequest(),
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """Run YOLO object detection on uploaded image. Saves detections to DB and caches in Redis."""
    try:
        return await service.detect_objects(
            image_id=image_id,
            user_id=current_user.id,
            conf_threshold=body.conf_threshold,
            classes=body.classes
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/remove/{bbox_id}", response_model=MLResultResponse)
async def remove_object(
    image_id: int,
    bbox_id: int,
    body: RemoveRequest = RemoveRequest(),
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """Remove detected object from image using LaMa inpainting."""
    try:
        return await service.remove_object(
            image_id=image_id,
            bbox_id=bbox_id,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_edge_blending=body.use_edge_blending
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/replace/{bbox_id}", response_model=MLResultResponse)
async def replace_object(
    image_id: int,
    bbox_id: int,
    replacement_file: UploadFile = File(..., description="Replacement object image"),
    expand_mask_pixels: int = Query(0, ge=0, le=50),
    use_color_matching: bool = Query(True),
    use_edge_blending: bool = Query(False),
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = Query('color_transfer'),
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """
    Replace detected object with uploaded replacement image.
    Pipeline: LaMa remove old object -> paste new object -> color match -> edge blend.
    """
    try:
        replacement_bytes = await replacement_file.read()
        return await service.replace_object(
            image_id=image_id,
            bbox_id=bbox_id,
            replace_image_bytes=replacement_bytes,
            user_id=current_user.id,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            use_edge_blending=use_edge_blending,
            color_match_method=color_match_method
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/remove-multiple", response_model=MLResultResponse)
async def remove_multiple_objects(
    image_id: int,
    body: RemoveMultipleRequest,
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """Remove multiple detected objects in a single inpainting pass."""
    try:
        return await service.remove_multiple_objects(
            image_id=image_id,
            bbox_ids=body.bbox_ids,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_edge_blending=body.use_edge_blending
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/reset")
async def reset_current_state(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """Reset current working state to original image."""
    try:
        await service._get_image_authorized(image_id, current_user.id)
        await service.reset_current_state(image_id)
        return {"detail": "State reset to original image"}
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.get("/classes", response_model=List[str])
async def get_supported_classes(
    current_user: User = Depends(get_current_user),
    service: MLService = Depends(get_ml_service)
):
    """Get list of all 80 COCO classes supported by YOLO detector."""
    return service.get_supported_classes()