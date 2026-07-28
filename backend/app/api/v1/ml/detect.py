from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.ml import DetectRequest
from app.services.ml.detector_service import DetectorService

from .deps import get_detector, _http_status

router = APIRouter(tags=["ML - Detection"])


@router.post("/images/{image_id}/detect")
async def detect_objects(
    image_id: int,
    body: DetectRequest = DetectRequest(),
    current_user: User = Depends(get_current_user),
    service: DetectorService = Depends(get_detector),
):
    """Run YOLO object detection. Saves detections to DB and caches in Redis."""
    try:
        return await service.detect_objects(
            image_id=image_id,
            user_id=current_user.id,
            conf_threshold=body.conf_threshold,
            classes=body.classes,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.get("/classes", response_model=List[str])
async def get_supported_classes(
    current_user: User = Depends(get_current_user),
    service: DetectorService = Depends(get_detector),
):
    """Get all 80 COCO classes supported by YOLO."""
    return service.get_supported_classes()