from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.ml import DetectRequest
from app.services.ml.detector_service import DetectorService
from app.services.ml_job_service import MLJobService
from app.services.ml.tracked_runner import run_tracked
from app.db.enums.ml_task_status import MLTaskType

from .deps import get_detector, get_base_deps, get_mljob_service, _http_status

router = APIRouter(tags=["ML - Detection"])


@router.post("/images/{image_id}/detect")
async def detect_objects(
    image_id: int,
    body: DetectRequest = DetectRequest(),
    current_user: User = Depends(get_current_user),
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Run YOLO object detection. Saves detections to DB and caches in Redis.
    Tracked in ml_jobs the same way the arq /async path is (see
    app.services.ml.tracked_runner.run_tracked)."""
    try:
        return await run_tracked(
            DetectorService, deps, mljob_service, "detect_objects",
            image_id, current_user.id, MLTaskType.DETECTION,
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