from fastapi import APIRouter, Depends, HTTPException

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.image import ImageResponse
from app.services.ml.version_history_service import VersionHistoryService
from app.core.logging import get_logger

from .deps import get_version_history, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - Session"])


@router.get("/images/{image_id}/current")
async def get_current_state(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    """
    Return the presigned URL that reflects the ACTUAL working state of the image

    The editor page must call this on mount instead of the plain image presigned
    URL, or a refresh/crash/reconnect will show the untouched original even though
    the backend still holds — and keeps building on top of — the edited state.
    """
    try:
        return await service.get_current_state(image_id=image_id, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/reset")
async def reset_current_state(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    """Reset working state to original image."""
    try:
        await service.reset_current_state(image_id, current_user.id)
        logger.info("image_state_reset", image_id=image_id)
        return {"detail": "State reset to original image"}
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/save", response_model=ImageResponse)
async def save_result(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    """Persist current working state as a new Image in the workspace."""
    try:
        result = await service.save_result(image_id=image_id, user_id=current_user.id)
        new_image_id = result["id"] if isinstance(result, dict) else result.id
        logger.info("image_result_saved", source_image_id=image_id, new_image_id=new_image_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/undo")
async def undo(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    try:
        return await service.undo(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/redo")
async def redo(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    try:
        return await service.redo(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.get("/images/{image_id}/history")
async def get_history(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: VersionHistoryService = Depends(get_version_history),
):
    try:
        return await service.get_history(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))