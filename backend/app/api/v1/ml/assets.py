from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.ml import (
    PasteRequest,
    PasteResponse,
    AssetResponse,
    RenameAssetRequest,
)
from app.services.ml.assets_service import AssetService
from app.core.logging import get_logger

from .deps import get_asset, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - Assets"])


@router.get("/assets", response_model=List[AssetResponse])
async def list_assets(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    """List extracted objects in the current user's asset library."""
    return await service.list_assets(current_user.id, limit=limit, offset=offset)


@router.get("/assets/{asset_id}/thumbnail")
async def get_asset_thumbnail(
    asset_id: str,
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    data = await service.get_asset_thumbnail(current_user.id, asset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Asset not found")
    return Response(content=data, media_type="image/png")


@router.get("/assets/{asset_id}/image")
async def get_asset_image(
    asset_id: str,
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    data = await service.get_asset_image(current_user.id, asset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Asset not found")
    return Response(content=data, media_type="image/png")


@router.patch("/assets/{asset_id}", response_model=AssetResponse)
async def rename_asset(
    asset_id: str,
    body: RenameAssetRequest,
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    try:
        result = await service.rename_asset(current_user.id, asset_id, body.label)
        logger.info("asset_renamed", asset_id=asset_id, label=body.label)
        return result
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.delete("/assets/{asset_id}")
async def delete_asset(
    asset_id: str,
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    try:
        await service.delete_asset(current_user.id, asset_id)
        logger.info("asset_deleted", asset_id=asset_id)
        return {"detail": "Asset deleted"}
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/paste", response_model=PasteResponse)
async def paste_extracted_object(
    image_id: int,
    body: PasteRequest,
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    """Paste an extracted object (from asset library or S3 URL) onto the current image."""
    try:
        return await service.paste_extracted_object(
            image_id=image_id,
            user_id=current_user.id,
            asset_id=body.asset_id,
            extracted_url=body.extracted_url,
            target_bbox=body.target_bbox.model_dump(),
            scale=body.scale,
            use_color_matching=body.use_color_matching,
            use_edge_blending=body.use_edge_blending,
            color_match_method=body.color_match_method,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))