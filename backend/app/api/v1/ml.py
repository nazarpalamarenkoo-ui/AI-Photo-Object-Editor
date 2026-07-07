from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SegmentByPolygonRequest,
    SegmentRequest,
    SegmentWithPromptRequest,
    SamRemoveRequest,
    SamReplaceRequest,
    ExtractRequest,
    PasteRequest,
    MLResultResponse,
    SegmentResponse,
    ExtractResponse,
    PasteResponse,
    AssetResponse,
    RenameAssetRequest,
)
from app.db.schemas.image import ImageResponse
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.services.ml.detector_service import DetectorService
from app.services.ml.editing_service import EditingService
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml.assets_service import AssetService
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory
from app.storage.redis.redis_assets import RedisAssetsStorage

router = APIRouter(prefix="/ml", tags=["ML"])


def _base_deps(db: AsyncSession) -> dict:
    return dict(
        db=db,
        s3_storage=S3Storage(),
        redis_storage=RedisStorage(),
        redis_history=RedisHistory(),
        redis_assets=RedisAssetsStorage(),
        image_repo=ImageRepository(db),
        detection_repo=DetectionRepository(db),
    )


def get_detector(db: AsyncSession = Depends(get_db)) -> DetectorService:
    return DetectorService(**_base_deps(db))


def get_editor(db: AsyncSession = Depends(get_db)) -> EditingService:
    return EditingService(**_base_deps(db))


def get_segmentation(db: AsyncSession = Depends(get_db)) -> SegmentationService:
    return SegmentationService(**_base_deps(db))


def get_asset(db: AsyncSession = Depends(get_db)) -> AssetService:
    return AssetService(**_base_deps(db))


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


@router.post("/images/{image_id}/remove/{bbox_id}", response_model=MLResultResponse)
async def remove_object(
    image_id: int,
    bbox_id: int,
    body: RemoveRequest = RemoveRequest(),
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """Remove a YOLO-detected object via LaMa inpainting."""
    try:
        return await service.remove_object(
            image_id=image_id,
            bbox_id=bbox_id,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_edge_blending=body.use_edge_blending,
            ldm_steps=body.ldm.ldm_steps,
            ldm_sampler=body.ldm.ldm_sampler,
            hd_strategy=body.ldm.hd_strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/remove-multiple", response_model=MLResultResponse)
async def remove_multiple_objects(
    image_id: int,
    body: RemoveMultipleRequest,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """Remove multiple YOLO-detected objects in one inpainting pass."""
    try:
        return await service.remove_multiple_objects(
            image_id=image_id,
            bbox_ids=body.bbox_ids,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_edge_blending=body.use_edge_blending,
            ldm_steps=body.ldm.ldm_steps,
            ldm_sampler=body.ldm.ldm_sampler,
            hd_strategy=body.ldm.hd_strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/replace/{bbox_id}", response_model=MLResultResponse)
async def replace_object(
    image_id: int,
    bbox_id: int,
    replacement_file: UploadFile = File(...),
    body: ReplaceRequest = Depends(),
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """Replace a YOLO-detected object with an uploaded image."""
    try:
        replacement_bytes = await replacement_file.read()
        return await service.replace_object(
            image_id=image_id,
            bbox_id=bbox_id,
            replace_image_bytes=replacement_bytes,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_color_matching=body.use_color_matching,
            use_edge_blending=body.use_edge_blending,
            color_match_method=body.color_match_method,
            ldm_steps=body.ldm.ldm_steps,
            ldm_sampler=body.ldm.ldm_sampler,
            hd_strategy=body.ldm.hd_strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/reset")
async def reset_current_state(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """Reset working state to original image."""
    try:
        await service._get_image_authorized(image_id, current_user.id)
        await service.reset_current_state(image_id)
        return {"detail": "State reset to original image"}
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/save", response_model=ImageResponse)
async def save_result(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """Persist current working state as a new Image in the workspace."""
    try:
        return await service.save_result(image_id=image_id, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/undo")
async def undo(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    try:
        return await service.undo(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/redo")
async def redo(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    try:
        return await service.redo(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.get("/images/{image_id}/history")
async def get_history(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    try:
        return await service.get_history(image_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/segment", response_model=SegmentResponse)
async def segment_objects(
    image_id: int,
    body: SegmentRequest = SegmentRequest(),
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Auto-segment all objects via SAM 2.1 (no prompts)."""
    try:
        return await service.segment_objects(
            image_id=image_id,
            user_id=current_user.id,
            min_area=body.min_area,
            max_segments=body.max_segments,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/segment/prompt", response_model=SegmentResponse)
async def segment_with_prompt(
    image_id: int,
    body: SegmentWithPromptRequest,
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Prompt-based SAM segmentation using points or a bbox."""
    try:
        bbox_dict = body.bbox.model_dump() if body.bbox else None
        return await service.segment_with_prompt(
            image_id=image_id,
            user_id=current_user.id,
            point_coords=body.point_coords,
            point_labels=body.point_labels,
            bbox=bbox_dict,
            multimask_output=body.multimask_output
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))

@router.post("/images/{image_id}/segment/polygon", response_model=SegmentResponse)
async def segment_by_polygon(
    image_id: int,
    body: SegmentByPolygonRequest,
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Exact segmentation by polygon points (lasso), without SAM2."""
    try:
        return await service.segment_by_polygon(
            image_id=image_id,
            user_id=current_user.id,
            points=body.points,
            smooth=body.smooth,
            smoothing_factor=body.smoothing_factor,
            feather_px=body.feather_px,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))
    
@router.post("/images/{image_id}/segment/{mask_id}/remove", response_model=MLResultResponse)
async def sam_remove_object(
    image_id: int,
    mask_id: int,
    body: SamRemoveRequest = SamRemoveRequest(),
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Remove SAM-segmented object via LaMa inpainting."""
    try:
        return await service.sam_remove_object(
            image_id=image_id,
            mask_id=mask_id,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_edge_blending=body.use_edge_blending,
            ldm_steps=body.ldm.ldm_steps,
            ldm_sampler=body.ldm.ldm_sampler,
            hd_strategy=body.ldm.hd_strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/segment/{mask_id}/replace", response_model=MLResultResponse)
async def sam_replace_object(
    image_id: int,
    mask_id: int,
    replacement_file: Optional[UploadFile] = File(None),
    asset_id: Optional[str] = Query(None),
    body: SamReplaceRequest = Depends(),
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
    asset_service: AssetService = Depends(get_asset),
):
    """Replace SAM-segmented object with an uploaded image OR a saved asset."""
    if not replacement_file and not asset_id:
        raise HTTPException(status_code=400, detail="Provide replacement_file or asset_id")

    try:
        if asset_id:
            replacement_bytes = await asset_service.get_asset_image(current_user.id, asset_id)
            if not replacement_bytes:
                raise HTTPException(status_code=404, detail="Asset not found")
            replacement_is_cutout = True
        else:
            replacement_bytes = await replacement_file.read()
            replacement_is_cutout = False

        return await service.sam_replace_object(
            image_id=image_id,
            mask_id=mask_id,
            replacement_image_bytes=replacement_bytes,
            user_id=current_user.id,
            expand_mask_pixels=body.expand_mask_pixels,
            use_color_matching=body.use_color_matching,
            use_edge_blending=body.use_edge_blending,
            color_match_method=body.color_match_method,
            ldm_steps=body.ldm.ldm_steps,
            ldm_sampler=body.ldm.ldm_sampler,
            hd_strategy=body.ldm.hd_strategy,
            replacement_is_cutout=replacement_is_cutout,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))

@router.post("/images/{image_id}/segment/{mask_id}/extract", response_model=ExtractResponse)
async def extract_object(
    image_id: int,
    mask_id: int,
    body: ExtractRequest = ExtractRequest(),
    current_user: User = Depends(get_current_user),
    service: AssetService = Depends(get_asset),
):
    """Extract SAM-segmented object as RGBA PNG, save into asset library (Redis)."""
    try:
        return await service.extract_object(
            image_id=image_id,
            mask_id=mask_id,
            user_id=current_user.id,
            padding_pixels=body.padding_pixels,
            label=body.label,
            persist_to_s3=body.persist_to_s3,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))

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
    from fastapi import Response
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
    from fastapi import Response
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
        return await service.rename_asset(current_user.id, asset_id, body.label)
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