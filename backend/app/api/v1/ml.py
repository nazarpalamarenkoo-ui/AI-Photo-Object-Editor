from typing import List, Literal, Optional

from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.config.settings import settings
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SegmentByPolygonRequest,
    SegmentHybridRequest,
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
from app.core.logging import get_logger
from app.core.tracing import inject_trace_context

logger = get_logger(__name__)

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


_arq_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    return _arq_pool


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


@router.post("/images/{image_id}/remove/{bbox_id}/async")
async def remove_object_async(
    image_id: int,
    bbox_id: int,
    body: RemoveRequest = RemoveRequest(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    """Same as /remove/{bbox_id}, but enqueues the job and returns immediately."""
    job = await pool.enqueue_job(
        "remove_object_task",
        image_id=image_id,
        bbox_id=bbox_id,
        user_id=current_user.id,
        expand_mask_pixels=body.expand_mask_pixels,
        use_edge_blending=body.use_edge_blending,
        ldm_steps=body.ldm.ldm_steps,
        ldm_sampler=body.ldm.ldm_sampler,
        hd_strategy=body.ldm.hd_strategy,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="remove_object_task", job_id=job.job_id)
    return {"job_id": job.job_id}


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


@router.post("/images/{image_id}/remove-multiple/async")
async def remove_multiple_objects_async(
    image_id: int,
    body: RemoveMultipleRequest,
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "remove_multiple_objects_task",
        image_id=image_id,
        bbox_ids=body.bbox_ids,
        user_id=current_user.id,
        expand_mask_pixels=body.expand_mask_pixels,
        use_edge_blending=body.use_edge_blending,
        ldm_steps=body.ldm.ldm_steps,
        ldm_sampler=body.ldm.ldm_sampler,
        hd_strategy=body.ldm.hd_strategy,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="remove_multiple_objects_task", job_id=job.job_id)
    return {"job_id": job.job_id}


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


@router.post("/images/{image_id}/replace/{bbox_id}/async")
async def replace_object_async(
    image_id: int,
    bbox_id: int,
    replacement_file: UploadFile = File(...),
    body: ReplaceRequest = Depends(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    replacement_bytes = await replacement_file.read()
    job = await pool.enqueue_job(
        "replace_object_task",
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
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="replace_object_task", job_id=job.job_id)
    return {"job_id": job.job_id}


@router.get("/images/{image_id}/current")
async def get_current_state(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
):
    """
    Return the presigned URL that reflects the ACTUAL working state of the image
    (Redis current_state if edits exist, otherwise the original upload).

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
    service: EditingService = Depends(get_editor),
):
    """Reset working state to original image."""
    try:
        await service._get_image_authorized(image_id, current_user.id)
        await service.reset_current_state(image_id)
        logger.info("image_state_reset", image_id=image_id)
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
        result = await service.save_result(image_id=image_id, user_id=current_user.id)
        logger.info("image_result_saved", source_image_id=image_id, new_image_id=result.id)
        return result
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


@router.post("/images/{image_id}/segment/async")
async def segment_objects_async(
    image_id: int,
    body: SegmentRequest = SegmentRequest(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "segment_objects_task",
        image_id=image_id,
        user_id=current_user.id,
        min_area=body.min_area,
        max_segments=body.max_segments,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="segment_objects_task", job_id=job.job_id)
    return {"job_id": job.job_id}


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


@router.post("/images/{image_id}/segment/prompt/async")
async def segment_with_prompt_async(
    image_id: int,
    body: SegmentWithPromptRequest,
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    bbox_dict = body.bbox.model_dump() if body.bbox else None
    job = await pool.enqueue_job(
        "segment_with_prompt_task",
        image_id=image_id,
        user_id=current_user.id,
        point_coords=body.point_coords,
        point_labels=body.point_labels,
        bbox=bbox_dict,
        multimask_output=body.multimask_output,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="segment_with_prompt_task", job_id=job.job_id)
    return {"job_id": job.job_id}


@router.post("/images/{image_id}/segment/polygon", response_model=SegmentResponse)
async def segment_by_polygon(
    image_id: int,
    body: SegmentByPolygonRequest,
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Exact segmentation by polygon points (lasso), without MobileSAM."""
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


@router.post("/images/{image_id}/segment/polygon/async")
async def segment_by_polygon_async(
    image_id: int,
    body: SegmentByPolygonRequest,
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "segment_by_polygon_task",
        image_id=image_id,
        user_id=current_user.id,
        points=body.points,
        smooth=body.smooth,
        smoothing_factor=body.smoothing_factor,
        feather_px=body.feather_px,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="segment_by_polygon_task", job_id=job.job_id)
    return {"job_id": job.job_id}

@router.post("/images/{image_id}/segment/hybrid", response_model=SegmentResponse)
async def segment_hybrid(
    image_id: int,
    body: SegmentHybridRequest = SegmentHybridRequest(),
    current_user: User = Depends(get_current_user),
    service: SegmentationService = Depends(get_segmentation),
):
    """Hybrid YOLO + MobileSAM segmentation: YOLO for common objects, sparse MobileSAM auto for the rest."""
    try:
        return await service.segment_hybrid(
            image_id=image_id,
            user_id=current_user.id,
            yolo_conf_threshold=body.yolo_conf_threshold,
            yolo_classes=body.yolo_classes,
            fallback_min_area=body.fallback_min_area,
            fallback_max_segments=body.fallback_max_segments,
            overlap_iou_thresh=body.overlap_iou_thresh,
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


@router.post("/images/{image_id}/segment/hybrid/async")
async def segment_hybrid_async(
    image_id: int,
    body: SegmentHybridRequest = SegmentHybridRequest(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "segment_hybrid_task",
        image_id=image_id,
        user_id=current_user.id,
        yolo_conf_threshold=body.yolo_conf_threshold,
        yolo_classes=body.yolo_classes,
        fallback_min_area=body.fallback_min_area,
        fallback_max_segments=body.fallback_max_segments,
        overlap_iou_thresh=body.overlap_iou_thresh,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="segment_hybrid_task", job_id=job.job_id)
    return {"job_id": job.job_id}

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


@router.post("/images/{image_id}/segment/{mask_id}/remove/async")
async def sam_remove_object_async(
    image_id: int,
    mask_id: int,
    body: SamRemoveRequest = SamRemoveRequest(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "sam_remove_object_task",
        image_id=image_id,
        mask_id=mask_id,
        user_id=current_user.id,
        expand_mask_pixels=body.expand_mask_pixels,
        use_edge_blending=body.use_edge_blending,
        ldm_steps=body.ldm.ldm_steps,
        ldm_sampler=body.ldm.ldm_sampler,
        hd_strategy=body.ldm.hd_strategy,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="sam_remove_object_task", job_id=job.job_id)
    return {"job_id": job.job_id}


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


@router.post("/images/{image_id}/segment/{mask_id}/replace/async")
async def sam_replace_object_async(
    image_id: int,
    mask_id: int,
    replacement_file: Optional[UploadFile] = File(None),
    asset_id: Optional[str] = Query(None),
    body: SamReplaceRequest = Depends(),
    current_user: User = Depends(get_current_user),
    asset_service: AssetService = Depends(get_asset),
    pool: ArqRedis = Depends(get_arq_pool),
):
    """
    Same contract as /segment/{mask_id}/replace, but enqueues the job.
    """
    if not replacement_file and not asset_id:
        raise HTTPException(status_code=400, detail="Provide replacement_file or asset_id")

    if asset_id:
        replacement_bytes = await asset_service.get_asset_image(current_user.id, asset_id)
        if not replacement_bytes:
            raise HTTPException(status_code=404, detail="Asset not found")
        replacement_is_cutout = True
    else:
        replacement_bytes = await replacement_file.read()
        replacement_is_cutout = False

    job = await pool.enqueue_job(
        "sam_replace_object_task",
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
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="sam_replace_object_task", job_id=job.job_id)
    return {"job_id": job.job_id}


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


@router.post("/images/{image_id}/segment/{mask_id}/extract/async")
async def extract_object_async(
    image_id: int,
    mask_id: int,
    body: ExtractRequest = ExtractRequest(),
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    job = await pool.enqueue_job(
        "sam_extract_object_task",
        image_id=image_id,
        mask_id=mask_id,
        user_id=current_user.id,
        padding_pixels=body.padding_pixels,
        label=body.label,
        persist_to_s3=body.persist_to_s3,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="sam_extract_object_task", job_id=job.job_id)
    return {"job_id": job.job_id}


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    pool: ArqRedis = Depends(get_arq_pool),
):
    """
    Poll status/result of an enqueued job.

    status: "deferred" | "queued" | "in_progress" | "complete" | "not_found"
    result: present only when status == "complete" and the task succeeded
    error:  present only when status == "complete" and the task raised
    """
    job = Job(job_id, pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": status.value}

    if status == JobStatus.complete:
        result_info = await job.result_info()
        if result_info is not None:
            if result_info.success:
                response["result"] = result_info.result
            else:
                response["error"] = str(result_info.result)

    return response


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