from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.segmentation import (
    SegmentRequest,
    SegmentWithPromptRequest,
    SegmentByPolygonRequest,
    SegmentHybridRequest,
    SegmentResponse,
)
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml_job_service import MLJobService
from app.services.ml.tracked_runner import run_tracked
from app.db.enums.ml_task_status import MLTaskType
from app.core.logging import get_logger
from app.core.tracing import inject_trace_context

from .deps import get_segmentation, get_arq_pool, get_base_deps, get_mljob_service, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - Segmentation"])


@router.post("/images/{image_id}/segment", response_model=SegmentResponse)
async def segment_objects(
    image_id: int,
    body: SegmentRequest = SegmentRequest(),
    current_user: User = Depends(get_current_user),
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Auto-segment all objects via SAM 2.1 (no prompts)."""
    try:
        return await run_tracked(
            SegmentationService, deps, mljob_service, "segment_objects",
            image_id, current_user.id, MLTaskType.SEGMENTATION,
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
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Prompt-based SAM segmentation using points or a bbox."""
    try:
        bbox_dict = body.bbox.model_dump() if body.bbox else None
        return await run_tracked(
            SegmentationService, deps, mljob_service, "segment_with_prompt",
            image_id, current_user.id, MLTaskType.SEGMENTATION_PROMPT,
            point_coords=body.point_coords,
            point_labels=body.point_labels,
            bbox=bbox_dict,
            multimask_output=body.multimask_output,
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
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Exact segmentation by polygon points (lasso), without MobileSAM."""
    try:
        return await run_tracked(
            SegmentationService, deps, mljob_service, "segment_by_polygon",
            image_id, current_user.id, MLTaskType.SEGMENTATION_POLYGON,
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
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Hybrid YOLO + MobileSAM segmentation: YOLO for common objects, sparse MobileSAM auto for the rest."""
    try:
        return await run_tracked(
            SegmentationService, deps, mljob_service, "segment_hybrid",
            image_id, current_user.id, MLTaskType.SEGMENTATION_HYBRID,
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