from typing import Optional

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.ml import (
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SamReplaceDiffusionRequest,
    MLResultResponse,
)
from app.services.ml.editing_service import EditingService
from app.services.ml.assets_service import AssetService
from app.services.ml_job_service import MLJobService
from app.services.ml.tracked_runner import run_tracked
from app.db.enums.ml_task_status import MLTaskType
from app.core.logging import get_logger
from app.core.tracing import inject_trace_context

from .deps import get_asset, get_arq_pool, get_base_deps, get_mljob_service, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - Editing"])


@router.post("/images/{image_id}/remove/{bbox_id}", response_model=MLResultResponse)
async def remove_object(
    image_id: int,
    bbox_id: int,
    body: RemoveRequest = RemoveRequest(),
    current_user: User = Depends(get_current_user),
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Remove a YOLO-detected object via LaMa inpainting."""
    try:
        return await run_tracked(
            EditingService, deps, mljob_service, "remove_object",
            image_id, current_user.id, MLTaskType.REMOVE_OBJECT,
            bbox_id=bbox_id,
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
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Remove multiple YOLO-detected objects in one inpainting pass."""
    try:
        return await run_tracked(
            EditingService, deps, mljob_service, "remove_multiple_objects",
            image_id, current_user.id, MLTaskType.REMOVE_MULTIPLE_OBJECTS,
            bbox_ids=body.bbox_ids,
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


@router.post("/images/{image_id}/replace/diffusion", response_model=MLResultResponse)
async def sam_replace_object_diffusion(
    image_id: int,
    mask_file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    asset_id: Optional[str] = Query(None),
    body: SamReplaceDiffusionRequest = Depends(),
    current_user: User = Depends(get_current_user),
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
    asset_service: AssetService = Depends(get_asset),
):
    """
    Replace a SAM-segmented object via diffusion (SD-inpainting + IP-Adapter),
    steered by an uploaded reference image or a saved asset, instead of a
    flat LaMa+paste composite.

    mask_file: binary SAM mask (PNG, L mode) for the object being replaced —
    SAM masks aren't persisted server-side, so the client resends the one
    it already has from the /segment call.
    """
    if not reference_file and not asset_id:
        raise HTTPException(status_code=400, detail="Provide reference_file or asset_id")

    try:
        mask_bytes = await mask_file.read()

        if asset_id:
            reference_bytes = await asset_service.get_asset_image(current_user.id, asset_id)
            if not reference_bytes:
                raise HTTPException(status_code=404, detail="Asset not found")
        else:
            reference_bytes = await reference_file.read()

        return await run_tracked(
            EditingService, deps, mljob_service, "sam_replace_object_diffusion",
            image_id, current_user.id, MLTaskType.DIFFUSION,
            mask_bytes=mask_bytes,
            bbox=body.bbox,
            reference_image_bytes=reference_bytes,
            prompt=body.prompt,
            use_color_matching=body.use_color_matching,
            color_match_method=body.color_match_method,
            negative_prompt=body.negative_prompt,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            ip_adapter_scale=body.ip_adapter_scale,
            strength=body.strength,
            seed=body.seed,
        )
    except (ValueError, RuntimeError) as e:
        status_code = _http_status(e) if isinstance(e, ValueError) else 502
        logger.warning("diffusion_replace_failed", image_id=image_id, error=str(e))
        raise HTTPException(status_code=status_code, detail=str(e))


@router.post("/images/{image_id}/replace/diffusion/async")
async def sam_replace_object_diffusion_async(
    image_id: int,
    mask_file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    asset_id: Optional[str] = Query(None),
    body: SamReplaceDiffusionRequest = Depends(),
    current_user: User = Depends(get_current_user),
    asset_service: AssetService = Depends(get_asset),
    pool: ArqRedis = Depends(get_arq_pool),
):
    """Same contract as /replace/diffusion, but enqueues the job."""
    if not reference_file and not asset_id:
        raise HTTPException(status_code=400, detail="Provide reference_file or asset_id")

    mask_bytes = await mask_file.read()

    if asset_id:
        reference_bytes = await asset_service.get_asset_image(current_user.id, asset_id)
        if not reference_bytes:
            raise HTTPException(status_code=404, detail="Asset not found")
    else:
        reference_bytes = await reference_file.read()

    job = await pool.enqueue_job(
        "sam_replace_object_diffusion_task",
        image_id=image_id,
        mask_bytes=mask_bytes,
        bbox=body.bbox,
        reference_image_bytes=reference_bytes,
        user_id=current_user.id,
        prompt=body.prompt,
        use_color_matching=body.use_color_matching,
        color_match_method=body.color_match_method,
        negative_prompt=body.negative_prompt,
        num_inference_steps=body.num_inference_steps,
        guidance_scale=body.guidance_scale,
        ip_adapter_scale=body.ip_adapter_scale,
        strength=body.strength,
        seed=body.seed,
        _trace_carrier=inject_trace_context(),
    )
    logger.info("ml_job_enqueued", task="sam_replace_object_diffusion_task", job_id=job.job_id)
    return {"job_id": job.job_id}


@router.post("/images/{image_id}/replace/{bbox_id}", response_model=MLResultResponse)
async def replace_object(
    image_id: int,
    bbox_id: int,
    replacement_file: UploadFile = File(...),
    body: ReplaceRequest = Depends(),
    current_user: User = Depends(get_current_user),
    deps: dict = Depends(get_base_deps),
    mljob_service: MLJobService = Depends(get_mljob_service),
):
    """Replace a YOLO-detected object with an uploaded image."""
    try:
        replacement_bytes = await replacement_file.read()
        return await run_tracked(
            EditingService, deps, mljob_service, "replace_object",
            image_id, current_user.id, MLTaskType.REPLACE_OBJECT,
            bbox_id=bbox_id,
            replace_image_bytes=replacement_bytes,
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