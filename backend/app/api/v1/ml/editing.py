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
from app.db.schemas.image import ImageResponse
from app.services.ml.editing_service import EditingService
from app.services.ml.assets_service import AssetService
from app.core.logging import get_logger
from app.core.tracing import inject_trace_context

from .deps import get_editor, get_asset, get_arq_pool, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - Editing"])


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


# NOTE: these two "diffusion" routes are intentionally registered BEFORE
# /images/{image_id}/replace/{bbox_id} and .../{bbox_id}/async below.
# FastAPI/Starlette match path routes in registration order, and
# "/replace/diffusion(/async)" has the exact same segment count as
# "/replace/{bbox_id}(/async)". If the {bbox_id} routes were registered
# first, a request to ".../replace/diffusion/async" would match THEM
# instead, with bbox_id literally set to the string "diffusion" — which
# is exactly the 422 (int_parsing on "diffusion" + missing
# replacement_file) you were seeing. Keep this block above the YOLO
# replace routes, or rename this path to something that can't collide
# (e.g. "/replace-diffusion") if you ever reorder things again.
@router.post("/images/{image_id}/replace/diffusion", response_model=MLResultResponse)
async def sam_replace_object_diffusion(
    image_id: int,
    mask_file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    asset_id: Optional[str] = Query(None),
    body: SamReplaceDiffusionRequest = Depends(),
    current_user: User = Depends(get_current_user),
    service: EditingService = Depends(get_editor),
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

        return await service.sam_replace_object_diffusion(
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
        )
    except ValueError as e:
        raise HTTPException(status_code=_http_status(e), detail=str(e))


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