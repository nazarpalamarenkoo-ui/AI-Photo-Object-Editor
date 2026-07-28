from typing import Optional

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from app.api.auth.auth import get_current_user
from app.db.models.user import User
from app.db.schemas.ml import (
    SamRemoveRequest,
    SamReplaceRequest,
    ExtractRequest,
    MLResultResponse,
    ExtractResponse,
)
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml.assets_service import AssetService
from app.core.logging import get_logger
from app.core.tracing import inject_trace_context

from .deps import get_segmentation, get_asset, get_arq_pool, _http_status

logger = get_logger(__name__)

router = APIRouter(tags=["ML - SAM Ops"])


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