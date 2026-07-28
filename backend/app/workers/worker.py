from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from arq.connections import RedisSettings

from app.config.settings import settings
from app.db.db_connect import get_db_session
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory
from app.storage.redis.redis_assets import RedisAssetsStorage
from app.db.models.user import User
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.ml.pipeline.pipeline import get_pipeline

from app.services.ml.editing_service import EditingService
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml.assets_service import AssetService

from app.core.logging import configure_logging, get_logger, log_job
from app.core.tracing import setup_tracing, trace_job

import asyncio
import threading
from collections import Counter
 
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    
    
configure_logging()
setup_tracing("image-editor-worker")
logger = get_logger("arq.worker")


@asynccontextmanager
async def _build_ml_deps(db):
    """
    Shared dependencies for ML services inside a task — the equivalent
    of _base_deps() from the router, but outside FastAPI DI.
    """
    s3_storage = S3Storage()
    redis_storage = RedisStorage()
    redis_history = RedisHistory()
    redis_assets = RedisAssetsStorage()
    image_repo = ImageRepository(db)
    detection_repo = DetectionRepository(db)
    pipeline = get_pipeline()

    try:
        yield {
            "db": db,
            "s3_storage": s3_storage,
            "redis_storage": redis_storage,
            "redis_history": redis_history,
            "redis_assets": redis_assets,
            "image_repo": image_repo,
            "detection_repo": detection_repo,
            "pipeline": pipeline,
        }
    finally:
        await redis_storage.close()
        await redis_history.close()
        await redis_assets.close()


async def _resource_monitor(interval_seconds: int = 60):
    process = psutil.Process() if _HAS_PSUTIL else None
    iteration = 0
 
    while True:
        await asyncio.sleep(interval_seconds)
        iteration += 1
 
        threads = threading.enumerate()
        name_counts = Counter()
        for t in threads:
            base = t.name.rsplit("_", 1)[0].rsplit("-", 1)[0]
            name_counts[base] += 1
 
        log_fields = {
            "iteration": iteration,
            "num_threads": len(threads),
            "thread_groups": dict(name_counts),
        }
 
        if process is not None:
            mem = process.memory_info()
            log_fields.update({
                "rss_mb": round(mem.rss / (1024 * 1024), 1),
                "vms_mb": round(mem.vms / (1024 * 1024), 1),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
                "open_files": len(process.open_files()),
                "connections": len(process.net_connections(kind="inet"))
                    if hasattr(process, "net_connections") else None,
            })

        try:
            import torch
            if torch.cuda.is_available():
                log_fields.update({
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
                })
        except Exception:
            pass
 
        logger.warning("worker_resource_snapshot", **log_fields)
        
@log_job(queue="segmentation")
@trace_job()
async def segment_objects_task(
    ctx, image_id: int, user_id: int, min_area: int = 500, max_segments: int = 50
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.segment_objects(
                image_id=image_id, user_id=user_id,
                min_area=min_area, max_segments=max_segments,
            )


@log_job(queue="segmentation")
@trace_job()
async def segment_with_prompt_task(
    ctx,
    image_id: int,
    user_id: int,
    point_coords: Optional[List[Tuple[int, int]]] = None,
    point_labels: Optional[List[int]] = None,
    bbox: Optional[dict] = None,
    multimask_output: Optional[bool] = None,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.segment_with_prompt(
                image_id=image_id, user_id=user_id,
                point_coords=point_coords, point_labels=point_labels,
                bbox=bbox, multimask_output=multimask_output,
            )


@log_job(queue="segmentation")
@trace_job()
async def segment_by_polygon_task(
    ctx,
    image_id: int,
    user_id: int,
    points: List[Tuple[int, int]],
    smooth: bool = True,
    smoothing_factor: float = 0.0,
    feather_px: int = 0,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.segment_by_polygon(
                image_id=image_id, user_id=user_id, points=points,
                smooth=smooth, smoothing_factor=smoothing_factor,
                feather_px=feather_px,
            )


@log_job(queue="segmentation")
@trace_job()
async def segment_hybrid_task(
    ctx,
    image_id: int,
    user_id: int,
    yolo_conf_threshold: float = 0.35,
    yolo_classes: Optional[List[str]] = None,
    fallback_min_area: int = 800,
    fallback_max_segments: int = 50,
    overlap_iou_thresh: float = 0.5,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.segment_hybrid(
                image_id=image_id, user_id=user_id,
                yolo_conf_threshold=yolo_conf_threshold,
                yolo_classes=yolo_classes,
                fallback_min_area=fallback_min_area,
                fallback_max_segments=fallback_max_segments,
                overlap_iou_thresh=overlap_iou_thresh,
            )


@log_job(queue="segmentation")
@trace_job()
async def sam_remove_object_task(
    ctx,
    image_id: int,
    mask_id: int,
    user_id: int,
    expand_mask_pixels: int = 12,
    use_edge_blending: bool = False,
    ldm_steps: int = 25,
    ldm_sampler: str = "plms",
    hd_strategy: str = "CROP",
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.sam_remove_object(
                image_id=image_id, mask_id=mask_id, user_id=user_id,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                ldm_steps=ldm_steps, ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )


@log_job(queue="segmentation")
@trace_job()
async def sam_replace_object_task(
    ctx,
    image_id: int,
    mask_id: int,
    replacement_image_bytes: bytes,
    user_id: int,
    expand_mask_pixels: int = 8,
    use_color_matching: bool = False,
    use_edge_blending: bool = False,
    color_match_method: str = "color_transfer",
    ldm_steps: int = 25,
    ldm_sampler: str = "plms",
    hd_strategy: str = "CROP",
    replacement_is_cutout: bool = False,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = SegmentationService(**deps)
            return await service.sam_replace_object(
                image_id=image_id, mask_id=mask_id,
                replacement_image_bytes=replacement_image_bytes, user_id=user_id,
                expand_mask_pixels=expand_mask_pixels,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                ldm_steps=ldm_steps, ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
                replacement_is_cutout=replacement_is_cutout,
            )

@log_job(queue="segmentation")
@trace_job()
async def sam_replace_object_diffusion_task(
    ctx,
    image_id: int,
    mask_bytes: bytes,
    bbox: dict,
    reference_image_bytes: bytes,
    user_id: int,
    prompt: str = "",
    use_color_matching: bool = False,
    color_match_method: str = "color_transfer",
    negative_prompt: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    ip_adapter_scale: Optional[float] = None,
    strength: Optional[float] = None,
    seed: int = 0,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = EditingService(**deps)
            return await service.sam_replace_object_diffusion(
                image_id=image_id,
                mask_bytes=mask_bytes,
                bbox=bbox,
                reference_image_bytes=reference_image_bytes,
                user_id=user_id,
                prompt=prompt,
                use_color_matching=use_color_matching,
                color_match_method=color_match_method,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_scale=ip_adapter_scale,
                strength=strength,
                seed=seed,
            )
            
@log_job(queue="inpainting")
@trace_job()
async def remove_object_task(
    ctx,
    image_id: int,
    bbox_id: int,
    user_id: int,
    expand_mask_pixels: int = 5,
    use_edge_blending: bool = True,
    ldm_steps: int = 25,
    ldm_sampler: str = "plms",
    hd_strategy: str = "CROP",
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = EditingService(**deps)
            return await service.remove_object(
                image_id=image_id, bbox_id=bbox_id, user_id=user_id,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                ldm_steps=ldm_steps, ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )


@log_job(queue="inpainting")
@trace_job()
async def remove_multiple_objects_task(
    ctx,
    image_id: int,
    bbox_ids: List[int],
    user_id: int,
    expand_mask_pixels: int = 5,
    use_edge_blending: bool = True,
    ldm_steps: int = 25,
    ldm_sampler: str = "plms",
    hd_strategy: str = "CROP",
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = EditingService(**deps)
            return await service.remove_multiple_objects(
                image_id=image_id, bbox_ids=bbox_ids, user_id=user_id,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                ldm_steps=ldm_steps, ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )


@log_job(queue="inpainting")
@trace_job()
async def replace_object_task(
    ctx,
    image_id: int,
    bbox_id: int,
    replace_image_bytes: bytes,
    user_id: int,
    expand_mask_pixels: int = 25,
    use_color_matching: bool = False,
    use_edge_blending: bool = False,
    color_match_method: str = "mean_std",
    ldm_steps: int = 25,
    ldm_sampler: str = "plms",
    hd_strategy: str = "CROP",
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            service = EditingService(**deps)
            return await service.replace_object(
                image_id=image_id, bbox_id=bbox_id,
                replace_image_bytes=replace_image_bytes, user_id=user_id,
                expand_mask_pixels=expand_mask_pixels,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                ldm_steps=ldm_steps, ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )


@log_job(queue="segmentation")
@trace_job()
async def sam_extract_object_task(
    ctx,
    image_id: int,
    mask_id: int,
    user_id: int,
    padding_pixels: int = 8,
    label: Optional[str] = None,
    persist_to_s3: bool = False,
) -> dict:
    async with get_db_session() as db:
        async with _build_ml_deps(db) as deps:
            # AssetService accepts redis_assets both explicitly and via
            deps = dict(deps)
            redis_assets = deps.pop("redis_assets")
            service = AssetService(redis_assets=redis_assets, **deps)
            return await service.extract_object(
                image_id=image_id, mask_id=mask_id, user_id=user_id,
                padding_pixels=padding_pixels, label=label,
                persist_to_s3=persist_to_s3,
            )


async def startup(ctx):
    """
    Warms up the ML model ONCE when the worker process starts — the
    equivalent of the lifespan-preload in app. arq calls this before
    processing the first task.
    """
    logger.info("ml_pipeline_warmup_started")
    get_pipeline()
    logger.info("ml_pipeline_warmup_finished")
    ctx["_resource_monitor_task"] = asyncio.create_task(_resource_monitor(interval_seconds=60))

async def shutdown(ctx):
    logger.info("worker_shutdown")
    task = ctx.get("_resource_monitor_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class WorkerSettings:
    functions = [
        segment_objects_task,
        segment_with_prompt_task,
        segment_by_polygon_task,
        sam_remove_object_task,
        sam_replace_object_task,
        sam_replace_object_diffusion_task,
        segment_hybrid_task,
        remove_object_task,
        remove_multiple_objects_task,
        replace_object_task,
        sam_extract_object_task,
    ]
    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 1

    job_timeout = 100_000_000

    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)