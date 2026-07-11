import logging
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

logger = logging.getLogger("arq.worker")

ML_DEVICE = getattr(settings, "ML_DEVICE", "cpu")


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
    pipeline = get_pipeline(device=ML_DEVICE)

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
    logger.info("Warming up ML pipeline (device=%s)...", ML_DEVICE)
    get_pipeline(device=ML_DEVICE)
    logger.info("ML pipeline ready.")


async def shutdown(ctx):
    logger.info("Worker shutting down.")


class WorkerSettings:
    functions = [
        segment_objects_task,
        segment_with_prompt_task,
        segment_by_polygon_task,
        sam_remove_object_task,
        sam_replace_object_task,
        segment_hybrid_task,
        remove_object_task,
        remove_multiple_objects_task,
        replace_object_task,
        sam_extract_object_task,
    ]
    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 1

    job_timeout = 300

    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)