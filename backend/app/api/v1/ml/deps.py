from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.db_connect import get_db_session
from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.repository.image_content_repo import ImageContentRepository
from app.repository.detection_repo import DetectionRepository
from app.repository.segmentation_repo import SegmentationRepository
from app.repository.edit_history_repo import ImageEditHistoryRepository
from app.repository.assets_repo import AssetRepository
from app.repository.mljob_repo import MLJobRepository
from app.ml.pipeline.pipeline import get_pipeline
from app.services.ml.detector_service import DetectorService
from app.services.ml.editing_service import EditingService
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml.assets_service import AssetService
from app.services.ml.version_history_service import VersionHistoryService
from app.services.ml_job_service import MLJobService
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory
from app.storage.redis.redis_assets import RedisAssetsStorage


def _base_deps() -> dict:
    return dict(
        s3_storage=S3Storage(),
        redis_storage=RedisStorage(),
        redis_history=RedisHistory(),
        redis_assets=RedisAssetsStorage(),
        image_repo=ImageRepository(get_db_session),
        image_version_repo=ImageVersionRepository(get_db_session),
        image_content_repo=ImageContentRepository(get_db_session),
        detection_repo=DetectionRepository(get_db_session),
        segmentation_repo=SegmentationRepository(get_db_session),
        edit_history_repo=ImageEditHistoryRepository(get_db_session),
        assets_repo=AssetRepository(get_db_session),
        pipeline=get_pipeline(),
    )


def get_base_deps() -> dict:
    """
    Public entry point to the same dep-set _base_deps builds, for sync
    routes that need to hand a raw deps dict to
    app.services.ml.tracked_runner.run_tracked (which instantiates the
    service itself, so a pre-built service instance from get_editor/
    get_detector/etc. won't do here).
    """
    return _base_deps()

def get_detector() -> DetectorService:
    return DetectorService(**_base_deps())

def get_editor() -> EditingService:
    return EditingService(**_base_deps())

def get_segmentation() -> SegmentationService:
    return SegmentationService(**_base_deps())

def get_asset() -> AssetService:
    return AssetService(**_base_deps())

def get_version_history() -> VersionHistoryService:
    return VersionHistoryService(**_base_deps())

def get_mljob_service() -> MLJobService:
    return MLJobService(
        mljob_repo=MLJobRepository(get_db_session),
        image_repo=ImageRepository(get_db_session),
        image_version_repo=ImageVersionRepository(get_db_session),
    )

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