from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.db_connect import get_db
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