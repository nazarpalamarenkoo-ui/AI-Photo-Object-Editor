from typing import Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.pipeline.pipeline import MLPipeline, get_pipeline
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory
from app.storage.redis.redis_assets import RedisAssetsStorage
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.db.models.image import Image
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseMLService:
    """
    Base class with shared dependencies and helpers.
    Contains NO business logic — only infrastructure wiring.
    """

    def __init__(
        self,
        db: AsyncSession,
        s3_storage: S3Storage,
        redis_storage: RedisStorage,
        redis_history: RedisHistory,
        redis_assets: RedisStorage,
        image_repo: ImageRepository,
        detection_repo: DetectionRepository,
        pipeline: MLPipeline | None = None,
        device: str = "cpu",
    ):
        self.db = db
        self.s3 = s3_storage
        self.redis_storage = redis_storage
        self.redis_history = redis_history
        self.redis_assets = redis_assets
        self.image_repo = image_repo
        self.detection_repo = detection_repo
        self._pipeline = pipeline
        self._device = device
    @property
    def pipeline(self) -> MLPipeline:
        """Lazy initialization of the ML pipeline."""
        if self._pipeline is None:
            self._pipeline = get_pipeline()
        return self._pipeline

    async def _get_image_authorized(self, image_id: int, user_id: int) -> Image:
        """Fetch image from DB and verify ownership."""
        image = await self.image_repo.get_by_id(image_id)
        if not image:
            logger.warning("image_not_found", image_id=image_id)
            raise ValueError(f"Image {image_id} not found")
        if image.user_id != user_id:
            logger.warning(
                "image_access_unauthorized",
                image_id=image_id,
                owner_user_id=image.user_id,
                requesting_user_id=user_id,
            )
            raise ValueError("Unauthorized: image belongs to different user")
        return image


    async def _get_current_image_bytes(self, image_id: int, storage_path: str) -> bytes:
        """Get working image bytes — Redis current_state first, S3 fallback."""
        cached = await self.redis_storage.get_cache_image(image_id, suffix="current_state")
        if cached:
            return cached
        return await self.s3.download(storage_path)

    async def _get_current_state_url(self, image_id: int, user_id: int, storage_path: str) -> Tuple[str, bool]:
        """
        Return for the image's current working state.
        """
        cached = await self.redis_storage.get_cache_image(image_id, suffix="current_state")
        if cached:
            url = await self._get_temp_url_from_bytes(
                image_id=image_id, user_id=user_id, image_bytes=cached, op="current"
            )
            return url, True

        url = await self.s3.get_presigned_url(path=storage_path, expiration=3600)
        return url, False

    async def _save_current_state(self, image_id: int, image_bytes: bytes) -> None:
        """Persist working state to Redis (TTL 2 h)."""
        await self.redis_storage.cache_image(
            image_id=image_id,
            image_data=image_bytes,
            suffix="current_state",
            ttl=7200,
        )

    async def _upload_result(
        self,
        result_bytes: bytes,
        path: str,
        content_type: str = "image/jpeg",
    ) -> Tuple[str, str]:
        """Upload bytes to S3 and return (s3_uri, presigned_url)."""
        result_url = await self.s3.upload_bytes(
            data=result_bytes, path=path, content_type=content_type
        )
        presigned_url = await self.s3.get_presigned_url(path=path, expiration=3600)
        return result_url, presigned_url

    async def _get_temp_url_from_bytes(
        self, image_id: int, user_id: int, image_bytes: bytes, op: str
    ) -> str:
        """Upload bytes to a temp S3 path and return presigned URL."""
        path = f"temp/{user_id}/{image_id}/{op}_{int(datetime.utcnow().timestamp())}.jpg"
        await self.s3.upload_bytes(data=image_bytes, path=path, content_type="image/jpeg")
        return await self.s3.get_presigned_url(path=path, expiration=3600)

    async def _get_segment_or_raise(self, image_id: int, mask_id: int) -> dict:
        """Fetch cached segment by mask_id or raise ValueError."""
        segments = await self.redis_storage.get_cached_segments(image_id)
        if not segments:
            logger.warning("segments_not_cached", image_id=image_id)
            raise ValueError(
                f"No segments found for image {image_id}. Run segment_objects first."
            )
        segment = next((s for s in segments if s["mask_id"] == mask_id), None)
        if not segment:
            logger.warning("segment_not_found", image_id=image_id, mask_id=mask_id)
            raise ValueError(f"Segment with mask_id={mask_id} not found.")
        return segment