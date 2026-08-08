import io
from typing import Optional, Tuple
from datetime import datetime
from PIL import Image as PILImage

from app.ml.pipeline.pipeline import MLPipeline, get_pipeline
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory
from app.storage.redis.redis_assets import RedisAssetsStorage
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.repository.segmentation_repo import SegmentationRepository
from app.repository.edit_history_repo import ImageEditHistoryRepository
from app.repository.assets_repo import AssetRepository
from app.repository.image_content_repo import ImageContentRepository
from app.db.models.image import Image
from app.db.models.image_content import ImageContent
from app.repository.image_version_repo import ImageVersionRepository
from app.db.models.image_version import ImageVersion
from app.db.schemas.model_meta import ModelMeta
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseMLService:

    def __init__(
        self,
        s3_storage: S3Storage,
        redis_storage: RedisStorage,
        redis_history: RedisHistory,
        redis_assets: RedisStorage,
        image_repo: ImageRepository,
        image_version_repo: ImageVersionRepository,
        image_content_repo: ImageContentRepository,
        detection_repo: DetectionRepository,
        segmentation_repo: SegmentationRepository,
        edit_history_repo: ImageEditHistoryRepository,
        assets_repo: AssetRepository,
        pipeline: MLPipeline | None = None,
        device: str = "cuda",
    ):
        self.s3 = s3_storage
        self.redis_storage = redis_storage
        self.redis_history = redis_history
        self.redis_assets = redis_assets
        self.image_repo = image_repo
        self.image_version_repo = image_version_repo
        self.image_content_repo = image_content_repo
        self.detection_repo = detection_repo
        self.segmentation_repo = segmentation_repo
        self.edit_history_repo = edit_history_repo
        self.assets_repo = assets_repo
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

    async def _get_current_version_authorized(
        self, image_id: int, user_id: int
    ) -> tuple[Image, ImageVersion]:
        image = await self._get_image_authorized(image_id, user_id)
        version = await self.image_version_repo.get_current(image)
        if version is None:
            logger.error("image_missing_current_version", image_id=image_id)
            raise ValueError(f"Image {image_id} has no current version")
        return image, version

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

    async def _fork_version(
        self,
        image: Image,
        result_bytes: bytes,
        storage_path: str,
    ) -> ImageVersion:
        content_hash = ImageContent.hash_bytes(result_bytes)
        width, height = self._read_dimensions(result_bytes)

        content, created = await self.image_content_repo.get_or_create(
            content_hash=content_hash,
            storage_path=storage_path,
            width=width,
            height=height,
            file_size=len(result_bytes),
        )

        new_version = await self.image_version_repo.create_next(
            image, storage_path=storage_path, content_id=content.id
        )

        logger.info(
            "version_forked",
            image_id=image.id,
            new_version_id=new_version.id,
            content_id=content.id,
            content_deduplicated=not created,
        )
        return new_version

    @staticmethod
    def _read_dimensions(image_bytes: bytes) -> tuple[int, int]:
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            return img.width, img.height

    async def _get_segment_or_raise(
        self, content_id: int, mask_id: int, image_id: int
    ) -> dict:
        masks = await self.segmentation_repo.get_by_content(
            content_id, active_only=True
        )
        match = next((m for m in masks if m.mask_id == mask_id), None)
        if not match:
            logger.warning(
                "segment_not_found",
                content_id=content_id,
                mask_id=mask_id,
            )
            raise ValueError(
                f"Segment with mask_id={mask_id} not found for content {content_id}."
            )

        cache_suffix = f"mask:{content_id}:{mask_id}"
        mask_bytes = await self.redis_storage.get_cache_image(image_id, suffix=cache_suffix)
        if not mask_bytes:
            mask_bytes = await self.s3.download(match.mask_storage_path)
            await self.redis_storage.cache_image(
                image_id=image_id,
                image_data=mask_bytes,
                suffix=cache_suffix,
                ttl=7200,
            )

        return {
            "id": match.id,
            "mask_id": match.mask_id,
            "bbox": {"x1": match.x1, "y1": match.y1, "x2": match.x2, "y2": match.y2},
            "area": match.area,
            "score": match.score,
            "mask_bytes": mask_bytes,
        }

    @staticmethod
    def _extract_model_meta(
        result_or_metrics: Optional[dict],
        default_name: str,
        default_version: str = "unknown",
    ) -> ModelMeta:
        result_or_metrics = result_or_metrics or {}
        metrics = result_or_metrics.get("metrics", result_or_metrics) or {}

        model_name = (
            result_or_metrics.get("model_name")
            or metrics.get("model_name")
            or default_name
        )
        model_version = (
            result_or_metrics.get("model_version")
            or metrics.get("model_version")
            or default_version
        )

        inference_time_ms = metrics.get("inference_time_ms")
        if inference_time_ms is None:
            inference_time_ms = metrics.get("processing_time_ms")
        if inference_time_ms is None:
            inference_time_s = metrics.get("inference_time_s")
            if inference_time_s is None:
                inference_time_s = metrics.get("processing_time_s")
            inference_time_ms = inference_time_s * 1000 if inference_time_s is not None else 0.0

        return ModelMeta(
            model_name=model_name,
            model_version=model_version,
            inference_time_ms=float(inference_time_ms),
        )

    @staticmethod
    def _extract_processing_time_ms(metrics: Optional[dict]) -> Optional[int]:
        """
        Best-effort read of an *overall op* duration out of pipeline
        metrics, for ImageEditHistory.processing_time_ms.
        """
        if not metrics:
            return None
        for key in ("processing_time_ms", "total_time_ms", "inference_time_ms"):
            if key in metrics:
                return int(metrics[key])
        if "inference_time_s" in metrics:
            return int(metrics["inference_time_s"] * 1000)
        return None