import io
from typing import List, Optional
from fastapi import UploadFile
from PIL import Image as PILImage

from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.repository.image_content_repo import ImageContentRepository
from app.storage.s3_storage import S3Storage
from app.storage.redis.redis_storage import RedisStorage
from app.db.models.image import Image
from app.db.models.image_version import ImageVersion
from app.db.models.image_content import ImageContent
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class ImageService:

    def __init__(
        self,
        s3: S3Storage,
        redis_cache: RedisStorage,
        image_repo: ImageRepository,
        image_version_repo: ImageVersionRepository,
        image_content_repo: ImageContentRepository,
    ):
        self.s3 = s3
        self.redis = redis_cache
        self.image_repo = image_repo
        self.image_version_repo = image_version_repo
        self.image_content_repo = image_content_repo

    async def upload_image(self, file: UploadFile, user_id: int) -> Image:
        with log_execution(
            "service_upload_image",
            logger=logger,
            user_id=user_id,
            filename=file.filename,
            content_type=file.content_type,
        ):
            self._validate_file(file)

            file_content = await file.read()
            width, height = self._read_dimensions(file_content, file.filename)

            storage_path = f'uploads/{user_id}/{file.filename}'

            s3_url = await self.s3.upload_bytes(
                data=file_content,
                path=storage_path,
                content_type=file.content_type
            )

            # Create the row first (no placeholder cache write under a
            # shared id=0 key — that key collided across concurrent
            # uploads from different users). Once we have the real id,
            # cache under it directly.
            image = await self.image_repo.create(
                filename=file.filename,
                storage_path=s3_url,
                user_id=user_id,
                mime_type=file.content_type,
                width=width,
                height=height,
                file_size=len(file_content),
                cache_key=None,
            )

            cache_key = await self.redis.cache_image(
                image_id=image.id,
                image_data=file_content,
                suffix='original'
            )
            image.cache_key = cache_key
            await self.image_repo.update(image)

            # Dedup point: if these exact bytes were uploaded before (by
            # this user or anyone else — content has no owner), version 0
            # points straight at the existing ImageContent and inherits
            # whatever detections/segmentation_masks already exist there.
            # No ML job needs to run again for this content.
            content_hash = ImageContent.hash_bytes(file_content)
            content, content_created = await self.image_content_repo.get_or_create(
                content_hash=content_hash,
                storage_path=s3_url,
                width=width,
                height=height,
                file_size=len(file_content),
            )

            # version 0 — "original" pixel state; also sets image.current_version_id.
            # create_original runs on its own short-lived session (session-per-
            # operation repos) and mutates a *merged copy* of `image` there — the
            # `image` instance we're holding here never sees that write. Refetch
            # by id instead of the old self.db.refresh(image), which no longer
            # applies (no shared session) and wouldn't have picked up a merged
            # copy's changes anyway.
            version = await self.image_version_repo.create_original(image, content_id=content.id)
            image = await self.image_repo.get_by_id(image.id)

            logger.info(
                "image_uploaded",
                image_id=image.id,
                user_id=user_id,
                version_id=version.id,
                content_id=content.id,
                content_deduplicated=not content_created,
                size_bytes=len(file_content),
            )

        return image

    async def get_image(
        self,
        image_id: int,
        user_id: int
    ) -> Image:

        image = await self.image_repo.get_by_id(image_id)

        if not image:
            logger.warning('image_not_found', image_id=image_id)
            raise ValueError(f'Image {image_id} not found')

        if image.user_id != user_id:
            logger.warning(
                'image_access_unauthorized',
                image_id=image_id,
                owner_user_id=image.user_id,
                requesting_user_id=user_id,
            )
            raise ValueError('Unauthorized: image belongs to different user')

        return image

    async def get_current_version(
        self,
        image_id: int,
        user_id: int
    ) -> ImageVersion:
        """
        Resolve the current pixel-state pointer for an image.

        Other services (Detection/SegmentationMask/MLJob) scope by
        version.content_id, not image_id or image_version_id directly —
        this is the entry point they should call to get that version
        (and its content_id) after an auth-checked lookup.
        """
        image = await self.get_image(image_id, user_id)
        version = await self.image_version_repo.get_current(image)
        if version is None:
            logger.error('image_missing_current_version', image_id=image_id)
            raise ValueError(f'Image {image_id} has no current version')
        return version

    async def get_user_image(
        self,
        user_id: int,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Image]:

        images = await self.image_repo.get_user_images(user_id)

        if offset is not None:
            images = images[offset:]
        if limit is not None:
            images = images[:limit]

        return images

    async def delete_image(
        self,
        image_id: int,
        user_id: int
    ) -> bool:

        image = await self.get_image(image_id, user_id)

        await self.s3.delete(image.storage_path)
        await self.redis.invalidate_image(image_id)

        success = await self.image_repo.delete(image_id)

        logger.info('image_deleted', image_id=image_id, user_id=user_id, success=success)

        return success

    async def download_image(
        self,
        image_id: int,
        user_id: int
    ) -> bytes:

        image = await self.get_image(image_id, user_id)
        image_bytes = await self.s3.download(image.storage_path)

        return image_bytes

    async def get_presigned_url(
        self,
        image_id: int,
        user_id: int,
        expiration: int = 3600
    ) -> str:

        image = await self.get_image(image_id, user_id)

        url = await self.s3.get_presigned_url(
            path=image.storage_path,
            expiration=expiration
        )

        return url

    def _validate_file(self, file: UploadFile) -> None:
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if file.content_type not in allowed_types:
            logger.warning(
                'upload_rejected_invalid_type',
                content_type=file.content_type,
                filename=file.filename,
            )
            raise ValueError(
                f"Invalid file type: {file.content_type}. "
                f"Allowed types: {', '.join(allowed_types)}"
            )

        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            logger.warning(
                'upload_rejected_too_large',
                size_bytes=file.size,
                filename=file.filename,
            )
            raise ValueError(
                f"File too large: {file.size / (1024*1024):.2f}MB. "
                f"Max size: 10MB"
            )

    def _read_dimensions(self, file_content: bytes, filename: str) -> tuple[int, int]:
        """Pillow needs real pixel dims for Image.width/height (NOT NULL)."""
        try:
            with PILImage.open(io.BytesIO(file_content)) as img:
                return img.width, img.height
        except Exception as e:
            logger.warning('image_dimension_read_failed', filename=filename, exc_info=e)
            raise ValueError(f"Could not read image dimensions: {e}")