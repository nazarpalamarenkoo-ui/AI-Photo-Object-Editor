import pytest
from io import BytesIO
from fastapi import UploadFile

from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.repository.image_content_repo import ImageContentRepository
from app.services.image_service import ImageService

pytestmark = pytest.mark.integration


def _make_service(db_session, mock_s3_storage, mock_redis_cache) -> ImageService:
    """ImageService.__init__ takes s3/redis_cache/image_repo/
    image_version_repo/image_content_repo — no db kwarg."""
    return ImageService(
        s3=mock_s3_storage,
        redis_cache=mock_redis_cache,
        image_repo=ImageRepository(db_session),
        image_version_repo=ImageVersionRepository(db_session),
        image_content_repo=ImageContentRepository(db_session),
    )


def _upload_file(image_bytes: bytes, filename: str = "test.png") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(image_bytes), headers={"content-type": "image/png"})


class TestUploadImage:
    @pytest.mark.asyncio
    async def test_creates_image_and_original_version(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_user, image_bytes,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)

        image = await service.upload_image(_upload_file(image_bytes), sample_user.id)

        assert image.user_id == sample_user.id
        assert image.width == 20
        assert image.height == 20
        version = await service.get_current_version(image.id, sample_user.id)
        assert version.image_id == image.id
        assert version.content_id is not None

    @pytest.mark.asyncio
    async def test_duplicate_content_dedupes_to_same_content_id(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_user, image_bytes,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)

        first = await service.upload_image(_upload_file(image_bytes, "a.png"), sample_user.id)
        second = await service.upload_image(_upload_file(image_bytes, "b.png"), sample_user.id)

        v1 = await service.get_current_version(first.id, sample_user.id)
        v2 = await service.get_current_version(second.id, sample_user.id)
        assert v1.content_id == v2.content_id

    @pytest.mark.asyncio
    async def test_rejects_invalid_content_type(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_user, image_bytes,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        bad_file = UploadFile(filename="x.txt", file=BytesIO(b"not an image"), headers={"content-type": "text/plain"})

        with pytest.raises(ValueError, match="Invalid file type"):
            await service.upload_image(bad_file, sample_user.id)

    @pytest.mark.asyncio
    async def test_rejects_unreadable_dimensions(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_user,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        bad_file = UploadFile(filename="x.png", file=BytesIO(b"not-actually-a-png"), headers={"content-type": "image/png"})

        with pytest.raises(ValueError, match="Could not read image dimensions"):
            await service.upload_image(bad_file, sample_user.id)


class TestGetImage:
    @pytest.mark.asyncio
    async def test_success(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        result = await service.get_image(sample_image.id, sample_user.id)
        assert result.id == sample_image.id
        assert result.user_id == sample_user.id

    @pytest.mark.asyncio
    async def test_not_found(self, db_session, mock_s3_storage, mock_redis_cache):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="not found"):
            await service.get_image(99999, 1)

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_image(sample_image.id, sample_user.id + 999)


class TestGetCurrentVersion:
    @pytest.mark.asyncio
    async def test_returns_current_version(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        version = await service.get_current_version(sample_image.id, sample_user.id)
        assert version.id == sample_image_version.id

    @pytest.mark.asyncio
    async def test_raises_when_no_version(
        self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user,
    ):
        # sample_image has no ImageVersion unless sample_image_version was also requested
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="no current version"):
            await service.get_current_version(sample_image.id, sample_user.id)


class TestGetUserImage:
    @pytest.mark.asyncio
    async def test_lists_users_images(
        self, db_session, mock_s3_storage, mock_redis_cache, multiple_images, sample_user,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        images = await service.get_user_image(sample_user.id)
        assert len(images) == 3

    @pytest.mark.asyncio
    async def test_applies_limit_and_offset(
        self, db_session, mock_s3_storage, mock_redis_cache, multiple_images, sample_user,
    ):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        images = await service.get_user_image(sample_user.id, limit=1, offset=1)
        assert len(images) == 1


class TestDeleteImage:
    @pytest.mark.asyncio
    async def test_success(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        result = await service.delete_image(sample_image.id, sample_user.id)
        assert result is True
        mock_s3_storage.delete.assert_awaited_once_with(sample_image.storage_path)

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.delete_image(sample_image.id, sample_user.id + 999)


class TestDownloadImage:
    @pytest.mark.asyncio
    async def test_success(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        data = await service.download_image(sample_image.id, sample_user.id)
        assert data == b"fake downloaded data"

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.download_image(sample_image.id, sample_user.id + 999)


class TestGetPresignedUrl:
    @pytest.mark.asyncio
    async def test_success(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        url = await service.get_presigned_url(sample_image.id, sample_user.id)
        assert url.startswith("https://")

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
        service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_presigned_url(sample_image.id, sample_user.id + 999)