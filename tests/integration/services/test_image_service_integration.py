import pytest
from app.repository.image_repo import ImageRepository
from app.services.image_service import ImageService


def _make_service(db_session, mock_s3_storage, mock_redis_cache) -> ImageService:
    return ImageService(
        db=db_session,
        s3=mock_s3_storage,
        redis_cache=mock_redis_cache,
        image_repo=ImageRepository(db_session),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_success(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    result = await service.get_image(sample_image.id, sample_user.id)
    assert result.id == sample_image.id
    assert result.user_id == sample_user.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_not_found(db_session, mock_s3_storage, mock_redis_cache):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    with pytest.raises(ValueError, match="not found"):
        await service.get_image(99999, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_unauthorized(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.get_image(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_success(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    result = await service.delete_image(sample_image.id, sample_user.id)
    assert result is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_unauthorized(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.delete_image(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_image(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    data = await service.download_image(sample_image.id, sample_user.id)
    assert data == b"fake downloaded data"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_image_unauthorized(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.download_image(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_presigned_url(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    url = await service.get_presigned_url(sample_image.id, sample_user.id)
    assert "presigned" in url or url.startswith("http")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_presigned_url_unauthorized(db_session, mock_s3_storage, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.get_presigned_url(sample_image.id, sample_user.id + 999)