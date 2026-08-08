import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import UploadFile
from PIL import Image as PILImage

from app.services.image_service import ImageService
from app.db.models.image import Image
from app.db.models.image_version import ImageVersion
from app.db.models.image_content import ImageContent


def _png_bytes(size=(20, 20)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, "black").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_s3():
    s3 = MagicMock()
    s3.upload_bytes = AsyncMock(return_value='s3://bucket/uploads/456/test.png')
    s3.download = AsyncMock(return_value=b'fake_image_bytes')
    s3.delete = AsyncMock(return_value=True)
    s3.get_presigned_url = AsyncMock(return_value='https://presigned.url/test.png')
    return s3


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.invalidate_image = AsyncMock()
    redis.cache_image = AsyncMock(return_value='image:123:original')
    return redis


@pytest.fixture
def mock_image_repo():
    repo = MagicMock()

    mock_image = MagicMock(spec=Image)
    mock_image.id = 123
    mock_image.user_id = 456
    mock_image.filename = 'test.png'
    mock_image.storage_path = 's3://bucket/uploads/456/test.png'
    mock_image.cache_key = None

    repo.create = AsyncMock(return_value=mock_image)
    repo.get_by_id = AsyncMock(return_value=mock_image)
    repo.get_user_images = AsyncMock(return_value=[mock_image])
    repo.delete = AsyncMock(return_value=True)
    repo.update = AsyncMock(return_value=mock_image)

    return repo


@pytest.fixture
def mock_image_version_repo():
    repo = MagicMock()

    mock_version = MagicMock(spec=ImageVersion)
    mock_version.id = 1
    mock_version.content_id = 10

    repo.create_original = AsyncMock(return_value=mock_version)
    repo.get_current = AsyncMock(return_value=mock_version)
    return repo


@pytest.fixture
def mock_image_content_repo():
    repo = MagicMock()

    mock_content = MagicMock(spec=ImageContent)
    mock_content.id = 10

    repo.get_or_create = AsyncMock(return_value=(mock_content, True))
    return repo


@pytest.fixture
def image_service(mock_s3, mock_redis, mock_image_repo, mock_image_version_repo, mock_image_content_repo):
    return ImageService(
        s3=mock_s3,
        redis_cache=mock_redis,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
        image_content_repo=mock_image_content_repo,
    )


@pytest.fixture
def valid_upload_file():
    file_content = _png_bytes()
    file = MagicMock(spec=UploadFile)
    file.filename = 'test.png'
    file.file = io.BytesIO(file_content)
    file.content_type = 'image/png'
    file.size = len(file_content)
    file.read = AsyncMock(return_value=file_content)
    return file


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_success(
    image_service, valid_upload_file, mock_s3, mock_image_repo,
    mock_image_version_repo, mock_image_content_repo,
):
    result = await image_service.upload_image(file=valid_upload_file, user_id=456)

    mock_s3.upload_bytes.assert_called_once()
    mock_image_repo.create.assert_called_once()
    mock_image_content_repo.get_or_create.assert_called_once()
    mock_image_version_repo.create_original.assert_called_once()
    assert mock_image_repo.get_by_id.await_count == 1

    assert result.id == 123
    assert result.filename == 'test.png'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_dedup_uses_existing_content(
    image_service, valid_upload_file, mock_image_content_repo, mock_image_version_repo,
):
    existing_content = MagicMock(spec=ImageContent)
    existing_content.id = 77
    mock_image_content_repo.get_or_create = AsyncMock(return_value=(existing_content, False))

    await image_service.upload_image(file=valid_upload_file, user_id=456)

    _, kwargs = mock_image_version_repo.create_original.call_args
    assert kwargs['content_id'] == 77


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_invalid_type(image_service):
    invalid_file = MagicMock(spec=UploadFile)
    invalid_file.filename = 'test.pdf'
    invalid_file.file = io.BytesIO(b'fake content')
    invalid_file.content_type = 'application/pdf'
    invalid_file.size = len(b'fake content')

    with pytest.raises(ValueError, match="Invalid file type"):
        await image_service.upload_image(file=invalid_file, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_too_large(image_service):
    large_file = MagicMock(spec=UploadFile)
    large_file.filename = 'large.jpg'
    large_file.file = io.BytesIO(b'x')
    large_file.content_type = 'image/jpeg'
    large_file.size = 11 * 1024 * 1024

    with pytest.raises(ValueError, match="File too large"):
        await image_service.upload_image(file=large_file, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_bad_dimensions(image_service):
    bad_file = MagicMock(spec=UploadFile)
    bad_file.filename = 'not-really-an-image.jpg'
    bad_file.file = io.BytesIO(b'not an image')
    bad_file.content_type = 'image/jpeg'
    bad_file.size = 12
    bad_file.read = AsyncMock(return_value=b'not an image')

    with pytest.raises(ValueError, match="Could not read image dimensions"):
        await image_service.upload_image(file=bad_file, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_success(image_service):
    result = await image_service.get_image(image_id=123, user_id=456)

    assert result.id == 123
    assert result.user_id == 456


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_not_found(image_service, mock_image_repo):
    mock_image_repo.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Image 123 not found"):
        await image_service.get_image(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_unauthorized(image_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.get_image(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_version_success(image_service, mock_image_version_repo):
    result = await image_service.get_current_version(image_id=123, user_id=456)

    mock_image_version_repo.get_current.assert_awaited_once()
    assert result.id == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_version_missing(image_service, mock_image_version_repo):
    mock_image_version_repo.get_current = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="has no current version"):
        await image_service.get_current_version(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_version_unauthorized(image_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.get_current_version(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_images(image_service, mock_image_repo):
    result = await image_service.get_user_image(user_id=456)

    mock_image_repo.get_user_images.assert_called_once_with(456)
    assert len(result) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_images_with_pagination(image_service, mock_image_repo):
    mock_images = [MagicMock(id=i) for i in range(20)]
    mock_image_repo.get_user_images = AsyncMock(return_value=mock_images)

    result = await image_service.get_user_image(user_id=456, limit=10, offset=5)

    assert len(result) == 10
    assert result[0].id == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_success(image_service, mock_s3, mock_redis, mock_image_repo):
    result = await image_service.delete_image(image_id=123, user_id=456)

    mock_s3.delete.assert_called_once()
    mock_redis.invalidate_image.assert_called_once_with(123)
    mock_image_repo.delete.assert_called_once_with(123)

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_unauthorized(image_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.delete_image(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_success(image_service, mock_s3):
    result = await image_service.download_image(image_id=123, user_id=456)

    mock_s3.download.assert_called_once()
    assert result == b'fake_image_bytes'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_unauthorized(image_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.download_image(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_success(image_service, mock_s3):
    result = await image_service.get_presigned_url(image_id=123, user_id=456, expiration=3600)

    mock_s3.get_presigned_url.assert_called_once()
    assert result == 'https://presigned.url/test.png'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_unauthorized(image_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.get_presigned_url(image_id=123, user_id=999)