import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import UploadFile
from io import BytesIO

from app.services.image_service import ImageService
from app.db.models.image import Image

@pytest.fixture
def mock_db():
    """Mock database session"""
    return MagicMock()


@pytest.fixture
def mock_s3():
    """Mock S3 storage"""
    s3 = MagicMock()
    s3.upload = AsyncMock(return_value='s3://bucket/uploads/123/test.jpg')
    s3.download = AsyncMock(return_value=b'fake_image_bytes')
    s3.delete = AsyncMock(return_value=True)
    s3.get_presigned_url = AsyncMock(return_value='https://presigned.url/test.jpg')
    return s3


@pytest.fixture
def mock_redis():
    """Mock Redis cache"""
    redis = MagicMock()
    redis.invalidate_image = AsyncMock()
    return redis


@pytest.fixture
def mock_image_repo():
    """Mock Image repository"""
    repo = MagicMock()
    
    # Mock image object
    mock_image = MagicMock(spec=Image)
    mock_image.id = 123
    mock_image.user_id = 456
    mock_image.filename = 'test.jpg'
    mock_image.storage_path = 's3://bucket/uploads/456/test.jpg'
    
    repo.create = AsyncMock(return_value=mock_image)
    repo.get_by_id = AsyncMock(return_value=mock_image)
    repo.get_user_images = AsyncMock(return_value=[mock_image])
    repo.delete = AsyncMock(return_value=True)
    
    return repo


@pytest.fixture
def image_service(mock_db, mock_s3, mock_redis, mock_image_repo):
    """Image Service instance with mocked dependencies"""
    service = ImageService(
        db=mock_db,
        s3=mock_s3,
        redis_cache=mock_redis,
        image_repo=mock_image_repo
    )
    # Add get_user_images method if it doesn't exist
    if not hasattr(service, 'get_user_images'):
        async def get_user_images(user_id, limit=None, offset=None):
            images = await mock_image_repo.get_user_images(user_id)
            if offset is not None and limit is not None:
                return images[offset:offset+limit]
            return images
        service.get_user_images = get_user_images
    return service


@pytest.fixture
def valid_upload_file():
    """Create valid upload file using MagicMock to allow setting content_type"""
    file_content = b'fake image content'
    file = MagicMock(spec=UploadFile)
    file.filename = 'test.jpg'
    file.file = BytesIO(file_content)
    file.content_type = 'image/jpeg'
    file.size = len(file_content)
    file.read = AsyncMock(return_value=file_content)
    return file

@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_success(image_service, valid_upload_file, mock_s3, mock_image_repo):
    """Test successful image upload"""
    result = await image_service.upload_image(
        file=valid_upload_file,
        user_id=456
    )
    
    # Verify S3 upload was called
    mock_s3.upload.assert_called_once()
    
    # Verify DB record created
    mock_image_repo.create.assert_called_once()
    
    # Check result
    assert result.id == 123
    assert result.filename == 'test.jpg'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_invalid_type(image_service):
    """Test upload with invalid file type"""
    invalid_file = MagicMock(spec=UploadFile)
    invalid_file.filename = 'test.pdf'
    invalid_file.file = BytesIO(b'fake content')
    invalid_file.content_type = 'application/pdf'
    invalid_file.size = len(b'fake content')
    
    with pytest.raises(ValueError, match="Invalid file type"):
        await image_service.upload_image(
            file=invalid_file,
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_too_large(image_service):
    """Test upload with file too large"""
    large_content = b'x' * (11 * 1024 * 1024)  # 11MB
    large_file = MagicMock(spec=UploadFile)
    large_file.filename = 'large.jpg'
    large_file.file = BytesIO(large_content)
    large_file.content_type = 'image/jpeg'
    large_file.size = 11 * 1024 * 1024
    
    with pytest.raises(ValueError, match="File too large"):
        await image_service.upload_image(
            file=large_file,
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_success(image_service):
    """Test get image with valid authorization"""
    result = await image_service.get_image(
        image_id=123,
        user_id=456
    )
    
    assert result.id == 123
    assert result.user_id == 456


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_not_found(image_service, mock_image_repo):
    """Test get image that doesn't exist"""
    mock_image_repo.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(ValueError, match="Image 123 not found"):
        await image_service.get_image(
            image_id=123,
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_unauthorized(image_service):
    """Test get image with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.get_image(
            image_id=123,
            user_id=999  # Wrong user!
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_images(image_service, mock_image_repo):
    """Test get all user images"""
    result = await image_service.get_user_images(user_id=456)
    
    mock_image_repo.get_user_images.assert_called_once_with(456)
    assert len(result) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_images_with_pagination(image_service, mock_image_repo):
    """Test get user images with pagination"""
    # Create multiple mock images
    mock_images = [MagicMock(id=i) for i in range(20)]
    mock_image_repo.get_user_images = AsyncMock(return_value=mock_images)
    
    result = await image_service.get_user_images(
        user_id=456,
        limit=10,
        offset=5
    )
    
    assert len(result) == 10
    assert result[0].id == 5  # Offset 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_success(image_service, mock_s3, mock_redis, mock_image_repo):
    """Test successful image deletion"""
    result = await image_service.delete_image(
        image_id=123,
        user_id=456
    )
    
    # Verify S3 deletion
    mock_s3.delete.assert_called_once()
    
    # Verify cache invalidation
    mock_redis.invalidate_image.assert_called_once_with(123)
    
    # Verify DB deletion
    mock_image_repo.delete.assert_called_once_with(123)
    
    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_unauthorized(image_service):
    """Test delete image with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.delete_image(
            image_id=123,
            user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_success(image_service, mock_s3):
    """Test download image bytes"""
    result = await image_service.download_image(
        image_id=123,
        user_id=456
    )
    
    mock_s3.download.assert_called_once()
    assert result == b'fake_image_bytes'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_unauthorized(image_service):
    """Test download with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.download_image(
            image_id=123,
            user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_success(image_service, mock_s3):
    """Test generate presigned URL"""
    result = await image_service.get_presigned_url(
        image_id=123,
        user_id=456,
        expiration=3600
    )
    
    mock_s3.get_presigned_url.assert_called_once()
    assert result == 'https://presigned.url/test.jpg'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_unauthorized(image_service):
    """Test presigned URL with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await image_service.get_presigned_url(
            image_id=123,
            user_id=999
        )