import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.detection_service import DetectionService
from app.db.models.detection import Detection
from app.db.models.image import Image


@pytest.fixture
def mock_db():
    """Mock database session"""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Mock Redis cache"""
    redis = MagicMock()
    redis.get_cached_detections = AsyncMock(return_value=None)
    redis.cache_detections = AsyncMock()
    redis.invalidate_image = AsyncMock()
    return redis


@pytest.fixture
def mock_detection_repo():
    """Mock Detection repository"""
    repo = MagicMock()
    
    # Mock detections
    mock_detections = [
        MagicMock(
            id=1,
            image_id=123,
            bbox_id=0,
            detected_class='car',
            confidence=0.95,
            x1=100, y1=100, x2=200, y2=200
        ),
        MagicMock(
            id=2,
            image_id=123,
            bbox_id=1,
            detected_class='person',
            confidence=0.88,
            x1=300, y1=150, x2=400, y2=350
        )
    ]
    
    repo.get_by_image = AsyncMock(return_value=mock_detections)
    repo.delete_by_image = AsyncMock(return_value=2)
    
    return repo


@pytest.fixture
def mock_image_repo():
    """Mock Image repository"""
    repo = MagicMock()
    
    mock_image = MagicMock(spec=Image)
    mock_image.id = 123
    mock_image.user_id = 456
    mock_image.filename = 'test.jpg'
    
    repo.get_by_id = AsyncMock(return_value=mock_image)
    
    return repo


@pytest.fixture
def detection_service(mock_db, mock_redis, mock_detection_repo, mock_image_repo):
    """Detection Service instance with mocked dependencies"""
    return DetectionService(
        db=mock_db,
        redis_cache=mock_redis,
        detection_repo=mock_detection_repo,
        image_repo=mock_image_repo
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_success(detection_service, mock_detection_repo):
    """Test get detections with valid authorization"""
    result = await detection_service.get_image_detections(
        image_id=123,
        user_id=456
    )
    
    mock_detection_repo.get_by_image.assert_called_once_with(123)
    assert len(result) == 2
    assert result[0].detected_class == 'car'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_image_not_found(detection_service, mock_image_repo):
    """Test get detections for non-existent image"""
    mock_image_repo.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(ValueError, match="Image 123 not found"):
        await detection_service.get_image_detections(
            image_id=123,
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_unauthorized(detection_service):
    """Test get detections with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.get_image_detections(
            image_id=123,
            user_id=999  # Wrong user!
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_from_cache(detection_service, mock_redis, mock_detection_repo):
    """Test get detections from cache"""
    # Setup cache hit
    cached_data = [MagicMock(bbox_id=0)]
    mock_redis.get_cached_detections = AsyncMock(return_value=cached_data)
    
    result = await detection_service.get_image_detections(
        image_id=123,
        user_id=456,
        use_cache=True
    )
    
    # Should NOT call DB
    mock_detection_repo.get_by_image.assert_not_called()
    
    # Should return cached data
    assert result == cached_data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_cache_miss(detection_service, mock_redis, mock_detection_repo):
    """Test get detections with cache miss"""
    # Cache returns None
    mock_redis.get_cached_detections = AsyncMock(return_value=None)
    
    result = await detection_service.get_image_detections(
        image_id=123,
        user_id=456,
        use_cache=True
    )
    
    # Should call DB
    mock_detection_repo.get_by_image.assert_called_once()
    
    # Should cache results
    mock_redis.cache_detections.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_cache_disabled(detection_service, mock_redis, mock_detection_repo):
    """Test get detections with cache disabled"""
    result = await detection_service.get_image_detections(
        image_id=123,
        user_id=456,
        use_cache=False
    )
    
    # Should call DB
    mock_detection_repo.get_by_image.assert_called_once()
    
    # Should NOT check cache
    mock_redis.get_cached_detections.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_success(detection_service):
    """Test get specific detection by bbox_id"""
    result = await detection_service.get_detection_by_bbox_id(
        image_id=123,
        bbox_id=0,
        user_id=456
    )
    
    assert result.bbox_id == 0
    assert result.detected_class == 'car'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_not_found(detection_service):
    """Test get detection with non-existent bbox_id"""
    with pytest.raises(ValueError, match="Detection with bbox_id 999 not found"):
        await detection_service.get_detection_by_bbox_id(
            image_id=123,
            bbox_id=999,  # Doesn't exist!
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_unauthorized(detection_service):
    """Test get detection with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.get_detection_by_bbox_id(
            image_id=123,
            bbox_id=0,
            user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_success(detection_service, mock_detection_repo, mock_redis):
    """Test delete all detections for image"""
    count = await detection_service.delete_image_detections(
        image_id=123,
        user_id=456
    )
    
    # Verify deletion
    mock_detection_repo.delete_by_image.assert_called_once_with(123)
    
    # Verify cache invalidation
    mock_redis.invalidate_image.assert_called_once_with(123)
    
    assert count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_unauthorized(detection_service):
    """Test delete detections with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.delete_image_detections(
            image_id=123,
            user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_image_not_found(detection_service, mock_image_repo):
    """Test delete detections for non-existent image"""
    mock_image_repo.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(ValueError, match="Image 123 not found"):
        await detection_service.delete_image_detections(
            image_id=123,
            user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_success(detection_service):
    """Test get detection statistics"""
    stats = await detection_service.get_detection_stats(
        image_id=123,
        user_id=456
    )
    
    assert stats['total_detections'] == 2
    assert 'car' in stats['classes']
    assert 'person' in stats['classes']
    assert stats['avg_confidence'] == pytest.approx(0.915, rel=0.01)  # (0.95 + 0.88) / 2
    assert stats['min_confidence'] == 0.88
    assert stats['max_confidence'] == 0.95


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_no_detections(detection_service, mock_detection_repo):
    """Test get stats with no detections"""
    mock_detection_repo.get_by_image = AsyncMock(return_value=[])
    
    stats = await detection_service.get_detection_stats(
        image_id=123,
        user_id=456
    )
    
    assert stats['total_detections'] == 0
    assert stats['classes'] == []
    assert stats['avg_confidence'] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_unauthorized(detection_service):
    """Test get stats with wrong user"""
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.get_detection_stats(
            image_id=123,
            user_id=999
        )