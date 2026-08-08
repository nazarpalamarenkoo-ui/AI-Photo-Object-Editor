import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.detection_service import DetectionService
from app.db.models.image import Image
from app.db.models.image_version import ImageVersion


def make_detection(bbox_id, content_id=100, detected_class='car', confidence=0.95, is_active=True):
    d = MagicMock()
    d.id = bbox_id + 1000
    d.content_id = content_id
    d.bbox_id = bbox_id
    d.detected_class = detected_class
    d.confidence = confidence
    d.is_active = is_active
    d.x1, d.y1, d.x2, d.y2 = 100, 100, 200, 200
    return d


@pytest.fixture
def sample_image():
    image = MagicMock(spec=Image)
    image.id = 123
    image.user_id = 456
    return image


@pytest.fixture
def sample_version():
    version = MagicMock(spec=ImageVersion)
    version.id = 1
    version.image_id = 123
    version.content_id = 100
    return version


@pytest.fixture
def mock_detection_repo():
    repo = MagicMock()
    detections = [
        make_detection(0, detected_class='car', confidence=0.95),
        make_detection(1, detected_class='person', confidence=0.88),
    ]
    repo.get_by_content = AsyncMock(return_value=detections)
    repo.delete_by_content = AsyncMock(return_value=2)
    repo.soft_delete = AsyncMock(side_effect=lambda det_id: MagicMock(id=det_id, is_active=False))
    return repo


@pytest.fixture
def mock_image_repo(sample_image):
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=sample_image)
    return repo


@pytest.fixture
def mock_image_version_repo(sample_version):
    repo = MagicMock()
    repo.get_current = AsyncMock(return_value=sample_version)
    repo.get_by_id = AsyncMock(return_value=sample_version)
    return repo


@pytest.fixture
def detection_service(mock_detection_repo, mock_image_repo, mock_image_version_repo):
    return DetectionService(
        detection_repo=mock_detection_repo,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_uses_current_version_content(
    detection_service, mock_detection_repo, mock_image_version_repo,
):
    result = await detection_service.get_detections(image_id=123, user_id=456)

    mock_image_version_repo.get_current.assert_awaited_once()
    mock_detection_repo.get_by_content.assert_called_once_with(100, active_only=True)
    assert len(result) == 2
    assert result[0].detected_class == 'car'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_explicit_version_id(
    detection_service, mock_detection_repo, mock_image_version_repo, sample_version,
):
    result = await detection_service.get_detections(image_id=123, user_id=456, version_id=1)

    mock_image_version_repo.get_by_id.assert_awaited_once_with(1)
    mock_detection_repo.get_by_content.assert_called_once_with(100, active_only=True)
    assert len(result) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_version_mismatch_raises(
    detection_service, mock_image_version_repo, sample_version,
):
    sample_version.image_id = 999  # belongs to a different image

    with pytest.raises(ValueError, match="Version 1 not found for image 123"):
        await detection_service.get_detections(image_id=123, user_id=456, version_id=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_image_not_found(detection_service, mock_image_repo):
    mock_image_repo.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Image 123 not found"):
        await detection_service.get_detections(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_unauthorized(detection_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.get_detections(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detections_no_current_version(detection_service, mock_image_version_repo):
    mock_image_version_repo.get_current = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="has no current version"):
        await detection_service.get_detections(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_success(detection_service):
    result = await detection_service.get_detection_by_bbox_id(image_id=123, bbox_id=0, user_id=456)

    assert result.bbox_id == 0
    assert result.detected_class == 'car'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_not_found(detection_service):
    with pytest.raises(ValueError, match="Detection with bbox_id 999 not found"):
        await detection_service.get_detection_by_bbox_id(image_id=123, bbox_id=999, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_soft_delete_detection_success(detection_service, mock_detection_repo):
    result = await detection_service.soft_delete_detection(image_id=123, bbox_id=0, user_id=456)

    mock_detection_repo.soft_delete.assert_awaited_once()
    assert result.is_active is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_soft_delete_detection_not_found(detection_service):
    with pytest.raises(ValueError, match="Detection with bbox_id 999 not found"):
        await detection_service.soft_delete_detection(image_id=123, bbox_id=999, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_version_detections_success(detection_service, mock_detection_repo):
    count = await detection_service.delete_version_detections(image_id=123, user_id=456)

    mock_detection_repo.delete_by_content.assert_called_once_with(100)
    assert count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_version_detections_unauthorized(detection_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.delete_version_detections(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_success(detection_service):
    stats = await detection_service.get_detection_stats(image_id=123, user_id=456)

    assert stats['total_detections'] == 2
    assert 'car' in stats['classes']
    assert 'person' in stats['classes']
    assert stats['avg_confidence'] == pytest.approx(0.915, rel=0.01)
    assert stats['min_confidence'] == 0.88
    assert stats['max_confidence'] == 0.95


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_no_detections(detection_service, mock_detection_repo):
    mock_detection_repo.get_by_content = AsyncMock(return_value=[])

    stats = await detection_service.get_detection_stats(image_id=123, user_id=456)

    assert stats['total_detections'] == 0
    assert stats['classes'] == []
    assert stats['avg_confidence'] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_unauthorized(detection_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await detection_service.get_detection_stats(image_id=123, user_id=999)