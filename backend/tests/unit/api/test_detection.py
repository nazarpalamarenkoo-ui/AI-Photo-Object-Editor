import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException

from app.api.v1.detection import (
    get_image_detections,
    get_detection_by_bbox,
    get_detection_stats,
    delete_image_detections
)


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.get_image_detections = AsyncMock()
    service.get_detection_by_bbox_id = AsyncMock()
    service.get_detection_stats = AsyncMock()
    service.delete_image_detections = AsyncMock()
    return service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_success(mock_user, mock_service):
    mock_service.get_image_detections.return_value = [{"id": 1}]

    result = await get_image_detections(
        image_id=10,
        use_cache=True,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_image_detections.assert_called_once_with(
        image_id=10,
        user_id=1,
        use_cache=True
    )
    assert len(result) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_not_found(mock_user, mock_service):
    mock_service.get_image_detections.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await get_image_detections(image_id=10, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_unauthorized(mock_user, mock_service):
    mock_service.get_image_detections.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await get_image_detections(image_id=10, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_detections_other_error(mock_user, mock_service):
    mock_service.get_image_detections.side_effect = ValueError("some error")

    with pytest.raises(HTTPException) as exc:
        await get_image_detections(image_id=10, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_success(mock_user, mock_service):
    mock_service.get_detection_by_bbox_id.return_value = {"id": 1}

    result = await get_detection_by_bbox(
        image_id=1,
        bbox_id=2,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_detection_by_bbox_id.assert_called_once_with(
        image_id=1,
        bbox_id=2,
        user_id=1
    )
    assert result["id"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_not_found(mock_user, mock_service):
    mock_service.get_detection_by_bbox_id.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await get_detection_by_bbox(image_id=1, bbox_id=2, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_by_bbox_unauthorized(mock_user, mock_service):
    mock_service.get_detection_by_bbox_id.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await get_detection_by_bbox(image_id=1, bbox_id=2, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_success(mock_user, mock_service):
    mock_service.get_detection_stats.return_value = {"count": 3}

    result = await get_detection_stats(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_detection_stats.assert_called_once_with(image_id=1, user_id=1)
    assert result["count"] == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_not_found(mock_user, mock_service):
    mock_service.get_detection_stats.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await get_detection_stats(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_detection_stats_unauthorized(mock_user, mock_service):
    mock_service.get_detection_stats.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await get_detection_stats(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_success(mock_user, mock_service):
    mock_service.delete_image_detections.return_value = 5

    result = await delete_image_detections(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.delete_image_detections.assert_called_once_with(image_id=1, user_id=1)
    assert result["deleted"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_not_found(mock_user, mock_service):
    mock_service.delete_image_detections.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await delete_image_detections(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_detections_unauthorized(mock_user, mock_service):
    mock_service.delete_image_detections.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await delete_image_detections(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403