import pytest
from unittest.mock import AsyncMock, MagicMock
from app.repository.detection_repo import DetectionRepository
from app.db.models.detection import Detection


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_many():
    db = AsyncMock()
    repo = DetectionRepository(db)

    detections = [Detection(), Detection()]

    db.add_all = MagicMock()
    db.commit = AsyncMock()

    result = await repo.create_many(detections)

    db.add_all.assert_called_once_with(detections)
    assert result == detections


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_image():
    db = AsyncMock()
    repo = DetectionRepository(db)

    detections = [Detection(), Detection()]

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = detections
    db.execute = AsyncMock(return_value=result_mock)

    result = await repo.get_by_image(1)

    assert result == detections


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_by_image():
    db = AsyncMock()
    repo = DetectionRepository(db)

    detections = [Detection(), Detection(), Detection()]

    repo.get_by_image = AsyncMock(return_value=detections)
    db.delete = AsyncMock()
    db.commit = AsyncMock()

    count = await repo.delete_by_image(1)

    assert count == 3
    assert db.delete.call_count == 3