import pytest
from app.repository.detection_repo import DetectionRepository
from app.db.models.detection import Detection


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_many_detections(db_session, sample_image):
    repo = DetectionRepository(db_session)
    dets = [
        Detection(image_id=sample_image.id, x1=10, y1=10, x2=100, y2=100, detected_class="person", confidence=0.9),
        Detection(image_id=sample_image.id, x1=200, y1=200, x2=300, y2=300, detected_class="car", confidence=0.8)
    ]
    created = await repo.create_many(dets)
    assert len(created) == 2


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_detections_by_image(db_session, sample_detection, sample_image):
    repo = DetectionRepository(db_session)
    dets = await repo.get_by_image(sample_image.id)
    assert len(dets) >= 1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_by_image(db_session, sample_detection, sample_image):
    repo = DetectionRepository(db_session)
    count = await repo.delete_by_image(sample_image.id)
    assert count >= 1
    dets = await repo.get_by_image(sample_image.id)
    assert len(dets) == 0