import pytest
from app.repository.detection_repo import DetectionRepository
from app.db.models.detection import Detection


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_many_detections(db_session, sample_image):
    """Test creating multiple detections - FIXED with bbox_id"""
    repo = DetectionRepository(db_session)
    
    dets = [
        Detection(
            image_id=sample_image.id,
            bbox_id=0, 
            x1=10,
            y1=10,
            x2=100,
            y2=100,
            detected_class="person",
            confidence=0.9
        ),
        Detection(
            image_id=sample_image.id,
            bbox_id=1,
            x1=200,
            y1=200,
            x2=300,
            y2=300,
            detected_class="car",
            confidence=0.8
        )
    ]
    
    created = await repo.create_many(dets)
    assert len(created) == 2
    
    # Verify bbox_ids
    assert created[0].bbox_id == 0
    assert created[1].bbox_id == 1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_detections_by_image(db_session, sample_detection, sample_image):
    """Test getting all detections for image"""
    repo = DetectionRepository(db_session)
    
    dets = await repo.get_by_image(sample_image.id)
    assert len(dets) >= 1
    
    # Verify bbox_id exists
    assert all(hasattr(d, 'bbox_id') for d in dets)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_by_image(db_session, sample_detection, sample_image):
    """Test deleting all detections for image"""
    repo = DetectionRepository(db_session)
    
    count = await repo.delete_by_image(sample_image.id)
    assert count >= 1
    
    # Verify deleted
    dets = await repo.get_by_image(sample_image.id)
    assert len(dets) == 0


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id(db_session, sample_image):
    """Test finding detection by bbox_id"""
    repo = DetectionRepository(db_session)
    
    # Create detections with different bbox_ids
    det1 = Detection(
        image_id=sample_image.id,
        bbox_id=0,
        x1=10, y1=10, x2=50, y2=50,
        detected_class="person",
        confidence=0.95
    )
    det2 = Detection(
        image_id=sample_image.id,
        bbox_id=1,
        x1=100, y1=100, x2=150, y2=150,
        detected_class="car",
        confidence=0.88
    )
    
    await repo.create_many([det1, det2])
    
    # Get all detections
    all_dets = await repo.get_by_image(sample_image.id)
    
    # Find by bbox_id
    found = next((d for d in all_dets if d.bbox_id == 1), None)
    assert found is not None
    assert found.detected_class == "car"
    assert found.confidence == 0.88