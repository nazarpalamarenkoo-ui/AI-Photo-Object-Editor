import pytest
from app.repository.detection_repo import DetectionRepository
from app.db.models.detection import Detection


def _make_detection(content_id, bbox_id, x1, y1, x2, y2, detected_class, confidence):
    return Detection(
        content_id=content_id,
        bbox_id=bbox_id,
        x1=x1, y1=y1, x2=x2, y2=y2,
        detected_class=detected_class,
        confidence=confidence,
        model_name="yolo",
        model_version="v8",
        inference_time_ms=15.0,
    )


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_many_detections(db_session, sample_image_content):
    """Test bulk-inserting multiple Detection records"""
    repo = DetectionRepository(db_session)

    dets = [
        _make_detection(sample_image_content.id, 0, 10, 10, 100, 100, "person", 0.9),
        _make_detection(sample_image_content.id, 1, 200, 200, 300, 300, "car", 0.8),
    ]

    created = await repo.create_many(dets)

    assert len(created) == 2
    assert created[0].bbox_id == 0
    assert created[1].bbox_id == 1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_content_active_only(db_session, sample_image_content):
    """get_by_content with active_only=True must exclude soft-deleted detections"""
    repo = DetectionRepository(db_session)

    dets = [
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.9),
        _make_detection(sample_image_content.id, 1, 100, 100, 150, 150, "dog", 0.75),
    ]
    created = await repo.create_many(dets)

    # Soft-delete the first one
    await repo.soft_delete(created[0].id)

    active = await repo.get_by_content(sample_image_content.id, active_only=True)

    assert all(d.is_active for d in active)
    assert not any(d.id == created[0].id for d in active)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_content_all(db_session, sample_image_content):
    """get_by_content with active_only=False must include soft-deleted detections"""
    repo = DetectionRepository(db_session)

    created = await repo.create_many([
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.9),
    ])
    await repo.soft_delete(created[0].id)

    all_dets = await repo.get_by_content(sample_image_content.id, active_only=False)

    assert any(d.id == created[0].id for d in all_dets)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id(db_session, sample_image_content):
    """Test fetching a Detection by primary key"""
    repo = DetectionRepository(db_session)

    created = await repo.create_many([
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.95),
    ])

    fetched = await repo.get_by_id(created[0].id)

    assert fetched is not None
    assert fetched.id == created[0].id
    assert fetched.detected_class == "person"
    assert fetched.confidence == 0.95


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id_not_found(db_session):
    """Test fetching non-existent detection returns None"""
    repo = DetectionRepository(db_session)

    detection = await repo.get_by_id(999999)

    assert detection is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id(db_session, sample_image_content):
    """Test finding a detection among results by bbox_id"""
    repo = DetectionRepository(db_session)

    det1 = _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.95)
    det2 = _make_detection(sample_image_content.id, 1, 100, 100, 150, 150, "car", 0.88)

    await repo.create_many([det1, det2])

    all_dets = await repo.get_by_content(sample_image_content.id)

    found = next((d for d in all_dets if d.bbox_id == 1), None)
    assert found is not None
    assert found.detected_class == "car"
    assert found.confidence == 0.88


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_max_bbox_id(db_session, sample_image_content):
    """max_bbox_id must return the highest bbox_id across all detections (including inactive)"""
    repo = DetectionRepository(db_session)

    created = await repo.create_many([
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "a", 0.5),
        _make_detection(sample_image_content.id, 3, 20, 20, 60, 60, "b", 0.6),
    ])
    # Soft-delete the one with bbox_id=3 — it must still count toward max
    await repo.soft_delete(created[1].id)

    max_id = await repo.max_bbox_id(sample_image_content.id)

    assert max_id == 3


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_max_bbox_id_empty_returns_minus_one(db_session, sample_image_content):
    """max_bbox_id must return -1 when there are no detections"""
    repo = DetectionRepository(db_session)

    max_id = await repo.max_bbox_id(sample_image_content.id)

    assert max_id == -1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_soft_delete(db_session, sample_image_content):
    """soft_delete must set is_active=False without removing the row"""
    repo = DetectionRepository(db_session)

    created = await repo.create_many([
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.9),
    ])
    detection_id = created[0].id

    deleted = await repo.soft_delete(detection_id)

    assert deleted is not None
    assert deleted.is_active is False

    # Row still exists
    still_there = await repo.get_by_id(detection_id)
    assert still_there is not None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_soft_delete_not_found(db_session):
    """soft_delete on a non-existent id must not raise and returns None"""
    repo = DetectionRepository(db_session)

    result = await repo.soft_delete(999999)

    assert result is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_by_content(db_session, sample_image_content):
    """delete_by_content must hard-delete all detections and return the count"""
    repo = DetectionRepository(db_session)

    await repo.create_many([
        _make_detection(sample_image_content.id, 0, 10, 10, 50, 50, "person", 0.9),
        _make_detection(sample_image_content.id, 1, 100, 100, 150, 150, "car", 0.8),
    ])

    count = await repo.delete_by_content(sample_image_content.id)

    assert count == 2
    remaining = await repo.get_by_content(sample_image_content.id, active_only=False)
    assert len(remaining) == 0


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_by_content_empty_returns_zero(db_session, sample_image_content):
    """delete_by_content on content with no detections must return 0"""
    repo = DetectionRepository(db_session)

    count = await repo.delete_by_content(sample_image_content.id)

    assert count == 0