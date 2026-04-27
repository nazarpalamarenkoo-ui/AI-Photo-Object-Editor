import pytest
from app.db.models.detection import Detection
from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.services.detection_service import DetectionService


def _make_service(db_session, mock_redis_cache) -> DetectionService:
    return DetectionService(
        db=db_session,
        redis_cache=mock_redis_cache,
        detection_repo=DetectionRepository(db_session),
        image_repo=ImageRepository(db_session),
    )


async def _add_detection(db_session, image_id: int, bbox_id: int, cls: str = "person", conf: float = 0.9):
    repo = DetectionRepository(db_session)
    detections = [Detection(
        image_id=image_id,
        bbox_id=bbox_id,
        detected_class=cls,
        confidence=conf,
        x1=10, y1=10, x2=100, y2=200,
    )]
    created = await repo.create_many(detections)
    return created[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_from_db(db_session, mock_redis_cache, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_redis_cache)

    detections = await service.get_image_detections(sample_image.id, sample_user.id, use_cache=False)

    assert len(detections) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_from_cache(db_session, mock_redis_cache, sample_image, sample_user):
    fake = [{"bbox_id": 0, "detected_class": "car"}]
    await mock_redis_cache.cache_detections(sample_image.id, fake)
    service = _make_service(db_session, mock_redis_cache)

    detections = await service.get_image_detections(sample_image.id, sample_user.id, use_cache=True)

    assert detections == fake


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_image_not_found(db_session, mock_redis_cache):
    service = _make_service(db_session, mock_redis_cache)
    with pytest.raises(ValueError, match="not found"):
        await service.get_image_detections(99999, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_unauthorized(db_session, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.get_image_detections(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_success(db_session, mock_redis_cache, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=3)
    service = _make_service(db_session, mock_redis_cache)

    det = await service.get_detection_by_bbox_id(sample_image.id, 3, sample_user.id)

    assert det.bbox_id == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_by_bbox_id_not_found(db_session, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_redis_cache)
    with pytest.raises(ValueError, match="not found"):
        await service.get_detection_by_bbox_id(sample_image.id, 99, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_detections(db_session, mock_redis_cache, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_redis_cache)

    count = await service.delete_image_detections(sample_image.id, sample_user.id)

    assert count == 2
    remaining = await service.get_image_detections(sample_image.id, sample_user.id, use_cache=False)
    assert len(remaining) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_detections_unauthorized(db_session, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_redis_cache)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.delete_image_detections(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_stats(db_session, mock_redis_cache, sample_image, sample_user):
    repo = DetectionRepository(db_session)
    await repo.create_many([
        Detection(image_id=sample_image.id, bbox_id=0, detected_class="person",
                  confidence=0.9, x1=0, y1=0, x2=10, y2=10),
        Detection(image_id=sample_image.id, bbox_id=1, detected_class="car",
                  confidence=0.7, x1=0, y1=0, x2=10, y2=10),
    ])
    service = _make_service(db_session, mock_redis_cache)

    stats = await service.get_detection_stats(sample_image.id, sample_user.id)

    assert stats["total_detections"] == 2
    assert stats["avg_confidence"] == pytest.approx(0.8)
    assert stats["min_confidence"] == pytest.approx(0.7)
    assert stats["max_confidence"] == pytest.approx(0.9)
    assert set(stats["classes"]) == {"person", "car"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_stats_empty(db_session, mock_redis_cache, sample_image, sample_user):
    service = _make_service(db_session, mock_redis_cache)
    stats = await service.get_detection_stats(sample_image.id, sample_user.id)
    assert stats["total_detections"] == 0
    assert stats["avg_confidence"] == 0.0