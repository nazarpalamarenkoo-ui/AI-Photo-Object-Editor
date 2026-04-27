import pytest
from unittest.mock import AsyncMock, MagicMock
from app.db.models.detection import Detection
from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.services.ml_service import MLService


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.detect_objects = AsyncMock(return_value={
        "detections": [
            {"bbox_id": 0, "detected_class": "person", "confidence": 0.95,
             "x1": 10, "y1": 10, "x2": 100, "y2": 200},
            {"bbox_id": 1, "detected_class": "dog", "confidence": 0.80,
             "x1": 200, "y1": 50, "x2": 300, "y2": 150},
        ],
        "image_size": (640, 480),
        "metrics": {"inference_time_ms": 120},
        "timestamp": "2024-01-01T00:00:00",
    })
    pipeline.remove_object = AsyncMock(return_value={
        "result_bytes": b"removed_image",
        "metrics": {"processing_time_ms": 800},
        "timestamp": "2024-01-01T00:00:01",
    })
    pipeline.replace_object = AsyncMock(return_value={
        "result_bytes": b"replaced_image",
        "metrics": {"processing_time_ms": 1200},
        "timestamp": "2024-01-01T00:00:02",
    })
    pipeline.remove_multiple_objects = AsyncMock(return_value={
        "result_bytes": b"multi_removed_image",
        "metrics": {"processing_time_ms": 1500},
        "timestamp": "2024-01-01T00:00:03",
    })
    pipeline.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])
    return pipeline


@pytest.fixture
def mock_redis_ml():
    """Redis mock з підтримкою get_cache_image для MLService."""
    _store = {}

    redis = MagicMock()

    async def cache_image(image_id, image_data, suffix="original", ttl=None):
        key = f"image:{image_id}:{suffix}"
        _store[key] = image_data
        return key

    async def get_cache_image(image_id, suffix="original"):
        return _store.get(f"image:{image_id}:{suffix}")

    async def cache_detections(image_id, detections, ttl=None):
        _store[f"detections:{image_id}"] = detections

    async def get_cached_detections(image_id):
        return _store.get(f"detections:{image_id}")

    async def invalidate_image(image_id):
        for k in list(_store.keys()):
            if str(image_id) in k:
                _store.pop(k)

    async def delete(key):
        _store.pop(key, None)

    redis.cache_image = cache_image
    redis.get_cache_image = get_cache_image
    redis.cache_detections = cache_detections
    redis.get_cached_detections = get_cached_detections
    redis.invalidate_image = invalidate_image
    redis.delete = delete
    return redis


def _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline) -> MLService:
    return MLService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_ml,
        image_repo=ImageRepository(db_session),
        detection_repo=DetectionRepository(db_session),
        pipeline=mock_pipeline,
    )


async def _add_detection(db_session, image_id: int, bbox_id: int, cls: str = "person") -> Detection:
    repo = DetectionRepository(db_session)
    created = await repo.create_many([Detection(
        image_id=image_id, bbox_id=bbox_id, detected_class=cls,
        confidence=0.9, x1=10, y1=10, x2=100, y2=200,
    )])
    return created[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_success(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.detect_objects(sample_image.id, sample_user.id)

    assert "detections" in result
    assert len(result["detections"]) == 2
    mock_s3_storage.download.assert_called_once()
    mock_pipeline.detect_objects.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_saves_to_db(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)

    repo = DetectionRepository(db_session)
    saved = await repo.get_by_image(sample_image.id)
    assert len(saved) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_image_not_found(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    with pytest.raises(ValueError, match="not found"):
        await service.detect_objects(99999, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_unauthorized(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.detect_objects(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_success(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.remove_object(sample_image.id, 0, sample_user.id)

    assert "result_url" in result
    assert "presigned_url" in result
    mock_pipeline.remove_object.assert_called_once()
    mock_s3_storage.upload_bytes.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_detection_not_found(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    with pytest.raises(ValueError, match="bbox_id 99 not found"):
        await service.remove_object(sample_image.id, 99, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_unauthorized(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    with pytest.raises(ValueError, match="Unauthorized"):
        await service.remove_object(sample_image.id, 0, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_success(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.replace_object(sample_image.id, 0, b"replacement_bytes", sample_user.id)

    assert "result_url" in result
    assert "presigned_url" in result
    mock_pipeline.replace_object.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_success(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.remove_multiple_objects(sample_image.id, [0, 1], sample_user.id)

    assert "result_url" in result
    mock_pipeline.remove_multiple_objects.assert_called_once()
    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert len(call_kwargs["selected_bboxes"]) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_no_valid_detections(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    with pytest.raises(ValueError, match="No valid detections"):
        await service.remove_multiple_objects(sample_image.id, [99, 100], sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_partial(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.remove_multiple_objects(sample_image.id, [0, 99], sample_user.id)

    assert "result_url" in result
    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert len(call_kwargs["selected_bboxes"]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_supported_classes(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    classes = service.get_supported_classes()
    assert "person" in classes
    assert len(classes) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_detect_then_remove(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    detect_result = await service.detect_objects(sample_image.id, sample_user.id)
    assert len(detect_result["detections"]) == 2

    remove_result = await service.remove_object(sample_image.id, 0, sample_user.id)
    assert "result_url" in remove_result

    mock_pipeline.detect_objects.assert_called_once()
    mock_pipeline.remove_object.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_fallback_to_s3(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    mock_s3_storage.download.assert_called_once_with(sample_image.storage_path)