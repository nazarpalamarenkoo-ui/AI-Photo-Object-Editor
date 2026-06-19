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
    """Redis mock with full undo/redo/history/current_state support."""
    _image_store = {}
    _undo_stacks = {}
    _redo_stacks = {}

    redis = MagicMock()

    async def cache_image(image_id, image_data, suffix="original", ttl=None):
        key = f"image:{image_id}:{suffix}"
        _image_store[key] = image_data
        return key

    async def get_cache_image(image_id, suffix="original"):
        return _image_store.get(f"image:{image_id}:{suffix}")

    async def cache_detections(image_id, detections, ttl=None):
        _image_store[f"detections:{image_id}"] = detections

    async def get_cached_detections(image_id):
        return _image_store.get(f"detections:{image_id}")

    async def delete(key):
        _image_store.pop(key, None)

    async def push_undo_state(image_id, data, label=""):
        _undo_stacks.setdefault(image_id, []).append({'bytes': data, 'label': label})

    async def pop_undo_state(image_id):
        stack = _undo_stacks.get(image_id, [])
        return stack.pop() if stack else None

    async def push_redo_state(image_id, data, label=""):
        _redo_stacks.setdefault(image_id, []).append({'bytes': data, 'label': label})

    async def pop_redo_state(image_id):
        stack = _redo_stacks.get(image_id, [])
        return stack.pop() if stack else None

    async def get_history_labels(image_id):
        return [e['label'] for e in _undo_stacks.get(image_id, [])]

    async def clear_history(image_id):
        _undo_stacks.pop(image_id, None)
        _redo_stacks.pop(image_id, None)

    redis.cache_image = cache_image
    redis.get_cache_image = get_cache_image
    redis.cache_detections = cache_detections
    redis.get_cached_detections = get_cached_detections
    redis.delete = delete
    redis.push_undo_state = push_undo_state
    redis.pop_undo_state = pop_undo_state
    redis.push_redo_state = push_redo_state
    redis.pop_redo_state = pop_redo_state
    redis.get_history_labels = get_history_labels
    redis.clear_history = clear_history

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
async def test_detect_objects_uses_redis_cache_skips_s3(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """When Redis has current_state, S3 download must be skipped."""
    await mock_redis_ml.cache_image(sample_image.id, b'cached_bytes', suffix='current_state')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)

    mock_s3_storage.download.assert_not_called()
    call_kwargs = mock_pipeline.detect_objects.call_args.kwargs
    assert call_kwargs['image_bytes'] == b'cached_bytes'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_deletes_old_detections_before_saving(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """Old detections for image must be cleared before new ones are saved."""
    await _add_detection(db_session, sample_image.id, bbox_id=5, cls="cat")
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)

    repo = DetectionRepository(db_session)
    saved = await repo.get_by_image(sample_image.id)
    # Only newly detected bbox_ids (0, 1) should remain — old bbox_id=5 removed
    saved_ids = {d.bbox_id for d in saved}
    assert 5 not in saved_ids
    assert {0, 1} == saved_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_caches_detections_in_redis(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)

    cached = await mock_redis_ml.get_cached_detections(sample_image.id)
    assert cached is not None
    assert len(cached) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_forwards_conf_threshold_and_classes(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(
        sample_image.id, sample_user.id,
        conf_threshold=0.8, classes=['person']
    )

    call_kwargs = mock_pipeline.detect_objects.call_args.kwargs
    assert call_kwargs['conf_threshold'] == 0.8
    assert call_kwargs['classes'] == ['person']


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_pushes_to_undo_stack(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    labels = await mock_redis_ml.get_history_labels(sample_image.id)
    assert any('remove' in lbl and '0' in lbl for lbl in labels)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_saves_result_as_current_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state == b'removed_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_clears_detections_from_db(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    repo = DetectionRepository(db_session)
    remaining = await repo.get_by_image(sample_image.id)
    assert remaining == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_scene_bboxes_includes_all_detections(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """scene_bboxes forwarded to pipeline must reflect all detections in DB."""
    await _add_detection(db_session, sample_image.id, bbox_id=0, cls='car')
    await _add_detection(db_session, sample_image.id, bbox_id=1, cls='person')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    assert len(scene) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_passes_ldm_params(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(
        sample_image.id, 0, sample_user.id,
        ldm_steps=50, ldm_sampler='ddim', hd_strategy='RESIZE'
    )

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 50
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_uses_current_state_from_redis(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """Second operation must use Redis current_state, not original S3."""
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await mock_redis_ml.cache_image(sample_image.id, b'edited_state', suffix='current_state')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    mock_s3_storage.download.assert_not_called()
    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    assert call_kwargs['image_bytes'] == b'edited_state'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_detection_not_found(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="bbox_id 77 not found"):
        await service.replace_object(sample_image.id, 77, b'rep', sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.replace_object(sample_image.id, 0, b'rep', sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_pushes_to_undo_stack(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.replace_object(sample_image.id, 0, b'rep', sample_user.id)

    labels = await mock_redis_ml.get_history_labels(sample_image.id)
    assert any('replace' in lbl and '0' in lbl for lbl in labels)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_saves_result_as_current_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.replace_object(sample_image.id, 0, b'rep', sample_user.id)

    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state == b'replaced_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_clears_detections_from_db(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.replace_object(sample_image.id, 0, b'rep', sample_user.id)

    repo = DetectionRepository(db_session)
    remaining = await repo.get_by_image(sample_image.id)
    assert remaining == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_ldm_params(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.replace_object(
        sample_image.id, 0, b'rep', sample_user.id,
        ldm_steps=10, ldm_sampler='ddim', hd_strategy='ORIGINAL'
    )

    call_kwargs = mock_pipeline.replace_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'ORIGINAL'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_replacement_bytes_to_pipeline(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.replace_object(sample_image.id, 0, b'my_replacement', sample_user.id)

    call_kwargs = mock_pipeline.replace_object.call_args.kwargs
    assert call_kwargs['replacement_image_bytes'] == b'my_replacement'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_scene_bboxes_excludes_selected(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """scene_bboxes must contain only detections NOT being removed."""
    await _add_detection(db_session, sample_image.id, bbox_id=0, cls='car')
    await _add_detection(db_session, sample_image.id, bbox_id=1, cls='person')
    await _add_detection(db_session, sample_image.id, bbox_id=2, cls='dog')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(sample_image.id, [0, 1], sample_user.id)

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    # Only bbox_id=2 remains in scene
    assert len(scene) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_scene_bboxes_none_when_all_selected(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(sample_image.id, [0, 1], sample_user.id)

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert not call_kwargs['scene_bboxes']


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_deletes_only_selected_from_db(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """Only selected detections should be deleted; others must remain."""
    await _add_detection(db_session, sample_image.id, bbox_id=0, cls='car')
    await _add_detection(db_session, sample_image.id, bbox_id=1, cls='person')
    await _add_detection(db_session, sample_image.id, bbox_id=2, cls='dog')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(sample_image.id, [0, 1], sample_user.id)

    repo = DetectionRepository(db_session)
    remaining = await repo.get_by_image(sample_image.id)
    remaining_ids = {d.bbox_id for d in remaining}
    assert remaining_ids == {2}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_pushes_undo_with_count_label(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(sample_image.id, [0, 1], sample_user.id)

    labels = await mock_redis_ml.get_history_labels(sample_image.id)
    assert any('2' in lbl for lbl in labels)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_saves_result_as_current_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(sample_image.id, [0], sample_user.id)

    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state == b'multi_removed_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_ldm_params(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_multiple_objects(
        sample_image.id, [0], sample_user.id,
        ldm_steps=30, ldm_sampler='plms', hd_strategy='CROP'
    )

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 30
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.remove_multiple_objects(sample_image.id, [0], sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_after_remove_restores_previous_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    # Set original bytes as S3 response
    original_bytes = b'original_image_bytes'
    mock_s3_storage.download = AsyncMock(return_value=original_bytes)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    # current_state is now 'removed_image'; undo should restore original
    result = await service.undo(sample_image.id, sample_user.id)

    assert 'presigned_url' in result
    assert 'label' in result
    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state == original_bytes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_raises_when_stack_empty(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Nothing to undo"):
        await service.undo(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_pushes_current_to_redo_stack(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)
    # current_state = b'removed_image'
    await service.undo(sample_image.id, sample_user.id)

    # redo stack should now have 'removed_image'
    redo_entry = await mock_redis_ml.pop_redo_state(sample_image.id)
    assert redo_entry is not None
    assert redo_entry['bytes'] == b'removed_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_after_undo_restores_removed_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    mock_s3_storage.download = AsyncMock(return_value=b'original_bytes')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)
    await service.undo(sample_image.id, sample_user.id)

    result = await service.redo(sample_image.id, sample_user.id)

    assert 'presigned_url' in result
    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state == b'removed_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_raises_when_stack_empty(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Nothing to redo"):
        await service.redo(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.undo(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.redo(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_reflects_operations(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    await _add_detection(db_session, sample_image.id, bbox_id=1)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)

    result = await service.get_history(sample_image.id, sample_user.id)

    assert 'history' in result
    assert len(result['history']) == 1
    assert 'remove' in result['history'][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_empty_before_any_operation(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    result = await service.get_history(sample_image.id, sample_user.id)

    assert result == {'history': []}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.get_history(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_success(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    # Run removal so current_state exists
    await service.remove_object(sample_image.id, 0, sample_user.id)
    result = await service.save_result(sample_image.id, sample_user.id)

    assert result is not None
    mock_s3_storage.upload_bytes.assert_called()
    # upload_bytes called twice: once for remove result, once for save_result
    assert mock_s3_storage.upload_bytes.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_raises_when_no_current_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="No processed result to save"):
        await service.save_result(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_unauthorized(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    with pytest.raises(ValueError, match="Unauthorized"):
        await service.save_result(sample_image.id, sample_user.id + 999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_uploads_to_saved_path(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)
    await service.remove_object(sample_image.id, 0, sample_user.id)

    await service.save_result(sample_image.id, sample_user.id)

    # Last upload_bytes call should use 'saved/' path
    last_call = mock_s3_storage.upload_bytes.call_args_list[-1]
    path = last_call.kwargs.get('path', last_call.args[1] if len(last_call.args) > 1 else '')
    assert path.startswith('saved/')
    assert str(sample_user.id) in path
    assert str(sample_image.id) in path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_clears_redis_state(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await mock_redis_ml.cache_image(sample_image.id, b'some_state', suffix='current_state')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.reset_current_state(sample_image.id)

    state = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_clears_undo_history(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    await _add_detection(db_session, sample_image.id, bbox_id=0)
    mock_s3_storage.download = AsyncMock(return_value=b'orig')
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.remove_object(sample_image.id, 0, sample_user.id)
    assert len(await mock_redis_ml.get_history_labels(sample_image.id)) == 1

    await service.reset_current_state(sample_image.id)

    labels = await mock_redis_ml.get_history_labels(sample_image.id)
    assert labels == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_detect_remove_undo_redo(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    original_bytes = b'original_image'
    mock_s3_storage.download = AsyncMock(return_value=original_bytes)
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)
    await service.remove_object(sample_image.id, 0, sample_user.id)

    state_after_remove = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state_after_remove == b'removed_image'

    await service.undo(sample_image.id, sample_user.id)
    state_after_undo = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state_after_undo == original_bytes

    await service.redo(sample_image.id, sample_user.id)
    state_after_redo = await mock_redis_ml.get_cache_image(sample_image.id, suffix='current_state')
    assert state_after_redo == b'removed_image'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_detect_replace_save(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)
    await service.replace_object(sample_image.id, 0, b'replacement', sample_user.id)
    result = await service.save_result(sample_image.id, sample_user.id)

    assert result is not None
    assert result.status == 'processed'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sequential_removes_build_undo_stack(
    db_session, mock_s3_storage, mock_redis_ml, mock_pipeline, sample_image, sample_user
):
    """Two removals should produce two entries in undo stack."""
    mock_pipeline.detect_objects = AsyncMock(return_value={
        "detections": [
            {"bbox_id": 0, "detected_class": "car", "confidence": 0.9,
             "x1": 10, "y1": 10, "x2": 50, "y2": 50},
            {"bbox_id": 1, "detected_class": "dog", "confidence": 0.8,
             "x1": 60, "y1": 60, "x2": 100, "y2": 100},
        ],
        "image_size": (640, 480),
        "metrics": {}, "timestamp": "2024-01-01T00:00:00",
    })
    service = _make_service(db_session, mock_s3_storage, mock_redis_ml, mock_pipeline)

    await service.detect_objects(sample_image.id, sample_user.id)
    await service.remove_object(sample_image.id, 0, sample_user.id)

    await _add_detection(db_session, sample_image.id, bbox_id=1, cls='dog')
    await service.remove_object(sample_image.id, 1, sample_user.id)

    labels = await mock_redis_ml.get_history_labels(sample_image.id)
    assert len(labels) == 2