import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch
from datetime import datetime

from app.services.ml_service import MLService
from app.db.models.detection import Detection
from app.db.models.image import Image


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.fixture
def mock_s3():
    s3 = MagicMock()
    s3.download = AsyncMock(return_value=b'fake_image_bytes')
    s3.upload_bytes = AsyncMock(return_value='s3://bucket/result.jpg')
    s3.get_presigned_url = AsyncMock(return_value='https://presigned.url/result.jpg')
    return s3


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.cache_detections = AsyncMock()
    redis.get_cached_detections = AsyncMock(return_value=None)
    redis.get_cache_image = AsyncMock(return_value=None)
    redis.cache_image = AsyncMock(return_value='image:123:current_state')
    redis.delete = AsyncMock()
    redis.exists = AsyncMock(return_value=False)
    redis.push_undo_state = AsyncMock()
    redis.push_redo_state = AsyncMock()
    redis.pop_undo_state = AsyncMock(return_value=None)
    redis.pop_redo_state = AsyncMock(return_value=None)
    redis.get_history_labels = AsyncMock(return_value=[])
    redis.clear_history = AsyncMock()
    return redis


@pytest.fixture
def mock_image_repo():
    repo = MagicMock()
    mock_image = MagicMock(spec=Image)
    mock_image.id = 123
    mock_image.user_id = 456
    mock_image.storage_path = 's3://bucket/image.jpg'
    mock_image.filename = 'test.jpg'
    repo.get_by_id = AsyncMock(return_value=mock_image)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def mock_detection_repo():
    repo = MagicMock()
    mock_detections = [
        MagicMock(bbox_id=0, detected_class='car', confidence=0.95,
                  x1=100, y1=100, x2=200, y2=200),
        MagicMock(bbox_id=1, detected_class='person', confidence=0.88,
                  x1=300, y1=150, x2=400, y2=350),
    ]
    repo.get_by_image = AsyncMock(return_value=mock_detections)
    repo.create_many = AsyncMock(return_value=mock_detections)
    repo.delete_by_image = AsyncMock()
    return repo


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.detect_objects = AsyncMock(return_value={
        'detections': [
            {'bbox_id': 0, 'detected_class': 'car', 'confidence': 0.95,
             'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        ],
        'image_size': (640, 480),
        'metrics': {'inference_time_ms': 500},
        'timestamp': '2024-01-01T00:00:00'
    })
    pipeline.remove_object = AsyncMock(return_value={
        'result_bytes': b'fake_result_image',
        'metrics': {'processing_time': 1.5},
        'timestamp': '2024-01-01T00:00:00'
    })
    pipeline.replace_object = AsyncMock(return_value={
        'result_bytes': b'fake_replaced_image',
        'metrics': {'processing_time': 2.0},
        'timestamp': '2024-01-01T00:00:00'
    })
    pipeline.remove_multiple_objects = AsyncMock(return_value={
        'result_bytes': b'fake_multi_result',
        'metrics': {'processing_time': 3.0},
        'timestamp': '2024-01-01T00:00:00'
    })
    pipeline.get_supported_classes = MagicMock(return_value=['car', 'person', 'dog'])
    return pipeline


@pytest.fixture
def ml_service(mock_db, mock_s3, mock_redis, mock_image_repo, mock_detection_repo, mock_pipeline):
    return MLService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
        device='cpu'
    )

@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_uses_redis_cache_when_available(ml_service, mock_redis, mock_s3, mock_pipeline):
    """When Redis has current_state, S3 download must be skipped."""
    mock_redis.get_cache_image = AsyncMock(return_value=b'cached_image_bytes')

    await ml_service.detect_objects(image_id=123, user_id=456)

    mock_s3.download.assert_not_called()
    call_kwargs = mock_pipeline.detect_objects.call_args.kwargs
    assert call_kwargs['image_bytes'] == b'cached_image_bytes'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_falls_back_to_s3_when_cache_miss(ml_service, mock_redis, mock_s3):
    """When Redis returns None, S3 download must be called."""
    mock_redis.get_cache_image = AsyncMock(return_value=None)

    await ml_service.detect_objects(image_id=123, user_id=456)

    mock_s3.download.assert_called_once_with('s3://bucket/image.jpg')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_deletes_old_detections_before_saving(ml_service, mock_detection_repo):
    """delete_by_image must be called before create_many to avoid duplicates."""
    call_order = []
    mock_detection_repo.delete_by_image = AsyncMock(side_effect=lambda *a, **kw: call_order.append('delete'))
    mock_detection_repo.create_many = AsyncMock(side_effect=lambda *a, **kw: call_order.append('create') or [])

    await ml_service.detect_objects(image_id=123, user_id=456)

    assert call_order.index('delete') < call_order.index('create')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_conf_threshold_forwarded(ml_service, mock_pipeline):
    await ml_service.detect_objects(image_id=123, user_id=456, conf_threshold=0.75)

    call_kwargs = mock_pipeline.detect_objects.call_args.kwargs
    assert call_kwargs['conf_threshold'] == 0.75


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_pushes_undo_state_before_pipeline(ml_service, mock_redis, mock_pipeline):
    """undo state must be pushed before pipeline is called."""
    call_order = []
    mock_redis.push_undo_state = AsyncMock(side_effect=lambda *a, **kw: call_order.append('undo'))
    mock_pipeline.remove_object = AsyncMock(
        side_effect=lambda *a, **kw: call_order.append('pipeline') or {
            'result_bytes': b'r', 'metrics': {}, 'timestamp': ''
        }
    )

    await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    assert call_order.index('undo') < call_order.index('pipeline')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_saves_current_state_to_redis(ml_service, mock_redis):
    """Result bytes must be cached in Redis as current_state after removal."""
    await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    mock_redis.cache_image.assert_called_once()
    call_kwargs = mock_redis.cache_image.call_args.kwargs
    assert call_kwargs['image_id'] == 123
    assert call_kwargs['suffix'] == 'current_state'
    assert call_kwargs['image_data'] == b'fake_result_image'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_deletes_detections_after_removal(ml_service, mock_detection_repo, mock_redis):
    """Detections must be invalidated in DB and Redis after removal."""
    await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    mock_detection_repo.delete_by_image.assert_called_once_with(123)
    mock_redis.delete.assert_called_with('image:123:detections')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_builds_scene_bboxes_from_all_detections(ml_service, mock_pipeline):
    """scene_bboxes must include ALL detections (pipeline decides what to exclude)."""
    await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    assert isinstance(scene, list)
    assert len(scene) == 2
    assert {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200} in scene
    assert {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 350} in scene


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_passes_ldm_params(ml_service, mock_pipeline):
    await ml_service.remove_object(
        image_id=123, bbox_id=0, user_id=456,
        ldm_steps=50, ldm_sampler='ddim', hd_strategy='RESIZE'
    )

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 50
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_selected_bbox_matches_detection(ml_service, mock_pipeline):
    """selected_bbox passed to pipeline must match stored detection coordinates."""
    await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    assert call_kwargs['selected_bbox'] == {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_pushes_undo_state_before_pipeline(ml_service, mock_redis, mock_pipeline):
    call_order = []
    mock_redis.push_undo_state = AsyncMock(side_effect=lambda *a, **kw: call_order.append('undo'))
    mock_pipeline.replace_object = AsyncMock(
        side_effect=lambda *a, **kw: call_order.append('pipeline') or {
            'result_bytes': b'r', 'metrics': {}, 'timestamp': ''
        }
    )

    await ml_service.replace_object(
        image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=456
    )

    assert call_order.index('undo') < call_order.index('pipeline')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_saves_current_state_to_redis(ml_service, mock_redis):
    await ml_service.replace_object(
        image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=456
    )

    mock_redis.cache_image.assert_called_once()
    call_kwargs = mock_redis.cache_image.call_args.kwargs
    assert call_kwargs['image_data'] == b'fake_replaced_image'
    assert call_kwargs['suffix'] == 'current_state'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_deletes_detections_after_replacement(ml_service, mock_detection_repo, mock_redis):
    await ml_service.replace_object(
        image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=456
    )

    mock_detection_repo.delete_by_image.assert_called_once_with(123)
    mock_redis.delete.assert_called_with('image:123:detections')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_builds_scene_bboxes(ml_service, mock_pipeline):
    await ml_service.replace_object(
        image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=456
    )

    call_kwargs = mock_pipeline.replace_object.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    assert len(scene) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_passes_ldm_params(ml_service, mock_pipeline):
    await ml_service.replace_object(
        image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=456,
        ldm_steps=10, ldm_sampler='ddim', hd_strategy='ORIGINAL'
    )

    call_kwargs = mock_pipeline.replace_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'ORIGINAL'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_detection_not_found(ml_service):
    with pytest.raises(ValueError, match="Detection with bbox_id 999 not found"):
        await ml_service.replace_object(
            image_id=123, bbox_id=999, replace_image_bytes=b'rep', user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.replace_object(
            image_id=123, bbox_id=0, replace_image_bytes=b'rep', user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_scene_bboxes_excludes_selected(ml_service, mock_pipeline):
    """scene_bboxes must NOT include the bboxes being removed."""
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0], user_id=456
    )

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    # bbox_id=0 removed; scene_bboxes should only have bbox_id=1
    assert {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 350} in scene
    assert {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200} not in scene


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_scene_bboxes_none_when_all_removed(ml_service, mock_pipeline):
    """When all detections are selected, scene_bboxes should be None (or empty)."""
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0, 1], user_id=456
    )

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    scene = call_kwargs['scene_bboxes']
    assert not scene  # None or []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_pushes_undo_state(ml_service, mock_redis):
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0, 1], user_id=456
    )

    mock_redis.push_undo_state.assert_called_once()
    call_args = mock_redis.push_undo_state.call_args
    assert call_args.args[0] == 123 or call_args.kwargs.get('image_id') == 123


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_ldm_params(ml_service, mock_pipeline):
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0], user_id=456,
        ldm_steps=30, ldm_sampler='plms', hd_strategy='CROP'
    )

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 30
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_deletes_from_db_and_redis(ml_service, mock_db, mock_redis, mock_detection_repo):
    """Selected detections must be deleted from DB; Redis detection cache cleared."""
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0, 1], user_id=456
    )

    assert mock_db.delete.call_count == 2
    mock_db.commit.assert_called_once()
    mock_redis.delete.assert_called_with('image:123:detections')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_saves_current_state(ml_service, mock_redis):
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0], user_id=456
    )

    mock_redis.cache_image.assert_called_once()
    call_kwargs = mock_redis.cache_image.call_args.kwargs
    assert call_kwargs['suffix'] == 'current_state'
    assert call_kwargs['image_data'] == b'fake_multi_result'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.remove_multiple_objects(
            image_id=123, bbox_ids=[0], user_id=999
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_returns_previous_state(ml_service, mock_redis, mock_s3):
    mock_redis.pop_undo_state = AsyncMock(return_value={
        'bytes': b'prev_state_bytes',
        'label': 'remove bbox_id=0'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current_bytes')

    result = await ml_service.undo(image_id=123, user_id=456)

    assert 'presigned_url' in result
    assert result['label'] == 'remove bbox_id=0'
    assert 'history' in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_pushes_current_to_redo_stack(ml_service, mock_redis):
    mock_redis.pop_undo_state = AsyncMock(return_value={
        'bytes': b'prev_bytes', 'label': 'op'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current_bytes')

    await ml_service.undo(image_id=123, user_id=456)

    mock_redis.push_redo_state.assert_called_once()
    call_args = mock_redis.push_redo_state.call_args
    assert b'current_bytes' in call_args.args or call_args.kwargs.get('data') == b'current_bytes'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_raises_when_nothing_to_undo(ml_service, mock_redis):
    mock_redis.pop_undo_state = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Nothing to undo"):
        await ml_service.undo(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_saves_prev_state_as_current(ml_service, mock_redis):
    mock_redis.pop_undo_state = AsyncMock(return_value={
        'bytes': b'prev_bytes', 'label': 'remove'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current')

    await ml_service.undo(image_id=123, user_id=456)

    mock_redis.cache_image.assert_called_once()
    call_kwargs = mock_redis.cache_image.call_args.kwargs
    assert call_kwargs['image_data'] == b'prev_bytes'
    assert call_kwargs['suffix'] == 'current_state'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.undo(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_no_redo_push_when_no_current_state(ml_service, mock_redis):
    """When there is no current state in Redis, redo stack must not be touched."""
    mock_redis.pop_undo_state = AsyncMock(return_value={
        'bytes': b'prev', 'label': 'op'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=None)

    await ml_service.undo(image_id=123, user_id=456)

    mock_redis.push_redo_state.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_returns_next_state(ml_service, mock_redis):
    mock_redis.pop_redo_state = AsyncMock(return_value={
        'bytes': b'next_bytes', 'label': 'redo'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current_bytes')

    result = await ml_service.redo(image_id=123, user_id=456)

    assert 'presigned_url' in result
    assert result['label'] == 'redo'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_pushes_current_to_undo_stack(ml_service, mock_redis):
    mock_redis.pop_redo_state = AsyncMock(return_value={
        'bytes': b'next', 'label': 'redo'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current')

    await ml_service.redo(image_id=123, user_id=456)

    mock_redis.push_undo_state.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_raises_when_nothing_to_redo(ml_service, mock_redis):
    mock_redis.pop_redo_state = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Nothing to redo"):
        await ml_service.redo(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_saves_next_state_as_current(ml_service, mock_redis):
    mock_redis.pop_redo_state = AsyncMock(return_value={
        'bytes': b'next_bytes', 'label': 'redo'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=b'current')

    await ml_service.redo(image_id=123, user_id=456)

    mock_redis.cache_image.assert_called_once()
    call_kwargs = mock_redis.cache_image.call_args.kwargs
    assert call_kwargs['image_data'] == b'next_bytes'
    assert call_kwargs['suffix'] == 'current_state'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.redo(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_no_undo_push_when_no_current_state(ml_service, mock_redis):
    mock_redis.pop_redo_state = AsyncMock(return_value={
        'bytes': b'next', 'label': 'redo'
    })
    mock_redis.get_cache_image = AsyncMock(return_value=None)

    await ml_service.redo(image_id=123, user_id=456)

    mock_redis.push_undo_state.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_returns_labels(ml_service, mock_redis):
    mock_redis.get_history_labels = AsyncMock(return_value=['remove bbox_id=0', 'replace bbox_id=1'])

    result = await ml_service.get_history(image_id=123, user_id=456)

    assert result == {'history': ['remove bbox_id=0', 'replace bbox_id=1']}
    mock_redis.get_history_labels.assert_called_once_with(123)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_empty(ml_service, mock_redis):
    mock_redis.get_history_labels = AsyncMock(return_value=[])

    result = await ml_service.get_history(image_id=123, user_id=456)

    assert result == {'history': []}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.get_history(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_success(ml_service, mock_redis, mock_image_repo, mock_s3):
    mock_redis.get_cache_image = AsyncMock(return_value=b'processed_bytes')

    saved_image = MagicMock(spec=Image)
    saved_image.status = None
    mock_image_repo.create = AsyncMock(return_value=saved_image)

    result = await ml_service.save_result(image_id=123, user_id=456)

    mock_s3.upload_bytes.assert_called_once()
    mock_image_repo.create.assert_called_once()
    mock_image_repo.update.assert_called_once()
    assert result.status == 'processed'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_raises_when_no_cached_result(ml_service, mock_redis):
    mock_redis.get_cache_image = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="No processed result to save"):
        await ml_service.save_result(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.save_result(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_uploads_to_correct_path(ml_service, mock_redis, mock_s3, mock_image_repo):
    mock_redis.get_cache_image = AsyncMock(return_value=b'processed_bytes')
    mock_image_repo.create = AsyncMock(return_value=MagicMock(spec=Image, status=None))

    await ml_service.save_result(image_id=123, user_id=456)

    upload_call = mock_s3.upload_bytes.call_args.kwargs
    assert 'saved/456/123' in upload_call['path']
    assert upload_call['content_type'] == 'image/jpeg'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_current_state_clears_redis(ml_service, mock_redis):
    await ml_service.reset_current_state(image_id=123)

    mock_redis.delete.assert_called_once_with('image:123:current_state')
    mock_redis.clear_history.assert_called_once_with(123)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_s3_upload_failure_propagates(ml_service, mock_s3):
    mock_s3.upload_bytes = AsyncMock(side_effect=Exception("S3 upload failed"))

    with pytest.raises(Exception, match="S3 upload failed"):
        await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_pipeline_failure_propagates(ml_service, mock_pipeline):
    mock_pipeline.replace_object = AsyncMock(side_effect=RuntimeError("LaMa OOM"))

    with pytest.raises(RuntimeError, match="LaMa OOM"):
        await ml_service.replace_object(
            image_id=123, bbox_id=0, replace_image_bytes=b'r', user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_pipeline_failure_propagates(ml_service, mock_pipeline):
    mock_pipeline.remove_multiple_objects = AsyncMock(side_effect=RuntimeError("CUDA OOM"))

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        await ml_service.remove_multiple_objects(
            image_id=123, bbox_ids=[0], user_id=456
        )