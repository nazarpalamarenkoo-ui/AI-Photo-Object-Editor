import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from app.services.ml_service import MLService
from app.db.models.detection import Detection
from app.db.models.image import Image


@pytest.fixture
def mock_db():
    return MagicMock()


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
    return repo


@pytest.fixture
def mock_detection_repo():
    repo = MagicMock()

    mock_detections = [
        MagicMock(bbox_id=0, detected_class='car', confidence=0.95,
                  x1=100, y1=100, x2=200, y2=200),
        MagicMock(bbox_id=1, detected_class='person', confidence=0.88,
                  x1=300, y1=150, x2=400, y2=350)
    ]

    repo.get_by_image = AsyncMock(return_value=mock_detections)
    repo.create_many = AsyncMock(return_value=mock_detections)
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
async def test_detect_objects_success(ml_service, mock_pipeline, mock_s3, mock_redis, mock_detection_repo):
    result = await ml_service.detect_objects(image_id=123, user_id=456, conf_threshold=0.5)

    mock_s3.download.assert_called_once_with('s3://bucket/image.jpg')
    mock_pipeline.detect_objects.assert_called_once()
    mock_detection_repo.create_many.assert_called_once()
    mock_redis.cache_detections.assert_called_once()

    assert 'detections' in result
    assert 'image_size' in result
    assert len(result['detections']) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_image_not_found(ml_service, mock_image_repo):
    mock_image_repo.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Image 123 not found"):
        await ml_service.detect_objects(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.detect_objects(image_id=123, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_with_class_filter(ml_service, mock_pipeline):
    await ml_service.detect_objects(image_id=123, user_id=456, classes=['car', 'person'])

    call_kwargs = mock_pipeline.detect_objects.call_args.kwargs
    assert call_kwargs['classes'] == ['car', 'person']


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_success(ml_service, mock_pipeline, mock_s3):
    result = await ml_service.remove_object(image_id=123, bbox_id=0, user_id=456)

    mock_pipeline.remove_object.assert_called_once()
    mock_s3.upload_bytes.assert_called_once()
    mock_s3.get_presigned_url.assert_called_once()

    assert 'result_url' in result
    assert 'presigned_url' in result
    assert 'metrics' in result
    assert result['result_url'] == 's3://bucket/result.jpg'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_detection_not_found(ml_service):
    with pytest.raises(ValueError, match="Detection with bbox_id 999 not found"):
        await ml_service.remove_object(image_id=123, bbox_id=999, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_unauthorized(ml_service):
    with pytest.raises(ValueError, match="Unauthorized"):
        await ml_service.remove_object(image_id=123, bbox_id=0, user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_with_custom_params(ml_service, mock_pipeline):
    await ml_service.remove_object(
        image_id=123, bbox_id=0, user_id=456,
        expand_mask_pixels=10, use_edge_blending=False
    )

    call_kwargs = mock_pipeline.remove_object.call_args.kwargs
    assert call_kwargs['expand_mask_pixels'] == 10
    assert call_kwargs['use_edge_blending'] == False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_success(ml_service, mock_pipeline, mock_s3):
    result = await ml_service.replace_object(
        image_id=123, bbox_id=0,
        replace_image_bytes=b'fake_replacement_image',
        user_id=456
    )

    mock_pipeline.replace_object.assert_called_once()
    mock_s3.upload_bytes.assert_called_once()

    assert 'result_url' in result
    assert 'presigned_url' in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_with_processors(ml_service, mock_pipeline):
    await ml_service.replace_object(
        image_id=123, bbox_id=0,
        replace_image_bytes=b'replacement',
        user_id=456,
        use_color_matching=True,
        use_edge_blending=True,
        color_match_method='histogram'
    )

    call_kwargs = mock_pipeline.replace_object.call_args.kwargs
    assert call_kwargs['use_color_matching'] == True
    assert call_kwargs['use_edge_blending'] == True
    assert call_kwargs['color_match_method'] == 'histogram'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_success(ml_service, mock_pipeline, mock_s3):
    result = await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0, 1], user_id=456
    )

    mock_pipeline.remove_multiple_objects.assert_called_once()
    mock_s3.upload_bytes.assert_called_once()

    assert 'result_url' in result
    assert 'presigned_url' in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_no_valid_detections(ml_service):
    with pytest.raises(ValueError, match="No valid detections found"):
        await ml_service.remove_multiple_objects(
            image_id=123, bbox_ids=[999, 888], user_id=456
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_partial_valid(ml_service, mock_pipeline):
    await ml_service.remove_multiple_objects(
        image_id=123, bbox_ids=[0, 999], user_id=456
    )

    call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
    assert len(call_kwargs['selected_bboxes']) == 1


@pytest.mark.unit
def test_get_supported_classes(ml_service, mock_pipeline):
    classes = ml_service.get_supported_classes()

    assert isinstance(classes, list)
    assert len(classes) == 3
    assert 'car' in classes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_workflow_detect_then_remove(ml_service, mock_pipeline, mock_s3, mock_detection_repo):
    detect_result = await ml_service.detect_objects(image_id=123, user_id=456)

    assert len(detect_result['detections']) > 0

    bbox_id = detect_result['detections'][0]['bbox_id']
    remove_result = await ml_service.remove_object(image_id=123, bbox_id=bbox_id, user_id=456)

    assert 'result_url' in remove_result
    mock_pipeline.detect_objects.assert_called_once()
    mock_pipeline.remove_object.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handling_s3_download_fails(ml_service, mock_s3):
    mock_s3.download = AsyncMock(side_effect=Exception("S3 connection error"))

    with pytest.raises(Exception, match="S3 connection error"):
        await ml_service.detect_objects(image_id=123, user_id=456)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handling_pipeline_fails(ml_service, mock_pipeline):
    mock_pipeline.detect_objects = AsyncMock(side_effect=Exception("YOLO model not loaded"))

    with pytest.raises(Exception, match="YOLO model not loaded"):
        await ml_service.detect_objects(image_id=123, user_id=456)