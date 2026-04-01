import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from app.ml.pipeline import MLPipeline


@pytest.fixture
def mock_mode():
    """Mock YoloLamaMode"""
    mode = MagicMock()
    
    # Mock detect_objects
    mode.detect_objects = AsyncMock(return_value={
        'detections': [
            {
                'bbox_id': 0,
                'class': 'car',
                'confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
            },
            {
                'bbox_id': 1,
                'class': 'person',
                'confidence': 0.88,
                'bbox': {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 350}
            }
        ],
        'image_size': (640, 480),
        'metrics': {'inference_time': 0.5}
    })
    
    # Mock remove_object
    mode.remove_object = AsyncMock(return_value={
        'result_bytes': b'fake_result_image',
        'metrics': {'processing_time': 1.2}
    })
    
    # Mock replace_object
    mode.replace_object = AsyncMock(return_value={
        'result_bytes': b'fake_replaced_image',
        'metrics': {'processing_time': 1.5}
    })
    
    # Mock remove_multiple_objects
    mode.remove_multiple_objects = AsyncMock(return_value={
        'result_bytes': b'fake_result_multiple',
        'metrics': {'processing_time': 2.0}
    })
    
    # Mock get_supported_classes
    mode.get_supported_classes = MagicMock(return_value=['car', 'person', 'dog'])
    
    return mode


@pytest.fixture
def mock_tracker():
    """Mock ExperimentTracker"""
    tracker = MagicMock()
    tracker.log_detection_metrics = MagicMock()
    tracker.log_metrics = MagicMock()
    return tracker


@pytest.fixture
def pipeline(mock_mode, mock_tracker):
    """MLPipeline instance with mocked dependencies"""
    return MLPipeline(mode=mock_mode, tracker=mock_tracker, device='cpu')


@pytest.fixture
def test_image_bytes():
    """Create valid test image bytes"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_bbox():
    """Test bounding box"""
    return {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}


@pytest.mark.unit
def test_pipeline_init(pipeline):
    """Test MLPipeline initialization"""
    assert pipeline is not None
    assert pipeline.device == 'cpu'
    assert pipeline.mode is not None
    assert pipeline.tracker is not None


@pytest.mark.unit
def test_pipeline_singleton():
    """Test singleton pattern"""
    from app.ml.pipeline import get_pipeline
    
    # Reset singleton
    import app.ml.pipeline
    app.ml.pipeline._pipeline_instance = None
    
    with patch('app.ml.pipeline.get_yolo_lama_mode'):
        with patch('app.ml.pipeline.get_tracker'):
            pipeline1 = get_pipeline(device='cpu')
            pipeline2 = get_pipeline(device='cpu')
    
    assert pipeline1 is pipeline2
    
    # Cleanup
    app.ml.pipeline._pipeline_instance = None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects(pipeline, test_image_bytes):
    """Test detect_objects method"""
    result = await pipeline.detect_objects(
        image_bytes=test_image_bytes,
        conf_threshold=0.5
    )
    
    assert 'detections' in result
    assert 'image_size' in result
    assert 'timestamp' in result
    
    assert len(result['detections']) == 2
    assert result['detections'][0]['class'] == 'car'
    assert result['image_size'] == (640, 480)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_with_classes(pipeline, test_image_bytes):
    """Test detect_objects with class filter"""
    result = await pipeline.detect_objects(
        image_bytes=test_image_bytes,
        conf_threshold=0.5,
        classes=['car', 'person']
    )
    
    assert 'detections' in result
    
    # Verify mode was called with correct params
    pipeline.mode.detect_objects.assert_called_once()
    call_kwargs = pipeline.mode.detect_objects.call_args.kwargs
    assert call_kwargs['classes'] == ['car', 'person']


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_tracks_metrics(pipeline, test_image_bytes):
    """Test that detect_objects tracks metrics"""
    result = await pipeline.detect_objects(
        image_bytes=test_image_bytes,
        track_metrics=True
    )
    
    # Verify tracker was called
    pipeline.tracker.log_detection_metrics.assert_called_once()
    call_kwargs = pipeline.tracker.log_detection_metrics.call_args.kwargs
    
    assert call_kwargs['num_detections'] == 2
    assert 'inference_time' in call_kwargs
    assert 'avg_confidence' in call_kwargs
    
    # avg_confidence should be (0.95 + 0.88) / 2 = 0.915
    assert abs(call_kwargs['avg_confidence'] - 0.915) < 0.01


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_invalid_image(pipeline):
    """Test detect_objects with invalid image bytes"""
    with pytest.raises(ValueError, match="Invalid image bytes"):
        await pipeline.detect_objects(image_bytes=b'not_an_image')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_empty_bytes(pipeline):
    """Test detect_objects with empty bytes"""
    with pytest.raises(ValueError, match="image_bytes cannot be empty"):
        await pipeline.detect_objects(image_bytes=b'')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object(pipeline, test_image_bytes, test_bbox):
    """Test remove_object method"""
    result = await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox
    )
    
    assert 'result_bytes' in result
    assert 'timestamp' in result
    assert result['result_bytes'] == b'fake_result_image'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_with_params(pipeline, test_image_bytes, test_bbox):
    """Test remove_object with custom parameters"""
    result = await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        expand_mask_pixels=10,
        use_edge_blending=False
    )
    
    assert 'result_bytes' in result
    
    # Verify mode was called with correct params
    pipeline.mode.remove_object.assert_called_once()
    call_kwargs = pipeline.mode.remove_object.call_args.kwargs
    assert call_kwargs['expand_mask_pixels'] == 10
    assert call_kwargs['use_edge_blending'] == False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_tracks_metrics(pipeline, test_image_bytes, test_bbox):
    """Test that remove_object tracks metrics"""
    result = await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        track_metrics=True
    )
    
    # Verify tracker was called
    pipeline.tracker.log_metrics.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_invalid_bbox(pipeline, test_image_bytes):
    """Test remove_object with invalid bbox"""
    invalid_bbox = {'x1': 200, 'y1': 100, 'x2': 100, 'y2': 200}  # x1 >= x2
    
    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await pipeline.remove_object(
            image_bytes=test_image_bytes,
            selected_bbox=invalid_bbox
        )



@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object(pipeline, test_image_bytes, test_bbox):
    """Test replace_object method"""
    replacement_bytes = test_image_bytes  # Use same image as replacement
    
    result = await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=replacement_bytes
    )
    
    assert 'result_bytes' in result
    assert 'timestamp' in result
    assert result['result_bytes'] == b'fake_replaced_image'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_with_processors(pipeline, test_image_bytes, test_bbox):
    """Test replace_object with processors enabled"""
    result = await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=test_image_bytes,
        use_color_matching=True,
        use_edge_blending=True,
        color_match_method='histogram'
    )
    
    assert 'result_bytes' in result
    
    # Verify mode was called with correct params
    call_kwargs = pipeline.mode.replace_object.call_args.kwargs
    assert call_kwargs['use_color_matching'] == True
    assert call_kwargs['use_edge_blending'] == True
    assert call_kwargs['color_match_method'] == 'histogram'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_invalid_replacement(pipeline, test_image_bytes, test_bbox):
    """Test replace_object with invalid replacement image"""
    with pytest.raises(ValueError, match="Invalid image bytes"):
        await pipeline.replace_object(
            image_bytes=test_image_bytes,
            selected_bbox=test_bbox,
            replacement_image_bytes=b'invalid'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects(pipeline, test_image_bytes):
    """Test remove_multiple_objects method"""
    bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 250}
    ]
    
    result = await pipeline.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=bboxes
    )
    
    assert 'result_bytes' in result
    assert 'timestamp' in result
    assert result['result_bytes'] == b'fake_result_multiple'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_empty_list(pipeline, test_image_bytes):
    """Test remove_multiple_objects with empty bbox list"""
    with pytest.raises(ValueError, match="selected_bboxes cannot be empty"):
        await pipeline.remove_multiple_objects(
            image_bytes=test_image_bytes,
            selected_bboxes=[]
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_invalid_bbox_in_list(pipeline, test_image_bytes):
    """Test remove_multiple_objects with invalid bbox in list"""
    bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 400, 'y1': 150, 'x2': 300, 'y2': 250}  # Invalid: x1 >= x2
    ]
    
    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await pipeline.remove_multiple_objects(
            image_bytes=test_image_bytes,
            selected_bboxes=bboxes
        )


@pytest.mark.unit
def test_validate_bbox_missing_keys(pipeline):
    """Test bbox validation with missing keys"""
    invalid_bbox = {'x1': 100, 'y1': 100}  # Missing x2, y2
    
    with pytest.raises(ValueError, match="bbox missing required key"):
        pipeline._validate_bbox(invalid_bbox)


@pytest.mark.unit
def test_validate_bbox_negative_coordinates(pipeline):
    """Test bbox validation with negative coordinates"""
    invalid_bbox = {'x1': -10, 'y1': 100, 'x2': 200, 'y2': 200}
    
    with pytest.raises(ValueError, match="bbox coordinates must be >= 0"):
        pipeline._validate_bbox(invalid_bbox)


@pytest.mark.unit
def test_validate_bbox_not_dict(pipeline):
    """Test bbox validation with non-dict input"""
    with pytest.raises(ValueError, match="bbox must be a dict"):
        pipeline._validate_bbox([100, 100, 200, 200])


@pytest.mark.unit
def test_validate_image_bytes_not_bytes(pipeline):
    """Test image validation with non-bytes input"""
    with pytest.raises(ValueError, match="image_bytes must be bytes"):
        pipeline._validate_image_bytes("not_bytes")


@pytest.mark.unit
def test_get_supported_classes(pipeline):
    """Test get_supported_classes method"""
    classes = pipeline.get_supported_classes()
    
    assert isinstance(classes, list)
    assert len(classes) == 3
    assert 'car' in classes
    assert 'person' in classes
    assert 'dog' in classes