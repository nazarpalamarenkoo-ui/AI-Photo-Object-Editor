"""
Integration Tests for YOLODetector - ВИПРАВЛЕНО

Location: tests/integration/ml/test_detector_integration.py
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from PIL import Image
 
from app.ml.detector import YOLODetector


@pytest.fixture
def mock_yolo_model():
    model = MagicMock()
    model.names = {0: 'person', 1: 'bicycle', 2: 'car'}
    
    # Mock inference результат
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    # Mock box 1
    box1 = MagicMock()
    box1.xyxy = [Mock(cpu=lambda: Mock(numpy=lambda: [100, 100, 200, 200]))]
    box1.cls = [Mock(cpu=lambda: Mock(numpy=lambda: 0))]  # person
    box1.conf = [Mock(cpu=lambda: Mock(numpy=lambda: 0.95))]
    
    # Mock box 2
    box2 = MagicMock()
    box2.xyxy = [Mock(cpu=lambda: Mock(numpy=lambda: [300, 300, 400, 400]))]
    box2.cls = [Mock(cpu=lambda: Mock(numpy=lambda: 2))]  # car
    box2.conf = [Mock(cpu=lambda: Mock(numpy=lambda: 0.87))]
    
    mock_boxes.__iter__ = Mock(return_value=iter([box1, box2]))
    mock_boxes.__len__ = Mock(return_value=2)
    mock_result.boxes = mock_boxes
    
    model.return_value = [mock_result]
    
    return model
 
 
@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    tracker.log_detection_metrics = MagicMock()
    return tracker
 
 
@pytest.fixture
def detector(mock_tracker):
    with patch('app.ml.detector.YOLO') as mock_yolo:
        mock_yolo.return_value = MagicMock()
        mock_yolo.return_value.names = {0: 'person', 1: 'bicycle', 2: 'car'}
        
        detector = YOLODetector(
            model_path='yolov10n.pt',
            device='cpu',
            tracker=mock_tracker
        )
        
        yield detector
 
 
@pytest.fixture
def test_image_path(tmp_path):
    img = Image.new('RGB', (640, 480), color='white')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return str(img_path)
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_with_tracking(detector, test_image_path, mock_yolo_model):
    detector.model = mock_yolo_model
    
    result = await detector.detect(
        test_image_path,
        conf_threshold=0.5,
        track_metrics=True
    )
    
    # Verify result structure
    assert 'detections' in result
    assert 'metrics' in result
    
    detections = result['detections']
    assert len(detections) == 2
    assert detections[0]['detected_class'] == 'person'
    assert detections[1]['detected_class'] == 'car'
    
    # Verify metrics
    metrics = result['metrics']
    assert metrics['num_detections'] == 2
    assert 'avg_confidence' in metrics
    assert 'inference_time_ms' in metrics
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_no_tracking(detector, test_image_path, mock_yolo_model):
    detector.model = mock_yolo_model
    
    result = await detector.detect(
        test_image_path,
        conf_threshold=0.5,
        track_metrics=False
    )
    
    assert 'detections' in result
    assert len(result['detections']) == 2
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_with_class_filter(detector, test_image_path, mock_yolo_model):
    detector.model = mock_yolo_model
    
    result = await detector.detect(
        test_image_path,
        classes=['person'],
        track_metrics=False
    )
    
    detections = result['detections']
    assert all(d['detected_class'] == 'person' for d in detections)
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_custom_threshold(detector, test_image_path, mock_yolo_model):
    detector.model = mock_yolo_model
    
    result = await detector.detect(
        test_image_path,
        conf_threshold=0.9,
        track_metrics=False
    )
    
    assert 'detections' in result
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_empty_result(detector, test_image_path):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.return_value = [mock_result]
    detector.model = mock_model
    
    result = await detector.detect(
        test_image_path,
        track_metrics=False
    )
    
    assert result['detections'] == []
    assert result['metrics']['num_detections'] == 0
 
 
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_invalid_image_path(detector):
    """Test detection with invalid image path - ВИПРАВЛЕНО!"""
    # Mock model щоб кидав FileNotFoundError
    def mock_detect_error(*args, **kwargs):
        raise FileNotFoundError("Image not found")
    
    detector.model = MagicMock(side_effect=mock_detect_error)
    
    with pytest.raises(FileNotFoundError):
        await detector.detect(
            "/nonexistent/image.jpg",
            track_metrics=False
        )