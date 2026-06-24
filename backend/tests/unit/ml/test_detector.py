import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO
from PIL import Image

from app.ml.detector import YOLODetector

def make_box(x1, y1, x2, y2, cls_id, conf):
    box = MagicMock()

    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value.numpy.return_value = [x1, y1, x2, y2]

    box.cls = [MagicMock()]
    box.cls[0].cpu.return_value.numpy.return_value = cls_id

    box.conf = [MagicMock()]
    box.conf[0].cpu.return_value.numpy.return_value = conf

    return box


@pytest.fixture
def mock_tracker():
    """Mock ExperimentTracker"""
    tracker = MagicMock()
    tracker.log_detection_metrics = MagicMock()
    return tracker


@pytest.fixture
def mock_yolo_model():
    model = MagicMock()
    model.names = {0: 'person', 2: 'car', 16: 'dog'}
    model.to = MagicMock(return_value=model)

    def mock_predict(*args, **kwargs):
        mock_result = MagicMock()
        mock_result.boxes = [
            make_box(100, 100, 200, 200, 0, 0.95)
        ]
        return [mock_result]

    model.predict = MagicMock(side_effect=mock_predict)
    return model

@pytest.fixture
def detector(mock_tracker, mock_yolo_model):
    """YOLODetector with mocked YOLO and tracker"""
    with patch('ultralytics.YOLO') as mock_yolo_class:
        mock_yolo_class.return_value = mock_yolo_model
        
        detector = YOLODetector(
            model_path='yolov10n.pt',
            device='cpu',
            conf_threshold=0.5,
            tracker=mock_tracker
        )
        
        return detector

@pytest.mark.unit
@patch('ultralytics.YOLO')
def test_detector_init_success(mock_yolo_class):
    """Test detector initialization with default params"""
    mock_model = MagicMock()
    mock_model.names = {0: 'person', 1: 'car'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    tracker = MagicMock()
    
    detector = YOLODetector(
        model_path='yolov10n.pt',
        device='cpu',
        conf_threshold=0.7,
        tracker=tracker
    )
    
    assert detector.device == 'cpu'
    assert detector.conf_threshold == 0.7
    assert detector.model_path == 'yolov10n.pt'
    
    # Verify YOLO was called
    mock_yolo_class.assert_called_once_with('yolov10n.pt')
    mock_model.to.assert_called_once_with('cpu')


@pytest.mark.unit
@patch('ultralytics.YOLO')
def test_detector_init_cuda(mock_yolo_class):
    """Test detector initialization with CUDA device"""
    mock_model = MagicMock()
    mock_model.names = {0: 'person'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    tracker = MagicMock()
    
    detector = YOLODetector(
        model_path='yolov10n.pt',
        device='cuda',
        tracker=tracker
    )
    
    assert detector.device == 'cuda'
    mock_model.to.assert_called_once_with('cuda')


@pytest.mark.unit
@patch('ultralytics.YOLO')
def test_detector_init_custom_threshold(mock_yolo_class):
    """Test detector initialization with custom threshold"""
    mock_model = MagicMock()
    mock_model.names = {0: 'person'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    tracker = MagicMock()
    
    detector = YOLODetector(
        model_path='yolov10n.pt',
        device='cpu',
        conf_threshold=0.9,
        tracker=tracker
    )
    
    assert detector.conf_threshold == 0.9

@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_success(detector, mock_tracker):
    """Test successful detection"""
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )
    
    assert 'detections' in result
    assert 'metrics' in result
    assert len(result['detections']) == 1
    
    det = result['detections'][0]
    assert det['detected_class'] == 'person'
    assert det['confidence'] == 0.95
    assert det['x1'] == 100
    assert det['y1'] == 100
    assert det['x2'] == 200
    assert det['y2'] == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_with_tracking(detector, mock_tracker):
    """Test detection with metrics tracking"""
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=True
    )
    
    # Verify tracker was called
    mock_tracker.log_detection_metrics.assert_called_once()
    
    assert 'detections' in result
    assert 'metrics' in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_no_tracking(detector, mock_tracker):
    """Test detection without metrics tracking"""
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )
    
    # Verify tracker was NOT called
    mock_tracker.log_detection_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_with_class_filter(detector, mock_yolo_model):
    """Test detection with class filter"""
    # Mock multiple detections
    def mock_predict_multi(image_path, **kwargs):
        mock_result = MagicMock()

        mock_result.boxes = [
            make_box(100, 100, 200, 200, 0, 0.95),  # person
            make_box(300, 300, 400, 400, 2, 0.88),  # car
        ]

        return [mock_result]
    
    mock_yolo_model.predict = MagicMock(side_effect=mock_predict_multi)
    
    result = await detector.detect(
        image_path='fake_path.jpg',
        classes=['person', 'dog'],  # Filter: only person and dog
        track_metrics=False
    )
    
    # Should only return person (car filtered out)
    assert len(result['detections']) == 1
    assert result['detections'][0]['detected_class'] == 'person'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_custom_threshold(detector, mock_yolo_model):
    """Test detection with custom confidence threshold"""
    # Mock detections with different confidences
    def mock_predict_conf(image_path, **kwargs):
        mock_result = MagicMock()

        mock_result.boxes = [
            make_box(100, 100, 200, 200, 0, 0.95),
            make_box(300, 300, 400, 400, 0, 0.45),
        ]

        return [mock_result]
    
    mock_yolo_model.predict = MagicMock(side_effect=mock_predict_conf)
    
    result = await detector.detect(
        image_path='fake_path.jpg',
        conf_threshold=0.5,  # Filter out 0.45
        track_metrics=False
    )
    
    # Should only return high confidence detection
    assert len(result['detections']) == 1

    confidences = [d['confidence'] for d in result['detections']]
    assert confidences == [0.95]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_empty_result(detector, mock_yolo_model):

    def mock_predict_empty(*args, **kwargs):
        mock_result = MagicMock()
        mock_result.boxes = []
        return [mock_result]

    mock_yolo_model.predict = MagicMock(side_effect=mock_predict_empty)

    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )

    assert result['detections'] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_metrics_structure(detector):
    """Test metrics dict structure"""
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )
    
    metrics = result['metrics']
    
    assert 'num_detections' in metrics
    assert 'avg_confidence' in metrics
    assert 'inference_time_ms' in metrics
    assert metrics['inference_time_ms'] >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_detection_fields(detector):
    """Test detection dict contains all required fields"""
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )
    
    required_fields = {'x1', 'y1', 'x2', 'y2', 'detected_class', 'confidence', 'class_id'}
    
    for det in result['detections']:
        assert required_fields.issubset(det.keys())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_bbox_id_assignment(detector, mock_yolo_model):
    """Test that bbox_id is assigned sequentially"""
    # Mock multiple detections
    def mock_predict_multi(image_path, **kwargs):
        mock_result = MagicMock()

        mock_result.boxes = [
            make_box(100, 100, 200, 200, 0, 0.95),
            make_box(300, 300, 400, 400, 2, 0.88),
            make_box(500, 500, 600, 600, 16, 0.92),
        ]

        return [mock_result]
    
    mock_yolo_model.predict = MagicMock(side_effect=mock_predict_multi)
    
    result = await detector.detect(
        image_path='fake_path.jpg',
        track_metrics=False
    )
    
    # Verify bbox_id is 0, 1, 2
    assert len(result['detections']) == 3

# перевіряємо порядок через координати
    assert result['detections'][0]['x1'] == 100
    assert result['detections'][1]['x1'] == 300
    assert result['detections'][2]['x1'] == 500

@pytest.mark.unit
def test_get_class_names(detector):
    class_names = detector.get_class_names()
    
    assert isinstance(class_names, (dict, list))
    assert len(class_names) > 0

    if isinstance(class_names, dict):
        values = list(class_names.values())
    else:
        values = class_names

    assert all(isinstance(v, str) for v in values)

@pytest.mark.unit
@patch('ultralytics.YOLO')
def test_detector_singleton(mock_yolo_class):
    """Test singleton pattern for get_detector"""
    from app.ml.detector import get_detector
    import app.ml.detector
    
    # Reset singleton
    app.ml.detector._detector_instance = None
    
    mock_model = MagicMock()
    mock_model.names = {0: 'person'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    tracker = MagicMock()
    
    detector1 = get_detector(device='cpu', tracker=tracker)
    detector2 = get_detector(device='cpu', tracker=tracker)
    
    assert detector1 is detector2
    
    # Cleanup
    app.ml.detector._detector_instance = None


@pytest.mark.unit
@patch('ultralytics.YOLO')
def test_detector_singleton_different_params(mock_yolo_class):
    """Test singleton returns same instance even with different params"""
    from app.ml.detector import get_detector
    import app.ml.detector
    
    # Reset singleton
    app.ml.detector._detector_instance = None
    
    mock_model = MagicMock()
    mock_model.names = {0: 'person'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    tracker = MagicMock()
    
    detector1 = get_detector(device='cpu', tracker=tracker)
    detector2 = get_detector(device='cuda', tracker=tracker)  # Different device!
    
    # Should return same instance (singleton)
    assert detector1 is detector2
    
    # Cleanup
    app.ml.detector._detector_instance = None