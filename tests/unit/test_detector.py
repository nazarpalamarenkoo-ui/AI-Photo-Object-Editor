import pytest
from unittest.mock import MagicMock, patch, Mock

# ВАЖЛИВО: Patch get_tracker ПЕРЕД import!
with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.detector import YOLODetector

@pytest.mark.unit
@patch('app.ml.detector.get_tracker')
@patch('app.ml.detector.YOLO')
def test_detector_init(mock_yolo_class, mock_get_tracker):
    """Test detector initialization"""
    # Mock tracker
    mock_tracker = MagicMock()
    mock_get_tracker.return_value = mock_tracker
    
    # Mock YOLO class
    mock_model = MagicMock()
    mock_model.names = {0: 'person', 1: 'car'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    # Create detector
    detector = YOLODetector(
        model_path='yolov10n.pt',
        device='cpu',
        conf_threshold=0.7,
        tracker=mock_tracker  # Передаємо tracker явно!
    )
    
    assert detector.device == 'cpu'
    assert detector.conf_threshold == 0.7
    assert detector.model_path == 'yolov10n.pt'
    
    # Verify YOLO was called
    mock_yolo_class.assert_called_once_with('yolov10n.pt')
    mock_model.to.assert_called_once_with('cpu')


@pytest.mark.unit
@patch('app.ml.detector.get_tracker')
@patch('app.ml.detector.YOLO')
def test_get_class_names(mock_yolo_class, mock_get_tracker):
    """Test getting class names"""
    # Mock tracker
    mock_tracker = MagicMock()
    mock_get_tracker.return_value = mock_tracker
    
    # Mock YOLO
    mock_model = MagicMock()
    mock_model.names = {0: 'person', 1: 'bicycle', 2: 'car'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    detector = YOLODetector(device='cpu', tracker=mock_tracker)
    classes = detector.get_class_names()
    
    assert isinstance(classes, list)
    assert 'person' in classes
    assert 'bicycle' in classes
    assert 'car' in classes
    assert len(classes) == 3


@pytest.mark.unit
def test_calculate_metrics_with_detections():
    """Test metrics calculation with detections"""
    detector = YOLODetector.__new__(YOLODetector)
    
    detections = [
        {'confidence': 0.95, 'detected_class': 'person'},
        {'confidence': 0.87, 'detected_class': 'car'},
        {'confidence': 0.92, 'detected_class': 'person'}
    ]
    
    metrics = detector._calculate_metrics(detections, 100.5)
    
    assert metrics['num_detections'] == 3
    assert metrics['avg_confidence'] == pytest.approx((0.95 + 0.87 + 0.92) / 3)
    assert metrics['inference_time_ms'] == 100.5
    assert set(metrics['classes_detected']) == {'person', 'car'}
    assert metrics['min_confidence'] == 0.87
    assert metrics['max_confidence'] == 0.95


@pytest.mark.unit
def test_calculate_metrics_no_detections():
    """Test metrics calculation with no detections"""
    detector = YOLODetector.__new__(YOLODetector)
    
    metrics = detector._calculate_metrics([], 50.0)
    
    assert metrics['num_detections'] == 0
    assert metrics['avg_confidence'] == 0.0
    assert metrics['inference_time_ms'] == 50.0
    assert metrics['classes_detected'] == []


@pytest.mark.unit
def test_calculate_metrics_single_detection():
    """Test metrics with single detection"""
    detector = YOLODetector.__new__(YOLODetector)
    
    detections = [{'confidence': 0.95, 'detected_class': 'person'}]
    
    metrics = detector._calculate_metrics(detections, 50.0)
    
    assert metrics['num_detections'] == 1
    assert metrics['avg_confidence'] == 0.95
    assert metrics['min_confidence'] == 0.95
    assert metrics['max_confidence'] == 0.95
    assert metrics['classes_detected'] == ['person']


@pytest.mark.unit
@patch('app.ml.detector.get_tracker')
@patch('app.ml.detector.YOLO')
def test_detector_singleton(mock_yolo_class, mock_get_tracker):
    """Test singleton pattern"""
    from app.ml.detector import get_detector
    
    # Reset singleton
    import app.ml.detector
    app.ml.detector._detector_instance = None
    
    # Mock tracker
    mock_tracker = MagicMock()
    mock_get_tracker.return_value = mock_tracker
    
    # Mock YOLO
    mock_model = MagicMock()
    mock_model.names = {0: 'person'}
    mock_model.to = MagicMock(return_value=mock_model)
    mock_yolo_class.return_value = mock_model
    
    detector1 = get_detector(device='cpu', tracker=mock_tracker)
    detector2 = get_detector(device='cpu', tracker=mock_tracker)
    
    assert detector1 is detector2
    
    # Cleanup
    app.ml.detector._detector_instance = None
