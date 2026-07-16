import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
from PIL import Image
import mlflow

from app.ml.detector import YOLODetector, get_detector
from app.ml.experiment_tracker import ExperimentTracker


@pytest.fixture(autouse=True)
def _no_mlflow_network(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    import mlflow
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(mlflow, "end_run", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_metric", MagicMock())
    yield


def _make_box(x1, y1, x2, y2, conf, cls_id):
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu().numpy.return_value = [x1, y1, x2, y2]
    box.conf = [MagicMock()]
    box.conf[0].cpu().numpy.return_value = conf
    box.cls = [MagicMock()]
    box.cls[0].cpu().numpy.return_value = cls_id
    return box


def _make_result(boxes_data, names=None):
    if names is None:
        names = {0: 'person', 2: 'car', 16: 'dog'}
    result = MagicMock()
    result.names = names
    result.boxes = [_make_box(*b) for b in boxes_data]
    return result


@pytest.fixture
def detector():
    with patch('ultralytics.YOLO') as mock_yolo:
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 2: 'car', 16: 'dog'}

        mock_model.to.return_value = mock_model

        mock_model.side_effect = None
        mock_model.predict.return_value = [
            _make_result([(100, 100, 200, 200, 0.95, 0)])
        ]
        mock_yolo.return_value = mock_model

        # мокаємо трекер повністю - жодних реальних HTTP/файлових викликів mlflow
        mock_tracker = MagicMock()
        mock_tracker.log_run = MagicMock()
        mock_tracker.log_detection_metrics = MagicMock()

        detector = YOLODetector(tracker=mock_tracker)
        yield detector


@pytest.fixture
def test_image_bytes():
    """Create test image bytes"""
    img = Image.new('RGB', (640, 480), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_with_tracking(detector, test_image_bytes, tmp_path):
    """Test detection with metrics tracking"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(
        str(image_path),
        track_metrics=True
    )

    assert 'detections' in result
    assert 'metrics' in result
    assert len(result['detections']) > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_no_tracking(detector, test_image_bytes, tmp_path):
    """Test detection without metrics tracking"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    with patch('mlflow.log_metric') as mock_log:
        result = await detector.detect(
            str(image_path),
            track_metrics=False
        )

    assert 'detections' in result
    mock_log.assert_not_called()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_with_class_filter(detector, test_image_bytes, tmp_path):
    """Test detection with class filter"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    detector.model.predict.return_value = [
        _make_result([
            (100, 100, 200, 200, 0.95, 0),   # person
            (200, 200, 300, 300, 0.87, 2),   # car
            (300, 300, 400, 400, 0.72, 16),  # dog
        ])
    ]

    result = await detector.detect(
        str(image_path),
        classes=['person', 'dog']
    )

    # Should only return filtered classes
    detected_classes = [d['detected_class'] for d in result['detections']]
    assert all(c in ['person', 'dog'] for c in detected_classes)
    assert 'car' not in detected_classes


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_custom_threshold(detector, test_image_bytes, tmp_path):
    """Test detection with custom confidence threshold"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    detector.model.predict.return_value =[
        _make_result([
            (100, 100, 200, 200, 0.95, 0),  # passes 0.9
            (200, 200, 300, 300, 0.80, 2),  # filtered out by threshold
        ])
    ]

    result = await detector.detect(
        str(image_path),
        conf_threshold=0.9
    )

    # All detections should have confidence >= 0.9
    for det in result['detections']:
        assert det['confidence'] >= 0.9


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_empty_result(detector, test_image_bytes, tmp_path):
    """Test detection when no objects found"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    empty = MagicMock()
    empty.boxes = []
    empty.names = {}
    detector.model.predict.return_value = [empty]

    result = await detector.detect(str(image_path))

    assert result['detections'] == []


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_invalid_image_path(detector):
    """Test detection with invalid image path"""
    detector.model.predict.side_effect = FileNotFoundError("/nonexistent/path/image.jpg")

    with pytest.raises(Exception):
        await detector.detect('/nonexistent/path/image.jpg')


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_returns_correct_bbox_fields(detector, test_image_bytes, tmp_path):
    """Test that each detection has all required fields"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(str(image_path), track_metrics=False)

    required_keys = {'x1', 'y1', 'x2', 'y2', 'detected_class', 'confidence', 'class_id'}
    for det in result['detections']:
        assert required_keys.issubset(det.keys()), f"Missing keys: {required_keys - det.keys()}"


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_metrics_structure(detector, test_image_bytes, tmp_path):
    """Test metrics dict contains required keys"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(str(image_path), track_metrics=False)

    m = result['metrics']
    assert 'num_detections' in m
    assert 'avg_confidence' in m
    assert 'inference_time_ms' in m
    assert m['num_detections'] == len(result['detections'])
    assert m['inference_time_ms'] >= 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_multiple_objects(detector, test_image_bytes, tmp_path):
    """Test detection returns all objects when multiple present"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    detector.model.predict.return_value = [
        _make_result([
            (100, 100, 200, 200, 0.95, 0),
            (200, 200, 300, 300, 0.87, 2),
            (300, 300, 400, 400, 0.72, 16),
        ])
    ]

    result = await detector.detect(str(image_path), track_metrics=False)

    assert len(result['detections']) == 3
    assert result['metrics']['num_detections'] == 3


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_avg_confidence_calculated_correctly(detector, test_image_bytes, tmp_path):
    """Test avg_confidence is mean of all detection confidences"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    detector.model.predict.return_value = [
        _make_result([
            (100, 100, 200, 200, 0.90, 0),
            (200, 200, 300, 300, 0.80, 2),
        ])
    ]

    result = await detector.detect(str(image_path), track_metrics=False)

    assert result['metrics']['avg_confidence'] == pytest.approx(0.85, rel=0.01)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_class_filter_excludes_all(detector, test_image_bytes, tmp_path):
    """Test that filtering by non-existent class returns empty list"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(
        str(image_path),
        classes=['bicycle'],  # not in mock results
        track_metrics=False
    )

    assert result['detections'] == []
    assert result['metrics']['num_detections'] == 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_bbox_coordinates_are_integers(detector, test_image_bytes, tmp_path):
    """Test that bbox coordinates are cast to int"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(str(image_path), track_metrics=False)

    for det in result['detections']:
        assert isinstance(det['x1'], int)
        assert isinstance(det['y1'], int)
        assert isinstance(det['x2'], int)
        assert isinstance(det['y2'], int)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_confidence_is_float(detector, test_image_bytes, tmp_path):
    """Test that confidence is a float"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    result = await detector.detect(str(image_path), track_metrics=False)

    for det in result['detections']:
        assert isinstance(det['confidence'], float)
        assert 0.0 <= det['confidence'] <= 1.0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_uses_default_threshold_when_not_specified(detector, test_image_bytes, tmp_path):
    """Test that detector uses instance conf_threshold when none provided"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    # Default conf_threshold is 0.5, our mock returns conf=0.95 → should pass
    result = await detector.detect(str(image_path), track_metrics=False)

    assert len(result['detections']) > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_get_class_names(detector):
    """Test get_class_names returns list of strings"""
    names = detector.get_class_names()

    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_min_max_confidence_in_metrics(detector, test_image_bytes, tmp_path):
    """Test min/max confidence present in metrics when detections exist"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    detector.model.predict.return_value = [
        _make_result([
            (100, 100, 200, 200, 0.95, 0),
            (200, 200, 300, 300, 0.75, 2),
        ])
    ]

    result = await detector.detect(str(image_path), track_metrics=False)

    m = result['metrics']
    assert 'min_confidence' in m
    assert 'max_confidence' in m
    assert m['min_confidence'] == pytest.approx(0.75, rel=0.01)
    assert m['max_confidence'] == pytest.approx(0.95, rel=0.01)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_empty_metrics_when_no_detections(detector, test_image_bytes, tmp_path):
    """Test metrics structure when no objects detected"""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(test_image_bytes)

    empty = MagicMock()
    empty.boxes = []
    empty.names = {}
    detector.model.predict.return_value = [empty]

    result = await detector.detect(str(image_path), track_metrics=False)

    m = result['metrics']
    assert m['num_detections'] == 0
    assert m['avg_confidence'] == 0.0
    assert m['classes_detected'] == []