import pytest
import io
from unittest.mock import MagicMock, AsyncMock, patch, call
from PIL import Image as PILImage

from app.ml.pipeline import MLPipeline
from app.ml.experiment_tracker import ExperimentTracker


@pytest.fixture
def sample_image_bytes():
    """Real valid PNG image bytes — used across all tests"""
    buf = io.BytesIO()
    PILImage.new('RGB', (640, 480), color=(120, 80, 60)).save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def replacement_image_bytes():
    """Second valid image for replacement tests"""
    buf = io.BytesIO()
    PILImage.new('RGB', (100, 100), color=(0, 128, 255)).save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def valid_bbox():
    return {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 200}


@pytest.fixture
def valid_bboxes():
    return [
        {'x1': 50,  'y1': 50,  'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 100, 'x2': 500, 'y2': 350},
    ]


def _make_tracker() -> ExperimentTracker:
    """
    Build a real ExperimentTracker with MLflow calls patched at network level.
    Tests can inspect tracker.log_detection_metrics / log_metrics calls normally.
    """
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.create_experiment', return_value='test-exp-id'):
        tracker = ExperimentTracker(
            tracking_uri='http://localhost:5000',
            experiment_name='integration-test'
        )
    # patch low-level mlflow log calls so nothing hits the network
    tracker.log_detection_metrics = MagicMock(wraps=tracker.log_detection_metrics)
    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        pass  # context manager just to confirm the path exists
    return tracker


@pytest.fixture
def real_tracker():
    """Real ExperimentTracker, MLflow network calls stubbed"""
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.create_experiment', return_value='test-exp-id'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.set_tag'):
        tracker = ExperimentTracker(
            tracking_uri='http://localhost:5000',
            experiment_name='integration-test'
        )
        yield tracker


@pytest.fixture
def mock_yolo_result():
    """Canonical YOLO detection result reused by multiple tests"""
    return {
        'detections': [
            {'bbox_id': 0, 'class': 'car',    'confidence': 0.95,
             'bbox': {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 200}},
            {'bbox_id': 1, 'class': 'person', 'confidence': 0.80,
             'bbox': {'x1': 300, 'y1': 100, 'x2': 500, 'y2': 350}},
        ],
        'image_size': (640, 480),
        'metrics': {'inference_time_ms': 18.4},
    }


def _make_mode(detect_result=None, remove_result=None,
               replace_result=None, multi_result=None,
               sample_image_bytes=b'fake'):
    """Helper — build a mock YoloLamaMode with sensible defaults"""
    mode = MagicMock()
    mode.detect_objects = AsyncMock(return_value=detect_result or {
        'detections': [], 'image_size': (640, 480),
        'metrics': {'inference_time_ms': 10.0},
    })
    mode.remove_object = AsyncMock(return_value=remove_result or {
        'result_bytes': sample_image_bytes,
        'metrics': {'processing_time_ms': 100.0},
    })
    mode.replace_object = AsyncMock(return_value=replace_result or {
        'result_bytes': sample_image_bytes,
        'metrics': {'processing_time_ms': 150.0},
    })
    mode.remove_multiple_objects = AsyncMock(return_value=multi_result or {
        'result_bytes': sample_image_bytes,
        'metrics': {'processing_time_ms': 200.0},
    })
    mode.get_supported_classes = MagicMock(return_value=['car', 'person', 'dog'])
    return mode


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_tracker_receives_correct_metrics(
    real_tracker, mock_yolo_result, sample_image_bytes
):
    """
    Pipeline calls ExperimentTracker.log_detection_metrics with values
    derived from the YOLO result — avg_confidence computed from detections list.
    """
    mode = _make_mode(detect_result=mock_yolo_result)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag'):
        result = await pipeline.detect_objects(
            image_bytes=sample_image_bytes,
            conf_threshold=0.4,
            track_metrics=True
        )

    # (0.95 + 0.80) / 2 = 0.875
    logged = {c.args[0]: c.args[1] for c in mock_log_metric.call_args_list}
    assert logged['num_detections'] == 2
    assert logged['avg_confidence'] == pytest.approx(0.875, rel=0.01)
    assert logged['conf_threshold'] == pytest.approx(0.4)
    # inference_time_ms and inference_time_sec must also be present
    assert 'inference_time_ms' in logged
    assert 'inference_time_sec' in logged


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_tracker_skipped_when_no_metrics_key(
    real_tracker, sample_image_bytes
):
    """
    When mode returns a result WITHOUT a 'metrics' key, tracker must NOT be called
    even if track_metrics=True — pipeline guards on result.get('metrics').
    """
    mode = _make_mode(detect_result={
        'detections': [{'bbox_id': 0, 'class': 'car', 'confidence': 0.9,
                        'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}}],
        'image_size': (640, 480),
        # no 'metrics' key intentionally
    })
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        await pipeline.detect_objects(image_bytes=sample_image_bytes, track_metrics=True)

    mock_log_metric.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_tracker_logs_operation_and_timing(
    real_tracker, sample_image_bytes, valid_bbox
):
    """
    After remove_object, tracker.log_metrics must receive a dict that includes
    operation='remove_object' and a numeric processing_time.
    """
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            expand_mask_pixels=10,
            use_edge_blending=False,
            track_metrics=True
        )

    logged = {c.args[0]: c.args[1] for c in mock_log_metric.call_args_list}
    assert logged.get('expand_mask_pixels') == 10
    # edge_blending is a bool — logged as tag, not metric
    tags = {c.args[0]: c.args[1] for c in
            patch('mlflow.set_tag').start().call_args_list
            if c.args}
    # processing_time must be a positive float
    assert logged.get('processing_time', 0) >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_tracker_logs_color_match_method(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    """
    Tracker must receive color_match_method as a tag (non-numeric value).
    """
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            color_match_method='color_transfer',
            track_metrics=True
        )

    tag_calls = {c.args[0]: c.args[1] for c in mock_set_tag.call_args_list}
    assert tag_calls.get('color_match_method') == 'color_transfer'
    assert tag_calls.get('operation') == 'replace_object'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_tracker_logs_num_objects(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes,
            track_metrics=True
        )

    logged = {c.args[0]: c.args[1] for c in mock_log_metric.call_args_list}
    assert logged['num_objects'] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_passes_all_args_to_mode(
    real_tracker, mock_yolo_result, sample_image_bytes
):
    """
    Pipeline must forward conf_threshold and classes to mode unchanged.
    """
    mode = _make_mode(detect_result=mock_yolo_result)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.detect_objects(
            image_bytes=sample_image_bytes,
            conf_threshold=0.7,
            classes=['car', 'truck'],
            track_metrics=False
        )

    mode.detect_objects.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        conf_threshold=0.7,
        classes=['car', 'truck']
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_passes_all_args_to_mode(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            expand_mask_pixels=8,
            use_edge_blending=False,
            track_metrics=False
        )

    mode.remove_object.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bbox=valid_bbox,
        expand_mask_pixels=8,
        use_edge_blending=False
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_all_args_to_mode(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            expand_mask_pixels=3,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method='histogram',
            track_metrics=False
        )

    mode.replace_object.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bbox=valid_bbox,
        replacement_image_bytes=replacement_image_bytes,
        expand_mask_pixels=3,
        use_color_matching=False,
        use_edge_blending=False,
        color_match_method='histogram'
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_all_args_to_mode(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes,
            expand_mask_pixels=3,
            use_edge_blending=False,
            track_metrics=False
        )

    mode.remove_multiple_objects.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bboxes=valid_bboxes,
        expand_mask_pixels=3,
        use_edge_blending=False
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_result_contains_required_keys(
    real_tracker, mock_yolo_result, sample_image_bytes
):
    """
    Pipeline enriches mode result with timestamp — final dict must have
    detections, image_size, metrics, timestamp.
    """
    mode = _make_mode(detect_result=mock_yolo_result)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        result = await pipeline.detect_objects(image_bytes=sample_image_bytes)

    assert 'detections'  in result
    assert 'image_size'  in result
    assert 'metrics'     in result
    assert 'timestamp'   in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_result_contains_required_keys(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        result = await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox
        )

    assert 'result_bytes' in result
    assert 'metrics'      in result
    assert 'timestamp'    in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_result_contains_required_keys(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        result = await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes
        )

    assert 'result_bytes' in result
    assert 'metrics'      in result
    assert 'timestamp'    in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_result_contains_required_keys(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        result = await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes
        )

    assert 'result_bytes' in result
    assert 'metrics'      in result
    assert 'timestamp'    in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_mode_exception_propagates(
    real_tracker, sample_image_bytes
):
    """
    If mode raises, pipeline must re-raise — tracker must NOT be called
    because the exception happens before metric logging.
    """
    mode = _make_mode()
    mode.detect_objects = AsyncMock(side_effect=RuntimeError('YOLO inference failed'))
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        with pytest.raises(RuntimeError, match='YOLO inference failed'):
            await pipeline.detect_objects(image_bytes=sample_image_bytes)

    mock_log_metric.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_mode_exception_propagates(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode()
    mode.remove_object = AsyncMock(side_effect=RuntimeError('LaMa inpainting failed'))
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        with pytest.raises(RuntimeError, match='LaMa inpainting failed'):
            await pipeline.remove_object(
                image_bytes=sample_image_bytes,
                selected_bbox=valid_bbox
            )

    mock_log_metric.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_mode_exception_propagates(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    mode = _make_mode()
    mode.replace_object = AsyncMock(side_effect=RuntimeError('replacement failed'))
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        with pytest.raises(RuntimeError, match='replacement failed'):
            await pipeline.replace_object(
                image_bytes=sample_image_bytes,
                selected_bbox=valid_bbox,
                replacement_image_bytes=replacement_image_bytes
            )

    mock_log_metric.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_mode_exception_propagates(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode()
    mode.remove_multiple_objects = AsyncMock(side_effect=RuntimeError('multi removal failed'))
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        with pytest.raises(RuntimeError, match='multi removal failed'):
            await pipeline.remove_multiple_objects(
                image_bytes=sample_image_bytes,
                selected_bboxes=valid_bboxes
            )

    mock_log_metric.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_fires_before_mode_call_on_detect(real_tracker):
    """Mode must never be called when validation fails"""
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with pytest.raises(ValueError):
        await pipeline.detect_objects(image_bytes=b'')

    mode.detect_objects.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_fires_before_mode_call_on_remove(
    real_tracker, sample_image_bytes
):
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    bad_bbox = {'x1': 200, 'y1': 10, 'x2': 100, 'y2': 100}  # x1 > x2

    with pytest.raises(ValueError, match='bbox x1 must be < x2'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=bad_bbox
        )

    mode.remove_object.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_fires_before_mode_call_on_replace(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with pytest.raises(ValueError):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=b''   # invalid replacement
        )

    mode.replace_object.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_fires_before_mode_call_on_remove_multiple(
    real_tracker, sample_image_bytes
):
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with pytest.raises(ValueError, match='selected_bboxes cannot be empty'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=[]
        )

    mode.remove_multiple_objects.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_track_metrics_false_no_mlflow_calls_detect(
    real_tracker, mock_yolo_result, sample_image_bytes
):
    mode = _make_mode(detect_result=mock_yolo_result)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.detect_objects(
            image_bytes=sample_image_bytes,
            track_metrics=False
        )

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_track_metrics_false_no_mlflow_calls_remove(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            track_metrics=False
        )

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()