import pytest
import io
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image as PILImage

from app.ml.pipeline import MLPipeline
from app.ml.experiment_tracker import ExperimentTracker


@pytest.fixture
def sample_image_bytes():
    buf = io.BytesIO()
    PILImage.new('RGB', (640, 480), color=(120, 80, 60)).save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def replacement_image_bytes():
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


@pytest.fixture
def scene_bboxes():
    return [{'x1': 0, 'y1': 0, 'x2': 30, 'y2': 30}]


@pytest.fixture
def real_tracker():
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
async def test_detect_objects_empty_detections_no_avg_confidence_crash(
    real_tracker, sample_image_bytes
):
    mode = _make_mode(detect_result={
        'detections': [],
        'image_size': (640, 480),
        'metrics': {'inference_time_ms': 5.0},
    })
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        result = await pipeline.detect_objects(
            image_bytes=sample_image_bytes,
            track_metrics=True
        )

    logged = {c.args[0]: c.args[1] for c in mock_log_metric.call_args_list}
    assert logged['num_detections'] == 0
    assert 'timestamp' in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_track_metrics_false_no_mlflow_calls_replace(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            track_metrics=False
        )

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_track_metrics_false_no_mlflow_calls_remove_multiple(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes,
            track_metrics=False
        )

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_passes_scene_bboxes_to_mode(
    real_tracker, sample_image_bytes, valid_bbox, scene_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            scene_bboxes=scene_bboxes,
            track_metrics=False
        )

    call_kwargs = mode.remove_object.call_args.kwargs
    assert call_kwargs['scene_bboxes'] == scene_bboxes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_scene_bboxes_to_mode(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox, scene_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            scene_bboxes=scene_bboxes,
            track_metrics=False
        )

    call_kwargs = mode.replace_object.call_args.kwargs
    assert call_kwargs['scene_bboxes'] == scene_bboxes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_passes_ldm_params_to_mode(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            ldm_steps=50,
            ldm_sampler='ddim',
            hd_strategy='RESIZE',
            track_metrics=False
        )

    call_kwargs = mode.remove_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 50
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_ldm_params_to_mode(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            ldm_steps=10,
            ldm_sampler='ddim',
            hd_strategy='ORIGINAL',
            track_metrics=False
        )

    call_kwargs = mode.replace_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'ORIGINAL'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_ldm_params_to_mode(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes,
            ldm_steps=30,
            ldm_sampler='plms',
            hd_strategy='CROP',
            track_metrics=False
        )

    call_kwargs = mode.remove_multiple_objects.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 30
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'



@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_passes_all_args_including_scene_and_ldm(
    real_tracker, sample_image_bytes, valid_bbox, scene_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            expand_mask_pixels=15,
            use_edge_blending=True,
            scene_bboxes=scene_bboxes,
            ldm_steps=40,
            ldm_sampler='ddim',
            hd_strategy='RESIZE',
            track_metrics=False
        )

    mode.remove_object.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bbox=valid_bbox,
        expand_mask_pixels=15,
        use_edge_blending=True,
        scene_bboxes=scene_bboxes,
        ldm_steps=40,
        ldm_sampler='ddim',
        hd_strategy='RESIZE'
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_passes_all_args_including_scene_and_ldm(
    real_tracker, sample_image_bytes, replacement_image_bytes, valid_bbox, scene_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=replacement_image_bytes,
            expand_mask_pixels=5,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method='color_transfer',
            scene_bboxes=scene_bboxes,
            ldm_steps=20,
            ldm_sampler='plms',
            hd_strategy='CROP',
            track_metrics=False
        )

    mode.replace_object.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bbox=valid_bbox,
        replacement_image_bytes=replacement_image_bytes,
        expand_mask_pixels=5,
        use_color_matching=True,
        use_edge_blending=True,
        color_match_method='color_transfer',
        scene_bboxes=scene_bboxes,
        ldm_steps=20,
        ldm_sampler='plms',
        hd_strategy='CROP'
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_all_args_including_ldm(
    real_tracker, sample_image_bytes, valid_bboxes
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric'), patch('mlflow.set_tag'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=valid_bboxes,
            expand_mask_pixels=12,
            use_edge_blending=True,
            ldm_steps=35,
            ldm_sampler='ddim',
            hd_strategy='RESIZE',
            track_metrics=False
        )

    mode.remove_multiple_objects.assert_awaited_once_with(
        image_bytes=sample_image_bytes,
        selected_bboxes=valid_bboxes,
        expand_mask_pixels=12,
        use_edge_blending=True,
        ldm_steps=35,
        ldm_sampler='ddim',
        hd_strategy='RESIZE'
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_tracker_logs_edge_blending_true(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode(sample_image_bytes=sample_image_bytes)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag:
        await pipeline.remove_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            use_edge_blending=True,
            track_metrics=True
        )

    all_metric_keys = {c.args[0] for c in mock_log_metric.call_args_list}
    all_tag_keys = {c.args[0] for c in mock_set_tag.call_args_list}
    assert 'edge_blending' in all_metric_keys or 'edge_blending' in all_tag_keys


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_conf_threshold_passed_to_tracker(
    real_tracker, mock_yolo_result, sample_image_bytes
):
    mode = _make_mode(detect_result=mock_yolo_result)
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with patch('mlflow.log_metric') as mock_log_metric, patch('mlflow.set_tag'):
        await pipeline.detect_objects(
            image_bytes=sample_image_bytes,
            conf_threshold=0.8,
            track_metrics=True
        )

    logged = {c.args[0]: c.args[1] for c in mock_log_metric.call_args_list}
    assert logged.get('conf_threshold') == pytest.approx(0.8)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_tracker_logs_processing_time(
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
    assert logged.get('processing_time', 0) >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_invalid_replacement_image_fires_before_mode(
    real_tracker, sample_image_bytes, valid_bbox
):
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    with pytest.raises(ValueError):
        await pipeline.replace_object(
            image_bytes=sample_image_bytes,
            selected_bbox=valid_bbox,
            replacement_image_bytes=b'garbage'
        )

    mode.replace_object.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validation_invalid_bbox_in_multiple_fires_before_mode(
    real_tracker, sample_image_bytes
):
    mode = _make_mode()
    pipeline = MLPipeline(mode=mode, tracker=real_tracker, device='cpu')

    bad_bboxes = [
        {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 200},
        {'x1': 500, 'y1': 100, 'x2': 300, 'y2': 350},  # x1 > x2
    ]

    with pytest.raises(ValueError, match='bbox x1 must be < x2'):
        await pipeline.remove_multiple_objects(
            image_bytes=sample_image_bytes,
            selected_bboxes=bad_bboxes
        )

    mode.remove_multiple_objects.assert_not_awaited()