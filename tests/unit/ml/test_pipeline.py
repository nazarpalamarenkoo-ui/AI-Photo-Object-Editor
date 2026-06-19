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

    mode.remove_object = AsyncMock(return_value={
        'result_bytes': b'fake_result_image',
        'metrics': {'processing_time': 1.2}
    })

    mode.replace_object = AsyncMock(return_value={
        'result_bytes': b'fake_replaced_image',
        'metrics': {'processing_time': 1.5}
    })

    mode.remove_multiple_objects = AsyncMock(return_value={
        'result_bytes': b'fake_result_multiple',
        'metrics': {'processing_time': 2.0}
    })

    mode.get_supported_classes = MagicMock(return_value=['car', 'person', 'dog'])
    return mode


@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    tracker.log_detection_metrics = MagicMock()
    tracker.log_metrics = MagicMock()
    return tracker


@pytest.fixture
def pipeline(mock_mode, mock_tracker):
    return MLPipeline(mode=mock_mode, tracker=mock_tracker, device='cpu')


@pytest.fixture
def test_image_bytes():
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_bbox():
    return {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_no_track_metrics_tracker_not_called(pipeline, test_image_bytes, mock_tracker):
    await pipeline.detect_objects(
        image_bytes=test_image_bytes,
        conf_threshold=0.5,
        track_metrics=False
    )
    mock_tracker.log_detection_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_empty_detections_avg_confidence_none(pipeline, test_image_bytes, mock_tracker, mock_mode):
    mock_mode.detect_objects = AsyncMock(return_value={
        'detections': [],
        'image_size': (640, 480),
        'metrics': {'inference_time': 0.1}
    })

    await pipeline.detect_objects(image_bytes=test_image_bytes, track_metrics=True)

    call_kwargs = mock_tracker.log_detection_metrics.call_args.kwargs
    assert call_kwargs['num_detections'] == 0
    assert call_kwargs['avg_confidence'] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_no_metrics_key_tracker_not_called(pipeline, test_image_bytes, mock_tracker, mock_mode):
    mock_mode.detect_objects = AsyncMock(return_value={
        'detections': [],
        'image_size': (640, 480)
    })

    await pipeline.detect_objects(image_bytes=test_image_bytes, track_metrics=True)
    mock_tracker.log_detection_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_passes_ldm_params_to_mode(pipeline, test_image_bytes, test_bbox):
    await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        ldm_steps=50,
        ldm_sampler='ddim',
        hd_strategy='RESIZE',
        track_metrics=False
    )

    call_kwargs = pipeline.mode.remove_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 50
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_passes_scene_bboxes_to_mode(pipeline, test_image_bytes, test_bbox):
    scene = [{'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}]

    await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        scene_bboxes=scene,
        track_metrics=False
    )

    call_kwargs = pipeline.mode.remove_object.call_args.kwargs
    assert call_kwargs['scene_bboxes'] == scene


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_no_track_metrics_tracker_not_called(pipeline, test_image_bytes, test_bbox, mock_tracker):
    await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        track_metrics=False
    )
    mock_tracker.log_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_passes_ldm_params_to_mode(pipeline, test_image_bytes, test_bbox):
    await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=test_image_bytes,
        ldm_steps=10,
        ldm_sampler='ddim',
        hd_strategy='ORIGINAL',
        track_metrics=False
    )

    call_kwargs = pipeline.mode.replace_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'ORIGINAL'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_passes_scene_bboxes_to_mode(pipeline, test_image_bytes, test_bbox):
    scene = [{'x1': 10, 'y1': 10, 'x2': 60, 'y2': 60}]

    await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=test_image_bytes,
        scene_bboxes=scene,
        track_metrics=False
    )

    call_kwargs = pipeline.mode.replace_object.call_args.kwargs
    assert call_kwargs['scene_bboxes'] == scene


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_no_track_metrics_tracker_not_called(pipeline, test_image_bytes, test_bbox, mock_tracker):
    await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=test_image_bytes,
        track_metrics=False
    )
    mock_tracker.log_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_passes_ldm_params_to_mode(pipeline, test_image_bytes):
    bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 150, 'x2': 400, 'y2': 250}
    ]

    await pipeline.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=bboxes,
        ldm_steps=30,
        ldm_sampler='plms',
        hd_strategy='CROP',
        track_metrics=False
    )

    call_kwargs = pipeline.mode.remove_multiple_objects.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 30
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_no_track_metrics_tracker_not_called(pipeline, test_image_bytes, mock_tracker):
    bboxes = [{'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}]

    await pipeline.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=bboxes,
        track_metrics=False
    )
    mock_tracker.log_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_objects_single_bbox(pipeline, test_image_bytes):
    bboxes = [{'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150}]

    result = await pipeline.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=bboxes,
        track_metrics=False
    )

    assert 'result_bytes' in result


@pytest.mark.unit
def test_validate_bbox_y1_equals_y2(pipeline):
    bbox = {'x1': 10, 'y1': 100, 'x2': 200, 'y2': 100}
    with pytest.raises(ValueError, match="bbox y1 must be < y2"):
        pipeline._validate_bbox(bbox)


@pytest.mark.unit
def test_validate_bbox_y1_greater_than_y2(pipeline):
    bbox = {'x1': 10, 'y1': 200, 'x2': 200, 'y2': 100}
    with pytest.raises(ValueError, match="bbox y1 must be < y2"):
        pipeline._validate_bbox(bbox)


@pytest.mark.unit
def test_validate_bbox_x1_equals_x2(pipeline):
    bbox = {'x1': 100, 'y1': 10, 'x2': 100, 'y2': 200}
    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        pipeline._validate_bbox(bbox)


@pytest.mark.unit
def test_validate_bbox_zero_coordinates_are_valid(pipeline):
    bbox = {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}
    pipeline._validate_bbox(bbox)


@pytest.mark.unit
def test_validate_bbox_all_negative(pipeline):
    bbox = {'x1': -10, 'y1': -20, 'x2': -5, 'y2': -1}
    with pytest.raises(ValueError, match="bbox coordinates must be >= 0"):
        pipeline._validate_bbox(bbox)


@pytest.mark.unit
def test_validate_image_bytes_empty_string(pipeline):
    with pytest.raises(ValueError, match="image_bytes cannot be empty"):
        pipeline._validate_image_bytes("")


@pytest.mark.unit
def test_validate_image_bytes_none(pipeline):
    with pytest.raises(ValueError, match="image_bytes cannot be empty"):
        pipeline._validate_image_bytes(None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_object_default_ldm_params(pipeline, test_image_bytes, test_bbox):
    await pipeline.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        track_metrics=False
    )

    call_kwargs = pipeline.mode.remove_object.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 25
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_default_color_match_method(pipeline, test_image_bytes, test_bbox):
    await pipeline.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=test_bbox,
        replacement_image_bytes=test_image_bytes,
        track_metrics=False
    )

    call_kwargs = pipeline.mode.replace_object.call_args.kwargs
    assert call_kwargs['color_match_method'] == 'mean_std'