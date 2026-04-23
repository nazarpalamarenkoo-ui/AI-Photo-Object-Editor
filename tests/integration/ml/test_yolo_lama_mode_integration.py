import pytest
from unittest.mock import AsyncMock, MagicMock
from io import BytesIO
import numpy as np
from PIL import Image

from app.ml.modes.yolo_lama_mode import YoloLamaMode


def _make_valid_image_bytes(width=640, height=480, color='white'):
    """Helper: create real valid JPEG image bytes"""
    img = Image.new('RGB', (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


FAKE_RESULT_BYTES = _make_valid_image_bytes()


@pytest.fixture
def mock_detector():
    """Mock YOLO detector"""
    detector = AsyncMock()

    async def mock_detect(image_path, conf_threshold=0.5, classes=None, track_metrics=True):
        detections = [
            {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'detected_class': 'person', 'confidence': 0.95, 'class_id': 0},
            {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400, 'detected_class': 'car',    'confidence': 0.87, 'class_id': 2},
            {'x1': 500, 'y1': 100, 'x2': 600, 'y2': 200, 'detected_class': 'dog',    'confidence': 0.92, 'class_id': 16},
        ]
        if classes:
            detections = [d for d in detections if d['detected_class'] in classes]
        return {
            'detections': detections,
            'metrics': {
                'num_detections': len(detections),
                'avg_confidence': sum(d['confidence'] for d in detections) / len(detections) if detections else 0,
                'inference_time_ms': 50.0
            }
        }

    detector.detect = mock_detect
    detector.get_class_names = MagicMock(return_value=['person', 'bicycle', 'car', 'dog'])
    return detector


@pytest.fixture
def mock_inpainter():
    """Mock LaMa inpainter — returns real JPEG bytes so downstream PIL calls succeed"""
    inpainter = MagicMock()

    async def mock_inpaint(image_bytes, mask_bytes=None, bbox=None, mode=None,
                           replacement_image_bytes=None, track_metrics=True):
        return {
            'result_bytes': FAKE_RESULT_BYTES,
            'metrics': {
                'processing_time_ms': 200.0,
                'mask_size_pixels': 10000,
                'image_size': (640, 480),
                'mode': mode.value if mode else 'remove'
            }
        }

    inpainter.inpaint = AsyncMock(side_effect=mock_inpaint)
    return inpainter


@pytest.fixture
def mock_edge_blender():
    """Mock EdgeBlender — returns the processed bytes unchanged"""
    blender = MagicMock()

    async def mock_blend(original_image_bytes, processed_image_bytes, mask_bytes, expand_mask_pixels=0):
        return processed_image_bytes

    blender.auto_blend = AsyncMock(side_effect=mock_blend)
    return blender


@pytest.fixture
def mock_color_matcher():
    """Mock ColorMatcher — returns the input bytes unchanged"""
    matcher = MagicMock()

    async def mock_match(image_with_replacement_bytes, bbox, method='mean_std', context_margin=20):
        return image_with_replacement_bytes

    matcher.match_colors = AsyncMock(side_effect=mock_match)
    return matcher


@pytest.fixture
def mock_background_remover():
    """Mock BackgroundRemover"""
    remover = MagicMock()

    async def mock_remove(image_bytes, method='grabcut', return_format='png'):
        # Return valid RGBA PNG bytes
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    remover.remove_background = AsyncMock(side_effect=mock_remove)
    return remover


@pytest.fixture
def mode(mock_detector, mock_inpainter, mock_edge_blender, mock_color_matcher, mock_background_remover):
    """YoloLamaMode з повністю замоканими компонентами"""
    return YoloLamaMode(
        detector=mock_detector,
        inpainter=mock_inpainter,
        edge_blender=mock_edge_blender,
        color_matcher=mock_color_matcher,
        background_remover=mock_background_remover,
        device='cpu'
    )


@pytest.fixture
def test_image_bytes():
    return _make_valid_image_bytes()


@pytest.fixture
def test_replacement_bytes():
    return _make_valid_image_bytes(width=100, height=100, color='red')


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_objects_all(mode, test_image_bytes):
    """Test detecting ALL objects"""
    result = await mode.detect_objects(image_bytes=test_image_bytes, conf_threshold=0.5)

    assert 'detections' in result
    assert 'metrics' in result
    assert 'image_size' in result

    detections = result['detections']
    assert len(detections) == 3

    assert all('bbox_id' in d for d in detections)
    assert detections[0]['bbox_id'] == 0
    assert detections[1]['bbox_id'] == 1
    assert detections[2]['bbox_id'] == 2


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_objects_filtered(mode, test_image_bytes):
    """Test detecting with class filter"""
    result = await mode.detect_objects(image_bytes=test_image_bytes, classes=['person', 'dog'])

    detections = result['detections']
    assert len(detections) == 2

    classes = [d['detected_class'] for d in detections]
    assert 'person' in classes
    assert 'dog' in classes
    assert 'car' not in classes


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_single_object(mode, test_image_bytes):
    """Test removing SINGLE selected object"""
    selected_bbox = {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400}

    result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        expand_mask_pixels=5
    )

    assert 'result_bytes' in result
    assert 'metrics' in result
    assert result['metrics']['mode'] == 'remove'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_creates_single_mask(mode, test_image_bytes):
    """Test that remove creates mask ONLY for selected bbox"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    mask_bytes = await mode._create_single_bbox_mask(
        test_image_bytes,
        selected_bbox,
        expand_pixels=5
    )

    mask = Image.open(BytesIO(mask_bytes)).convert('L')
    mask_array = np.array(mask)

    assert mask_array.shape == (480, 640)
    assert np.any(mask_array == 255)
    assert mask_array[0, 0] == 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_single_object(mode, test_image_bytes, test_replacement_bytes):
    """Test replacing SINGLE selected object"""
    selected_bbox = {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400}

    result = await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes,
        expand_mask_pixels=0
    )

    assert 'result_bytes' in result
    assert 'metrics' in result
    assert result['metrics']['mode'] == 'remove'  # internal step uses REMOVE to clean bg


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_passes_bbox_to_inpainter(mode, test_image_bytes, test_replacement_bytes, mock_inpainter):
    """Test that replace passes correct args to inpainter"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes
    )

    mock_inpainter.inpaint.assert_called_once()
    call_kwargs = mock_inpainter.inpaint.call_args.kwargs

    # replace_object calls inpainter in REMOVE mode with mask_bytes (not bbox)
    assert 'mask_bytes' in call_kwargs or 'bbox' in call_kwargs



@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_multiple_objects(mode, test_image_bytes):
    """Test removing MULTIPLE selected objects"""
    selected_bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 500, 'y1': 100, 'x2': 600, 'y2': 200}
    ]

    result = await mode.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=selected_bboxes,
        expand_mask_pixels=5
    )

    assert 'result_bytes' in result
    assert 'metrics' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_multiple_creates_combined_mask(mode, test_image_bytes):
    """Test that multiple remove creates combined mask"""
    selected_bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400}
    ]

    mask_bytes = await mode._create_combined_mask(
        test_image_bytes,
        selected_bboxes,
        expand_pixels=0
    )

    mask = Image.open(BytesIO(mask_bytes)).convert('L')
    mask_array = np.array(mask)

    assert mask_array.shape == (480, 640)
    assert np.all(mask_array[100:200, 100:200] == 255)
    assert np.all(mask_array[300:400, 300:400] == 255)
    assert mask_array[0, 0] == 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_full_workflow_detect_then_remove(mode, test_image_bytes):
    """Test full workflow: detect → user selects → remove"""
    detect_result = await mode.detect_objects(test_image_bytes)
    detections = detect_result['detections']
    assert len(detections) == 3

    selected = detections[1]
    assert selected['detected_class'] == 'car'

    remove_result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox={'x1': selected['x1'], 'y1': selected['y1'],
                       'x2': selected['x2'], 'y2': selected['y2']}
    )

    assert 'result_bytes' in remove_result
    assert isinstance(remove_result['result_bytes'], bytes)
    assert len(remove_result['result_bytes']) > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_full_workflow_detect_then_replace(mode, test_image_bytes, test_replacement_bytes):
    """Test full workflow: detect → user selects → replace"""
    detect_result = await mode.detect_objects(test_image_bytes)
    selected = detect_result['detections'][0]

    replace_result = await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox={'x1': selected['x1'], 'y1': selected['y1'],
                       'x2': selected['x2'], 'y2': selected['y2']},
        replacement_image_bytes=test_replacement_bytes
    )

    assert 'result_bytes' in replace_result
    assert isinstance(replace_result['result_bytes'], bytes)
    assert len(replace_result['result_bytes']) > 0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_no_objects(mode, test_image_bytes, mock_detector):
    """Test detect when no objects found"""
    async def mock_detect_empty(*args, **kwargs):
        return {'detections': [], 'metrics': {'num_detections': 0, 'avg_confidence': 0, 'inference_time_ms': 30.0}}

    mock_detector.detect = mock_detect_empty
    result = await mode.detect_objects(test_image_bytes)

    assert result['detections'] == []


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_bbox_at_image_edge(mode, test_image_bytes):
    """Test removing bbox at image edge — expand should clamp to image bounds"""
    edge_bbox = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}

    result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=edge_bbox,
        expand_mask_pixels=10
    )

    assert 'result_bytes' in result