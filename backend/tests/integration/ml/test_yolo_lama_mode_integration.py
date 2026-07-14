import pytest
from unittest.mock import AsyncMock, MagicMock
from io import BytesIO
import numpy as np
from PIL import Image

from app.ml.modes.yolo_lama_mode import YoloLamaMode


def _make_valid_image_bytes(width=640, height=480, color='white'):
    img = Image.new('RGB', (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


FAKE_RESULT_BYTES = _make_valid_image_bytes()


@pytest.fixture
def mock_detector():
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
    inpainter = MagicMock()

    def real_create_remove_mask(image_shape, bbox, expand_pixels=12, other_bboxes=None):
        H, W = image_shape
        x1 = max(0, bbox['x1'] - expand_pixels)
        y1 = max(0, bbox['y1'] - expand_pixels)
        x2 = min(W, bbox['x2'] + expand_pixels)
        y2 = min(H, bbox['y2'] + expand_pixels)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    inpainter.create_remove_mask = MagicMock(side_effect=real_create_remove_mask)

    async def mock_inpaint(image_bytes, mask_bytes=None, bbox=None, mode=None,
                           replacement_image_bytes=None, track_metrics=True,
                           ldm_steps=25, ldm_sampler='plms', hd_strategy='CROP'):
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
    blender = MagicMock()

    async def mock_blend(original_image_bytes, processed_image_bytes, mask_bytes, expand_mask_pixels=0):
        return processed_image_bytes

    blender.auto_blend = AsyncMock(side_effect=mock_blend)
    return blender


@pytest.fixture
def mock_color_matcher():
    matcher = MagicMock()

    def mock_match(result_bytes, original_image_bytes, bbox, method='mean_std'):
        return result_bytes

    matcher.match_against_original = MagicMock(side_effect=mock_match)
    matcher.match_colors = AsyncMock(side_effect=lambda **kw: kw.get('image_with_replacement_bytes', FAKE_RESULT_BYTES))
    return matcher


@pytest.fixture
def mock_background_remover():
    remover = MagicMock()

    async def mock_remove_and_resize(image_bytes, size):
        w, h = size
        img = Image.new('RGBA', (w, h), color=(255, 0, 0, 255))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    remover.remove_and_resize = AsyncMock(side_effect=mock_remove_and_resize)

    async def mock_remove_background(image_bytes, method='grabcut', return_format='png'):
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    remover.remove_background = AsyncMock(side_effect=mock_remove_background)
    return remover


@pytest.fixture
def mock_compositor():
    compositor = MagicMock()

    def mock_compose(clean_bg_bytes, replacement_rgba_bytes, bbox, edge_softness=0):
        return FAKE_RESULT_BYTES

    compositor.compose = MagicMock(side_effect=mock_compose)
    return compositor


@pytest.fixture
def mode(mock_detector, mock_inpainter, mock_edge_blender, mock_color_matcher,
         mock_background_remover, mock_compositor):
    m = YoloLamaMode(
        detector=mock_detector,
        inpainter=mock_inpainter,
        edge_blender=mock_edge_blender,
        color_matcher=mock_color_matcher,
        background_remover=mock_background_remover,
    )
    m.compositor = mock_compositor
    return m


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
async def test_remove_object_creates_single_mask(mode, test_image_bytes, mock_inpainter):
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        expand_mask_pixels=5,
        use_edge_blending=False
    )

    mock_inpainter.create_remove_mask.assert_called_once()
    call_args = mock_inpainter.create_remove_mask.call_args
    assert call_args.args[1] == selected_bbox


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_mask_covers_bbox_area(mode, test_image_bytes, mock_inpainter):
    """Test that mask passed to inpainter.inpaint covers the bbox area"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        expand_mask_pixels=0,
        use_edge_blending=False
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    mask_bytes = call_kwargs['mask_bytes']

    mask = Image.open(BytesIO(mask_bytes)).convert('L')
    mask_array = np.array(mask)

    assert mask_array.shape == (480, 640)
    assert np.any(mask_array[100:200, 100:200] == 255)
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
    assert result['metrics']['mode'] == 'remove'


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
    assert 'mask_bytes' in call_kwargs or 'bbox' in call_kwargs


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_calls_remove_and_resize(mode, test_image_bytes, test_replacement_bytes, mock_background_remover):
    """replace_object calls remove_and_resize with correct bbox size"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 250}  # 200x150

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes
    )

    mock_background_remover.remove_and_resize.assert_called_once()
    call_args = mock_background_remover.remove_and_resize.call_args
    actual_size = call_args.args[1] if call_args.args else call_args.kwargs.get('size')
    assert actual_size == (200, 150)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_calls_inpainter_remove_mode(mode, test_image_bytes, test_replacement_bytes, mock_inpainter):
    """replace_object calls inpainter in REMOVE mode to clean background"""
    from app.ml.inpainter import InpaintMode

    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes
    )

    mock_inpainter.inpaint.assert_called_once()
    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['mode'] == InpaintMode.REMOVE


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_calls_compositor(mode, test_image_bytes, test_replacement_bytes, mock_compositor):
    """replace_object calls compositor.compose to paste replacement"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes
    )

    mock_compositor.compose.assert_called_once()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_without_color_matching(mode, test_image_bytes, test_replacement_bytes, mock_color_matcher):
    """Test that color matching is skipped when use_color_matching=False"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes,
        use_color_matching=False
    )

    mock_color_matcher.match_against_original.assert_not_called()

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
async def test_remove_multiple_calls_inpainter_once(mode, test_image_bytes, mock_inpainter):
    """remove_multiple_objects runs ONE inpainting pass for all bboxes"""
    selected_bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
        {'x1': 500, 'y1': 100, 'x2': 600, 'y2': 200},
    ]

    await mode.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=selected_bboxes
    )

    assert mock_inpainter.inpaint.call_count == 1

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
        return {
            'detections': [],
            'metrics': {'num_detections': 0, 'avg_confidence': 0, 'inference_time_ms': 30.0}
        }

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


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_without_edge_blending(mode, test_image_bytes, mock_edge_blender):
    """Test that edge blending is skipped when use_edge_blending=False"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        use_edge_blending=False
    )

    mock_edge_blender.auto_blend.assert_not_called()

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_ldm_fast_preset(mode, test_image_bytes, mock_inpainter):
    """Fast preset — 10 steps, plms, CROP."""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        use_edge_blending=False,
        ldm_steps=10,
        ldm_sampler='plms',
        hd_strategy='CROP'
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'plms'
    assert call_kwargs['hd_strategy'] == 'CROP'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_ldm_quality_preset(mode, test_image_bytes, mock_inpainter):
    """Quality preset — 25 steps."""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP'
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 25


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_ldm_custom_ddim(mode, test_image_bytes, mock_inpainter):
    """Custom preset — ddim sampler."""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        use_edge_blending=False,
        ldm_steps=30,
        ldm_sampler='ddim',
        hd_strategy='RESIZE'
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_ldm_params_forwarded(mode, test_image_bytes, test_replacement_bytes, mock_inpainter):
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes,
        ldm_steps=15,
        ldm_sampler='ddim',
        hd_strategy='ORIGINAL'
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 15
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'ORIGINAL'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_multiple_ldm_params_forwarded(mode, test_image_bytes, mock_inpainter):
    selected_bboxes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
        {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
    ]

    await mode.remove_multiple_objects(
        image_bytes=test_image_bytes,
        selected_bboxes=selected_bboxes,
        ldm_steps=10,
        ldm_sampler='ddim',
        hd_strategy='RESIZE'
    )

    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    assert call_kwargs['ldm_steps'] == 10
    assert call_kwargs['ldm_sampler'] == 'ddim'
    assert call_kwargs['hd_strategy'] == 'RESIZE'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_with_scene_bboxes(mode, test_image_bytes, mock_inpainter):
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    scene = [
        {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
        {'x1': 500, 'y1': 100, 'x2': 600, 'y2': 200},
    ]

    result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        scene_bboxes=scene,
        use_edge_blending=False
    )

    assert 'result_bytes' in result
    mock_inpainter.create_remove_mask.assert_called()
    call_args = mock_inpainter.create_remove_mask.call_args
    other = call_args.kwargs.get('other_bboxes') or (call_args.args[2] if len(call_args.args) > 2 else None)
    if other is not None:
        assert selected_bbox not in other


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_remove_object_no_scene_bboxes(mode, test_image_bytes):
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}

    result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        scene_bboxes=None,
        use_edge_blending=False
    )

    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_with_scene_bboxes(mode, test_image_bytes, test_replacement_bytes):
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    scene = [{'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400}]

    result = await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes,
        scene_bboxes=scene
    )

    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_create_remove_mask_excludes_target_from_neighbors(mode, test_image_bytes):
    target = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    neighbor = {'x1': 300, 'y1': 100, 'x2': 400, 'y2': 200}
    all_bboxes = [target, neighbor]

    mask_bytes = await mode._create_remove_mask(
        test_image_bytes,
        target,
        expand_pixels=5,
        all_bboxes=all_bboxes
    )

    mask = np.array(Image.open(BytesIO(mask_bytes)).convert('L'))
    assert np.any(mask[100:200, 100:200] == 255)
    assert np.all(mask[100:200, 300:400] == 0)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_create_combined_mask_with_scene_bboxes(mode, test_image_bytes):
    selected = [
        {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150},
    ]
    scene = [
        {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400},
    ]

    mask_bytes = await mode._create_combined_mask(
        test_image_bytes,
        selected,
        expand_pixels=0,
        scene_bboxes=scene
    )

    mask = np.array(Image.open(BytesIO(mask_bytes)).convert('L'))
    assert np.any(mask[50:150, 50:150] == 255)
    assert np.all(mask[300:400, 300:400] == 0)
    
@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_get_supported_classes(mode):
    """Test get_supported_classes delegates to detector"""
    classes = mode.get_supported_classes()

    assert isinstance(classes, list)
    assert 'person' in classes
    assert 'car' in classes
    
