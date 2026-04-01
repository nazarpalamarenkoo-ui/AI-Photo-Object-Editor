"""
Integration Tests for YoloLamaMode - ВИПРАВЛЕНО

Location: tests/integration/ml/test_yolo_lama_mode_integration.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from io import BytesIO
import numpy as np
from PIL import Image

from app.ml.modes.yolo_lama_mode import YoloLamaMode


# ==================== FIXTURES ====================

@pytest.fixture
def mock_detector():
    """Mock YOLO detector"""
    detector = AsyncMock()
    
    async def mock_detect(image_path, conf_threshold=0.5, classes=None, track_metrics=True):
        detections = [
            {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'detected_class': 'person', 'confidence': 0.95, 'class_id': 0},
            {'x1': 300, 'y1': 300, 'x2': 400, 'y2': 400, 'detected_class': 'car', 'confidence': 0.87, 'class_id': 2},
            {'x1': 500, 'y1': 100, 'x2': 600, 'y2': 200, 'detected_class': 'dog', 'confidence': 0.92, 'class_id': 16}
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
    """Mock LaMa inpainter - ВИПРАВЛЕНО!"""
    inpainter = MagicMock()
    
    async def mock_inpaint(image_bytes, mask_bytes=None, bbox=None, mode=None, replacement_image_bytes=None, track_metrics=True):
        return {
            'result_bytes': b"fake processed image data",
            'metrics': {
                'processing_time_ms': 200.0,
                'mask_size_pixels': 10000,
                'image_size': (640, 480),
                'mode': mode.value if mode else 'remove'
            }
        }
    
    # ВАЖЛИВО: AsyncMock wrapper щоб працював assert_called_once!
    inpainter.inpaint = AsyncMock(side_effect=mock_inpaint)
    
    return inpainter


@pytest.fixture
def mode(mock_detector, mock_inpainter):
    """YoloLamaMode з mock компонентами"""
    return YoloLamaMode(
        detector=mock_detector,
        inpainter=mock_inpainter,
        device='cpu'
    )


@pytest.fixture
def test_image_bytes():
    """Create test image bytes (640x480)"""
    img = Image.new('RGB', (640, 480), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def test_replacement_bytes():
    """Create replacement image bytes"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


# ==================== INTEGRATION TESTS - DETECT ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_objects_all(mode, test_image_bytes):
    """Test detecting ALL objects"""
    result = await mode.detect_objects(
        image_bytes=test_image_bytes,
        conf_threshold=0.5
    )
    
    assert 'detections' in result
    assert 'metrics' in result
    assert 'image_size' in result
    
    detections = result['detections']
    assert len(detections) == 3
    
    # Check bbox_id
    assert all('bbox_id' in d for d in detections)
    assert detections[0]['bbox_id'] == 0
    assert detections[1]['bbox_id'] == 1
    assert detections[2]['bbox_id'] == 2


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_detect_objects_filtered(mode, test_image_bytes):
    """Test detecting with class filter"""
    result = await mode.detect_objects(
        image_bytes=test_image_bytes,
        classes=['person', 'dog']
    )
    
    detections = result['detections']
    assert len(detections) == 2
    
    classes = [d['detected_class'] for d in detections]
    assert 'person' in classes
    assert 'dog' in classes
    assert 'car' not in classes


# ==================== INTEGRATION TESTS - REMOVE SINGLE ====================

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
    assert result['result_bytes'] == b"fake processed image data"
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
    
    # Check size matches image (640x480)
    assert mask_array.shape == (480, 640)
    
    # Expanded bbox should be white
    assert np.any(mask_array == 255)
    
    # Outside should be black
    assert mask_array[0, 0] == 0


# ==================== INTEGRATION TESTS - REPLACE SINGLE ====================

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
    assert result['metrics']['mode'] == 'replace'


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_object_passes_bbox_to_inpainter(mode, test_image_bytes, test_replacement_bytes, mock_inpainter):
    """Test that replace passes bbox to inpainter"""
    selected_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    
    await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox=selected_bbox,
        replacement_image_bytes=test_replacement_bytes
    )
    
    # ТЕПЕР ЦЕ ПРАЦЮЄ БО inpainter.inpaint - AsyncMock!
    mock_inpainter.inpaint.assert_called_once()
    call_kwargs = mock_inpainter.inpaint.call_args.kwargs
    
    assert 'bbox' in call_kwargs
    assert call_kwargs['bbox'] == selected_bbox


# ==================== INTEGRATION TESTS - REMOVE MULTIPLE ====================

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
    
    # Check size
    assert mask_array.shape == (480, 640)
    
    # Both bbox areas should be white
    assert np.all(mask_array[100:200, 100:200] == 255)
    assert np.all(mask_array[300:400, 300:400] == 255)
    
    # Outside should be black
    assert mask_array[0, 0] == 0


# ==================== WORKFLOW TESTS ====================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_full_workflow_detect_then_remove(mode, test_image_bytes):
    """Test full workflow: detect → user selects → remove"""
    # Step 1: Detect
    detect_result = await mode.detect_objects(test_image_bytes)
    detections = detect_result['detections']
    assert len(detections) == 3
    
    # Step 2: User selects car
    selected = detections[1]
    assert selected['detected_class'] == 'car'
    
    # Step 3: Remove
    remove_result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox={'x1': selected['x1'], 'y1': selected['y1'], 'x2': selected['x2'], 'y2': selected['y2']}
    )
    
    assert remove_result['result_bytes'] == b"fake processed image data"


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_full_workflow_detect_then_replace(mode, test_image_bytes, test_replacement_bytes):
    """Test full workflow: detect → user selects → replace"""
    # Step 1: Detect
    detect_result = await mode.detect_objects(test_image_bytes)
    selected = detect_result['detections'][0]
    
    # Step 2: Replace
    replace_result = await mode.replace_object(
        image_bytes=test_image_bytes,
        selected_bbox={'x1': selected['x1'], 'y1': selected['y1'], 'x2': selected['x2'], 'y2': selected['y2']},
        replacement_image_bytes=test_replacement_bytes
    )
    
    assert replace_result['result_bytes'] == b"fake processed image data"


# ==================== EDGE CASES ====================

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
    """Test removing bbox at image edge"""
    edge_bbox = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}
    
    result = await mode.remove_object(
        image_bytes=test_image_bytes,
        selected_bbox=edge_bbox,
        expand_mask_pixels=10
    )
    
    assert 'result_bytes' in result