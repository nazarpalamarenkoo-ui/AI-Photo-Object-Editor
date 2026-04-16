import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from io import BytesIO

with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.inpainter import LaMaInpainter, InpaintMode


@pytest.mark.unit
@patch('app.ml.inpainter.get_tracker')
@patch('app.ml.inpainter.ModelManager')
def test_inpainter_init(mock_model_manager_class, mock_get_tracker):
    """Test inpainter initialization"""
    # Mock tracker
    mock_tracker = MagicMock()
    mock_get_tracker.return_value = mock_tracker
    
    # Mock ModelManager
    mock_model = MagicMock()
    mock_model_manager_class.return_value = mock_model
    
    # Create inpainter
    inpainter = LaMaInpainter(device='cpu', tracker=mock_tracker)
    
    assert inpainter.device == 'cpu'
    assert inpainter.model_manager is not None
    assert inpainter.tracker is not None
    
    # Verify ModelManager was called
    mock_model_manager_class.assert_called_once_with(
        name='lama',
        device='cpu'
    )


@pytest.mark.unit
def test_create_mask_from_bbox():
    """Test mask creation from bbox"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    
    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    mask = inpainter._create_mask_from_bbox((480, 640), bbox)
    
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    
    # Check bbox area is white
    assert np.all(mask[100:200, 100:200] == 255)
    
    # Check outside is black
    assert np.all(mask[0:100, 0:100] == 0)


@pytest.mark.unit
def test_get_bbox_from_mask():
    """Test bbox extraction from mask"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    
    # Create mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 150:250] = 255
    
    bbox = inpainter._get_bbox_from_mask(mask)
    
    assert bbox['x1'] == 150
    assert bbox['x2'] == 249
    assert bbox['y1'] == 100
    assert bbox['y2'] == 199


@pytest.mark.unit
def test_get_bbox_from_empty_mask():
    """Test bbox extraction from empty mask"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    
    mask = np.zeros((480, 640), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Empty mask"):
        inpainter._get_bbox_from_mask(mask)


@pytest.mark.unit
@patch('app.ml.inpainter.get_tracker')
@patch('app.ml.inpainter.ModelManager')
def test_inpainter_singleton(mock_model_manager_class, mock_get_tracker):
    """Test singleton pattern"""
    from app.ml.inpainter import get_inpainter
    
    # Reset singleton
    import app.ml.inpainter
    app.ml.inpainter._inpainter_instance = None
    
    # Mock tracker
    mock_tracker = MagicMock()
    mock_get_tracker.return_value = mock_tracker
    
    # Mock ModelManager
    mock_model = MagicMock()
    mock_model_manager_class.return_value = mock_model
    
    inpainter1 = get_inpainter(device='cpu', tracker=mock_tracker)
    inpainter2 = get_inpainter(device='cpu', tracker=mock_tracker)
    
    assert inpainter1 is inpainter2
    
    # Cleanup
    app.ml.inpainter._inpainter_instance = None


@pytest.mark.unit
def test_create_mask_edge_cases():
    """Test mask creation at image edges"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    
    # Bbox at corner
    bbox = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}
    mask = inpainter._create_mask_from_bbox((480, 640), bbox)
    
    assert mask.shape == (480, 640)
    assert np.all(mask[0:50, 0:50] == 255)


@pytest.mark.unit
def test_create_mask_full_image():
    """Test mask creation covering entire image"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)
    
    bbox = {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 480}
    mask = inpainter._create_mask_from_bbox((480, 640), bbox)
    
    # Entire mask should be white
    assert np.all(mask == 255)
