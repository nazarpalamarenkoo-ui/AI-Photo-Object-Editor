import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from PIL import Image
from io import BytesIO
import sys
with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.inpainter import LaMaInpainter, InpaintMode

mock_lama = MagicMock()
mock_lama.model_manager.ModelManager = MagicMock()
mock_lama.schema.Config = MagicMock()
mock_lama.schema.HDStrategy = MagicMock()
sys.modules['lama_cleaner'] = mock_lama
sys.modules['lama_cleaner.model_manager'] = mock_lama.model_manager
sys.modules['lama_cleaner.schema'] = mock_lama.schema

from app.ml.inpainter import LaMaInpainter, InpaintMode, get_inpainter

@pytest.mark.unit
def test_inpainter_init():
    """Test inpainter initialization"""
    mock_tracker = MagicMock()

    inpainter = LaMaInpainter(device='cpu', tracker=mock_tracker)

    assert inpainter.device == 'cpu'
    assert inpainter.tracker is mock_tracker
    assert inpainter.model_manager is not None


@pytest.mark.unit
def test_create_mask_from_bbox():
    """Test mask creation from bbox"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert np.all(mask[100:200, 100:200] == 255)
    assert np.all(mask[0:100, 0:100] == 0)


@pytest.mark.unit
def test_get_bbox_from_mask():
    """Test bbox extraction from mask"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

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
def test_inpainter_singleton():
    """Test singleton pattern"""
    import app.ml.inpainter
    app.ml.inpainter._inpainter_instance = None

    mock_tracker = MagicMock()

    inpainter1 = get_inpainter(device='cpu', tracker=mock_tracker)
    inpainter2 = get_inpainter(device='cpu', tracker=mock_tracker)

    assert inpainter1 is inpainter2

    app.ml.inpainter._inpainter_instance = None


@pytest.mark.unit
def test_create_mask_edge_cases():
    """Test mask creation at image edges"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)

    assert mask.shape == (480, 640)
    assert np.all(mask[0:50, 0:50] == 255)


@pytest.mark.unit
def test_create_mask_full_image():
    """Test mask creation covering entire image"""
    inpainter = LaMaInpainter.__new__(LaMaInpainter)

    bbox = {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 480}
    mask = inpainter.create_remove_mask((480, 640), bbox, expand_pixels=0)
    # Entire mask should be white
    assert np.all(mask == 255)

    assert np.all(mask == 255)