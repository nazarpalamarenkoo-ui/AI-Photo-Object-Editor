"""
Unit Tests for YoloLamaMode

Location: tests/unit/test_yolo_lama_mode.py

КРИТИЧНО: Використовуємо ТІЛЬКИ mock компоненти, нічого не завантажуємо!
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ml.modes.yolo_lama_mode import YoloLamaMode


# ==================== UNIT TESTS ====================

@pytest.mark.unit
def test_mode_init():
    """Test mode initialization"""
    # Mock detector
    mock_detector = MagicMock()
    mock_detector.get_class_names = MagicMock(return_value=['person', 'car'])
    
    # Mock inpainter
    mock_inpainter = MagicMock()
    
    # Create mode з ГОТОВИМИ mock компонентами
    mode = YoloLamaMode(
        detector=mock_detector,
        inpainter=mock_inpainter,
        device='cpu'
    )
    
    assert mode.device == 'cpu'
    assert mode.detector is mock_detector
    assert mode.inpainter is mock_inpainter


@pytest.mark.unit
def test_get_supported_classes():
    """Test getting supported classes"""
    # Mock detector
    mock_detector = MagicMock()
    mock_detector.get_class_names = MagicMock(
        return_value=['person', 'car', 'dog']
    )
    
    # Mock inpainter
    mock_inpainter = MagicMock()
    
    # Create mode
    mode = YoloLamaMode(
        detector=mock_detector,
        inpainter=mock_inpainter,
        device='cpu'
    )
    
    classes = mode.get_supported_classes()
    
    assert isinstance(classes, list)
    assert 'person' in classes
    assert 'car' in classes
    assert 'dog' in classes


@pytest.mark.unit
@patch('app.ml.modes.yolo_lama_mode.get_inpainter')
@patch('app.ml.modes.yolo_lama_mode.get_detector')
def test_mode_singleton(mock_get_detector, mock_get_inpainter):
    """Test singleton pattern"""
    from app.ml.modes.yolo_lama_mode import get_yolo_lama_mode
    
    # Reset singleton
    import app.ml.modes.yolo_lama_mode
    app.ml.modes.yolo_lama_mode._yolo_lama_mode_instance = None
    
    # Mock components (повертаємо готові mock об'єкти)
    mock_detector = MagicMock()
    mock_inpainter = MagicMock()
    mock_get_detector.return_value = mock_detector
    mock_get_inpainter.return_value = mock_inpainter
    
    # Get singleton
    mode1 = get_yolo_lama_mode(device='cpu')
    mode2 = get_yolo_lama_mode(device='cpu')
    
    assert mode1 is mode2
    
    # Verify mock functions were called
    mock_get_detector.assert_called_once_with(device='cpu')
    mock_get_inpainter.assert_called_once_with(device='cpu')
    
    # Cleanup
    app.ml.modes.yolo_lama_mode._yolo_lama_mode_instance = None