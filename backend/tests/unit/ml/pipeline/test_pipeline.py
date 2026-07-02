import pytest
from unittest.mock import MagicMock, patch

from app.ml.pipeline.pipeline import MLPipeline
import app.ml.pipeline.pipeline as pipeline_module

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_pipeline_singleton():
    """Ensure the module-level singleton does not leak between tests."""
    pipeline_module._pipeline_instance = None
    yield
    pipeline_module._pipeline_instance = None


@pytest.fixture
def factory_mocks():
    """Patch every default-factory dependency used by MLPipeline.__init__."""
    with patch("app.ml.pipeline.pipeline.get_yolo_lama_mode") as get_yolo, \
         patch("app.ml.pipeline.pipeline.get_sam_mode") as get_sam, \
         patch("app.ml.pipeline.pipeline.get_tracker") as get_tracker, \
         patch("app.ml.pipeline.pipeline.get_validator") as get_validator:
        get_yolo.return_value = MagicMock(name="default_yolo_lama_mode")
        get_sam.return_value = MagicMock(name="default_sam_lama_mode")
        get_tracker.return_value = MagicMock(name="default_tracker")
        get_validator.return_value = MagicMock(name="default_validator")
        yield {
            "get_yolo_lama_mode": get_yolo,
            "get_sam_mode": get_sam,
            "get_tracker": get_tracker,
            "get_validator": get_validator,
        }


def test_constructor_default_initialization_uses_factories(factory_mocks):
    pipeline = MLPipeline(device="cpu")

    factory_mocks["get_yolo_lama_mode"].assert_called_once_with(device="cpu")
    factory_mocks["get_sam_mode"].assert_called_once_with(device="cpu")
    factory_mocks["get_tracker"].assert_called_once()
    factory_mocks["get_validator"].assert_called_once()
    assert pipeline.device == "cpu"
    assert pipeline.yolo_lama_mode is factory_mocks["get_yolo_lama_mode"].return_value
    assert pipeline.sam_lama_mode is factory_mocks["get_sam_mode"].return_value
    assert pipeline.tracker is factory_mocks["get_tracker"].return_value
    assert pipeline.validator is factory_mocks["get_validator"].return_value


def test_constructor_default_device_is_cuda(factory_mocks):
    MLPipeline()

    factory_mocks["get_yolo_lama_mode"].assert_called_once_with(device="cuda")
    factory_mocks["get_sam_mode"].assert_called_once_with(device="cuda")


def test_constructor_custom_dependency_injection_skips_factories(factory_mocks):
    custom_mode = MagicMock(name="custom_yolo_lama_mode")
    custom_sam_mode = MagicMock(name="custom_sam_lama_mode")
    custom_tracker = MagicMock(name="custom_tracker")
    custom_validator = MagicMock(name="custom_validator")

    pipeline = MLPipeline(
        mode=custom_mode,
        sam_mode=custom_sam_mode,
        tracker=custom_tracker,
        validator=custom_validator,
        device="cpu",
    )

    factory_mocks["get_yolo_lama_mode"].assert_not_called()
    factory_mocks["get_sam_mode"].assert_not_called()
    factory_mocks["get_tracker"].assert_not_called()
    factory_mocks["get_validator"].assert_not_called()
    assert pipeline.yolo_lama_mode is custom_mode
    assert pipeline.sam_lama_mode is custom_sam_mode
    assert pipeline.tracker is custom_tracker
    assert pipeline.validator is custom_validator


def test_constructor_partial_dependency_injection_falls_back_to_factories(factory_mocks):
    custom_mode = MagicMock(name="custom_yolo_lama_mode")

    pipeline = MLPipeline(mode=custom_mode, device="cpu")

    assert pipeline.yolo_lama_mode is custom_mode
    factory_mocks["get_yolo_lama_mode"].assert_not_called()
    factory_mocks["get_sam_mode"].assert_called_once()
    factory_mocks["get_tracker"].assert_called_once()
    factory_mocks["get_validator"].assert_called_once()


def test_get_supported_classes_delegates_to_yolo_lama_mode(factory_mocks):
    custom_mode = MagicMock()
    custom_mode.get_supported_classes = MagicMock(return_value=["car", "person"])
    pipeline = MLPipeline(mode=custom_mode, device="cpu")

    classes = pipeline.get_supported_classes()

    assert classes == ["car", "person"]
    custom_mode.get_supported_classes.assert_called_once()


def test_get_pipeline_returns_singleton(factory_mocks):
    first = pipeline_module.get_pipeline(device="cpu")
    second = pipeline_module.get_pipeline(device="cpu")

    assert first is second
    factory_mocks["get_yolo_lama_mode"].assert_called_once()


def test_get_pipeline_creates_instance_with_given_device(factory_mocks):
    pipeline = pipeline_module.get_pipeline(device="cpu")

    assert isinstance(pipeline, MLPipeline)
    assert pipeline.device == "cpu"


def test_get_pipeline_ignores_device_on_subsequent_calls(factory_mocks):
    first = pipeline_module.get_pipeline(device="cpu")
    second = pipeline_module.get_pipeline(device="cuda")

    assert first is second
    assert second.device == "cpu"