import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.removal import RemovalMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Remover(RemovalMixin):
    def __init__(self, yolo_lama_mode, sam_lama_mode, tracker, validator):
        self.yolo_lama_mode = yolo_lama_mode
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def yolo_lama_mode() -> MagicMock:
    mode = MagicMock(name="YoloLamaMode")
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"removed"})
    mode.remove_multiple_objects = AsyncMock(return_value={"result_bytes": b"removed_multi"})
    return mode


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    mode.remove_object = AsyncMock(return_value={"result_bytes": b"sam_removed"})
    return mode


@pytest.fixture
def tracker() -> MagicMock:
    t = MagicMock(name="ExperimentTracker")
    t.log_metrics = MagicMock()
    return t


@pytest.fixture
def validator() -> MagicMock:
    v = MagicMock(name="Validator")
    v.validate_image_bytes = MagicMock()
    v.validate_mask_bytes = MagicMock()
    v.validate_bbox = MagicMock()
    return v


@pytest.fixture
def remover(yolo_lama_mode, sam_lama_mode, tracker, validator) -> _Remover:
    return _Remover(yolo_lama_mode, sam_lama_mode, tracker, validator)


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def mask_bytes() -> bytes:
    return b"mask"


@pytest.fixture
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


class TestRemoveObject:
    async def test_validates_image_and_bbox(self, remover, image_bytes, bbox, validator):
        await remover.remove_object(image_bytes=image_bytes, selected_bbox=bbox)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)
        validator.validate_bbox.assert_called_once_with(bbox)

    async def test_passes_scene_bboxes_and_ldm_params(self, remover, image_bytes, bbox, yolo_lama_mode):
        scene_bboxes = [bbox]
        await remover.remove_object(
            image_bytes=image_bytes, selected_bbox=bbox, scene_bboxes=scene_bboxes,
            ldm_steps=10, ldm_sampler="ddim", hd_strategy="RESIZE",
        )
        call_kwargs = yolo_lama_mode.remove_object.call_args.kwargs
        assert call_kwargs["scene_bboxes"] == scene_bboxes
        assert call_kwargs["ldm_steps"] == 10
        assert call_kwargs["ldm_sampler"] == "ddim"
        assert call_kwargs["hd_strategy"] == "RESIZE"

    async def test_track_metrics_false_skips_tracker(self, remover, image_bytes, bbox, tracker):
        await remover.remove_object(image_bytes=image_bytes, selected_bbox=bbox, track_metrics=False)
        tracker.log_metrics.assert_not_called()

    async def test_raises_on_invalid_bbox(self, remover, image_bytes, bbox, validator, yolo_lama_mode):
        validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

        with pytest.raises(ValueError, match="bbox x1 must be < x2"):
            await remover.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

        yolo_lama_mode.remove_object.assert_not_called()

    async def test_propagates_mode_exception(self, remover, image_bytes, bbox, yolo_lama_mode, tracker):
        yolo_lama_mode.remove_object = AsyncMock(side_effect=RuntimeError("lama crashed"))

        with pytest.raises(RuntimeError, match="lama crashed"):
            await remover.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

        tracker.log_metrics.assert_not_called()


class TestRemoveMultipleObjects:
    async def test_raises_when_no_bboxes_given(self, remover, image_bytes, yolo_lama_mode):
        with pytest.raises(ValueError, match="selected_bboxes cannot be empty"):
            await remover.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[])

        yolo_lama_mode.remove_multiple_objects.assert_not_called()

    async def test_validates_every_bbox_in_list(self, remover, image_bytes, bbox, validator):
        await remover.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox, bbox, bbox])
        assert validator.validate_bbox.call_count == 3

    async def test_raises_on_first_invalid_bbox(self, remover, image_bytes, bbox, validator, yolo_lama_mode):
        bad_bbox = {"x1": 5, "y1": 5, "x2": 1, "y2": 1}
        validator.validate_bbox.side_effect = [None, ValueError("bbox x1 must be < x2")]

        with pytest.raises(ValueError, match="bbox x1 must be < x2"):
            await remover.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox, bad_bbox])

        yolo_lama_mode.remove_multiple_objects.assert_not_called()

    async def test_logs_num_objects_metric(self, remover, image_bytes, bbox, tracker):
        await remover.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox, bbox, bbox])

        payload = tracker.log_metrics.call_args.args[0]
        assert payload["num_objects"] == 3
        assert payload["operation"] == "remove_multiple_objects"

    async def test_propagates_mode_exception(self, remover, image_bytes, bbox, yolo_lama_mode, tracker):
        yolo_lama_mode.remove_multiple_objects = AsyncMock(side_effect=RuntimeError("batch failed"))

        with pytest.raises(RuntimeError, match="batch failed"):
            await remover.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox])

        tracker.log_metrics.assert_not_called()


class TestSamRemoveObject:
    async def test_validates_image_and_mask(self, remover, image_bytes, mask_bytes, validator):
        await remover.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)
        validator.validate_mask_bytes.assert_called_once_with(mask_bytes)

    async def test_does_not_validate_bbox(self, remover, image_bytes, mask_bytes, validator):
        await remover.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)
        validator.validate_bbox.assert_not_called()

    async def test_passes_expand_mask_pixels(self, remover, image_bytes, mask_bytes, sam_lama_mode):
        await remover.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes, expand_mask_pixels=20)
        call_kwargs = sam_lama_mode.remove_object.call_args.kwargs
        assert call_kwargs["expand_mask_pixels"] == 20

    async def test_raises_on_invalid_mask(self, remover, image_bytes, mask_bytes, validator, sam_lama_mode):
        validator.validate_mask_bytes.side_effect = ValueError("Invalid mask bytes")

        with pytest.raises(ValueError, match="Invalid mask bytes"):
            await remover.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

        sam_lama_mode.remove_object.assert_not_called()

    async def test_propagates_mode_exception(self, remover, image_bytes, mask_bytes, sam_lama_mode, tracker):
        sam_lama_mode.remove_object = AsyncMock(side_effect=RuntimeError("sam lama crashed"))

        with pytest.raises(RuntimeError, match="sam lama crashed"):
            await remover.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

        tracker.log_metrics.assert_not_called()