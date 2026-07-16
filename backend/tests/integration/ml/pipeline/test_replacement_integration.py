import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.replacement import ReplacementMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Replacer(ReplacementMixin):
    def __init__(self, yolo_lama_mode, sam_lama_mode, tracker, validator):
        self.yolo_lama_mode = yolo_lama_mode
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def yolo_lama_mode() -> MagicMock:
    mode = MagicMock(name="YoloLamaMode")
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"replaced"})
    return mode


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    mode.replace_object = AsyncMock(return_value={"result_bytes": b"sam_replaced"})
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
def replacer(yolo_lama_mode, sam_lama_mode, tracker, validator) -> _Replacer:
    return _Replacer(yolo_lama_mode, sam_lama_mode, tracker, validator)


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def mask_bytes() -> bytes:
    return b"mask"


@pytest.fixture
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


class TestReplaceObject:
    async def test_validates_image_bbox_and_replacement(self, replacer, image_bytes, bbox, validator):
        await replacer.replace_object(
            image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
        )
        assert validator.validate_image_bytes.call_count == 2
        validator.validate_bbox.assert_called_once_with(bbox)

    async def test_passes_color_match_method(self, replacer, image_bytes, bbox, yolo_lama_mode):
        await replacer.replace_object(
            image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
            color_match_method="histogram",
        )
        call_kwargs = yolo_lama_mode.replace_object.call_args.kwargs
        assert call_kwargs["color_match_method"] == "histogram"

    async def test_raises_on_invalid_replacement_bytes(
        self, replacer, image_bytes, bbox, validator, yolo_lama_mode
    ):
        validator.validate_image_bytes.side_effect = [None, ValueError("Invalid image bytes")]

        with pytest.raises(ValueError, match="Invalid image bytes"):
            await replacer.replace_object(
                image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=b"",
            )

        yolo_lama_mode.replace_object.assert_not_called()

    async def test_propagates_mode_exception(self, replacer, image_bytes, bbox, yolo_lama_mode, tracker):
        yolo_lama_mode.replace_object = AsyncMock(side_effect=RuntimeError("replace crashed"))

        with pytest.raises(RuntimeError, match="replace crashed"):
            await replacer.replace_object(
                image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
            )

        tracker.log_metrics.assert_not_called()


class TestSamReplaceObject:
    async def test_validates_image_mask_bbox_and_replacement(
        self, replacer, image_bytes, mask_bytes, bbox, validator
    ):
        await replacer.sam_replace_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            replacement_image_bytes=image_bytes,
        )
        validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
        validator.validate_bbox.assert_called_once_with(bbox)
        assert validator.validate_image_bytes.call_count == 2

    async def test_default_edge_blending_is_false(self, replacer, image_bytes, mask_bytes, bbox, sam_lama_mode):
        await replacer.sam_replace_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            replacement_image_bytes=image_bytes,
        )
        call_kwargs = sam_lama_mode.replace_object.call_args.kwargs
        assert call_kwargs["use_edge_blending"] is False

    async def test_passes_expand_mask_pixels(self, replacer, image_bytes, mask_bytes, bbox, sam_lama_mode):
        await replacer.sam_replace_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            replacement_image_bytes=image_bytes, expand_mask_pixels=30,
        )
        call_kwargs = sam_lama_mode.replace_object.call_args.kwargs
        assert call_kwargs["expand_mask_pixels"] == 30

    async def test_raises_on_invalid_bbox(self, replacer, image_bytes, mask_bytes, bbox, validator, sam_lama_mode):
        validator.validate_bbox.side_effect = ValueError("bbox missing required key")

        with pytest.raises(ValueError, match="bbox missing required key"):
            await replacer.sam_replace_object(
                image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
                replacement_image_bytes=image_bytes,
            )

        sam_lama_mode.replace_object.assert_not_called()

    async def test_propagates_mode_exception(
        self, replacer, image_bytes, mask_bytes, bbox, sam_lama_mode, tracker
    ):
        sam_lama_mode.replace_object = AsyncMock(side_effect=RuntimeError("sam replace crashed"))

        with pytest.raises(RuntimeError, match="sam replace crashed"):
            await replacer.sam_replace_object(
                image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
                replacement_image_bytes=image_bytes,
            )

        tracker.log_metrics.assert_not_called()