import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ml.pipeline.extraction import ExtractionMixin

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _Extractor(ExtractionMixin):
    def __init__(self, sam_lama_mode, tracker, validator):
        self.sam_lama_mode = sam_lama_mode
        self.tracker = tracker
        self.validator = validator


@pytest.fixture
def sam_lama_mode() -> MagicMock:
    mode = MagicMock(name="SAMLamaMode")
    mode.extract_object = AsyncMock(return_value={
        "extracted_bytes": b"extracted", "cropped_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
        "original_size": (640, 480), "object_size": (5, 5), "area_pixels": 25,
    })
    mode.paste_extracted_object = AsyncMock(return_value={
        "result_bytes": b"pasted", "paste_bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
        "object_size": (5, 5),
    })
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
def extractor(sam_lama_mode, tracker, validator) -> _Extractor:
    return _Extractor(sam_lama_mode, tracker, validator)


@pytest.fixture
def image_bytes() -> bytes:
    return b"image"


@pytest.fixture
def mask_bytes() -> bytes:
    return b"mask"


@pytest.fixture
def bbox() -> dict:
    return {"x1": 1, "y1": 1, "x2": 10, "y2": 10}


class TestSamExtractObject:
    async def test_validates_image_mask_and_bbox(self, extractor, image_bytes, mask_bytes, bbox, validator):
        await extractor.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)
        validator.validate_image_bytes.assert_called_once_with(image_bytes)
        validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
        validator.validate_bbox.assert_called_once_with(bbox)

    async def test_raises_on_invalid_output_format(self, extractor, image_bytes, mask_bytes, bbox, sam_lama_mode):
        with pytest.raises(ValueError, match="output_format must be"):
            await extractor.sam_extract_object(
                image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox, output_format="GIF",
            )

        sam_lama_mode.extract_object.assert_not_called()

    async def test_accepts_webp_output_format(self, extractor, image_bytes, mask_bytes, bbox, sam_lama_mode):
        await extractor.sam_extract_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox, output_format="WEBP",
        )
        call_kwargs = sam_lama_mode.extract_object.call_args.kwargs
        assert call_kwargs["output_format"] == "WEBP"

    async def test_raises_on_invalid_mask(self, extractor, image_bytes, mask_bytes, bbox, validator, sam_lama_mode):
        validator.validate_mask_bytes.side_effect = ValueError("Invalid mask bytes")

        with pytest.raises(ValueError, match="Invalid mask bytes"):
            await extractor.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

        sam_lama_mode.extract_object.assert_not_called()

    async def test_propagates_mode_exception(self, extractor, image_bytes, mask_bytes, bbox, sam_lama_mode, tracker):
        sam_lama_mode.extract_object = AsyncMock(side_effect=RuntimeError("extract crashed"))

        with pytest.raises(RuntimeError, match="extract crashed"):
            await extractor.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

        tracker.log_metrics.assert_not_called()


class TestSamPasteExtractedObject:
    async def test_validates_both_images_and_bbox(self, extractor, image_bytes, bbox, validator):
        await extractor.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
        )
        assert validator.validate_image_bytes.call_count == 2
        validator.validate_bbox.assert_called_once_with(bbox)

    @pytest.mark.parametrize("scale", [0.05, 3.1, -1.0])
    async def test_raises_on_scale_out_of_range(self, extractor, image_bytes, bbox, scale, sam_lama_mode):
        with pytest.raises(ValueError, match="scale must be between"):
            await extractor.sam_paste_extracted_object(
                image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox, scale=scale,
            )

        sam_lama_mode.paste_extracted_object.assert_not_called()

    @pytest.mark.parametrize("scale", [0.1, 1.0, 3.0])
    async def test_accepts_scale_at_boundaries(self, extractor, image_bytes, bbox, scale, sam_lama_mode):
        await extractor.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox, scale=scale,
        )
        sam_lama_mode.paste_extracted_object.assert_called_once()

    async def test_passes_color_match_method(self, extractor, image_bytes, bbox, sam_lama_mode):
        await extractor.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
            color_match_method="histogram",
        )
        call_kwargs = sam_lama_mode.paste_extracted_object.call_args.kwargs
        assert call_kwargs["color_match_method"] == "histogram"

    async def test_propagates_mode_exception(self, extractor, image_bytes, bbox, sam_lama_mode, tracker):
        sam_lama_mode.paste_extracted_object = AsyncMock(side_effect=RuntimeError("paste crashed"))

        with pytest.raises(RuntimeError, match="paste crashed"):
            await extractor.sam_paste_extracted_object(
                image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
            )

        tracker.log_metrics.assert_not_called()