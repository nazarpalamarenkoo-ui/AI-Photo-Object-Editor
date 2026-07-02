import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_sam_extract_object_success(host, image_bytes, mask_bytes, bbox):
    result = await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

    assert result["extracted_bytes"] == b"extracted_object"
    assert result["area_pixels"] == 350
    assert "timestamp" in result


async def test_sam_extract_object_validates_image_mask_and_bbox(host, image_bytes, mask_bytes, bbox):
    await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)
    host.validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_sam_extract_object_invalid_output_format_raises(host, image_bytes, mask_bytes, bbox):
    with pytest.raises(ValueError, match="output_format must be 'PNG' or 'WEBP'"):
        await host.sam_extract_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox, output_format="GIF",
        )

    host.sam_lama_mode.extract_object.assert_not_called()


async def test_sam_extract_object_calls_sam_lama_mode_with_params(host, image_bytes, mask_bytes, bbox):
    await host.sam_extract_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        padding_pixels=16, output_format="WEBP",
    )

    host.sam_lama_mode.extract_object.assert_called_once_with(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        padding_pixels=16, output_format="WEBP",
    )


async def test_sam_extract_object_tracker_called_when_enabled(host, image_bytes, mask_bytes, bbox):
    await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox, track_metrics=True)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_extract_object"
    assert payload["area_pixels"] == 350


async def test_sam_extract_object_tracker_not_called_when_disabled(host, image_bytes, mask_bytes, bbox):
    await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox, track_metrics=False)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_extract_object_invalid_mask_raises(host, image_bytes, mask_bytes, bbox):
    host.validator.validate_mask_bytes.side_effect = ValueError("Invalid mask bytes")

    with pytest.raises(ValueError, match="Invalid mask bytes"):
        await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)

    host.sam_lama_mode.extract_object.assert_not_called()


async def test_sam_extract_object_propagates_mode_exception(host, image_bytes, mask_bytes, bbox):
    host.sam_lama_mode.extract_object = AsyncMock(side_effect=RuntimeError("extraction failed"))

    with pytest.raises(RuntimeError, match="extraction failed"):
        await host.sam_extract_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox)


# ---------------------------------------------------------------------------
# sam_paste_extracted_object
# ---------------------------------------------------------------------------

async def test_sam_paste_extracted_object_success(host, image_bytes, bbox):
    result = await host.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
    )

    assert result["result_bytes"] == b"pasted_image"
    assert "timestamp" in result


async def test_sam_paste_extracted_object_validates_images_and_bbox(host, image_bytes, bbox):
    await host.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
    )

    assert host.validator.validate_image_bytes.call_count == 2
    host.validator.validate_bbox.assert_called_once_with(bbox)


@pytest.mark.parametrize("invalid_scale", [0.05, 3.1, -1.0])
async def test_sam_paste_extracted_object_invalid_scale_raises(host, image_bytes, bbox, invalid_scale):
    with pytest.raises(ValueError, match="scale must be between 0.1 and 3.0"):
        await host.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox, scale=invalid_scale,
        )

    host.sam_lama_mode.paste_extracted_object.assert_not_called()


async def test_sam_paste_extracted_object_calls_mode_with_params(host, image_bytes, bbox):
    await host.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
        scale=1.5, use_color_matching=False, use_edge_blending=False,
        color_match_method="histogram",
    )

    call_kwargs = host.sam_lama_mode.paste_extracted_object.call_args.kwargs
    assert call_kwargs["scale"] == 1.5
    assert call_kwargs["use_color_matching"] is False
    assert call_kwargs["use_edge_blending"] is False
    assert call_kwargs["color_match_method"] == "histogram"


async def test_sam_paste_extracted_object_tracker_called_when_enabled(host, image_bytes, bbox):
    await host.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox, track_metrics=True,
    )

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_paste_extracted_object"


async def test_sam_paste_extracted_object_tracker_not_called_when_disabled(host, image_bytes, bbox):
    await host.sam_paste_extracted_object(
        image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox, track_metrics=False,
    )

    host.tracker.log_metrics.assert_not_called()


async def test_sam_paste_extracted_object_invalid_bbox_raises(host, image_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox y1 must be < y2")

    with pytest.raises(ValueError, match="bbox y1 must be < y2"):
        await host.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
        )

    host.sam_lama_mode.paste_extracted_object.assert_not_called()


async def test_sam_paste_extracted_object_propagates_mode_exception(host, image_bytes, bbox):
    host.sam_lama_mode.paste_extracted_object = AsyncMock(side_effect=RuntimeError("paste failed"))

    with pytest.raises(RuntimeError, match="paste failed"):
        await host.sam_paste_extracted_object(
            image_bytes=image_bytes, extracted_bytes=image_bytes, target_bbox=bbox,
        )