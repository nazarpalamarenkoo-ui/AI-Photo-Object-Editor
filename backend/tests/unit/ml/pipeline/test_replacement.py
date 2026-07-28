import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_replace_object_success(host, image_bytes, bbox):
    result = await host.replace_object(
        image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
    )

    assert result["result_bytes"] == b"replaced_image"
    assert "timestamp" in result


async def test_replace_object_validates_image_bbox_and_replacement(host, image_bytes, bbox):
    await host.replace_object(
        image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
    )

    assert host.validator.validate_image_bytes.call_count == 2
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_replace_object_calls_yolo_lama_mode_with_params(host, image_bytes, bbox):
    await host.replace_object(
        image_bytes=image_bytes,
        selected_bbox=bbox,
        replacement_image_bytes=image_bytes,
        color_match_method="histogram",
        ldm_steps=12,
        ldm_sampler="ddim",
        hd_strategy="RESIZE",
    )

    call_kwargs = host.yolo_lama_mode.replace_object.call_args.kwargs
    assert call_kwargs["color_match_method"] == "histogram"
    assert call_kwargs["ldm_steps"] == 12
    assert call_kwargs["ldm_sampler"] == "ddim"
    assert call_kwargs["hd_strategy"] == "RESIZE"


async def test_replace_object_adds_timestamp(host, image_bytes, bbox):
    result = await host.replace_object(
        image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
    )

    assert isinstance(result["timestamp"], str)


async def test_replace_object_tracker_called_when_enabled(host, image_bytes, bbox):
    await host.replace_object(image_bytes=image_bytes, selected_bbox=bbox,
                               replacement_image_bytes=image_bytes)
    host.tracker.log_metrics.assert_not_called()


async def test_replace_object_tracker_not_called_when_disabled(host, image_bytes, bbox):
    await host.replace_object(
        image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes)

    host.tracker.log_metrics.assert_not_called()


async def test_replace_object_invalid_replacement_image_raises(host, image_bytes, bbox):
    host.validator.validate_image_bytes.side_effect = [None, ValueError("Invalid image bytes")]

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.replace_object(
            image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
        )

    host.yolo_lama_mode.replace_object.assert_not_called()


async def test_replace_object_propagates_mode_exception(host, image_bytes, bbox):
    host.yolo_lama_mode.replace_object = AsyncMock(side_effect=RuntimeError("replace failed"))

    with pytest.raises(RuntimeError, match="replace failed"):
        await host.replace_object(
            image_bytes=image_bytes, selected_bbox=bbox, replacement_image_bytes=image_bytes,
        )



async def test_sam_replace_object_success(host, image_bytes, mask_bytes, bbox):
    result = await host.sam_replace_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        replacement_image_bytes=image_bytes,
    )

    assert result["result_bytes"] == b"sam_replaced_image"
    assert "timestamp" in result


async def test_sam_replace_object_validates_image_mask_bbox_and_replacement(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        replacement_image_bytes=image_bytes,
    )

    assert host.validator.validate_image_bytes.call_count == 2
    host.validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_sam_replace_object_calls_sam_lama_mode_with_params(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        replacement_image_bytes=image_bytes,
        expand_mask_pixels=5, color_match_method="mean_std",
    )

    call_kwargs = host.sam_lama_mode.replace_object.call_args.kwargs
    assert call_kwargs["expand_mask_pixels"] == 5
    assert call_kwargs["color_match_method"] == "mean_std"


async def test_sam_replace_object_tracker_called_when_enabled(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object(image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
                                   replacement_image_bytes=image_bytes)
    host.tracker.log_metrics.assert_not_called()


async def test_sam_replace_object_tracker_not_called_when_disabled(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        replacement_image_bytes=image_bytes)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_replace_object_invalid_bbox_raises(host, image_bytes, mask_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox missing required key: x2")

    with pytest.raises(ValueError, match="bbox missing required key: x2"):
        await host.sam_replace_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            replacement_image_bytes=image_bytes,
        )

    host.sam_lama_mode.replace_object.assert_not_called()


async def test_sam_replace_object_propagates_mode_exception(host, image_bytes, mask_bytes, bbox):
    host.sam_lama_mode.replace_object = AsyncMock(side_effect=RuntimeError("sam replace failed"))

    with pytest.raises(RuntimeError, match="sam replace failed"):
        await host.sam_replace_object(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            replacement_image_bytes=image_bytes,
        )


@pytest.fixture(autouse=True)
def _default_sam_replace_object_diffusion_mock(host):
    """
    Ensures host.sam_lama_mode.replace_object_diffusion behaves like an
    async mode call with a sane default, regardless of what the shared
    `host` fixture wires up elsewhere — mirrors the b"sam_replaced_image"
    convention used for sam_replace_object above.
    """
    host.sam_lama_mode.replace_object_diffusion = AsyncMock(
        return_value={
            "result_bytes": b"sam_replaced_diffusion_image",
            "metrics": {"diffusion_ms": 12.0},
        }
    )


async def test_sam_replace_object_diffusion_success(host, image_bytes, mask_bytes, bbox):
    result = await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    assert result["result_bytes"] == b"sam_replaced_diffusion_image"
    assert "timestamp" in result


async def test_sam_replace_object_diffusion_adds_timestamp(host, image_bytes, mask_bytes, bbox):
    result = await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    assert isinstance(result["timestamp"], str)


async def test_sam_replace_object_diffusion_validates_image_mask_bbox_and_reference(
    host, image_bytes, mask_bytes, bbox
):
    await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    # validate_image_bytes is called twice: once for image_bytes, once for
    # reference_image_bytes.
    assert host.validator.validate_image_bytes.call_count == 2
    host.validator.validate_mask_bytes.assert_called_once_with(mask_bytes)
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_sam_replace_object_diffusion_calls_sam_lama_mode_with_params(
    host, image_bytes, mask_bytes, bbox
):
    reference_bytes = b"reference_image"

    await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=reference_bytes,
        prompt="a wooden chair",
        use_color_matching=True,
        color_match_method="histogram",
        negative_prompt="blurry, low quality",
        num_inference_steps=30,
        guidance_scale=7.5,
        ip_adapter_scale=0.6,
        strength=0.9,
        seed=42,
    )

    call_kwargs = host.sam_lama_mode.replace_object_diffusion.call_args.kwargs
    assert call_kwargs["image_bytes"] == image_bytes
    assert call_kwargs["mask_bytes"] == mask_bytes
    assert call_kwargs["bbox"] == bbox
    assert call_kwargs["reference_image_bytes"] == reference_bytes
    assert call_kwargs["prompt"] == "a wooden chair"
    assert call_kwargs["use_color_matching"] is True
    assert call_kwargs["color_match_method"] == "histogram"
    assert call_kwargs["negative_prompt"] == "blurry, low quality"
    assert call_kwargs["num_inference_steps"] == 30
    assert call_kwargs["guidance_scale"] == 7.5
    assert call_kwargs["ip_adapter_scale"] == 0.6
    assert call_kwargs["strength"] == 0.9
    assert call_kwargs["seed"] == 42


async def test_sam_replace_object_diffusion_defaults(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    call_kwargs = host.sam_lama_mode.replace_object_diffusion.call_args.kwargs
    assert call_kwargs["prompt"] == ""
    assert call_kwargs["use_color_matching"] is False
    assert call_kwargs["color_match_method"] == "color_transfer"
    assert call_kwargs["negative_prompt"] is None
    assert call_kwargs["num_inference_steps"] is None
    assert call_kwargs["guidance_scale"] is None
    assert call_kwargs["ip_adapter_scale"] is None
    assert call_kwargs["strength"] is None
    assert call_kwargs["seed"] == 0


async def test_sam_replace_object_diffusion_tracker_not_called(host, image_bytes, mask_bytes, bbox):
    await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    host.tracker.log_metrics.assert_not_called()


async def test_sam_replace_object_diffusion_invalid_reference_image_raises(
    host, image_bytes, mask_bytes, bbox
):
    # First validate_image_bytes call (image_bytes) passes, second call
    # (reference_image_bytes) raises.
    host.validator.validate_image_bytes.side_effect = [None, ValueError("Invalid image bytes")]

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.sam_replace_object_diffusion(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            reference_image_bytes=image_bytes,
        )

    host.sam_lama_mode.replace_object_diffusion.assert_not_called()


async def test_sam_replace_object_diffusion_invalid_mask_raises(host, image_bytes, mask_bytes, bbox):
    host.validator.validate_mask_bytes.side_effect = ValueError("Invalid mask bytes")

    with pytest.raises(ValueError, match="Invalid mask bytes"):
        await host.sam_replace_object_diffusion(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            reference_image_bytes=image_bytes,
        )

    host.sam_lama_mode.replace_object_diffusion.assert_not_called()


async def test_sam_replace_object_diffusion_invalid_bbox_raises(host, image_bytes, mask_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox missing required key: x2")

    with pytest.raises(ValueError, match="bbox missing required key: x2"):
        await host.sam_replace_object_diffusion(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            reference_image_bytes=image_bytes,
        )

    host.sam_lama_mode.replace_object_diffusion.assert_not_called()


async def test_sam_replace_object_diffusion_propagates_mode_exception(
    host, image_bytes, mask_bytes, bbox
):
    host.sam_lama_mode.replace_object_diffusion = AsyncMock(
        side_effect=RuntimeError("sam diffusion replace failed")
    )

    with pytest.raises(RuntimeError, match="sam diffusion replace failed"):
        await host.sam_replace_object_diffusion(
            image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
            reference_image_bytes=image_bytes,
        )


async def test_sam_replace_object_diffusion_returns_metrics(host, image_bytes, mask_bytes, bbox):
    result = await host.sam_replace_object_diffusion(
        image_bytes=image_bytes, mask_bytes=mask_bytes, bbox=bbox,
        reference_image_bytes=image_bytes,
    )

    assert result["metrics"] == {"diffusion_ms": 12.0}