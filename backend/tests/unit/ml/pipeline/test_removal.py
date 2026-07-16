import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_remove_object_success(host, image_bytes, bbox):
    result = await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    assert result["result_bytes"] == b"removed_image"
    assert "timestamp" in result


async def test_remove_object_validates_image_and_bbox(host, image_bytes, bbox):
    await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_remove_object_calls_yolo_lama_mode_with_params(host, image_bytes, bbox):
    await host.remove_object(
        image_bytes=image_bytes,
        selected_bbox=bbox,
        expand_mask_pixels=20,
        use_edge_blending=False,
        ldm_steps=10,
        ldm_sampler="ddim",
        hd_strategy="RESIZE",
    )

    call_kwargs = host.yolo_lama_mode.remove_object.call_args.kwargs
    assert call_kwargs["expand_mask_pixels"] == 20
    assert call_kwargs["use_edge_blending"] is False
    assert call_kwargs["ldm_steps"] == 10
    assert call_kwargs["ldm_sampler"] == "ddim"
    assert call_kwargs["hd_strategy"] == "RESIZE"


async def test_remove_object_adds_timestamp(host, image_bytes, bbox):
    result = await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    assert isinstance(result["timestamp"], str)


async def test_remove_object_tracker_called_when_enabled(host, image_bytes, bbox):
    await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)
    host.tracker.log_metrics.assert_not_called() 


async def test_remove_object_tracker_not_called_when_disabled(host, image_bytes, bbox):
    await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    host.tracker.log_metrics.assert_not_called()


async def test_remove_object_invalid_bbox_raises(host, image_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)

    host.yolo_lama_mode.remove_object.assert_not_called()


async def test_remove_object_propagates_mode_exception(host, image_bytes, bbox):
    host.yolo_lama_mode.remove_object = AsyncMock(side_effect=RuntimeError("lama failed"))

    with pytest.raises(RuntimeError, match="lama failed"):
        await host.remove_object(image_bytes=image_bytes, selected_bbox=bbox)


async def test_remove_multiple_objects_success(host, image_bytes, bbox):
    result = await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox])

    assert result["result_bytes"] == b"removed_multiple_image"
    assert "timestamp" in result


async def test_remove_multiple_objects_validates_each_bbox(host, image_bytes, bbox):
    bboxes = [bbox, {"x1": 0, "y1": 0, "x2": 5, "y2": 5}]

    await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=bboxes)

    assert host.validator.validate_bbox.call_count == 2


async def test_remove_multiple_objects_empty_bbox_list_raises(host, image_bytes):
    with pytest.raises(ValueError, match="selected_bboxes cannot be empty"):
        await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[])

    host.yolo_lama_mode.remove_multiple_objects.assert_not_called()


async def test_remove_multiple_objects_invalid_bbox_in_list_raises(host, image_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox y1 must be < y2")

    with pytest.raises(ValueError, match="bbox y1 must be < y2"):
        await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox])


async def test_remove_multiple_objects_tracker_called_with_num_objects(host, image_bytes, bbox):
    await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox, bbox])
    host.tracker.log_metrics.assert_not_called()


async def test_remove_multiple_objects_tracker_not_called_when_disabled(host, image_bytes, bbox):
    await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox])

    host.tracker.log_metrics.assert_not_called()


async def test_remove_multiple_objects_propagates_mode_exception(host, image_bytes, bbox):
    host.yolo_lama_mode.remove_multiple_objects = AsyncMock(side_effect=RuntimeError("batch failed"))

    with pytest.raises(RuntimeError, match="batch failed"):
        await host.remove_multiple_objects(image_bytes=image_bytes, selected_bboxes=[bbox])


async def test_sam_remove_object_success(host, image_bytes, mask_bytes):
    result = await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

    assert result["result_bytes"] == b"sam_removed_image"
    assert "timestamp" in result


async def test_sam_remove_object_validates_image_and_mask(host, image_bytes, mask_bytes):
    await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)
    host.validator.validate_mask_bytes.assert_called_once_with(mask_bytes)


async def test_sam_remove_object_calls_sam_lama_mode(host, image_bytes, mask_bytes):
    await host.sam_remove_object(
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        expand_mask_pixels=15,
        use_edge_blending=False,
    )

    call_kwargs = host.sam_lama_mode.remove_object.call_args.kwargs
    assert call_kwargs["expand_mask_pixels"] == 15
    assert call_kwargs["use_edge_blending"] is False


async def test_sam_remove_object_tracker_called_when_enabled(host, image_bytes, mask_bytes):
    await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)
    host.tracker.log_metrics.assert_not_called()


async def test_sam_remove_object_tracker_not_called_when_disabled(host, image_bytes, mask_bytes):
    await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_remove_object_invalid_mask_raises(host, image_bytes, mask_bytes):
    host.validator.validate_mask_bytes.side_effect = ValueError("Invalid mask bytes")

    with pytest.raises(ValueError, match="Invalid mask bytes"):
        await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)

    host.sam_lama_mode.remove_object.assert_not_called()


async def test_sam_remove_object_propagates_mode_exception(host, image_bytes, mask_bytes):
    host.sam_lama_mode.remove_object = AsyncMock(side_effect=RuntimeError("sam lama failed"))

    with pytest.raises(RuntimeError, match="sam lama failed"):
        await host.sam_remove_object(image_bytes=image_bytes, mask_bytes=mask_bytes)