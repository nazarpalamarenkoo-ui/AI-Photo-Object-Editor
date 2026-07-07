import pytest
from unittest.mock import AsyncMock

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def polygon_points() -> list:
    """Not provided by conftest.py - defined locally for the polygon tests below."""
    return [(1, 1), (10, 1), (10, 10), (1, 10)]


@pytest.fixture
def host(host):
    host.sam_lama_mode.segment_by_polygon = AsyncMock(
        return_value={
            "segments": [{
                "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
                "area": 25,
                "mask_bytes": b"m",
                "mask_id": 0,
                "bbox_id": 0,
                "stability_score": 0.9,
                "predicted_iou": 0.8,
            }],
            "image_size": (640, 480),
        }
    )
    return host


async def test_sam_segment_objects_success(host, image_bytes):
    result = await host.sam_segment_objects(image_bytes=image_bytes)

    assert len(result["segments"]) == 1
    assert result["image_size"] == (640, 480)
    assert "timestamp" in result


async def test_sam_segment_objects_validates_image_bytes(host, image_bytes):
    await host.sam_segment_objects(image_bytes=image_bytes)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)


async def test_sam_segment_objects_calls_sam_lama_mode_with_params(host, image_bytes):
    await host.sam_segment_objects(image_bytes=image_bytes, min_area=1000, max_segments=10)

    host.sam_lama_mode.segment_objects.assert_called_once_with(
        image_bytes=image_bytes, min_area=1000, max_segments=10,
    )


async def test_sam_segment_objects_tracker_called_when_enabled(host, image_bytes):
    await host.sam_segment_objects(image_bytes=image_bytes, track_metrics=True)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_segment_auto"
    assert payload["num_segments"] == 1


async def test_sam_segment_objects_tracker_not_called_when_disabled(host, image_bytes):
    await host.sam_segment_objects(image_bytes=image_bytes, track_metrics=False)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_segment_objects_invalid_image_raises(host, image_bytes):
    host.validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.sam_segment_objects(image_bytes=image_bytes)

    host.sam_lama_mode.segment_objects.assert_not_called()


async def test_sam_segment_objects_propagates_mode_exception(host, image_bytes):
    host.sam_lama_mode.segment_objects = AsyncMock(side_effect=RuntimeError("segmentation failed"))

    with pytest.raises(RuntimeError, match="segmentation failed"):
        await host.sam_segment_objects(image_bytes=image_bytes)


async def test_sam_segment_with_prompt_success_with_points(host, image_bytes):
    result = await host.sam_segment_with_prompt(
        image_bytes=image_bytes, point_coords=[(10, 10)], point_labels=[1],
    )

    assert len(result["segments"]) == 1
    assert "timestamp" in result


async def test_sam_segment_with_prompt_success_with_bbox(host, image_bytes, bbox):
    result = await host.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)

    assert len(result["segments"]) == 1
    host.validator.validate_bbox.assert_called_once_with(bbox)


async def test_sam_segment_with_prompt_missing_prompts_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least one of: point_coords or bbox"):
        await host.sam_segment_with_prompt(image_bytes=image_bytes)

    host.sam_lama_mode.segment_with_prompt.assert_not_called()


async def test_sam_segment_with_prompt_mismatched_points_and_labels_raises(host, image_bytes):
    with pytest.raises(ValueError, match="point_coords and point_labels must have the same length"):
        await host.sam_segment_with_prompt(
            image_bytes=image_bytes, point_coords=[(1, 1), (2, 2)], point_labels=[1],
        )


async def test_sam_segment_with_prompt_invalid_bbox_raises(host, image_bytes, bbox):
    host.validator.validate_bbox.side_effect = ValueError("bbox x1 must be < x2")

    with pytest.raises(ValueError, match="bbox x1 must be < x2"):
        await host.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)

    host.sam_lama_mode.segment_with_prompt.assert_not_called()


async def test_sam_segment_with_prompt_calls_mode_with_params(host, image_bytes, bbox):
    points = [(5, 5)]
    labels = [1]

    await host.sam_segment_with_prompt(
        image_bytes=image_bytes, point_coords=points, point_labels=labels, bbox=bbox,
    )

    host.sam_lama_mode.segment_with_prompt.assert_called_once_with(
        image_bytes=image_bytes, point_coords=points, point_labels=labels, bbox=bbox,
        multimask_output=None,
    )


async def test_sam_segment_with_prompt_tracker_called_when_enabled(host, image_bytes, bbox):
    await host.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox, track_metrics=True)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_segment_prompt"
    assert payload["has_bbox"] is True
    assert payload["has_points"] is False


async def test_sam_segment_with_prompt_tracker_not_called_when_disabled(host, image_bytes, bbox):
    await host.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox, track_metrics=False)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_segment_with_prompt_propagates_mode_exception(host, image_bytes, bbox):
    host.sam_lama_mode.segment_with_prompt = AsyncMock(side_effect=RuntimeError("prompt segmentation failed"))

    with pytest.raises(RuntimeError, match="prompt segmentation failed"):
        await host.sam_segment_with_prompt(image_bytes=image_bytes, bbox=bbox)


async def test_sam_segment_by_polygon_success(host, image_bytes, polygon_points):
    result = await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points)

    assert len(result["segments"]) == 1
    assert result["image_size"] == (640, 480)
    assert "timestamp" in result


async def test_sam_segment_by_polygon_validates_image_bytes(host, image_bytes, polygon_points):
    await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points)

    host.validator.validate_image_bytes.assert_called_once_with(image_bytes)


async def test_sam_segment_by_polygon_calls_mode_with_params(host, image_bytes, polygon_points):
    await host.sam_segment_by_polygon(
        image_bytes=image_bytes, points=polygon_points, smooth=False, smoothing_factor=1.5, feather_px=3,
    )

    host.sam_lama_mode.segment_by_polygon.assert_called_once_with(
        image_bytes=image_bytes, points=polygon_points, smooth=False, smoothing_factor=1.5, feather_px=3,
    )


async def test_sam_segment_by_polygon_uses_default_params(host, image_bytes, polygon_points):
    await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points)

    host.sam_lama_mode.segment_by_polygon.assert_called_once_with(
        image_bytes=image_bytes, points=polygon_points, smooth=True, smoothing_factor=0.0, feather_px=0,
    )


async def test_sam_segment_by_polygon_too_few_points_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least 3 points"):
        await host.sam_segment_by_polygon(image_bytes=image_bytes, points=[(1, 1), (2, 2)])

    host.sam_lama_mode.segment_by_polygon.assert_not_called()


async def test_sam_segment_by_polygon_none_points_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least 3 points"):
        await host.sam_segment_by_polygon(image_bytes=image_bytes, points=None)

    host.sam_lama_mode.segment_by_polygon.assert_not_called()


async def test_sam_segment_by_polygon_empty_points_raises(host, image_bytes):
    with pytest.raises(ValueError, match="Provide at least 3 points"):
        await host.sam_segment_by_polygon(image_bytes=image_bytes, points=[])

    host.sam_lama_mode.segment_by_polygon.assert_not_called()


async def test_sam_segment_by_polygon_tracker_called_when_enabled(host, image_bytes, polygon_points):
    await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points, track_metrics=True)

    payload = host.tracker.log_metrics.call_args.args[0]
    assert payload["operation"] == "sam_segment_polygon"
    assert payload["num_points"] == len(polygon_points)


async def test_sam_segment_by_polygon_tracker_not_called_when_disabled(host, image_bytes, polygon_points):
    await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points, track_metrics=False)

    host.tracker.log_metrics.assert_not_called()


async def test_sam_segment_by_polygon_invalid_image_raises(host, image_bytes, polygon_points):
    host.validator.validate_image_bytes.side_effect = ValueError("Invalid image bytes")

    with pytest.raises(ValueError, match="Invalid image bytes"):
        await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points)

    host.sam_lama_mode.segment_by_polygon.assert_not_called()


async def test_sam_segment_by_polygon_propagates_mode_exception(host, image_bytes, polygon_points):
    host.sam_lama_mode.segment_by_polygon = AsyncMock(side_effect=RuntimeError("polygon segmentation failed"))

    with pytest.raises(RuntimeError, match="polygon segmentation failed"):
        await host.sam_segment_by_polygon(image_bytes=image_bytes, points=polygon_points)