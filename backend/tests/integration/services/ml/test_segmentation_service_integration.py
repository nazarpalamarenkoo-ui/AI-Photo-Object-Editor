import pytest
from unittest.mock import AsyncMock

from app.services.ml.segmentation_service import SegmentationService

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def segmentation_service(db_session, mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets, image_repo, detection_repo, mock_pipeline):
    return SegmentationService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        detection_repo=detection_repo,
        pipeline=mock_pipeline,
    )


def _polygon_segment_result(mask_id=999):
    return {
        "segments": [
            {"mask_id": mask_id, "bbox_id": mask_id, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
             "area": 100, "stability_score": 0.9, "mask_bytes": b"mask-bytes"}
        ],
        "metrics": {"latency_ms": 12},
        "image_size": (100, 100),
    }


def _detection(x1=0, y1=0, x2=10, y2=10):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.9, "label": "cat"}


def _segment(mask_id, bbox, source=None):
    seg = {
        "mask_id": mask_id, "bbox_id": mask_id, "bbox": bbox,
        "area": 100, "stability_score": 0.9, "mask_bytes": b"mask-bytes",
    }
    if source:
        seg["source"] = source
    return seg


class TestSegmentByPolygon:
    @pytest.mark.asyncio
    async def test_success_caches_segments_and_strips_mask_bytes(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value=_polygon_segment_result())

        result = await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)],
        )

        assert "mask_bytes" not in result["segments"][0]
        # segment_by_polygon assigns the raw offset to every segment (no
        # enumerate), so with nothing cached yet the offset is 0.
        assert result["segments"][0]["mask_id"] == 0
        assert result["segments"][0]["bbox_id"] == 0
        mock_redis_cache.cache_segments.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, segmentation_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_by_polygon(
                sample_image.id, sample_image.user_id + 1, points=[(0, 0), (10, 0), (5, 10)],
            )

    @pytest.mark.asyncio
    async def test_propagates_pipeline_exception(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_segment_by_polygon = AsyncMock(side_effect=RuntimeError("polygon failed"))

        with pytest.raises(RuntimeError, match="polygon failed"):
            await segmentation_service.segment_by_polygon(
                sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)],
            )

    @pytest.mark.asyncio
    async def test_appends_to_existing_cached_segments(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        existing = [{"mask_id": 3, "bbox_id": 3, "bbox": {}, "area": 1, "stability_score": None, "mask_bytes": b"x"}]
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=existing)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value=_polygon_segment_result())

        await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)],
        )

        call_kwargs = mock_redis_cache.cache_segments.call_args.kwargs
        assert len(call_kwargs["segments"]) == 2
        # offset = max(3) + 1 = 4, applied to the new segment
        new_seg = [s for s in call_kwargs["segments"] if s is not existing[0]][0]
        assert new_seg["mask_id"] == 4

    @pytest.mark.asyncio
    async def test_passes_polygon_params_to_pipeline(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value=_polygon_segment_result())
        points = [(0, 0), (10, 0), (5, 10)]

        await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=points,
            smooth=False, smoothing_factor=0.4, feather_px=2,
        )

        call_kwargs = mock_pipeline.sam_segment_by_polygon.call_args.kwargs
        assert call_kwargs["points"] == points
        assert call_kwargs["smooth"] is False
        assert call_kwargs["smoothing_factor"] == 0.4
        assert call_kwargs["feather_px"] == 2


class TestSegmentHybrid:
    @pytest.mark.asyncio
    async def test_success_combines_yolo_and_fallback_sources(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": [_detection(0, 0, 10, 10)]})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [_segment(100, {"x1": 0, "y1": 0, "x2": 10, "y2": 10})],
        })
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(200, {"x1": 50, "y1": 50, "x2": 60, "y2": 60})],
            "image_size": (300, 300),
        })

        result = await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

        sources = {seg["source"] for seg in result["segments"]}
        assert sources == {"yolo", "sam_auto"}
        assert len(result["segments"]) == 2
        assert all("mask_bytes" not in seg for seg in result["segments"])
        assert result["image_size"] == (300, 300)
        assert "metrics" not in result
        mock_redis_cache.cache_segments.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_overlapping_fallback_segment_dropped(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": [_detection(0, 0, 10, 10)]})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [_segment(1, {"x1": 0, "y1": 0, "x2": 10, "y2": 10})],
        })
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(2, {"x1": 0, "y1": 0, "x2": 10, "y2": 10})],  # identical bbox
            "image_size": (100, 100),
        })

        result = await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id, overlap_iou_thresh=0.5,
        )

        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "yolo"

    @pytest.mark.asyncio
    async def test_no_yolo_detections_skips_batch_call(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(1, {"x1": 0, "y1": 0, "x2": 10, "y2": 10})],
            "image_size": (50, 50),
        })

        result = await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

        mock_pipeline.sam_segment_with_prompts_batch.assert_not_called()
        assert result["segments"][0]["source"] == "sam_auto"

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, segmentation_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_hybrid(sample_image.id, sample_image.user_id + 1)

    @pytest.mark.asyncio
    async def test_propagates_detect_objects_exception(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("yolo crashed"))

        with pytest.raises(RuntimeError, match="yolo crashed"):
            await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_propagates_fallback_exception(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("fallback crashed"))

        with pytest.raises(RuntimeError, match="fallback crashed"):
            await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_passes_yolo_and_fallback_params_to_pipeline(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={"segments": [], "image_size": (1, 1)})

        await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id, yolo_conf_threshold=0.7, yolo_classes=["dog"],
            fallback_min_area=900, fallback_max_segments=20,
        )

        detect_kwargs = mock_pipeline.detect_objects.call_args.kwargs
        assert detect_kwargs["conf_threshold"] == 0.7
        assert detect_kwargs["classes"] == ["dog"]
        fallback_kwargs = mock_pipeline.sam_segment_objects.call_args.kwargs
        assert fallback_kwargs["min_area"] == 900
        assert fallback_kwargs["max_segments"] == 20