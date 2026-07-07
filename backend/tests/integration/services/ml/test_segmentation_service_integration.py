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


def _segment_result():
    return {
        "segments": [
            {"mask_id": 1, "bbox_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "area": 100, "stability_score": 0.95, "mask_bytes": b"mask-bytes"}
        ],
        "metrics": {"latency_ms": 10},
        "image_size": (100, 100),
    }


class TestSegmentObjects:
    @pytest.mark.asyncio
    async def test_success_caches_segments_and_strips_mask_bytes(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_segments = AsyncMock()
        # segment_objects overwrites mask_id/bbox_id in-place via enumerate(), so we
        # capture the same object the pipeline returns and assert against it after
        # the call rather than a freshly-built dict (which would still have the
        # pre-mutation mask_id/bbox_id values).
        segment_result = _segment_result()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value=segment_result)

        result = await segmentation_service.segment_objects(sample_image.id, sample_user.id)

        assert "mask_bytes" not in result["segments"][0]
        assert result["segments"][0]["mask_id"] == 0
        assert result["segments"][0]["bbox_id"] == 0
        mock_redis_cache.cache_segments.assert_awaited_once_with(
            image_id=sample_image.id, segments=segment_result["segments"], ttl=7200
        )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, segmentation_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_objects(sample_image.id, sample_image.user_id + 1)

    @pytest.mark.asyncio
    async def test_propagates_pipeline_exception(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("segmentation failed"))

        with pytest.raises(RuntimeError, match="segmentation failed"):
            await segmentation_service.segment_objects(sample_image.id, sample_user.id)


class TestSegmentWithPrompt:
    @pytest.mark.asyncio
    async def test_success_with_bbox_prompt(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        # No segments cached yet for this image, so _next_mask_offset() and the
        # "existing" lookup both need an explicit falsy return value — an
        # unconfigured AsyncMock's default return is a truthy, non-empty-looking
        # MagicMock and breaks max()/offset logic.
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value=_segment_result())

        bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        result = await segmentation_service.segment_with_prompt(sample_image.id, sample_user.id, bbox=bbox)

        call_kwargs = mock_pipeline.sam_segment_with_prompt.call_args.kwargs
        assert call_kwargs["bbox"] == bbox
        assert "mask_bytes" not in result["segments"][0]

    @pytest.mark.asyncio
    async def test_success_with_point_prompt(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value=_segment_result())

        await segmentation_service.segment_with_prompt(
            sample_image.id, sample_user.id, point_coords=[(10, 10)], point_labels=[1]
        )

        call_kwargs = mock_pipeline.sam_segment_with_prompt.call_args.kwargs
        assert call_kwargs["point_coords"] == [(10, 10)]
        assert call_kwargs["point_labels"] == [1]

    @pytest.mark.asyncio
    async def test_cache_updated_on_success(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)
        mock_redis_cache.cache_segments = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value=_segment_result())

        await segmentation_service.segment_with_prompt(sample_image.id, sample_user.id, point_coords=[(1, 1)], point_labels=[1])

        mock_redis_cache.cache_segments.assert_awaited_once()


class TestSamRemoveObject:
    @pytest.mark.asyncio
    async def test_success_uploads_and_returns_result(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(
            return_value={"result_bytes": b"removed", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        result = await segmentation_service.sam_remove_object(sample_image.id, 1, sample_user.id)

        assert result["result_url"] == "s3://bucket/r.jpg"

    @pytest.mark.asyncio
    async def test_raises_when_cached_segment_missing(self, segmentation_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await segmentation_service.sam_remove_object(sample_image.id, 1, sample_user.id)

    @pytest.mark.asyncio
    async def test_pushes_undo_state(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(
            return_value={"result_bytes": b"removed", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await segmentation_service.sam_remove_object(sample_image.id, 1, sample_user.id)

        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_current_state(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(
            return_value={"result_bytes": b"removed", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await segmentation_service.sam_remove_object(sample_image.id, 1, sample_user.id)

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id, image_data=b"removed", suffix="current_state", ttl=7200
        )


class TestSamReplaceObject:
    @pytest.mark.asyncio
    async def test_success_passes_replacement_bytes(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(
            return_value={"result_bytes": b"replaced", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        result = await segmentation_service.sam_replace_object(
            sample_image.id, 1, b"replacement-bytes", sample_user.id
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        call_kwargs = mock_pipeline.sam_replace_object.call_args.kwargs
        assert call_kwargs["replacement_image_bytes"] == b"replacement-bytes"

    @pytest.mark.asyncio
    async def test_cache_updated_on_success(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(
            return_value={"result_bytes": b"replaced", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await segmentation_service.sam_replace_object(sample_image.id, 1, b"replacement-bytes", sample_user.id)

        mock_redis_cache.cache_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_uploads_to_s3(
        self, segmentation_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"mask"}]
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(
            return_value={"result_bytes": b"replaced", "metrics": {}, "timestamp": "ts"}
        )
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await segmentation_service.sam_replace_object(sample_image.id, 1, b"replacement-bytes", sample_user.id)

        mock_s3_storage.upload_bytes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_segment_missing(self, segmentation_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="No segments found"):
            await segmentation_service.sam_replace_object(sample_image.id, 1, b"bytes", sample_user.id)