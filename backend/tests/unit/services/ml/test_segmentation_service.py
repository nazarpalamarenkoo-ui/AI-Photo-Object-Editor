import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.segmentation_service import SegmentationService
from app.db.models.image import Image


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.download = AsyncMock(return_value=b"original-bytes")
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/path.jpg")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.example/path.jpg")
    return s3


@pytest.fixture
def mock_redis_storage():
    redis_storage = AsyncMock()
    redis_storage.get_cache_image = AsyncMock(return_value=None)
    redis_storage.cache_image = AsyncMock(return_value=None)
    redis_storage.delete = AsyncMock(return_value=None)
    redis_storage.cache_segments = AsyncMock(return_value=None)
    redis_storage.get_cached_segments = AsyncMock(return_value=None)
    return redis_storage


@pytest.fixture
def mock_redis_history():
    history = AsyncMock()
    history.push_undo_state = AsyncMock(return_value=None)
    history.pop_undo_state = AsyncMock(return_value=None)
    history.push_redo_state = AsyncMock(return_value=None)
    history.pop_redo_state = AsyncMock(return_value=None)
    history.get_history_labels = AsyncMock(return_value=[])
    history.clear_history = AsyncMock(return_value=None)
    return history


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock(spec=Image)
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    image.filename = "original.jpg"
    return image


@pytest.fixture
def service(
    mock_db, mock_s3, mock_redis_storage, mock_redis_history,
    mock_image_repo, mock_detection_repo, mock_pipeline,
):
    return SegmentationService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


def make_segment(mask_id=1, area=1000):
    return {
        "mask_id": mask_id,
        "bbox_id": mask_id,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "area": area,
        "stability_score": 0.95,
        "mask_bytes": b"mask-bytes",
    }

class TestSegmentObjects:
    async def test_segment_objects_success(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        segments = [make_segment(1), make_segment(2)]
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": segments,
            "metrics": {"latency_ms": 200},
            "image_size": (640, 480),
        })

        result = await service.segment_objects(image_id=1, user_id=42, min_area=500, max_segments=50)

        mock_pipeline.sam_segment_objects.assert_awaited_once()
        _, kwargs = mock_pipeline.sam_segment_objects.call_args
        assert kwargs["min_area"] == 500
        assert kwargs["max_segments"] == 50

        mock_redis_storage.cache_segments.assert_awaited_once_with(
            image_id=1, segments=segments, ttl=7200
        )

        assert all("mask_bytes" not in seg for seg in result["segments"])
        assert len(result["segments"]) == 2
        assert result["image_size"] == (640, 480)
        assert "metrics" in result
        assert "timestamp" in result and result["timestamp"]

    async def test_segment_objects_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_objects(image_id=999, user_id=42)

    async def test_segment_objects_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 1
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_objects(image_id=1, user_id=42)

    async def test_segment_objects_empty_segments(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_objects(image_id=1, user_id=42)

        assert result["segments"] == []

    async def test_segment_objects_boundary_min_area_zero(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })

        await service.segment_objects(image_id=1, user_id=42, min_area=0)

        _, kwargs = mock_pipeline.sam_segment_objects.call_args
        assert kwargs["min_area"] == 0

    async def test_segment_objects_pipeline_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("sam crashed"))

        with pytest.raises(RuntimeError, match="sam crashed"):
            await service.segment_objects(image_id=1, user_id=42)

    async def test_segment_objects_redis_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [make_segment()], "metrics": {}, "image_size": (1, 1),
        })
        mock_redis_storage.cache_segments = AsyncMock(side_effect=ConnectionError("redis down"))

        with pytest.raises(ConnectionError, match="redis down"):
            await service.segment_objects(image_id=1, user_id=42)

class TestSegmentWithPrompt:
    async def test_segment_with_prompt_success_points(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        segments = [make_segment(1)]
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value={
            "segments": segments, "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_with_prompt(
            image_id=1, user_id=42, point_coords=[(10, 10)], point_labels=[1],
        )

        _, kwargs = mock_pipeline.sam_segment_with_prompt.call_args
        assert kwargs["point_coords"] == [(10, 10)]
        assert kwargs["point_labels"] == [1]
        mock_redis_storage.cache_segments.assert_awaited_once()
        assert "mask_bytes" not in result["segments"][0]

    async def test_segment_with_prompt_success_bbox(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

        await service.segment_with_prompt(image_id=1, user_id=42, bbox=bbox)

        _, kwargs = mock_pipeline.sam_segment_with_prompt.call_args
        assert kwargs["bbox"] == bbox

    async def test_segment_with_prompt_no_prompt_params_passes_none(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_with_prompt = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })

        await service.segment_with_prompt(image_id=1, user_id=42)

        _, kwargs = mock_pipeline.sam_segment_with_prompt.call_args
        assert kwargs["point_coords"] is None
        assert kwargs["point_labels"] is None
        assert kwargs["bbox"] is None

    async def test_segment_with_prompt_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_with_prompt(image_id=1, user_id=42)

    async def test_segment_with_prompt_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_with_prompt(image_id=1, user_id=42)

    async def test_segment_with_prompt_pipeline_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_with_prompt = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await service.segment_with_prompt(image_id=1, user_id=42)


class TestSamRemoveObject:
    async def test_sam_remove_object_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_redis_history, mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(5)])
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {"latency_ms": 50}, "timestamp": "t",
        })

        result = await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

        assert mock_redis_history.push_undo_state.await_count == 1
        undo_call_order = mock_redis_history.method_calls
        pipeline_call_order = mock_pipeline.method_calls
        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_remove_object.assert_awaited_once()

        mock_redis_storage.cache_image.assert_awaited_once()
        mock_s3.upload_bytes.assert_awaited_once()
        mock_s3.get_presigned_url.assert_awaited_once()

        assert result["result_url"] == "s3://bucket/path.jpg"
        assert result["presigned_url"] == "https://presigned.example/path.jpg"
        assert "metrics" in result
        assert "timestamp" in result and result["timestamp"]

    async def test_sam_remove_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.sam_remove_object(image_id=1, mask_id=1, user_id=42)

    async def test_sam_remove_object_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.sam_remove_object(image_id=1, mask_id=1, user_id=42)

    async def test_sam_remove_object_no_segments_cached(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await service.sam_remove_object(image_id=1, mask_id=1, user_id=42)

    async def test_sam_remove_object_mask_id_not_found(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(1)])

        with pytest.raises(ValueError, match="Segment with mask_id=999 not found"):
            await service.sam_remove_object(image_id=1, mask_id=999, user_id=42)

    async def test_sam_remove_object_pipeline_exception_after_undo_pushed(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_redis_history, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(5)])
        mock_pipeline.sam_remove_object = AsyncMock(side_effect=RuntimeError("lama crashed"))

        with pytest.raises(RuntimeError, match="lama crashed"):
            await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()

    async def test_sam_remove_object_s3_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(5)])
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })
        mock_s3.upload_bytes = AsyncMock(side_effect=IOError("s3 down"))

        with pytest.raises(IOError, match="s3 down"):
            await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

    async def test_sam_remove_object_boundary_expand_mask_zero(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(5)])
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        await service.sam_remove_object(image_id=1, mask_id=5, user_id=42, expand_mask_pixels=0)

        _, kwargs = mock_pipeline.sam_remove_object.call_args
        assert kwargs["expand_mask_pixels"] == 0

class TestSamReplaceObject:
    async def test_sam_replace_object_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
        mock_redis_history, mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(7)])
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.sam_replace_object(
            image_id=1, mask_id=7, replacement_image_bytes=b"replacement", user_id=42,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_replace_object.assert_awaited_once()
        _, kwargs = mock_pipeline.sam_replace_object.call_args
        assert kwargs["replacement_image_bytes"] == b"replacement"

        mock_redis_storage.cache_image.assert_awaited_once()
        assert result["result_url"] == "s3://bucket/path.jpg"
        assert "timestamp" in result and result["timestamp"]

    async def test_sam_replace_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.sam_replace_object(
                image_id=1, mask_id=1, replacement_image_bytes=b"x", user_id=42
            )

    async def test_sam_replace_object_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.sam_replace_object(
                image_id=1, mask_id=1, replacement_image_bytes=b"x", user_id=42
            )

    async def test_sam_replace_object_segment_not_cached(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await service.sam_replace_object(
                image_id=1, mask_id=1, replacement_image_bytes=b"x", user_id=42
            )

    async def test_sam_replace_object_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(7)])
        mock_pipeline.sam_replace_object = AsyncMock(side_effect=RuntimeError("inpaint failed"))

        with pytest.raises(RuntimeError, match="inpaint failed"):
            await service.sam_replace_object(
                image_id=1, mask_id=7, replacement_image_bytes=b"x", user_id=42
            )

    async def test_sam_replace_object_default_edge_blending_false(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(7)])
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.sam_replace_object(
            image_id=1, mask_id=7, replacement_image_bytes=b"x", user_id=42
        )

        _, kwargs = mock_pipeline.sam_replace_object.call_args
        assert kwargs["use_edge_blending"] is False