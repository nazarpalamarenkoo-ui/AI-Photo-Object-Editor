import pytest
from unittest.mock import AsyncMock

from app.services.ml.editing_service import EditingService

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
def editing_service(db_session, mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets, image_repo, detection_repo, mock_pipeline):
    return EditingService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        detection_repo=detection_repo,
        pipeline=mock_pipeline,
    )


@pytest.fixture
async def sample_detection(detection_repo, sample_image):
    from app.db.models.detection import Detection

    det = Detection(
        image_id=sample_image.id, bbox_id=1, detected_class="person",
        confidence=0.9, x1=0, y1=0, x2=10, y2=10,
    )
    await detection_repo.create_many([det])
    return det


def _pipeline_result():
    return {"result_bytes": b"edited-bytes", "metrics": {"latency_ms": 5}, "timestamp": "2024-01-01T00:00:00"}


class TestRemoveObject:
    @pytest.mark.asyncio
    async def test_success_uploads_and_returns_result(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        result = await editing_service.remove_object(sample_image.id, sample_detection.bbox_id, sample_user.id)

        assert result["result_url"] == "s3://bucket/r.jpg"
        assert result["presigned_url"] == "https://url"

    @pytest.mark.asyncio
    async def test_raises_when_detection_missing(self, editing_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="not found"):
            await editing_service.remove_object(sample_image.id, 9999, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, editing_service, sample_image, sample_detection):
        with pytest.raises(ValueError, match="Unauthorized"):
            await editing_service.remove_object(sample_image.id, sample_detection.bbox_id, sample_image.user_id + 1)

    @pytest.mark.asyncio
    async def test_pushes_undo_state_before_editing(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_redis_history, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.remove_object(sample_image.id, sample_detection.bbox_id, sample_user.id)

        mock_redis_history.push_undo_state.assert_awaited_once()
        args = mock_redis_history.push_undo_state.call_args
        assert args[0][0] == sample_image.id
        assert args[0][1] == b"image-bytes"

    @pytest.mark.asyncio
    async def test_updates_current_state_in_redis(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.remove_object(sample_image.id, sample_detection.bbox_id, sample_user.id)

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id, image_data=b"edited-bytes", suffix="current_state", ttl=7200
        )

    @pytest.mark.asyncio
    async def test_removes_stale_detections_and_invalidates_cache(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.remove_object(sample_image.id, sample_detection.bbox_id, sample_user.id)

        remaining = await editing_service.detection_repo.get_by_image(sample_image.id)
        assert remaining == []
        mock_redis_cache.delete.assert_awaited_once_with(f"image:{sample_image.id}:detections")


class TestReplaceObject:
    @pytest.mark.asyncio
    async def test_success_passes_replacement_bytes_to_pipeline(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        result = await editing_service.replace_object(
            sample_image.id, sample_detection.bbox_id, b"replacement-bytes", sample_user.id
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        call_kwargs = mock_pipeline.replace_object.call_args.kwargs
        assert call_kwargs["replacement_image_bytes"] == b"replacement-bytes"

    @pytest.mark.asyncio
    async def test_pushes_undo_state(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_redis_history, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.replace_object(
            sample_image.id, sample_detection.bbox_id, b"replacement-bytes", sample_user.id
        )

        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uploads_result_to_s3(
        self, editing_service, sample_image, sample_user, sample_detection, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.replace_object(
            sample_image.id, sample_detection.bbox_id, b"replacement-bytes", sample_user.id
        )

        mock_s3_storage.upload_bytes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_detection_missing(self, editing_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="not found"):
            await editing_service.replace_object(sample_image.id, 9999, b"bytes", sample_user.id)


class TestRemoveMultipleObjects:
    @pytest.fixture
    async def two_detections(self, detection_repo, sample_image):
        from app.db.models.detection import Detection

        dets = [
            Detection(image_id=sample_image.id, bbox_id=1, detected_class="person", confidence=0.9, x1=0, y1=0, x2=10, y2=10),
            Detection(image_id=sample_image.id, bbox_id=2, detected_class="car", confidence=0.8, x1=20, y1=20, x2=30, y2=30),
        ]
        await detection_repo.create_many(dets)
        return dets

    @pytest.mark.asyncio
    async def test_success_removes_selected_objects(
        self, editing_service, sample_image, sample_user, two_detections, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        result = await editing_service.remove_multiple_objects(sample_image.id, [1, 2], sample_user.id)

        assert result["result_url"] == "s3://bucket/r.jpg"
        remaining = await editing_service.detection_repo.get_by_image(sample_image.id)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_raises_for_invalid_bbox_ids(self, editing_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="No valid detections"):
            await editing_service.remove_multiple_objects(sample_image.id, [999], sample_user.id)

    @pytest.mark.asyncio
    async def test_partial_bbox_list_keeps_unselected_in_scene_bboxes(
        self, editing_service, sample_image, sample_user, two_detections, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.remove_multiple_objects(sample_image.id, [1], sample_user.id)

        call_kwargs = mock_pipeline.remove_multiple_objects.call_args.kwargs
        assert len(call_kwargs["selected_bboxes"]) == 1
        assert len(call_kwargs["scene_bboxes"]) == 1

    @pytest.mark.asyncio
    async def test_redis_detections_cache_invalidated(
        self, editing_service, sample_image, sample_user, two_detections, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/r.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://url")

        await editing_service.remove_multiple_objects(sample_image.id, [1, 2], sample_user.id)

        mock_redis_cache.delete.assert_awaited_once_with(f"image:{sample_image.id}:detections")


class TestUndo:
    @pytest.mark.asyncio
    async def test_success_restores_previous_state(
        self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_undo_state = AsyncMock(return_value={"bytes": b"prev-bytes", "label": "remove bbox_id=1"})
        mock_redis_history.push_redo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/undo.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://undo-url")

        result = await editing_service.undo(sample_image.id, sample_user.id)

        assert result["presigned_url"] == "https://undo-url"
        assert result["label"] == "remove bbox_id=1"
        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id, image_data=b"prev-bytes", suffix="current_state", ttl=7200
        )

    @pytest.mark.asyncio
    async def test_raises_when_nothing_to_undo(self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_history.pop_undo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to undo"):
            await editing_service.undo(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_pushes_redo_state_when_current_exists(
        self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_undo_state = AsyncMock(return_value={"bytes": b"prev-bytes", "label": "label"})
        mock_redis_history.push_redo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/undo.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://undo-url")

        await editing_service.undo(sample_image.id, sample_user.id)

        mock_redis_history.push_redo_state.assert_awaited_once_with(sample_image.id, b"current-bytes", label="redo")


class TestRedo:
    @pytest.mark.asyncio
    async def test_success_restores_next_state(
        self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_redo_state = AsyncMock(return_value={"bytes": b"next-bytes", "label": "redo"})
        mock_redis_history.push_undo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/redo.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://redo-url")

        result = await editing_service.redo(sample_image.id, sample_user.id)

        assert result["presigned_url"] == "https://redo-url"
        assert result["label"] == "redo"

    @pytest.mark.asyncio
    async def test_raises_when_nothing_to_redo(self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_history.pop_redo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to redo"):
            await editing_service.redo(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_pushes_undo_checkpoint_when_current_exists(
        self, editing_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"current-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_redis_history.pop_redo_state = AsyncMock(return_value={"bytes": b"next-bytes", "label": "redo"})
        mock_redis_history.push_undo_state = AsyncMock()
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/redo.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://redo-url")

        await editing_service.redo(sample_image.id, sample_user.id)

        mock_redis_history.push_undo_state.assert_awaited_once_with(
            sample_image.id, b"current-bytes", label="redo_checkpoint"
        )


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_existing_history(self, editing_service, sample_image, sample_user, mock_redis_history):
        mock_redis_history.get_history_labels = AsyncMock(return_value=["remove bbox_id=1", "replace bbox_id=2"])

        result = await editing_service.get_history(sample_image.id, sample_user.id)

        assert result["history"] == ["remove bbox_id=1", "replace bbox_id=2"]

    @pytest.mark.asyncio
    async def test_returns_empty_history(self, editing_service, sample_image, sample_user, mock_redis_history):
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])

        result = await editing_service.get_history(sample_image.id, sample_user.id)

        assert result["history"] == []


class TestSaveResult:
    @pytest.mark.asyncio
    async def test_success_creates_processed_image(
        self, editing_service, sample_image, sample_user, mock_redis_cache, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"processed-bytes")
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/saved.jpg")

        saved = await editing_service.save_result(sample_image.id, sample_user.id)

        assert saved.status == "processed"
        assert saved.user_id == sample_user.id
        assert saved.storage_path == "s3://bucket/saved.jpg"

    @pytest.mark.asyncio
    async def test_raises_when_no_current_state(self, editing_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No processed result"):
            await editing_service.save_result(sample_image.id, sample_user.id)


class TestResetCurrentState:
    @pytest.mark.asyncio
    async def test_clears_redis_state_and_history(self, editing_service, sample_image, mock_redis_cache, mock_redis_history):
        mock_redis_cache.delete = AsyncMock()
        mock_redis_history.clear_history = AsyncMock()

        await editing_service.reset_current_state(sample_image.id)

        mock_redis_cache.delete.assert_awaited_once_with(f"image:{sample_image.id}:current_state")
        mock_redis_history.clear_history.assert_awaited_once_with(sample_image.id)