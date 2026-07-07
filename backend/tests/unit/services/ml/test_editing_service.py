from contextlib import redirect_stderr

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from app.services.ml.editing_service import EditingService
from app.db.models.image import Image


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.delete = AsyncMock(return_value=None)
    db.commit = AsyncMock(return_value=None)
    return db


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
def mock_redis_assets():
    ra = AsyncMock()
    ra.list_assets = AsyncMock(return_value=[])
    ra.get_thumbnail = AsyncMock(return_value=None)
    ra.get_asset = AsyncMock(return_value=None)
    ra.rename_asset = AsyncMock(return_value=None)
    ra.delete_asset = AsyncMock(return_value=False)
    return ra

@pytest.fixture
def mock_image_repo():
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def mock_detection_repo():
    repo = AsyncMock()
    repo.get_by_image = AsyncMock(return_value=[])
    repo.delete_by_image = AsyncMock(return_value=None)
    return repo


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
    mock_db, mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_detection_repo, mock_pipeline,
):
    return EditingService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


def make_detection(bbox_id):
    det = MagicMock()
    det.bbox_id = bbox_id
    det.x1, det.y1, det.x2, det.y2 = 0, 0, 10, 10
    return det

class TestRemoveObject:
    async def test_remove_object_success(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_redis_history, mock_pipeline, mock_redis_storage, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {"latency_ms": 30}, "timestamp": "t",
        })

        result = await service.remove_object(image_id=1, bbox_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.remove_object.assert_awaited_once()
        mock_redis_storage.cache_image.assert_awaited_once()
        mock_detection_repo.delete_by_image.assert_awaited_once_with(1)
        mock_redis_storage.delete.assert_awaited_once_with("image:1:detections")
        mock_s3.upload_bytes.assert_awaited_once()

        assert result["result_url"] == "s3://bucket/path.jpg"
        assert "metrics" in result
        assert "timestamp" in result and result["timestamp"]

    async def test_remove_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_remove_object_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_remove_object_detection_not_found(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(2)])

        with pytest.raises(ValueError, match="bbox_id=1 not found"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_remove_object_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_redis_history, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(side_effect=RuntimeError("lama failure"))

        with pytest.raises(RuntimeError, match="lama failure"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()  # pushed before failure

    async def test_remove_object_s3_exception(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })
        mock_s3.upload_bytes = AsyncMock(side_effect=IOError("s3 unreachable"))

        with pytest.raises(IOError, match="s3 unreachable"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_remove_object_boundary_expand_mask_zero(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.remove_object(image_id=1, bbox_id=1, user_id=42, expand_mask_pixels=0)

        _, kwargs = mock_pipeline.remove_object.call_args
        assert kwargs["expand_mask_pixels"] == 0

    async def test_remove_object_optional_params_forwarded(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.remove_object(
            image_id=1, bbox_id=1, user_id=42,
            ldm_steps=50, ldm_sampler="ddim", hd_strategy="RESIZE",
            use_edge_blending=False,
        )

        _, kwargs = mock_pipeline.remove_object.call_args
        assert kwargs["ldm_steps"] == 50
        assert kwargs["ldm_sampler"] == "ddim"
        assert kwargs["hd_strategy"] == "RESIZE"
        assert kwargs["use_edge_blending"] is False

class TestReplaceObject:
    async def test_replace_object_success(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_redis_history, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(3)])
        mock_pipeline.replace_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.replace_object(
            image_id=1, bbox_id=3, replace_image_bytes=b"new-obj", user_id=42,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.replace_object.assert_awaited_once()
        _, kwargs = mock_pipeline.replace_object.call_args
        assert kwargs["replacement_image_bytes"] == b"new-obj"
        mock_redis_storage.cache_image.assert_awaited_once()
        assert "timestamp" in result and result["timestamp"]

    async def test_replace_object_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42
            )

    async def test_replace_object_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42
            )

    async def test_replace_object_detection_not_found(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="bbox_id=1 not found"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42
            )

    async def test_replace_object_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.replace_object = AsyncMock(side_effect=RuntimeError("crash"))

        with pytest.raises(RuntimeError, match="crash"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42
            )

    async def test_replace_object_default_color_matching_true(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.replace_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.replace_object(
            image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42
        )

        _, kwargs = mock_pipeline.replace_object.call_args
        assert kwargs["use_color_matching"] is False

class TestRemoveMultipleObjects:
    async def test_remove_multiple_objects_success(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_redis_history, mock_pipeline, mock_redis_storage, mock_db,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        dets = [make_detection(1), make_detection(2), make_detection(3)]
        mock_detection_repo.get_by_image = AsyncMock(return_value=dets)
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.remove_multiple_objects(
            image_id=1, bbox_ids=[1, 2], user_id=42
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.remove_multiple_objects.assert_awaited_once()
        _, kwargs = mock_pipeline.remove_multiple_objects.call_args
        assert len(kwargs["selected_bboxes"]) == 2
        assert len(kwargs["scene_bboxes"]) == 1  # remaining detection (bbox_id=3)

        assert mock_db.delete.await_count == 2
        mock_db.commit.assert_awaited_once()
        mock_redis_storage.delete.assert_awaited_once_with("image:1:detections")
        assert "timestamp" in result and result["timestamp"]

    async def test_remove_multiple_objects_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

    async def test_remove_multiple_objects_unauthorized(
        self, service, mock_image_repo, sample_image,
    ):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

    async def test_remove_multiple_objects_empty_bbox_ids_raises(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])

        with pytest.raises(ValueError, match="No valid detections"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[], user_id=42)

    async def test_remove_multiple_objects_no_matching_detections(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])

        with pytest.raises(ValueError, match="No valid detections found for bbox_ids"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[99, 100], user_id=42)

    async def test_remove_multiple_objects_all_selected_scene_bboxes_empty(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        dets = [make_detection(1), make_detection(2)]
        mock_detection_repo.get_by_image = AsyncMock(return_value=dets)
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.remove_multiple_objects(image_id=1, bbox_ids=[1, 2], user_id=42)

        _, kwargs = mock_pipeline.remove_multiple_objects.call_args
        assert kwargs["scene_bboxes"] is None  # falsy list converted to None

    async def test_remove_multiple_objects_pipeline_exception(
        self, service, mock_image_repo, sample_image, mock_detection_repo, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_multiple_objects = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError, match="fail"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

    async def test_remove_multiple_objects_db_commit_exception(
        self, service, mock_image_repo, sample_image, mock_detection_repo,
        mock_pipeline, mock_db,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_detection_repo.get_by_image = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })
        mock_db.commit = AsyncMock(side_effect=RuntimeError("db commit failed"))

        with pytest.raises(RuntimeError, match="db commit failed"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

class TestUndoRedo:
    async def test_undo_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"current")
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"previous", "label": "remove bbox_id=1"}
        )
        mock_redis_history.get_history_labels = AsyncMock(return_value=["op1"])

        result = await service.undo(image_id=1, user_id=42)

        mock_redis_history.push_redo_state.assert_awaited_once_with(
            1, b"current", label="redo"
        )
        mock_redis_storage.cache_image.assert_awaited_once()
        assert result["label"] == "remove bbox_id=1"
        assert result["history"] == ["op1"]
        assert "presigned_url" in result

    async def test_undo_nothing_to_undo(
        self, service, mock_image_repo, sample_image, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.pop_undo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to undo"):
            await service.undo(image_id=1, user_id=42)

    async def test_undo_no_current_state_skips_redo_push(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.pop_undo_state = AsyncMock(
            return_value={"bytes": b"previous", "label": "label"}
        )

        await service.undo(image_id=1, user_id=42)

        mock_redis_history.push_redo_state.assert_not_called()

    async def test_undo_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.undo(image_id=1, user_id=42)

    async def test_undo_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.undo(image_id=1, user_id=42)

    async def test_undo_redis_exception(
        self, service, mock_image_repo, sample_image, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.pop_undo_state = AsyncMock(side_effect=ConnectionError("redis down"))

        with pytest.raises(ConnectionError, match="redis down"):
            await service.undo(image_id=1, user_id=42)

    async def test_redo_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"current")
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next", "label": "redo_label"}
        )
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])

        result = await service.redo(image_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once_with(
            1, b"current", label="redo_checkpoint"
        )
        mock_redis_storage.cache_image.assert_awaited_once()
        assert result["label"] == "redo_label"

    async def test_redo_nothing_to_redo(
        self, service, mock_image_repo, sample_image, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.pop_redo_state = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Nothing to redo"):
            await service.redo(image_id=1, user_id=42)

    async def test_redo_no_current_state_skips_undo_push(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_redis_history.pop_redo_state = AsyncMock(
            return_value={"bytes": b"next", "label": "x"}
        )

        await service.redo(image_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_not_called()

    async def test_get_history_success(
        self, service, mock_image_repo, sample_image, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.get_history_labels = AsyncMock(return_value=["a", "b"])

        result = await service.get_history(image_id=1, user_id=42)

        assert result == {"history": ["a", "b"]}

    async def test_get_history_empty(
        self, service, mock_image_repo, sample_image, mock_redis_history,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_history.get_history_labels = AsyncMock(return_value=[])

        result = await service.get_history(image_id=1, user_id=42)

        assert result == {"history": []}

    async def test_get_history_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_history(image_id=1, user_id=42)

class TestSaveResult:
    async def test_save_result_success(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"processed")

        saved_image = MagicMock(spec=Image)
        mock_image_repo.create = AsyncMock(return_value=saved_image)
        mock_image_repo.update = AsyncMock()

        result = await service.save_result(image_id=1, user_id=42)

        mock_s3.upload_bytes.assert_awaited_once()
        mock_image_repo.create.assert_awaited_once()
        assert saved_image.status == "processed"
        mock_image_repo.update.assert_awaited_once_with(saved_image)
        assert result is saved_image

    async def test_save_result_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.save_result(image_id=1, user_id=42)

    async def test_save_result_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.save_result(image_id=1, user_id=42)

    async def test_save_result_no_processed_result(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No processed result to save"):
            await service.save_result(image_id=1, user_id=42)

    async def test_save_result_s3_exception(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"processed")
        mock_s3.upload_bytes = AsyncMock(side_effect=IOError("s3 down"))

        with pytest.raises(IOError, match="s3 down"):
            await service.save_result(image_id=1, user_id=42)

    async def test_save_result_repository_exception(
        self, service, mock_image_repo, sample_image, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"processed")
        mock_image_repo.create = AsyncMock(side_effect=RuntimeError("db error"))

        with pytest.raises(RuntimeError, match="db error"):
            await service.save_result(image_id=1, user_id=42)


class TestResetCurrentState:
    async def test_reset_current_state_success(
        self, service, mock_redis_storage, mock_redis_history,
    ):
        await service.reset_current_state(image_id=1)

        mock_redis_storage.delete.assert_awaited_once_with("image:1:current_state")
        mock_redis_history.clear_history.assert_awaited_once_with(1)

    async def test_reset_current_state_redis_exception(
        self, service, mock_redis_storage,
    ):
        mock_redis_storage.delete = AsyncMock(side_effect=ConnectionError("redis down"))

        with pytest.raises(ConnectionError, match="redis down"):
            await service.reset_current_state(image_id=1)

    async def test_reset_current_state_history_exception(
        self, service, mock_redis_history,
    ):
        mock_redis_history.clear_history = AsyncMock(side_effect=RuntimeError("history error"))

        with pytest.raises(RuntimeError, match="history error"):
            await service.reset_current_state(image_id=1)