import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.editing_service import EditingService

pytestmark = pytest.mark.unit


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
    return history


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    repo = AsyncMock()
    repo.create_next = AsyncMock()
    return repo


@pytest.fixture
def mock_image_content_repo():
    repo = AsyncMock()
    content = MagicMock()
    content.id = 200
    repo.get_or_create = AsyncMock(return_value=(content, True))
    return repo


@pytest.fixture
def mock_detection_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.soft_delete = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_segmentation_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.soft_delete = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_edit_history_repo():
    return AsyncMock()


@pytest.fixture
def mock_assets_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.content_id = 100
    version.storage_path = "raw/42/1/original.jpg"
    return version


@pytest.fixture
def new_version():
    version = MagicMock()
    version.id = 11
    version.content_id = 200
    return version


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline, sample_image, sample_version, new_version,
):
    mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
    mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
    mock_image_version_repo.create_next = AsyncMock(return_value=new_version)

    svc = EditingService(
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
        image_content_repo=mock_image_content_repo,
        detection_repo=mock_detection_repo,
        segmentation_repo=mock_segmentation_repo,
        edit_history_repo=mock_edit_history_repo,
        assets_repo=mock_assets_repo,
        pipeline=mock_pipeline,
    )
    svc._read_dimensions = MagicMock(return_value=(100, 100))
    return svc


def make_detection(bbox_id, content_id=100):
    det = MagicMock()
    det.bbox_id = bbox_id
    det.content_id = content_id
    det.x1, det.y1, det.x2, det.y2 = 0, 0, 10, 10
    return det


class TestRemoveObject:

    async def test_success(
        self, service, mock_detection_repo, mock_redis_history, mock_pipeline,
        mock_redis_storage, mock_s3, mock_edit_history_repo, mock_image_version_repo,
    ):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {"latency_ms": 30}, "timestamp": "t",
        })

        result = await service.remove_object(image_id=1, bbox_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.remove_object.assert_awaited_once()
        mock_s3.upload_bytes.assert_awaited_once()
        mock_image_version_repo.create_next.assert_awaited_once()
        mock_edit_history_repo.create.assert_awaited_once()
        mock_redis_storage.cache_image.assert_awaited_once()

        assert result["result_url"] == "s3://bucket/path.jpg"
        assert result["image_version_id"] == 11

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_detection_not_found(self, service, mock_detection_repo):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(2)])

        with pytest.raises(ValueError, match="bbox_id=1 not found"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_pipeline_exception_propagates_after_undo_push(
        self, service, mock_detection_repo, mock_redis_history, mock_pipeline,
    ):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(side_effect=RuntimeError("lama failure"))

        with pytest.raises(RuntimeError, match="lama failure"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()

    async def test_s3_exception_propagates(self, service, mock_detection_repo, mock_pipeline, mock_s3):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })
        mock_s3.upload_bytes = AsyncMock(side_effect=IOError("s3 unreachable"))

        with pytest.raises(IOError, match="s3 unreachable"):
            await service.remove_object(image_id=1, bbox_id=1, user_id=42)

    async def test_forwards_optional_params(self, service, mock_detection_repo, mock_pipeline):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.remove_object(
            image_id=1, bbox_id=1, user_id=42,
            ldm_steps=50, ldm_sampler="ddim", hd_strategy="RESIZE", use_edge_blending=False,
        )

        _, kwargs = mock_pipeline.remove_object.call_args
        assert kwargs["ldm_steps"] == 50
        assert kwargs["ldm_sampler"] == "ddim"
        assert kwargs["hd_strategy"] == "RESIZE"
        assert kwargs["use_edge_blending"] is False


class TestReplaceObject:

    async def test_success(
        self, service, mock_detection_repo, mock_redis_history, mock_pipeline, mock_redis_storage,
    ):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(3)])
        mock_pipeline.replace_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.replace_object(
            image_id=1, bbox_id=3, replace_image_bytes=b"new-obj", user_id=42,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        _, kwargs = mock_pipeline.replace_object.call_args
        assert kwargs["replacement_image_bytes"] == b"new-obj"
        mock_redis_storage.cache_image.assert_awaited_once()
        assert "image_version_id" in result

    async def test_detection_not_found(self, service, mock_detection_repo):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="bbox_id=1 not found"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42,
            )

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42,
            )

    async def test_pipeline_exception(self, service, mock_detection_repo, mock_pipeline):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.replace_object = AsyncMock(side_effect=RuntimeError("crash"))

        with pytest.raises(RuntimeError, match="crash"):
            await service.replace_object(
                image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42,
            )

    async def test_default_color_matching_false(self, service, mock_detection_repo, mock_pipeline):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.replace_object = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.replace_object(image_id=1, bbox_id=1, replace_image_bytes=b"x", user_id=42)

        _, kwargs = mock_pipeline.replace_object.call_args
        assert kwargs["use_color_matching"] is False


class TestRemoveMultipleObjects:

    async def test_success(
        self, service, mock_detection_repo, mock_redis_history, mock_pipeline, mock_edit_history_repo,
    ):
        dets = [make_detection(1), make_detection(2), make_detection(3)]
        mock_detection_repo.get_by_content = AsyncMock(return_value=dets)
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.remove_multiple_objects(image_id=1, bbox_ids=[1, 2], user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()
        _, kwargs = mock_pipeline.remove_multiple_objects.call_args
        assert len(kwargs["selected_bboxes"]) == 2
        assert len(kwargs["scene_bboxes"]) == 1  # remaining detection bbox_id=3
        mock_edit_history_repo.create.assert_awaited_once()
        assert "image_version_id" in result

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)

    async def test_no_matching_detections_raises(self, service, mock_detection_repo):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])

        with pytest.raises(ValueError, match="No valid detections found for bbox_ids"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[99, 100], user_id=42)

    async def test_all_selected_scene_bboxes_empty(
        self, service, mock_detection_repo, mock_pipeline,
    ):
        dets = [make_detection(1), make_detection(2)]
        mock_detection_repo.get_by_content = AsyncMock(return_value=dets)
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.remove_multiple_objects(image_id=1, bbox_ids=[1, 2], user_id=42)

        _, kwargs = mock_pipeline.remove_multiple_objects.call_args
        assert kwargs["scene_bboxes"] is None  # falsy list converted to None

    async def test_pipeline_exception(self, service, mock_detection_repo, mock_pipeline):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[make_detection(1)])
        mock_pipeline.remove_multiple_objects = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError, match="fail"):
            await service.remove_multiple_objects(image_id=1, bbox_ids=[1], user_id=42)


class TestSamReplaceObjectDiffusion:

    async def test_success(
        self, service, mock_redis_history, mock_pipeline, mock_redis_storage, mock_s3,
    ):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {"diffusion_ms": 12.0}, "timestamp": "t",
        })

        result = await service.sam_replace_object_diffusion(
            image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"reference", user_id=42,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_replace_object_diffusion.assert_awaited_once()
        mock_s3.upload_bytes.assert_awaited_once()
        assert result["result_url"] == "s3://bucket/path.jpg"
        assert result["metrics"] == {"diffusion_ms": 12.0}

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.sam_replace_object_diffusion(
                image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"reference", user_id=42,
            )

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.sam_replace_object_diffusion(
                image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"reference", user_id=42,
            )

    async def test_default_params_forwarded(self, service, mock_pipeline, mock_redis_storage):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.sam_replace_object_diffusion(
            image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"reference", user_id=42,
        )

        _, kwargs = mock_pipeline.sam_replace_object_diffusion.call_args
        assert kwargs["prompt"] == ""
        assert kwargs["use_color_matching"] is False
        assert kwargs["seed"] == 0

    async def test_optional_params_forwarded(self, service, mock_pipeline, mock_redis_storage):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(return_value={
            "result_bytes": b"r", "metrics": {}, "timestamp": "t",
        })

        await service.sam_replace_object_diffusion(
            image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"reference", user_id=42,
            prompt="a wooden chair", use_color_matching=True, seed=42,
        )

        _, kwargs = mock_pipeline.sam_replace_object_diffusion.call_args
        assert kwargs["prompt"] == "a wooden chair"
        assert kwargs["use_color_matching"] is True
        assert kwargs["seed"] == 42

    async def test_pipeline_exception_after_undo_push(self, service, mock_redis_history, mock_pipeline, mock_redis_storage):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(side_effect=RuntimeError("diffusion failure"))

        with pytest.raises(RuntimeError, match="diffusion failure"):
            await service.sam_replace_object_diffusion(
                image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"reference", user_id=42,
            )

        mock_redis_history.push_undo_state.assert_awaited_once()

    async def test_updates_current_state_in_redis(self, service, mock_pipeline, mock_redis_storage):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        await service.sam_replace_object_diffusion(
            image_id=1, mask_bytes=b"mask", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"reference", user_id=42,
        )

        mock_redis_storage.cache_image.assert_awaited_once_with(
            image_id=1, image_data=b"result", suffix="current_state", ttl=7200,
        )