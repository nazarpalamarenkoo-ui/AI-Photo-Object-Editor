import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.detector_service import DetectorService
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
    redis_storage.cache_detections = AsyncMock(return_value=None)
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
    repo = AsyncMock()
    repo.get_by_image = AsyncMock(return_value=[])
    repo.delete_by_image = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_pipeline():
    pipeline = AsyncMock()
    return pipeline


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
    mock_db,
    mock_s3,
    mock_redis_storage,
    mock_redis_history,
    mock_image_repo,
    mock_detection_repo,
    mock_pipeline,
):
    return DetectorService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


def make_detection(bbox_id=1, cls="person", conf=0.9):
    return {
        "bbox_id": bbox_id,
        "detected_class": cls,
        "confidence": conf,
        "x1": 1, "y1": 2, "x2": 3, "y2": 4,
    }


class TestDetectObjects:
    async def test_detect_objects_success(
        self, service, mock_image_repo, sample_image, mock_pipeline,
        mock_detection_repo, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        dets = [make_detection(1), make_detection(2, cls="car", conf=0.7)]
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": dets,
            "image_size": (640, 480),
            "metrics": {"latency_ms": 120},
            "timestamp": "2024-01-01T00:00:00",
        })

        result = await service.detect_objects(image_id=1, user_id=42, conf_threshold=0.6)

        mock_pipeline.detect_objects.assert_awaited_once()
        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["conf_threshold"] == 0.6
        assert kwargs["track_metrics"] is True

        mock_detection_repo.delete_by_image.assert_awaited_once_with(1)
        mock_redis_storage.delete.assert_awaited_once_with("image:1:detections")
        mock_detection_repo.create_many.assert_awaited_once()
        mock_redis_storage.cache_detections.assert_awaited_once_with(
            image_id=1, detections=dets, ttl=3600
        )

        assert result["detections"] == dets
        assert result["image_size"] == (640, 480)
        assert "metrics" in result
        assert "timestamp" in result and result["timestamp"]

    async def test_detect_objects_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.detect_objects(image_id=999, user_id=42)

    async def test_detect_objects_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 1  # different from requesting user
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.detect_objects(image_id=1, user_id=42)

    async def test_detect_objects_uses_redis_cache_when_present(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"cached-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42)

        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["image_bytes"] == b"cached-bytes"
        service.s3.download.assert_not_called()

    async def test_detect_objects_falls_back_to_s3_on_cache_miss(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_pipeline, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42)

        mock_s3.download.assert_awaited_once_with(sample_image.storage_path)

    async def test_detect_objects_empty_detections(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (640, 480), "metrics": {}, "timestamp": "t",
        })

        result = await service.detect_objects(image_id=1, user_id=42)

        assert result["detections"] == []
        mock_detection_repo.create_many.assert_awaited_once_with([])

    async def test_detect_objects_with_class_filter(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42, classes=["person", "car"])

        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["classes"] == ["person", "car"]

    async def test_detect_objects_boundary_conf_threshold_zero(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42, conf_threshold=0.0)

        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["conf_threshold"] == 0.0

    async def test_detect_objects_boundary_conf_threshold_one(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42, conf_threshold=1.0)

        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["conf_threshold"] == 1.0

    async def test_detect_objects_pipeline_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("model crashed"))

        with pytest.raises(RuntimeError, match="model crashed"):
            await service.detect_objects(image_id=1, user_id=42)

        mock_detection_repo.delete_by_image.assert_not_called()

    async def test_detect_objects_repository_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_detection_repo,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })
        mock_detection_repo.create_many = AsyncMock(side_effect=RuntimeError("db down"))

        with pytest.raises(RuntimeError, match="db down"):
            await service.detect_objects(image_id=1, user_id=42)

    async def test_detect_objects_redis_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })
        mock_redis_storage.cache_detections = AsyncMock(side_effect=ConnectionError("redis down"))

        with pytest.raises(ConnectionError, match="redis down"):
            await service.detect_objects(image_id=1, user_id=42)

    async def test_detect_objects_s3_exception_propagates_on_fallback(
        self, service, mock_image_repo, sample_image, mock_redis_storage, mock_s3,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cache_image = AsyncMock(return_value=None)
        mock_s3.download = AsyncMock(side_effect=IOError("s3 unreachable"))

        with pytest.raises(IOError, match="s3 unreachable"):
            await service.detect_objects(image_id=1, user_id=42)


class TestGetSupportedClasses:
    def test_get_supported_classes_returns_pipeline_classes(self, service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(
            return_value=["person", "car", "dog"]
        )

        result = service.get_supported_classes()

        assert result == ["person", "car", "dog"]
        mock_pipeline.get_supported_classes.assert_called_once()

    def test_get_supported_classes_empty_list(self, service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(return_value=[])

        result = service.get_supported_classes()

        assert result == []