import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.detector_service import DetectorService
from app.db.models.image import Image


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
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_content_repo():
    return AsyncMock()


def _mark_persisted(dets):
    """create_many() normally returns rows as they'd come back from the DB
    (is_active defaulted True on insert). Since these Detection() objects
    are never actually flushed here, set that explicitly rather than
    relying on SQLAlchemy's insert-time default."""
    for d in dets:
        d.is_active = True
    return dets


@pytest.fixture
def mock_detection_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.delete_by_content = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(side_effect=_mark_persisted)
    repo.max_bbox_id = AsyncMock(return_value=-1)
    return repo


@pytest.fixture
def mock_segmentation_repo():
    return AsyncMock()


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
    image = MagicMock(spec=Image)
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    image.filename = "original.jpg"
    image.width = 640
    image.height = 480
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.image_id = 1
    version.content_id = 100
    return version


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline, sample_image, sample_version,
):
    mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
    mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)

    return DetectorService(
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


def make_raw_detection(bbox_id=1, cls="person", conf=0.9):
    return {
        "bbox_id": bbox_id, "detected_class": cls, "confidence": conf,
        "x1": 1, "y1": 2, "x2": 3, "y2": 4,
    }


class TestDetectObjects:

    async def test_no_cache_runs_pipeline_and_persists(
        self, service, mock_pipeline, mock_detection_repo, sample_image,
    ):
        dets = [make_raw_detection(1), make_raw_detection(2, cls="car", conf=0.4)]
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": dets, "image_size": (640, 480),
            "metrics": {"latency_ms": 120}, "timestamp": "2024-01-01T00:00:00",
        })

        result = await service.detect_objects(image_id=1, user_id=42, conf_threshold=0.6)

        mock_pipeline.detect_objects.assert_awaited_once()
        _, kwargs = mock_pipeline.detect_objects.call_args
        assert kwargs["conf_threshold"] <= 0.6  # floored to DETECTION_FLOOR_THRESHOLD

        mock_detection_repo.create_many.assert_awaited_once()
        service.s3.download.assert_awaited_once_with(sample_image.storage_path)

        # only >= conf_threshold survives the read-time filter
        assert len(result["detections"]) == 1
        assert result["detections"][0]["detected_class"] == "person"
        assert result["image_size"] == (640, 480)

    async def test_served_from_content_cache_skips_pipeline(
        self, service, mock_pipeline, mock_detection_repo,
    ):
        cached = MagicMock(is_active=True, confidence=0.9, detected_class="car",
                            bbox_id=0, x1=0, y1=0, x2=1, y2=1)
        mock_detection_repo.get_by_content = AsyncMock(return_value=[cached])

        result = await service.detect_objects(image_id=1, user_id=42)

        mock_pipeline.detect_objects.assert_not_called()
        assert result["metrics"] == {"cache_hit": True}
        assert len(result["detections"]) == 1

    async def test_force_rerun_clears_cache_and_reruns(
        self, service, mock_detection_repo, mock_redis_storage, mock_pipeline,
    ):
        existing = MagicMock(is_active=True, confidence=0.9, detected_class="car",
                              bbox_id=0, x1=0, y1=0, x2=1, y2=1)
        mock_detection_repo.get_by_content = AsyncMock(return_value=[existing])
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42, force_rerun=True)

        mock_detection_repo.delete_by_content.assert_awaited_once_with(100)
        mock_redis_storage.delete.assert_awaited_once_with("image_content:100:detections")
        mock_pipeline.detect_objects.assert_awaited_once()

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.detect_objects(image_id=999, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 1
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.detect_objects(image_id=1, user_id=42)

    async def test_class_filter_on_read(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_raw_detection(1, cls="person"), make_raw_detection(2, cls="car")],
            "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        result = await service.detect_objects(image_id=1, user_id=42, classes=["car"])

        assert [d["detected_class"] for d in result["detections"]] == ["car"]

    async def test_bbox_id_offset_by_max_existing(self, service, mock_detection_repo, mock_pipeline):
        mock_detection_repo.max_bbox_id = AsyncMock(return_value=4)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_raw_detection(0)], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42)

        persisted = mock_detection_repo.create_many.call_args.args[0]
        assert persisted[0].bbox_id == 5

    async def test_pipeline_exception_propagates(self, service, mock_pipeline, mock_detection_repo):
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("model crashed"))

        with pytest.raises(RuntimeError, match="model crashed"):
            await service.detect_objects(image_id=1, user_id=42)

        mock_detection_repo.create_many.assert_not_called()

    async def test_repository_exception_propagates(self, service, mock_pipeline, mock_detection_repo):
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })
        mock_detection_repo.create_many = AsyncMock(side_effect=RuntimeError("db down"))

        with pytest.raises(RuntimeError, match="db down"):
            await service.detect_objects(image_id=1, user_id=42)

    async def test_cache_hit_uses_current_state_bytes_when_present(
        self, service, mock_redis_storage, mock_s3, mock_pipeline,
    ):
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"cached-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [], "image_size": (1, 1), "metrics": {}, "timestamp": "t",
        })

        await service.detect_objects(image_id=1, user_id=42)

        mock_s3.download.assert_not_called()


class TestGetSupportedClasses:

    def test_returns_pipeline_classes(self, service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])

        result = service.get_supported_classes()

        assert result == ["person", "car", "dog"]

    def test_empty_list(self, service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(return_value=[])

        assert service.get_supported_classes() == []