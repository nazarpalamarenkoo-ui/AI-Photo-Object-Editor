import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.detector_service import DetectorService

pytestmark = pytest.mark.integration

@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()

@pytest.fixture
def mock_helper():
    return MagicMock()

@pytest.fixture
def detector_service(db_session, mock_s3_storage, mock_redis_cache, mock_redis_history, image_repo, detection_repo, mock_pipeline):
    return DetectorService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        image_repo=image_repo,
        detection_repo=detection_repo,
        pipeline=mock_pipeline,
    )


def _detection_result():
    return {
        "detections": [
            {
                "bbox_id": 1,
                "detected_class": "person",
                "confidence": 0.9,
                "x1": 0, "y1": 0, "x2": 10, "y2": 10,
            }
        ],
        "image_size": (100, 100),
        "metrics": {"latency_ms": 12},
        "timestamp": "2024-01-01T00:00:00",
    }


class TestDetectObjects:
    @pytest.mark.asyncio
    async def test_success_persists_detections_and_caches(
        self, detector_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.delete = AsyncMock()
        mock_redis_cache.cache_detections = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())

        result = await detector_service.detect_objects(sample_image.id, sample_user.id)

        assert result["detections"][0]["detected_class"] == "person"
        persisted = await detector_service.detection_repo.get_by_image(sample_image.id)
        assert len(persisted) == 1
        assert persisted[0].bbox_id == 1
        mock_redis_cache.cache_detections.assert_awaited_once_with(
            image_id=sample_image.id, detections=_detection_result()["detections"], ttl=3600
        )

    @pytest.mark.asyncio
    async def test_deletes_previous_detections_before_persisting(
        self, detector_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.delete = AsyncMock()
        mock_redis_cache.cache_detections = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())

        await detector_service.detect_objects(sample_image.id, sample_user.id)
        await detector_service.detect_objects(sample_image.id, sample_user.id)

        persisted = await detector_service.detection_repo.get_by_image(sample_image.id)
        assert len(persisted) == 1
        mock_redis_cache.delete.assert_awaited_with(f"image:{sample_image.id}:detections")

    @pytest.mark.asyncio
    async def test_raises_when_image_missing(self, detector_service, sample_user):
        with pytest.raises(ValueError, match="not found"):
            await detector_service.detect_objects(999999, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, detector_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await detector_service.detect_objects(sample_image.id, sample_image.user_id + 1)

    @pytest.mark.asyncio
    async def test_propagates_pipeline_exception(
        self, detector_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("pipeline failure"))

        with pytest.raises(RuntimeError, match="pipeline failure"):
            await detector_service.detect_objects(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_propagates_repository_exception(
        self, detector_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())
        detector_service.detection_repo.delete_by_image = AsyncMock(side_effect=RuntimeError("db failure"))

        with pytest.raises(RuntimeError, match="db failure"):
            await detector_service.detect_objects(sample_image.id, sample_user.id)
