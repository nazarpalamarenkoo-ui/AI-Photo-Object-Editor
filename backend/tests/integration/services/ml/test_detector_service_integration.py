import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.detector_service import DetectorService

pytestmark = pytest.mark.integration


@pytest.fixture
def detector_service(ml_service_kwargs):
    return DetectorService(**ml_service_kwargs)


def _detection_result():
    return {
        "detections": [
            {"detected_class": "person", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        ],
        "metrics": {"latency_ms": 12},
    }


class TestDetectObjects:
    @pytest.mark.asyncio
    async def test_success_persists_detections(
        self, detector_service, sample_image_version, sample_image, sample_user,
        mock_redis_cache, mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())

        result = await detector_service.detect_objects(sample_image.id, sample_user.id)

        assert result["detections"][0]["detected_class"] == "person"
        assert result["metrics"] == {"latency_ms": 12}
        persisted = await detector_service.detection_repo.get_by_content(
            sample_image_version.content_id, active_only=False
        )
        assert len(persisted) == 1
        assert persisted[0].content_id == sample_image_version.content_id

    @pytest.mark.asyncio
    async def test_second_call_served_from_content_cache_skips_pipeline(
        self, detector_service, sample_image_version, sample_image, sample_user,
        mock_redis_cache, mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())

        await detector_service.detect_objects(sample_image.id, sample_user.id)
        mock_pipeline.detect_objects.reset_mock()

        result = await detector_service.detect_objects(sample_image.id, sample_user.id)

        mock_pipeline.detect_objects.assert_not_called()
        assert result["metrics"] == {"cache_hit": True}
        assert len(result["detections"]) == 1

    @pytest.mark.asyncio
    async def test_force_rerun_deletes_and_reruns_pipeline(
        self, detector_service, sample_image_version, sample_image, sample_user,
        mock_redis_cache, mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.delete = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value=_detection_result())

        await detector_service.detect_objects(sample_image.id, sample_user.id)
        mock_pipeline.detect_objects.reset_mock()

        await detector_service.detect_objects(sample_image.id, sample_user.id, force_rerun=True)

        mock_pipeline.detect_objects.assert_awaited_once()
        mock_redis_cache.delete.assert_awaited_once_with(
            f"image_content:{sample_image_version.content_id}:detections"
        )
        persisted = await detector_service.detection_repo.get_by_content(
            sample_image_version.content_id, active_only=False
        )
        assert len(persisted) == 1  # old rows hard-deleted, one fresh row persisted

    @pytest.mark.asyncio
    async def test_filters_by_confidence_and_class_on_read(
        self, detector_service, sample_image_version, sample_image, sample_user,
        mock_redis_cache, mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [
                {"detected_class": "person", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"detected_class": "car", "confidence": 0.1, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            ],
            "metrics": {},
        })

        result = await detector_service.detect_objects(
            sample_image.id, sample_user.id, conf_threshold=0.5
        )

        assert len(result["detections"]) == 1
        assert result["detections"][0]["detected_class"] == "person"
        # both rows still persisted below the floor threshold, just filtered on read
        persisted = await detector_service.detection_repo.get_by_content(
            sample_image_version.content_id, active_only=False
        )
        assert len(persisted) == 2

    @pytest.mark.asyncio
    async def test_image_not_found(self, detector_service, sample_user):
        with pytest.raises(ValueError, match="not found"):
            await detector_service.detect_objects(999999, sample_user.id)

    @pytest.mark.asyncio
    async def test_unauthorized(self, detector_service, sample_image_version, sample_image, sample_user):
        with pytest.raises(ValueError, match="Unauthorized"):
            await detector_service.detect_objects(sample_image.id, sample_user.id + 999)

    @pytest.mark.asyncio
    async def test_no_current_version_raises(self, detector_service, sample_image, sample_user):
        with pytest.raises(ValueError, match="no current version"):
            await detector_service.detect_objects(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_propagates_pipeline_exception(
        self, detector_service, sample_image_version, sample_image, sample_user,
        mock_redis_cache, mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("pipeline failure"))

        with pytest.raises(RuntimeError, match="pipeline failure"):
            await detector_service.detect_objects(sample_image.id, sample_user.id)


class TestGetSupportedClasses:
    def test_passthrough_to_pipeline(self, detector_service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(return_value=["person", "car"])

        assert detector_service.get_supported_classes() == ["person", "car"]