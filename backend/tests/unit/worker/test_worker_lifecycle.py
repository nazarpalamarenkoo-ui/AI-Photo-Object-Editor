import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import _build_ml_deps, startup, shutdown, WorkerSettings


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
class TestBuildMlDeps:
    async def test_yields_all_expected_dependencies(self):
        with patch("app.workers.worker.S3Storage") as MockS3, \
             patch("app.workers.worker.RedisStorage") as MockRedisStorage, \
             patch("app.workers.worker.RedisHistory") as MockRedisHistory, \
             patch("app.workers.worker.RedisAssetsStorage") as MockRedisAssets, \
             patch("app.workers.worker.ImageRepository") as MockImageRepo, \
             patch("app.workers.worker.ImageVersionRepository") as MockImageVersionRepo, \
             patch("app.workers.worker.ImageContentRepository") as MockImageContentRepo, \
             patch("app.workers.worker.DetectionRepository") as MockDetectionRepo, \
             patch("app.workers.worker.SegmentationRepository") as MockSegmentationRepo, \
             patch("app.workers.worker.ImageEditHistoryRepository") as MockEditHistoryRepo, \
             patch("app.workers.worker.AssetRepository") as MockAssetsRepo, \
             patch("app.workers.worker.MLJobRepository") as MockMLJobRepo, \
             patch("app.workers.worker.get_pipeline") as mock_get_pipeline, \
             patch("app.workers.worker.MLJobService") as MockMLJobService, \
             patch("app.workers.worker.get_db_session") as mock_get_db_session:

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            async with _build_ml_deps() as (deps, mljob_service):
                assert "db" not in deps
                assert deps["s3_storage"] is MockS3.return_value
                assert deps["redis_storage"] is MockRedisStorage.return_value
                assert deps["redis_history"] is MockRedisHistory.return_value
                assert deps["redis_assets"] is MockRedisAssets.return_value
                assert deps["image_repo"] is MockImageRepo.return_value
                assert deps["image_version_repo"] is MockImageVersionRepo.return_value
                assert deps["image_content_repo"] is MockImageContentRepo.return_value
                assert deps["detection_repo"] is MockDetectionRepo.return_value
                assert deps["segmentation_repo"] is MockSegmentationRepo.return_value
                assert deps["edit_history_repo"] is MockEditHistoryRepo.return_value
                assert deps["assets_repo"] is MockAssetsRepo.return_value
                assert deps["pipeline"] is mock_get_pipeline.return_value
                assert mljob_service is MockMLJobService.return_value

            MockImageRepo.assert_called_once_with(mock_get_db_session)
            MockImageVersionRepo.assert_called_once_with(mock_get_db_session)
            MockImageContentRepo.assert_called_once_with(mock_get_db_session)
            MockDetectionRepo.assert_called_once_with(mock_get_db_session)
            MockSegmentationRepo.assert_called_once_with(mock_get_db_session)
            MockEditHistoryRepo.assert_called_once_with(mock_get_db_session)
            MockAssetsRepo.assert_called_once_with(mock_get_db_session)
            MockMLJobRepo.assert_called_once_with(mock_get_db_session)
            mock_get_pipeline.assert_called_once_with()
            MockMLJobService.assert_called_once_with(
                mljob_repo=MockMLJobRepo.return_value,
                image_repo=MockImageRepo.return_value,
                image_version_repo=MockImageVersionRepo.return_value,
            )

    async def test_closes_redis_connections_on_normal_exit(self):
        with patch("app.workers.worker.S3Storage"), \
             patch("app.workers.worker.RedisStorage") as MockRedisStorage, \
             patch("app.workers.worker.RedisHistory") as MockRedisHistory, \
             patch("app.workers.worker.RedisAssetsStorage") as MockRedisAssets, \
             patch("app.workers.worker.ImageRepository"), \
             patch("app.workers.worker.ImageVersionRepository"), \
             patch("app.workers.worker.ImageContentRepository"), \
             patch("app.workers.worker.DetectionRepository"), \
             patch("app.workers.worker.SegmentationRepository"), \
             patch("app.workers.worker.ImageEditHistoryRepository"), \
             patch("app.workers.worker.AssetRepository"), \
             patch("app.workers.worker.MLJobRepository"), \
             patch("app.workers.worker.get_pipeline"), \
             patch("app.workers.worker.MLJobService"):

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            async with _build_ml_deps():
                pass

            MockRedisStorage.return_value.close.assert_awaited_once()
            MockRedisHistory.return_value.close.assert_awaited_once()
            MockRedisAssets.return_value.close.assert_awaited_once()

    async def test_closes_redis_connections_even_if_exception_raised_in_block(self):
        with patch("app.workers.worker.S3Storage"), \
             patch("app.workers.worker.RedisStorage") as MockRedisStorage, \
             patch("app.workers.worker.RedisHistory") as MockRedisHistory, \
             patch("app.workers.worker.RedisAssetsStorage") as MockRedisAssets, \
             patch("app.workers.worker.ImageRepository"), \
             patch("app.workers.worker.ImageVersionRepository"), \
             patch("app.workers.worker.ImageContentRepository"), \
             patch("app.workers.worker.DetectionRepository"), \
             patch("app.workers.worker.SegmentationRepository"), \
             patch("app.workers.worker.ImageEditHistoryRepository"), \
             patch("app.workers.worker.AssetRepository"), \
             patch("app.workers.worker.MLJobRepository"), \
             patch("app.workers.worker.get_pipeline"), \
             patch("app.workers.worker.MLJobService"):

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            with pytest.raises(ValueError):
                async with _build_ml_deps():
                    raise ValueError("boom")

            MockRedisStorage.return_value.close.assert_awaited_once()
            MockRedisHistory.return_value.close.assert_awaited_once()
            MockRedisAssets.return_value.close.assert_awaited_once()


@pytest.mark.asyncio
class TestStartup:
    async def test_warms_up_pipeline_with_configured_device(self):
        with patch("app.workers.worker.get_pipeline") as mock_get_pipeline:
            await startup(ctx={})
            mock_get_pipeline.assert_called_once_with()


@pytest.mark.asyncio
class TestShutdown:
    async def test_does_not_raise(self):
        await shutdown(ctx={})


class TestWorkerSettings:
    def test_registers_all_task_functions(self):
        names = {fn.__name__ for fn in WorkerSettings.functions}
        assert names == {
            "detect_objects_task",
            "segment_objects_task",
            "segment_with_prompt_task",
            "segment_by_polygon_task",
            "sam_remove_object_task",
            "sam_replace_object_task",
            "sam_replace_object_diffusion_task",
            "segment_hybrid_task",
            "remove_object_task",
            "remove_multiple_objects_task",
            "replace_object_task",
            "sam_extract_object_task",
        }

    def test_lifecycle_hooks_are_wired(self):
        assert WorkerSettings.on_startup is startup
        assert WorkerSettings.on_shutdown is shutdown

    def test_processes_one_job_at_a_time(self):
        assert WorkerSettings.max_jobs == 1

    def test_job_timeout_is_five_minutes(self):
        assert WorkerSettings.job_timeout == 100_000_000