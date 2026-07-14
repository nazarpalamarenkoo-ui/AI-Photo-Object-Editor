import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import _build_ml_deps, startup, shutdown, WorkerSettings


pytestmark = pytest.mark.unit
@pytest.mark.asyncio
class TestBuildMlDeps:
    async def test_yields_all_expected_dependencies(self):
        fake_db = MagicMock(name="db")

        with patch("app.workers.worker.S3Storage") as MockS3, patch(
            "app.workers.worker.RedisStorage"
        ) as MockRedisStorage, patch(
            "app.workers.worker.RedisHistory"
        ) as MockRedisHistory, patch(
            "app.workers.worker.RedisAssetsStorage"
        ) as MockRedisAssets, patch(
            "app.workers.worker.ImageRepository"
        ) as MockImageRepo, patch(
            "app.workers.worker.DetectionRepository"
        ) as MockDetectionRepo, patch(
            "app.workers.worker.get_pipeline"
        ) as mock_get_pipeline:

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            async with _build_ml_deps(fake_db) as deps:
                assert deps["db"] is fake_db
                assert deps["s3_storage"] is MockS3.return_value
                assert deps["redis_storage"] is MockRedisStorage.return_value
                assert deps["redis_history"] is MockRedisHistory.return_value
                assert deps["redis_assets"] is MockRedisAssets.return_value
                assert deps["image_repo"] is MockImageRepo.return_value
                assert deps["detection_repo"] is MockDetectionRepo.return_value
                assert deps["pipeline"] is mock_get_pipeline.return_value

            MockImageRepo.assert_called_once_with(fake_db)
            MockDetectionRepo.assert_called_once_with(fake_db)
            mock_get_pipeline.assert_called_once_with()

    async def test_closes_redis_connections_on_normal_exit(self):
        fake_db = MagicMock()

        with patch("app.workers.worker.S3Storage"), patch(
            "app.workers.worker.RedisStorage"
        ) as MockRedisStorage, patch(
            "app.workers.worker.RedisHistory"
        ) as MockRedisHistory, patch(
            "app.workers.worker.RedisAssetsStorage"
        ) as MockRedisAssets, patch(
            "app.workers.worker.ImageRepository"
        ), patch(
            "app.workers.worker.DetectionRepository"
        ), patch(
            "app.workers.worker.get_pipeline"
        ):

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            async with _build_ml_deps(fake_db):
                pass

            MockRedisStorage.return_value.close.assert_awaited_once()
            MockRedisHistory.return_value.close.assert_awaited_once()
            MockRedisAssets.return_value.close.assert_awaited_once()

    async def test_closes_redis_connections_even_if_exception_raised_in_block(self):
        fake_db = MagicMock()

        with patch("app.workers.worker.S3Storage"), patch(
            "app.workers.worker.RedisStorage"
        ) as MockRedisStorage, patch(
            "app.workers.worker.RedisHistory"
        ) as MockRedisHistory, patch(
            "app.workers.worker.RedisAssetsStorage"
        ) as MockRedisAssets, patch(
            "app.workers.worker.ImageRepository"
        ), patch(
            "app.workers.worker.DetectionRepository"
        ), patch(
            "app.workers.worker.get_pipeline"
        ):

            MockRedisStorage.return_value.close = AsyncMock()
            MockRedisHistory.return_value.close = AsyncMock()
            MockRedisAssets.return_value.close = AsyncMock()

            with pytest.raises(ValueError):
                async with _build_ml_deps(fake_db):
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
            "segment_objects_task",
            "segment_with_prompt_task",
            "segment_by_polygon_task",
            "sam_remove_object_task",
            "sam_replace_object_task",
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
        assert WorkerSettings.job_timeout == 300