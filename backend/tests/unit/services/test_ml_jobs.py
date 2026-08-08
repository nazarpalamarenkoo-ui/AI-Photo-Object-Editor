import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml_job_service import MLJobService

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_mljob_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    return AsyncMock()


@pytest.fixture
def service(mock_mljob_repo, mock_image_repo, mock_image_version_repo):
    return MLJobService(
        mljob_repo=mock_mljob_repo,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
    )


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.content_id = 100
    return version


@pytest.fixture
def task_type():
    """Opaque sentinel — MLJobService never branches on the concrete
    MLTaskType member, it just threads it through to the repo/logger."""
    return MagicMock(name="task_type")


class TestFindCompleted:

    async def test_returns_successful_job_for_current_version_content(
        self, service, mock_image_repo, mock_image_version_repo, mock_mljob_repo,
        sample_image, sample_version, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
        job = MagicMock()
        mock_mljob_repo.get_successful = AsyncMock(return_value=job)

        result = await service.find_completed(image_id=1, user_id=42, task_type=task_type)

        mock_image_version_repo.get_current.assert_awaited_once_with(sample_image)
        mock_mljob_repo.get_successful.assert_awaited_once_with(100, task_type)
        assert result is job

    async def test_returns_none_when_no_current_version(
        self, service, mock_image_repo, mock_image_version_repo, mock_mljob_repo,
        sample_image, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=None)

        result = await service.find_completed(image_id=1, user_id=42, task_type=task_type)

        assert result is None
        mock_mljob_repo.get_successful.assert_not_called()

    async def test_image_not_found(self, service, mock_image_repo, task_type):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Image 1 not found"):
            await service.find_completed(image_id=1, user_id=42, task_type=task_type)

    async def test_unauthorized(self, service, mock_image_repo, sample_image, task_type):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.find_completed(image_id=1, user_id=42, task_type=task_type)


class TestStart:

    async def test_creates_and_marks_running(
        self, service, mock_image_repo, mock_image_version_repo, mock_mljob_repo,
        sample_image, sample_version, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
        created_job = MagicMock(id=5)
        running_job = MagicMock(id=5)
        mock_mljob_repo.create = AsyncMock(return_value=created_job)
        mock_mljob_repo.mark_running = AsyncMock(return_value=running_job)

        result = await service.start(image_id=1, user_id=42, task_type=task_type)

        mock_mljob_repo.create.assert_awaited_once_with(100, 10, task_type)
        mock_mljob_repo.mark_running.assert_awaited_once_with(5)
        assert result is running_job

    async def test_no_current_version_raises(
        self, service, mock_image_repo, mock_image_version_repo, sample_image, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="has no current version"):
            await service.start(image_id=1, user_id=42, task_type=task_type)

    async def test_image_not_found(self, service, mock_image_repo, task_type):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.start(image_id=1, user_id=42, task_type=task_type)

    async def test_unauthorized(self, service, mock_image_repo, sample_image, task_type):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.start(image_id=1, user_id=42, task_type=task_type)


class TestComplete:

    async def test_marks_success(self, service, mock_mljob_repo):
        job = MagicMock(id=5)
        mock_mljob_repo.mark_success = AsyncMock(return_value=job)

        result = await service.complete(job_id=5, processing_time_ms=250)

        mock_mljob_repo.mark_success.assert_awaited_once_with(5, 250)
        assert result is job


class TestFail:

    async def test_marks_failed_with_truncated_message(self, service, mock_mljob_repo):
        job = MagicMock(id=5)
        mock_mljob_repo.mark_failed = AsyncMock(return_value=job)
        error = RuntimeError("x" * 3000)

        result = await service.fail(job_id=5, error=error)

        args = mock_mljob_repo.mark_failed.call_args.args
        assert args[0] == 5
        assert len(args[1]) == 2000
        assert result is job

    async def test_short_message_not_truncated(self, service, mock_mljob_repo):
        mock_mljob_repo.mark_failed = AsyncMock(return_value=MagicMock())
        error = ValueError("short message")

        await service.fail(job_id=5, error=error)

        args = mock_mljob_repo.mark_failed.call_args.args
        assert args[1] == "short message"


class TestTrack:
    """track() is the async-context-manager choke point used by
    tracked_runner.run_tracked — start on entry, complete on clean exit,
    fail + re-raise on exception."""

    async def test_success_path_starts_and_completes(
        self, service, mock_image_repo, mock_image_version_repo, mock_mljob_repo,
        sample_image, sample_version, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
        job = MagicMock(id=7)
        mock_mljob_repo.create = AsyncMock(return_value=job)
        mock_mljob_repo.mark_running = AsyncMock(return_value=job)
        mock_mljob_repo.mark_success = AsyncMock(return_value=job)

        async with service.track(image_id=1, user_id=42, task_type=task_type) as tracked_job:
            assert tracked_job is job

        mock_mljob_repo.mark_running.assert_awaited_once_with(7)
        mock_mljob_repo.mark_success.assert_awaited_once()
        args = mock_mljob_repo.mark_success.call_args.args
        assert args[0] == 7
        assert isinstance(args[1], int)
        assert args[1] >= 0
        mock_mljob_repo.mark_failed.assert_not_called()

    async def test_exception_path_marks_failed_and_reraises(
        self, service, mock_image_repo, mock_image_version_repo, mock_mljob_repo,
        sample_image, sample_version, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
        job = MagicMock(id=7)
        mock_mljob_repo.create = AsyncMock(return_value=job)
        mock_mljob_repo.mark_running = AsyncMock(return_value=job)
        mock_mljob_repo.mark_failed = AsyncMock(return_value=job)

        with pytest.raises(RuntimeError, match="boom"):
            async with service.track(image_id=1, user_id=42, task_type=task_type):
                raise RuntimeError("boom")

        mock_mljob_repo.mark_failed.assert_awaited_once()
        args = mock_mljob_repo.mark_failed.call_args.args
        assert args[0] == 7
        assert args[1] == "boom"
        mock_mljob_repo.mark_success.assert_not_called()

    async def test_start_failure_propagates_without_entering_body(
        self, service, mock_image_repo, task_type,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)
        entered = False

        with pytest.raises(ValueError, match="not found"):
            async with service.track(image_id=1, user_id=42, task_type=task_type):
                entered = True

        assert entered is False