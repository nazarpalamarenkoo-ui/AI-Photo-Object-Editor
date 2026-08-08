import pytest
from unittest.mock import AsyncMock

from app.repository.mljob_repo import MLJobRepository
from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.db.enums.ml_task_status import MLTaskType
from app.services.ml_job_service import MLJobService

pytestmark = pytest.mark.integration

TASK_TYPE = next(iter(MLTaskType))


def _make_service(db_session) -> MLJobService:
    """MLJobService has no redis/pipeline deps — just the three repos."""
    return MLJobService(
        mljob_repo=MLJobRepository(db_session),
        image_repo=ImageRepository(db_session),
        image_version_repo=ImageVersionRepository(db_session),
    )


class TestStart:
    @pytest.mark.asyncio
    async def test_creates_job_scoped_to_current_content(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)

        job = await service.start(sample_image.id, sample_user.id, TASK_TYPE)

        assert job.id is not None

    @pytest.mark.asyncio
    async def test_raises_when_no_current_version(
        self, db_session, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="no current version"):
            await service.start(sample_image.id, sample_user.id, TASK_TYPE)

    @pytest.mark.asyncio
    async def test_raises_when_image_not_found(self, db_session, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.start(999999, sample_user.id, TASK_TYPE)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.start(sample_image.id, sample_user.id + 999, TASK_TYPE)


class TestComplete:
    @pytest.mark.asyncio
    async def test_marks_job_as_the_one_find_completed_returns(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        job = await service.start(sample_image.id, sample_user.id, TASK_TYPE)

        completed = await service.complete(job.id, processing_time_ms=250)

        assert completed.id == job.id
        found = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert found is not None
        assert found.id == job.id

    @pytest.mark.asyncio
    async def test_forwards_processing_time_to_repo(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        job = await service.start(sample_image.id, sample_user.id, TASK_TYPE)
        spy = AsyncMock(wraps=service.mljob_repo.mark_success)
        service.mljob_repo.mark_success = spy

        await service.complete(job.id, processing_time_ms=999)

        spy.assert_awaited_once_with(job.id, 999)


class TestFail:
    @pytest.mark.asyncio
    async def test_failed_job_is_not_returned_by_find_completed(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        job = await service.start(sample_image.id, sample_user.id, TASK_TYPE)

        failed = await service.fail(job.id, RuntimeError("boom"))

        assert failed.id == job.id
        found = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert found is None

    @pytest.mark.asyncio
    async def test_truncates_error_message_to_2000_chars(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        job = await service.start(sample_image.id, sample_user.id, TASK_TYPE)
        spy = AsyncMock(wraps=service.mljob_repo.mark_failed)
        service.mljob_repo.mark_failed = spy

        await service.fail(job.id, RuntimeError("x" * 5000))

        forwarded_message = spy.call_args.args[1]
        assert len(forwarded_message) == 2000


class TestFindCompleted:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_job_ever_ran(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        result = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_while_job_still_running(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        await service.start(sample_image.id, sample_user.id, TASK_TYPE)

        result = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_current_version(
        self, db_session, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        result = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.find_completed(sample_image.id, sample_user.id + 999, TASK_TYPE)

    @pytest.mark.asyncio
    async def test_raises_when_image_not_found(self, db_session, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.find_completed(999999, sample_user.id, TASK_TYPE)


class TestTrack:
    @pytest.mark.asyncio
    async def test_success_path_completes_the_job(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)

        async with service.track(sample_image.id, sample_user.id, TASK_TYPE) as job:
            started_job_id = job.id

        found = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert found is not None
        assert found.id == started_job_id

    @pytest.mark.asyncio
    async def test_exception_path_fails_the_job_and_reraises(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)

        with pytest.raises(RuntimeError, match="pipeline exploded"):
            async with service.track(sample_image.id, sample_user.id, TASK_TYPE):
                raise RuntimeError("pipeline exploded")

        found = await service.find_completed(sample_image.id, sample_user.id, TASK_TYPE)
        assert found is None

    @pytest.mark.asyncio
    async def test_yields_a_job_with_an_id(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)

        async with service.track(sample_image.id, sample_user.id, TASK_TYPE) as job:
            assert job.id is not None

    @pytest.mark.asyncio
    async def test_computes_non_negative_elapsed_time_on_success(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        spy = AsyncMock(wraps=service.mljob_repo.mark_success)
        service.mljob_repo.mark_success = spy

        async with service.track(sample_image.id, sample_user.id, TASK_TYPE):
            pass

        elapsed_ms = spy.call_args.args[1]
        assert elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_does_not_call_complete_when_exception_raised(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        spy = AsyncMock(wraps=service.mljob_repo.mark_success)
        service.mljob_repo.mark_success = spy

        with pytest.raises(RuntimeError):
            async with service.track(sample_image.id, sample_user.id, TASK_TYPE):
                raise RuntimeError("boom")

        spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized_before_entering_context(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)
        entered = False

        with pytest.raises(ValueError, match="Unauthorized"):
            async with service.track(sample_image.id, sample_user.id + 999, TASK_TYPE):
                entered = True

        assert entered is False