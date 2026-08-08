import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.tracked_runner import run_tracked

pytestmark = pytest.mark.unit


class _AsyncTrackCM:
    """Stand-in for MLJobService.track()'s @asynccontextmanager."""

    def __init__(self, job):
        self._job = job

    async def __aenter__(self):
        return self._job

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def job():
    j = MagicMock()
    j.id = 55
    return j


@pytest.fixture
def mock_mljob_service(job):
    service = MagicMock()
    service.track = MagicMock(return_value=_AsyncTrackCM(job))
    return service


@pytest.fixture
def mock_service_instance():
    instance = MagicMock()
    instance.some_method = AsyncMock(return_value={"ok": True})
    return instance


@pytest.fixture
def mock_service_cls(mock_service_instance):
    return MagicMock(return_value=mock_service_instance)


@pytest.fixture
def task_type():
    return MagicMock(name="task_type")


class TestRunTracked:

    async def test_instantiates_service_with_deps(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        deps = {"s3_storage": "s3", "image_repo": "repo"}

        await run_tracked(
            mock_service_cls, deps, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
        )

        mock_service_cls.assert_called_once_with(s3_storage="s3", image_repo="repo")

    async def test_merges_extra_ctor_kwargs_over_deps(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        deps = {"a": 1, "b": 2}

        await run_tracked(
            mock_service_cls, deps, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
            extra_ctor_kwargs={"b": 99, "c": 3},
        )

        mock_service_cls.assert_called_once_with(a=1, b=99, c=3)

    async def test_no_extra_ctor_kwargs_uses_deps_as_is(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        deps = {"a": 1}

        await run_tracked(
            mock_service_cls, deps, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
        )

        mock_service_cls.assert_called_once_with(a=1)

    async def test_does_not_mutate_original_deps_dict(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        deps = {"a": 1}

        await run_tracked(
            mock_service_cls, deps, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
            extra_ctor_kwargs={"b": 2},
        )

        assert deps == {"a": 1}

    async def test_calls_track_with_image_user_and_task_type(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        await run_tracked(
            mock_service_cls, {}, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
        )

        mock_mljob_service.track.assert_called_once_with(1, 42, task_type)

    async def test_calls_named_method_with_image_user_and_extra_kwargs(
        self, mock_service_cls, mock_service_instance, mock_mljob_service, task_type,
    ):
        await run_tracked(
            mock_service_cls, {}, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
            conf_threshold=0.5, classes=["car"],
        )

        mock_service_instance.some_method.assert_awaited_once_with(
            image_id=1, user_id=42, conf_threshold=0.5, classes=["car"],
        )

    async def test_returns_method_result(
        self, mock_service_cls, mock_mljob_service, task_type,
    ):
        result = await run_tracked(
            mock_service_cls, {}, mock_mljob_service, "some_method",
            image_id=1, user_id=42, task_type=task_type,
        )

        assert result == {"ok": True}

    async def test_different_method_name_is_dispatched_dynamically(
        self, mock_service_cls, mock_service_instance, mock_mljob_service, task_type,
    ):
        mock_service_instance.remove_object = AsyncMock(return_value={"done": True})

        result = await run_tracked(
            mock_service_cls, {}, mock_mljob_service, "remove_object",
            image_id=1, user_id=42, task_type=task_type, bbox_id=3,
        )

        mock_service_instance.remove_object.assert_awaited_once_with(
            image_id=1, user_id=42, bbox_id=3,
        )
        assert result == {"done": True}

    async def test_method_exception_propagates(
        self, mock_service_cls, mock_service_instance, mock_mljob_service, task_type,
    ):
        mock_service_instance.some_method = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await run_tracked(
                mock_service_cls, {}, mock_mljob_service, "some_method",
                image_id=1, user_id=42, task_type=task_type,
            )