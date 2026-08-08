import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from app.db.enums.ml_task_status import MLTaskType
from app.services.ml_job_service import MLJobService
from app.services.ml.tracked_runner import run_tracked

pytestmark = pytest.mark.integration

TASK_TYPE = next(iter(MLTaskType))


class _FakeJob:
    """Stand-in for the MLJob row MLJobService.track() yields."""
    def __init__(self, id=1):
        self.id = id


def _make_mljob_service(job=None, on_track=None):
    """MagicMock MLJobService whose .track() is a *real* async context
    manager yielding `job` — a MagicMock alone can't support `async with`,
    so this wires up a working fake instead of mocking track() itself.
    """
    job = job or _FakeJob()

    @asynccontextmanager
    async def _track(image_id, user_id, task_type):
        if on_track:
            on_track(image_id, user_id, task_type)
        yield job

    service = MagicMock(spec=MLJobService)
    service.track = _track
    return service, job


class _FakeService:
    """Stand-in for a BaseMLService subclass — records ctor kwargs and
    exposes an async method run_tracked can dispatch to by name."""
    def __init__(self, **kwargs):
        self.ctor_kwargs = kwargs

    async def do_thing(self, image_id, user_id, **kwargs):
        return {"ok": True, "image_id": image_id, "user_id": user_id, "kwargs": kwargs}

class TestRunTracked:
    @pytest.mark.asyncio
    async def test_instantiates_service_with_deps(self):
        mljob_service, _ = _make_mljob_service()
        deps = {"s3_storage": "s3", "redis_storage": "redis"}
        captured = {}

        class RecordingService(_FakeService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured["kwargs"] = kwargs

        await run_tracked(
            RecordingService, deps, mljob_service, "do_thing",
            image_id=1, user_id=2, task_type=TASK_TYPE,
        )

        assert captured["kwargs"] == deps

    @pytest.mark.asyncio
    async def test_extra_ctor_kwargs_merge_over_deps(self):
        """extra_ctor_kwargs is applied AFTER deps, so it can override a
        key deps also sets (e.g. AssetService's redis_assets handling)."""
        mljob_service, _ = _make_mljob_service()
        deps = {"redis_assets": "base", "s3_storage": "s3"}
        captured = {}

        class RecordingService(_FakeService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured["kwargs"] = kwargs

        await run_tracked(
            RecordingService, deps, mljob_service, "do_thing",
            image_id=1, user_id=2, task_type=TASK_TYPE,
            extra_ctor_kwargs={"redis_assets": "overridden", "extra": "x"},
        )

        assert captured["kwargs"] == {
            "redis_assets": "overridden", "s3_storage": "s3", "extra": "x",
        }

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_deps_dict(self):
        """ctor_kwargs is a copy — the caller's `deps` dict (often reused
        across many run_tracked calls) must come back untouched."""
        mljob_service, _ = _make_mljob_service()
        deps = {"s3_storage": "s3"}

        await run_tracked(
            _FakeService, deps, mljob_service, "do_thing",
            image_id=1, user_id=2, task_type=TASK_TYPE,
            extra_ctor_kwargs={"extra": "x"},
        )

        assert deps == {"s3_storage": "s3"}

    @pytest.mark.asyncio
    async def test_calls_named_method_with_image_id_user_id_and_kwargs(self):
        mljob_service, _ = _make_mljob_service()

        result = await run_tracked(
            _FakeService, {}, mljob_service, "do_thing",
            image_id=42, user_id=7, task_type=TASK_TYPE,
            padding_pixels=8, label="obj",
        )

        assert result == {
            "ok": True, "image_id": 42, "user_id": 7,
            "kwargs": {"padding_pixels": 8, "label": "obj"},
        }

    @pytest.mark.asyncio
    async def test_wraps_call_in_mljob_track_with_correct_args(self):
        seen = {}
        mljob_service, job = _make_mljob_service(
            on_track=lambda *args: seen.setdefault("args", args)
        )

        await run_tracked(
            _FakeService, {}, mljob_service, "do_thing",
            image_id=5, user_id=6, task_type=TASK_TYPE,
        )

        assert seen["args"] == (5, 6, TASK_TYPE)

    @pytest.mark.asyncio
    async def test_returns_service_method_result_unchanged(self):
        mljob_service, _ = _make_mljob_service()

        class EchoService(_FakeService):
            async def do_thing(self, image_id, user_id, **kwargs):
                return {"detections": [1, 2, 3], "image_id": image_id}

        result = await run_tracked(
            EchoService, {}, mljob_service, "do_thing",
            image_id=3, user_id=4, task_type=TASK_TYPE,
        )

        assert result == {"detections": [1, 2, 3], "image_id": 3}

    @pytest.mark.asyncio
    async def test_propagates_method_exception(self):
        mljob_service, _ = _make_mljob_service()

        class FailingService(_FakeService):
            async def do_thing(self, image_id, user_id, **kwargs):
                raise RuntimeError("pipeline blew up")

        with pytest.raises(RuntimeError, match="pipeline blew up"):
            await run_tracked(
                FailingService, {}, mljob_service, "do_thing",
                image_id=1, user_id=2, task_type=TASK_TYPE,
            )

    @pytest.mark.asyncio
    async def test_no_extra_ctor_kwargs_leaves_deps_untouched(self):
        """extra_ctor_kwargs=None (the default) — ctor should get exactly
        deps, nothing added or removed."""
        mljob_service, _ = _make_mljob_service()
        deps = {"s3_storage": "s3", "redis_storage": "redis"}
        captured = {}

        class RecordingService(_FakeService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured["kwargs"] = kwargs

        await run_tracked(
            RecordingService, deps, mljob_service, "do_thing",
            image_id=1, user_id=2, task_type=TASK_TYPE,
        )

        assert captured["kwargs"] == deps