from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.logging.workerlogging import log_job

pytestmark = pytest.mark.unit

@pytest.fixture
def mock_logger():
    with patch("app.core.logging.workerlogging._logger") as mocked:
        yield mocked


@pytest.fixture
def mock_bind_worker_context():
    with patch("app.core.logging.workerlogging.bind_worker_context") as mocked:
        yield mocked


@pytest.fixture
def mock_clear_context():
    with patch("app.core.logging.workerlogging.clear_context") as mocked:
        yield mocked


class TestLogJobDecoratorSuccess:
    @pytest.mark.asyncio
    async def test_calls_wrapped_function_and_returns_its_result(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx, x, y):
            return x + y

        result = await my_task({"job_id": "j1"}, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_binds_worker_context_from_ctx(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job(queue="arq:high")
        async def my_task(ctx):
            return None

        await my_task({"job_id": "abc", "worker_name": "worker-1"})

        mock_bind_worker_context.assert_called_once_with(
            job_id="abc", worker_name="worker-1", queue="arq:high"
        )

    @pytest.mark.asyncio
    async def test_worker_name_falls_back_to_job_id_when_missing(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            return None

        await my_task({"job_id": "abc"})

        mock_bind_worker_context.assert_called_once_with(
            job_id="abc", worker_name="abc", queue="arq:queue"
        )

    @pytest.mark.asyncio
    async def test_worker_name_falls_back_to_worker_literal_when_ctx_empty(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            return None

        await my_task({})

        mock_bind_worker_context.assert_called_once_with(
            job_id="unknown", worker_name="worker", queue="arq:queue"
        )

    @pytest.mark.asyncio
    async def test_logs_started_and_finished_events(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            return "ok"

        await my_task({"job_id": "j1", "job_try": 1})

        mock_logger.info.assert_any_call("job_started", task="my_task", job_try=1)
        finished_calls = [
            c for c in mock_logger.info.call_args_list if c.args[0] == "job_finished"
        ]
        assert len(finished_calls) == 1
        assert finished_calls[0].kwargs["task"] == "my_task"
        assert "duration_ms" in finished_calls[0].kwargs

    @pytest.mark.asyncio
    async def test_clears_context_after_success(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            return None

        await my_task({"job_id": "j1"})
        mock_clear_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_through_extra_args_and_kwargs(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx, a, b, c=None):
            return (a, b, c)

        result = await my_task({"job_id": "j1"}, 1, 2, c=3)
        assert result == (1, 2, 3)

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_documented_task(ctx):
            """Does a thing."""
            return None

        assert my_documented_task.__name__ == "my_documented_task"
        assert my_documented_task.__doc__ == "Does a thing."


class TestLogJobDecoratorFailure:
    @pytest.mark.asyncio
    async def test_logs_job_failed_and_reraises(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            await my_task({"job_id": "j1"})

        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert args[0] == "job_failed"
        assert kwargs["task"] == "my_task"
        assert "duration_ms" in kwargs
        assert kwargs["exc_info"] is not None

    @pytest.mark.asyncio
    async def test_does_not_log_job_finished_on_failure(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            await my_task({"job_id": "j1"})

        finished_calls = [
            c for c in mock_logger.info.call_args_list if c.args[0] == "job_finished"
        ]
        assert finished_calls == []

    @pytest.mark.asyncio
    async def test_clears_context_even_on_failure(
        self, mock_logger, mock_bind_worker_context, mock_clear_context
    ):
        @log_job()
        async def my_task(ctx):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            await my_task({"job_id": "j1"})

        mock_clear_context.assert_called_once()