from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.logging.decorators import log_execution

pytestmark = pytest.mark.unit

@pytest.fixture
def fake_logger():
    logger = MagicMock()
    return logger


class TestLogExecutionAsContextManagerSync:
    def test_logs_started_and_finished_on_success(self, fake_logger):
        with log_execution("do_thing", logger=fake_logger):
            pass

        fake_logger.info.assert_any_call("do_thing_started")
        finished_calls = [
            c for c in fake_logger.info.call_args_list if c.args[0] == "do_thing_finished"
        ]
        assert len(finished_calls) == 1
        assert "duration_ms" in finished_calls[0].kwargs

    def test_passes_extra_fields_to_started_and_finished(self, fake_logger):
        with log_execution("do_thing", logger=fake_logger, user_id=42):
            pass

        fake_logger.info.assert_any_call("do_thing_started", user_id=42)
        finished_call = next(
            c for c in fake_logger.info.call_args_list if c.args[0] == "do_thing_finished"
        )
        assert finished_call.kwargs["user_id"] == 42

    def test_uses_default_logger_when_none_given(self):
        with log_execution("do_thing") as ctx:
            assert ctx.logger is not None

    def test_uses_given_level_for_started_and_finished(self, fake_logger):
        with log_execution("do_thing", logger=fake_logger, level="warning"):
            pass

        fake_logger.warning.assert_any_call("do_thing_started")
        finished_calls = [
            c for c in fake_logger.warning.call_args_list if c.args[0] == "do_thing_finished"
        ]
        assert len(finished_calls) == 1

    def test_logs_failed_and_reraises_on_exception(self, fake_logger):
        with pytest.raises(ValueError, match="oops"):
            with log_execution("do_thing", logger=fake_logger):
                raise ValueError("oops")

        fake_logger.error.assert_called_once()
        args, kwargs = fake_logger.error.call_args
        assert args[0] == "do_thing_failed"
        assert "duration_ms" in kwargs
        assert kwargs["exc_info"] is not None

    def test_does_not_log_finished_on_exception(self, fake_logger):
        with pytest.raises(ValueError):
            with log_execution("do_thing", logger=fake_logger):
                raise ValueError("oops")

        finished_calls = [
            c for c in fake_logger.info.call_args_list if c.args[0] == "do_thing_finished"
        ]
        assert finished_calls == []


class TestLogExecutionAsContextManagerAsync:
    @pytest.mark.asyncio
    async def test_logs_started_and_finished_on_success(self, fake_logger):
        async with log_execution("async_thing", logger=fake_logger):
            pass

        fake_logger.info.assert_any_call("async_thing_started")
        finished_calls = [
            c for c in fake_logger.info.call_args_list if c.args[0] == "async_thing_finished"
        ]
        assert len(finished_calls) == 1

    @pytest.mark.asyncio
    async def test_logs_failed_and_reraises_on_exception(self, fake_logger):
        with pytest.raises(RuntimeError, match="boom"):
            async with log_execution("async_thing", logger=fake_logger):
                raise RuntimeError("boom")

        fake_logger.error.assert_called_once()
        assert fake_logger.error.call_args.args[0] == "async_thing_failed"


class TestLogExecutionAsDecoratorSync:
    def test_wraps_sync_function_and_returns_its_result(self, fake_logger):
        @log_execution("compute", logger=fake_logger)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
        fake_logger.info.assert_any_call("compute_started")
        finished_calls = [
            c for c in fake_logger.info.call_args_list if c.args[0] == "compute_finished"
        ]
        assert len(finished_calls) == 1

    def test_preserves_function_metadata(self, fake_logger):
        @log_execution("compute", logger=fake_logger)
        def documented_add(a, b):
            """Adds two numbers."""
            return a + b

        assert documented_add.__name__ == "documented_add"
        assert documented_add.__doc__ == "Adds two numbers."

    def test_propagates_exception_and_logs_failure(self, fake_logger):
        @log_execution("compute", logger=fake_logger)
        def boom():
            raise KeyError("missing")

        with pytest.raises(KeyError):
            boom()

        fake_logger.error.assert_called_once()
        assert fake_logger.error.call_args.args[0] == "compute_failed"

    def test_each_call_gets_a_fresh_timer(self, fake_logger):
        @log_execution("compute", logger=fake_logger)
        def noop():
            return None

        noop()
        noop()

        finished_calls = [
            c for c in fake_logger.info.call_args_list if c.args[0] == "compute_finished"
        ]
        assert len(finished_calls) == 2


class TestLogExecutionAsDecoratorAsync:
    @pytest.mark.asyncio
    async def test_wraps_async_function_and_returns_its_result(self, fake_logger):
        @log_execution("async_compute", logger=fake_logger)
        async def add(a, b):
            return a + b

        result = await add(2, 3)
        assert result == 5
        fake_logger.info.assert_any_call("async_compute_started")

    @pytest.mark.asyncio
    async def test_preserves_function_metadata_for_async(self, fake_logger):
        @log_execution("async_compute", logger=fake_logger)
        async def documented_add(a, b):
            """Adds two numbers asynchronously."""
            return a + b

        assert documented_add.__name__ == "documented_add"
        assert documented_add.__doc__ == "Adds two numbers asynchronously."

    @pytest.mark.asyncio
    async def test_propagates_exception_and_logs_failure(self, fake_logger):
        @log_execution("async_compute", logger=fake_logger)
        async def boom():
            raise KeyError("missing")

        with pytest.raises(KeyError):
            await boom()

        fake_logger.error.assert_called_once()
        assert fake_logger.error.call_args.args[0] == "async_compute_failed"