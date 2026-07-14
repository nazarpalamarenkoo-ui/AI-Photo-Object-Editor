from __future__ import annotations

import uuid
from unittest.mock import patch
import pytest
import structlog

from app.core.logging.context import (
    bind_request_context,
    bind_user,
    bind_worker_context,
    clear_context,
    new_request_id,
    request_context,
)

pytestmark = pytest.mark.unit

class TestNewRequestId:
    def test_returns_a_string(self):
        assert isinstance(new_request_id(), str)

    def test_returns_a_valid_uuid4(self):
        value = new_request_id()
        parsed = uuid.UUID(value, version=4)
        assert str(parsed) == value

    def test_successive_calls_are_unique(self):
        assert new_request_id() != new_request_id()


class TestBindRequestContext:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()

    def test_binds_request_id_method_and_endpoint(self):
        bind_request_context(request_id="req-1", method="GET", endpoint="/health")
        bound = structlog.contextvars.get_contextvars()
        assert bound["request_id"] == "req-1"
        assert bound["method"] == "GET"
        assert bound["endpoint"] == "/health"
        assert bound["user_id"] is None

    def test_defaults_method_endpoint_and_user_id_to_none(self):
        bind_request_context(request_id="req-2")
        bound = structlog.contextvars.get_contextvars()
        assert bound["method"] is None
        assert bound["endpoint"] is None
        assert bound["user_id"] is None

    def test_accepts_extra_kwargs(self):
        bind_request_context(request_id="req-3", client_ip="127.0.0.1")
        bound = structlog.contextvars.get_contextvars()
        assert bound["client_ip"] == "127.0.0.1"

    def test_delegates_to_structlog_bind_contextvars(self):
        with patch("app.core.logging.context.structlog.contextvars.bind_contextvars") as mock_bind:
            bind_request_context(request_id="req-4", method="POST", endpoint="/x", user_id=42)
        mock_bind.assert_called_once_with(
            request_id="req-4", method="POST", endpoint="/x", user_id=42
        )


class TestBindWorkerContext:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()

    def test_binds_job_id_worker_name_and_queue(self):
        bind_worker_context(job_id="job-1", worker_name="worker-a", queue="arq:default")
        bound = structlog.contextvars.get_contextvars()
        assert bound["job_id"] == "job-1"
        assert bound["worker_name"] == "worker-a"
        assert bound["queue"] == "arq:default"

    def test_queue_defaults_to_none_when_omitted(self):
        bind_worker_context(job_id="job-2", worker_name="worker-b")
        bound = structlog.contextvars.get_contextvars()
        assert bound["queue"] is None

    def test_accepts_extra_kwargs(self):
        bind_worker_context(job_id="job-3", worker_name="worker-c", job_try=2)
        bound = structlog.contextvars.get_contextvars()
        assert bound["job_try"] == 2


class TestBindUser:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()

    def test_binds_user_id(self):
        bind_user(123)
        bound = structlog.contextvars.get_contextvars()
        assert bound["user_id"] == 123

    def test_does_not_clobber_other_bound_fields(self):
        bind_request_context(request_id="req-5")
        bind_user("u-abc")
        bound = structlog.contextvars.get_contextvars()
        assert bound["request_id"] == "req-5"
        assert bound["user_id"] == "u-abc"


class TestClearContext:
    def test_clears_all_bound_contextvars(self):
        bind_request_context(request_id="req-6")
        assert structlog.contextvars.get_contextvars()
        clear_context()
        assert structlog.contextvars.get_contextvars() == {}


class TestRequestContextManager:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()

    def test_binds_fields_within_the_block(self):
        with request_context(task="import", batch_id=7):
            bound = structlog.contextvars.get_contextvars()
            assert bound["task"] == "import"
            assert bound["batch_id"] == 7

    def test_resets_fields_after_the_block_exits(self):
        with request_context(task="import"):
            pass
        assert "task" not in structlog.contextvars.get_contextvars()

    def test_restores_previous_value_on_nested_use(self):
        bind_request_context(request_id="outer")
        with request_context(request_id="inner"):
            assert structlog.contextvars.get_contextvars()["request_id"] == "inner"
        assert structlog.contextvars.get_contextvars()["request_id"] == "outer"

    def test_resets_even_if_exception_raised_inside_block(self):
        with pytest_raises_generic():
            with request_context(task="risky"):
                raise ValueError("boom")
        assert "task" not in structlog.contextvars.get_contextvars()


def pytest_raises_generic():
    import pytest

    return pytest.raises(ValueError)