from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.core.tracing as tracing_module
from app.core.tracing import (
    setup_tracing,
    get_tracer,
    inject_trace_context,
    extract_trace_context,
    trace_job,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_initialized_flag():
    """setup_tracing() is guarded by a module-level `_initialized` flag —
    reset it around every test so tests don't leak state into each other."""
    original = tracing_module._initialized
    tracing_module._initialized = False
    yield
    tracing_module._initialized = original


class TestSetupTracing:
    def test_creates_resource_with_service_name(self):
        with patch("app.core.tracing.Resource") as mock_resource, \
             patch("app.core.tracing.TracerProvider") as mock_provider_cls, \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter"), \
             patch("app.core.tracing.trace"):

            setup_tracing("my-service")

            mock_resource.create.assert_called_once_with({"service.name": "my-service"})
            mock_provider_cls.assert_called_once_with(resource=mock_resource.create.return_value)

    def test_uses_default_otlp_endpoint(self):
        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider"), \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter") as mock_exporter_cls, \
             patch("app.core.tracing.trace"):

            setup_tracing("my-service")

            mock_exporter_cls.assert_called_once_with(endpoint="http://alloy:4317", insecure=True)

    def test_uses_custom_otlp_endpoint(self):
        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider"), \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter") as mock_exporter_cls, \
             patch("app.core.tracing.trace"):

            setup_tracing("my-service", otlp_endpoint="http://custom-collector:4317")

            mock_exporter_cls.assert_called_once_with(
                endpoint="http://custom-collector:4317", insecure=True
            )

    def test_wraps_exporter_in_batch_span_processor(self):
        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider") as mock_provider_cls, \
             patch("app.core.tracing.BatchSpanProcessor") as mock_bsp_cls, \
             patch("app.core.tracing.OTLPSpanExporter") as mock_exporter_cls, \
             patch("app.core.tracing.trace"):

            setup_tracing("my-service")

            mock_bsp_cls.assert_called_once_with(mock_exporter_cls.return_value)
            mock_provider_cls.return_value.add_span_processor.assert_called_once_with(
                mock_bsp_cls.return_value
            )

    def test_sets_global_tracer_provider(self):
        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider") as mock_provider_cls, \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter"), \
             patch("app.core.tracing.trace") as mock_trace:

            setup_tracing("my-service")

            mock_trace.set_tracer_provider.assert_called_once_with(mock_provider_cls.return_value)

    def test_is_idempotent_second_call_is_noop(self):
        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider") as mock_provider_cls, \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter"), \
             patch("app.core.tracing.trace") as mock_trace:

            setup_tracing("my-service")
            setup_tracing("my-service")
            setup_tracing("another-service", otlp_endpoint="http://other:4317")

            mock_trace.set_tracer_provider.assert_called_once()
            mock_provider_cls.assert_called_once()

    def test_sets_initialized_flag_after_first_call(self):
        assert tracing_module._initialized is False

        with patch("app.core.tracing.Resource"), \
             patch("app.core.tracing.TracerProvider"), \
             patch("app.core.tracing.BatchSpanProcessor"), \
             patch("app.core.tracing.OTLPSpanExporter"), \
             patch("app.core.tracing.trace"):

            setup_tracing("my-service")

        assert tracing_module._initialized is True


class TestGetTracer:
    def test_delegates_to_opentelemetry_trace_get_tracer(self):
        with patch("app.core.tracing.trace") as mock_trace:
            mock_trace.get_tracer.return_value = "tracer-instance"

            result = get_tracer("app.worker")

            mock_trace.get_tracer.assert_called_once_with("app.worker")
            assert result == "tracer-instance"


class TestInjectTraceContext:
    def test_returns_dict_populated_by_propagate_inject(self):
        def fake_inject(carrier):
            carrier["traceparent"] = "00-abc-def-01"

        with patch("app.core.tracing.propagate") as mock_propagate:
            mock_propagate.inject.side_effect = fake_inject

            result = inject_trace_context()

        assert result == {"traceparent": "00-abc-def-01"}

    def test_calls_propagate_inject_with_a_fresh_empty_dict(self):
        with patch("app.core.tracing.propagate") as mock_propagate:
            inject_trace_context()

            mock_propagate.inject.assert_called_once()
            (carrier_arg,), _ = mock_propagate.inject.call_args
            assert carrier_arg == {}

    def test_returns_empty_dict_when_no_active_span_context(self):
        with patch("app.core.tracing.propagate") as mock_propagate:
            mock_propagate.inject.side_effect = lambda carrier: None

            result = inject_trace_context()

        assert result == {}

    def test_return_value_is_a_plain_dict(self):
        with patch("app.core.tracing.propagate"):
            result = inject_trace_context()

        assert isinstance(result, dict)


class TestExtractTraceContext:
    def test_returns_none_for_none_carrier(self):
        assert extract_trace_context(None) is None

    def test_returns_none_for_empty_dict_carrier(self):
        assert extract_trace_context({}) is None

    def test_calls_propagate_extract_with_carrier_and_returns_its_result(self):
        carrier = {"traceparent": "00-abc-def-01"}
        with patch("app.core.tracing.propagate") as mock_propagate:
            mock_propagate.extract.return_value = "extracted-context"

            result = extract_trace_context(carrier)

            mock_propagate.extract.assert_called_once_with(carrier)
            assert result == "extracted-context"

    def test_does_not_call_propagate_extract_when_carrier_is_falsy(self):
        with patch("app.core.tracing.propagate") as mock_propagate:
            extract_trace_context(None)
            extract_trace_context({})

            mock_propagate.extract.assert_not_called()


def _make_tracer_mock():
    """A tracer whose start_as_current_span() returns a usable context
    manager. __exit__ must return False (not a truthy MagicMock default),
    otherwise it would silently swallow exceptions raised inside the
    `with` block — which would break exception-propagation tests."""
    tracer = MagicMock()
    span_cm = MagicMock()
    span_cm.__enter__ = MagicMock(return_value=MagicMock(name="span"))
    span_cm.__exit__ = MagicMock(return_value=False)
    tracer.start_as_current_span.return_value = span_cm
    return tracer


@pytest.mark.asyncio
class TestTraceJob:
    async def test_calls_wrapped_function_and_returns_its_result(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def my_task(ctx, x, y):
                return x + y

            result = await my_task({}, 2, 3)

        assert result == 5

    async def test_uses_function_name_as_span_name_by_default(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def remove_object_task(ctx):
                return None

            await remove_object_task({})

        args, _ = tracer.start_as_current_span.call_args
        assert args[0] == "remove_object_task"

    async def test_uses_explicit_operation_name_when_given(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job(operation_name="custom_span_name")
            async def some_task(ctx):
                return None

            await some_task({})

        args, _ = tracer.start_as_current_span.call_args
        assert args[0] == "custom_span_name"

    async def test_gets_tracer_named_app_worker(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer) as mock_get_tracer, \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def some_task(ctx):
                return None

            await some_task({})

        mock_get_tracer.assert_called_once_with("app.worker")

    async def test_pops_trace_carrier_from_kwargs_before_calling_func(self):
        tracer = _make_tracer_mock()
        received_kwargs = {}

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None) as mock_extract:

            @trace_job()
            async def some_task(ctx, **kwargs):
                received_kwargs.update(kwargs)
                return None

            await some_task({}, foo="bar", _trace_carrier={"traceparent": "00-x"})

        assert "_trace_carrier" not in received_kwargs
        assert received_kwargs == {"foo": "bar"}
        mock_extract.assert_called_once_with({"traceparent": "00-x"})

    async def test_extract_trace_context_called_with_none_when_no_carrier_present(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None) as mock_extract:

            @trace_job()
            async def some_task(ctx):
                return None

            await some_task({})

        mock_extract.assert_called_once_with(None)

    async def test_works_without_carrier_falls_back_to_root_span(self):
        """No _trace_carrier passed at all (e.g. a cron/scheduled job) —
        must not raise, and must still start a span."""
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def some_task(ctx, value):
                return value * 2

            result = await some_task({}, 21)

        assert result == 42
        tracer.start_as_current_span.assert_called_once()

    async def test_starts_span_with_extracted_context_as_parent(self):
        tracer = _make_tracer_mock()
        fake_parent_context = MagicMock(name="parent-context")

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=fake_parent_context):

            @trace_job()
            async def some_task(ctx):
                return None

            await some_task({}, _trace_carrier={"traceparent": "00-x"})

        _, kwargs = tracer.start_as_current_span.call_args
        assert kwargs["context"] is fake_parent_context

    async def test_passes_through_positional_and_keyword_args(self):
        tracer = _make_tracer_mock()
        captured = {}

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def some_task(ctx, image_id, bbox_id, user_id=None):
                captured["ctx"] = ctx
                captured["image_id"] = image_id
                captured["bbox_id"] = bbox_id
                captured["user_id"] = user_id
                return "done"

            ctx = {"redis": "pool"}
            result = await some_task(ctx, 1, 2, user_id=99, _trace_carrier=None)

        assert captured == {"ctx": ctx, "image_id": 1, "bbox_id": 2, "user_id": 99}
        assert result == "done"

    async def test_propagates_exception_raised_by_wrapped_function(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            @trace_job()
            async def failing_task(ctx):
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                await failing_task({})

    async def test_preserves_function_metadata_via_functools_wraps(self):
        @trace_job()
        async def my_named_task(ctx):
            """My docstring."""
            return None

        assert my_named_task.__name__ == "my_named_task"
        assert my_named_task.__doc__ == "My docstring."

    async def test_decorator_instance_is_reusable_across_multiple_functions(self):
        tracer = _make_tracer_mock()

        with patch("app.core.tracing.get_tracer", return_value=tracer), \
             patch("app.core.tracing.extract_trace_context", return_value=None):

            decorator = trace_job(operation_name="shared")

            @decorator
            async def task_a(ctx):
                return "a"

            @decorator
            async def task_b(ctx):
                return "b"

            result_a = await task_a({})
            result_b = await task_b({})

        assert result_a == "a"
        assert result_b == "b"
        assert tracer.start_as_current_span.call_count == 2