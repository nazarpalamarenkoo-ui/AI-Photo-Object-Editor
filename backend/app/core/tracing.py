from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from opentelemetry import trace, propagate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

_initialized = False


def setup_tracing(service_name: str, otlp_endpoint: str = "http://alloy:4317") -> None:
    """
    Configure the global OTel TracerProvider once per process.

    Call sites (exactly these two, nowhere else):
      - FastAPI: near the top of app/main.py, same place the old inline
        TracerProvider/BatchSpanProcessor/OTLPSpanExporter block used to live.
      - ARQ worker: near the top of app/worker.py, right after configure_logging().

    Idempotent — safe to call more than once (e.g. if a module gets
    re-imported), the second call is a no-op.
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str):
    return trace.get_tracer(name)


def inject_trace_context() -> dict:
    """
    Capture the CURRENT trace context (e.g. the active FastAPI request span,
    set up automatically by FastAPIInstrumentor) as a plain dict — safe to
    pass through arq job kwargs, since arq pickles kwargs and a raw OTel
    Context object isn't picklable, but a dict of strings is.

    Call this at the enqueue call site, inside the request handler / service
    method that pushes a job onto arq — that's where the request's span is
    still active:

        from app.core.tracing import inject_trace_context

        await redis_pool.enqueue_job(
            "remove_object_task",
            image_id=image_id, bbox_id=bbox_id, user_id=user_id,
            _trace_carrier=inject_trace_context(),
        )

    The receiving task function does NOT need a `_trace_carrier` parameter
    in its signature — the `trace_job` decorator below pops it out of
    kwargs before calling the real function.
    """
    carrier: dict = {}
    propagate.inject(carrier)
    return carrier


def extract_trace_context(carrier: Optional[dict]):
    """Rebuilds an OTel Context from a carrier dict produced by inject_trace_context()."""
    if not carrier:
        return None
    return propagate.extract(carrier)


def trace_job(operation_name: Optional[str] = None) -> Callable:
    """
    Decorator for arq task functions.

    Pops a `_trace_carrier` kwarg (added at enqueue time via
    inject_trace_context()) and, if present, starts the task's span as a
    CHILD of the HTTP request span that triggered it — stitching the
    app -> worker hop into a single trace in Tempo instead of two
    disconnected ones (one HTTP span, one orphaned worker span).

    Safe to use even when no carrier was passed (job enqueued from a
    non-instrumented context, a cron/scheduled job, etc) — falls back to
    starting a standalone root span instead of failing.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(ctx: dict, *args: Any, **kwargs: Any) -> Any:
            carrier = kwargs.pop("_trace_carrier", None)
            parent_context = extract_trace_context(carrier)

            tracer = get_tracer("app.worker")
            span_name = operation_name or func.__name__

            with tracer.start_as_current_span(span_name, context=parent_context):
                return await func(ctx, *args, **kwargs)

        return wrapper

    return decorator