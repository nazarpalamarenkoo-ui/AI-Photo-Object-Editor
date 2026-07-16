from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

from app.core.logging.context import bind_request_context, clear_context, new_request_id
from app.core.logging.logger import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def _resolve_route_pattern(request: Request) -> str:
    """
    Resolves the registered route *pattern* (e.g. '/images/{image_id}/url')
    instead of the raw path (e.g. '/images/153/url'). This keeps the
    'endpoint' field low-cardinality — critical for LogQL aggregations
    like `quantile_over_time(...) by (endpoint)` and Grafana dashboards,
    where one series per distinct image_id would otherwise blow up the
    legend and make percentile calculations meaningless (too few samples
    per series).
    """
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", request.url.path)
    return request.url.path


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    - Reuses an inbound `X-Request-ID` or generates a fresh uuid4 otherwise.
    - Resolves the FastAPI route *pattern* (not the raw path with IDs in it)
      so 'endpoint' stays low-cardinality across logs and metrics.
    - Binds request_id / method / endpoint into contextvars so *every* log
      line emitted anywhere during this request — including deep inside
      services, repositories, and the ML pipeline — automatically carries
      them, with no parameter threading required.
    - Logs `request_started`, `request_finished` (status + duration_ms),
      and `request_failed` (exception) for unhandled errors, then re-raises
      so FastAPI's normal exception handling still applies.
    - Echoes the request_id back in the response header for client-side
      correlation / bug reports.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        endpoint = _resolve_route_pattern(request)
        method = request.method

        bind_request_context(request_id=request_id, method=method, endpoint=endpoint)
        request.state.request_id = request_id

        start = time.perf_counter()
        logger.info(
            "request_started",
            client_ip=request.client.host if request.client else None,
            raw_path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "request_failed",
                status_code=500,
                duration_ms=duration_ms,
                exc_info=exc,
            )
            clear_context()
            raise
        else:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log = logger.warning if response.status_code >= 400 else logger.info
            log("request_finished", status_code=response.status_code, duration_ms=duration_ms)
            response.headers[REQUEST_ID_HEADER] = request_id
            clear_context()
            return response