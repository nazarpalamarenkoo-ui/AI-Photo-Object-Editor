
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging.context import bind_request_context, clear_context, new_request_id
from app.core.logging.logger import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    - Reuses an inbound `X-Request-ID` or generates a fresh uuid4 otherwise.
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
        endpoint = request.url.path
        method = request.method

        bind_request_context(request_id=request_id, method=method, endpoint=endpoint)
        request.state.request_id = request_id

        start = time.perf_counter()
        logger.info(
            "request_started",
            client_ip=request.client.host if request.client else None,
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