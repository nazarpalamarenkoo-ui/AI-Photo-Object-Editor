from __future__ import annotations
 
from unittest.mock import AsyncMock, patch
 
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
 
from app.core.logging.middleware import REQUEST_ID_HEADER, RequestLoggingMiddleware
 
pytestmark = pytest.mark.unit

def make_request(
    method: str = "GET",
    path: str = "/foo",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
        "client": client,
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)
 

 
@pytest.fixture
def middleware():

    return RequestLoggingMiddleware(app=AsyncMock())
 
 
@pytest.fixture
def patched_collaborators():
    with patch("app.core.logging.middleware.logger") as mock_logger, patch(
        "app.core.logging.middleware.bind_request_context"
    ) as mock_bind, patch(
        "app.core.logging.middleware.clear_context"
    ) as mock_clear, patch(
        "app.core.logging.middleware.new_request_id", return_value="generated-id"
    ) as mock_new_id:
        yield {
            "logger": mock_logger,
            "bind": mock_bind,
            "clear": mock_clear,
            "new_id": mock_new_id,
        }
 
 
class TestRequestIdHandling:
    @pytest.mark.asyncio
    async def test_reuses_inbound_request_id_header(self, middleware, patched_collaborators):
        request = make_request(headers=[(REQUEST_ID_HEADER.lower().encode(), b"inbound-id")])
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        response = await middleware.dispatch(request, call_next)
 
        assert response.headers[REQUEST_ID_HEADER] == "inbound-id"
        patched_collaborators["new_id"].assert_not_called()
 
    @pytest.mark.asyncio
    async def test_generates_request_id_when_header_absent(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        response = await middleware.dispatch(request, call_next)
 
        assert response.headers[REQUEST_ID_HEADER] == "generated-id"
        patched_collaborators["new_id"].assert_called_once()
 
    @pytest.mark.asyncio
    async def test_sets_request_id_on_request_state(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(req):
            assert req.state.request_id == "generated-id"
            return PlainTextResponse("ok")
 
        await middleware.dispatch(request, call_next)
 
 
class TestContextBinding:
    @pytest.mark.asyncio
    async def test_binds_request_id_method_and_endpoint(self, middleware, patched_collaborators):
        request = make_request(method="POST", path="/widgets")
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        await middleware.dispatch(request, call_next)
 
        patched_collaborators["bind"].assert_called_once_with(
            request_id="generated-id", method="POST", endpoint="/widgets"
        )
 
    @pytest.mark.asyncio
    async def test_clears_context_after_successful_response(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        await middleware.dispatch(request, call_next)
 
        patched_collaborators["clear"].assert_called_once()
 
    @pytest.mark.asyncio
    async def test_clears_context_after_unhandled_exception(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(_req):
            raise RuntimeError("downstream failure")
 
        with pytest.raises(RuntimeError):
            await middleware.dispatch(request, call_next)
 
        patched_collaborators["clear"].assert_called_once()
 
 
class TestLoggingBehavior:
    @pytest.mark.asyncio
    async def test_logs_request_started_with_client_ip(self, middleware, patched_collaborators):
        request = make_request(client=("10.0.0.5", 5555))
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        await middleware.dispatch(request, call_next)
 
        patched_collaborators["logger"].info.assert_any_call(
            "request_started", client_ip="10.0.0.5"
        )
 
    @pytest.mark.asyncio
    async def test_logs_request_started_with_none_client_ip_when_no_client(
        self, middleware, patched_collaborators
    ):
        request = make_request(client=None)
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        await middleware.dispatch(request, call_next)
 
        patched_collaborators["logger"].info.assert_any_call(
            "request_started", client_ip=None
        )
 
    @pytest.mark.asyncio
    async def test_logs_request_finished_at_info_for_2xx(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(_req):
            return PlainTextResponse("ok", status_code=200)
 
        await middleware.dispatch(request, call_next)
 
        finished = [
            c
            for c in patched_collaborators["logger"].info.call_args_list
            if c.args[0] == "request_finished"
        ]
        assert len(finished) == 1
        assert finished[0].kwargs["status_code"] == 200
        assert "duration_ms" in finished[0].kwargs
        patched_collaborators["logger"].warning.assert_not_called()
 
    @pytest.mark.asyncio
    async def test_logs_request_finished_at_warning_for_4xx_and_above(
        self, middleware, patched_collaborators
    ):
        request = make_request()
 
        async def call_next(_req):
            return PlainTextResponse("nope", status_code=404)
 
        await middleware.dispatch(request, call_next)
 
        finished = [
            c
            for c in patched_collaborators["logger"].warning.call_args_list
            if c.args[0] == "request_finished"
        ]
        assert len(finished) == 1
        assert finished[0].kwargs["status_code"] == 404
 
    @pytest.mark.asyncio
    async def test_logs_request_failed_with_exc_info_on_exception(
        self, middleware, patched_collaborators
    ):
        request = make_request()
        exc = ValueError("kaboom")
 
        async def call_next(_req):
            raise exc
 
        with pytest.raises(ValueError):
            await middleware.dispatch(request, call_next)
 
        patched_collaborators["logger"].error.assert_called_once()
        args, kwargs = patched_collaborators["logger"].error.call_args
        assert args[0] == "request_failed"
        assert kwargs["status_code"] == 500
        assert kwargs["exc_info"] is exc
        assert "duration_ms" in kwargs
 
    @pytest.mark.asyncio
    async def test_exception_is_reraised_not_swallowed(self, middleware, patched_collaborators):
        request = make_request()
 
        async def call_next(_req):
            raise KeyError("missing")
 
        with pytest.raises(KeyError):
            await middleware.dispatch(request, call_next)
 
 
class TestResponseHeader:
    @pytest.mark.asyncio
    async def test_echoes_request_id_header_on_response(self, middleware, patched_collaborators):
        request = make_request(headers=[(REQUEST_ID_HEADER.lower().encode(), b"round-trip-id")])
 
        async def call_next(_req):
            return PlainTextResponse("ok")
 
        response = await middleware.dispatch(request, call_next)
 
        assert response.headers[REQUEST_ID_HEADER] == "round-trip-id"
 
    @pytest.mark.asyncio
    async def test_does_not_set_response_header_when_exception_raised(
        self, middleware, patched_collaborators
    ):
        request = make_request()
 
        async def call_next(_req):
            raise RuntimeError("boom")
 
        with pytest.raises(RuntimeError):
            result = await middleware.dispatch(request, call_next)
            assert result is None  # unreachable; dispatch raises instead