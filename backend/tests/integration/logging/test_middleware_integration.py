from __future__ import annotations
 
from unittest.mock import AsyncMock, patch
 
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
 
from app.core.logging.middleware import REQUEST_ID_HEADER, RequestLoggingMiddleware

pytestmark = pytest.mark.integration

def build_app() -> Starlette:
    async def ok(request):
        return JSONResponse({"request_id": request.state.request_id})
 
    async def not_found(request):
        return PlainTextResponse("nope", status_code=404)
 
    async def boom(request):
        raise RuntimeError("integration boom")
 
    app = Starlette(
        routes=[
            Route("/ok", ok),
            Route("/missing", not_found),
            Route("/boom", boom),
        ]
    )
    app.add_middleware(RequestLoggingMiddleware)
    return app
 
 
@pytest.fixture
def client():
    return TestClient(build_app(), raise_server_exceptions=False)
 
 
class TestRequestLoggingMiddlewareIntegration:
    def test_successful_request_gets_200_and_request_id_header(self, client):
        response = client.get("/ok")
 
        assert response.status_code == 200
        assert REQUEST_ID_HEADER in response.headers
        assert response.headers[REQUEST_ID_HEADER]  # non-empty
 
    def test_generated_request_id_is_visible_to_the_route_handler(self, client):
        response = client.get("/ok")
 
        body = response.json()
        assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
 
    def test_inbound_request_id_header_is_echoed_back_unchanged(self, client):
        response = client.get("/ok", headers={REQUEST_ID_HEADER: "client-supplied-id"})
 
        assert response.headers[REQUEST_ID_HEADER] == "client-supplied-id"
        assert response.json()["request_id"] == "client-supplied-id"
 
    def test_each_request_without_header_gets_a_distinct_id(self, client):
        first = client.get("/ok").headers[REQUEST_ID_HEADER]
        second = client.get("/ok").headers[REQUEST_ID_HEADER]
        assert first != second
 
    def test_404_response_still_carries_request_id_header(self, client):
        response = client.get("/missing")
 
        assert response.status_code == 404
        assert REQUEST_ID_HEADER in response.headers
 
    def test_unhandled_exception_surfaces_as_500_to_the_client(self, client):
        response = client.get("/boom")
 
        assert response.status_code == 500
 
    def test_context_is_cleared_between_requests(self, client):
        # Bind a distinguishing request id on the first call, then make sure
        # a second call (with no inbound header) doesn't somehow inherit it
        # via leftover contextvars state from the first request.
        first = client.get("/ok", headers={REQUEST_ID_HEADER: "sticky-id"})
        assert first.json()["request_id"] == "sticky-id"
 
        second = client.get("/ok")
        assert second.json()["request_id"] != "sticky-id"