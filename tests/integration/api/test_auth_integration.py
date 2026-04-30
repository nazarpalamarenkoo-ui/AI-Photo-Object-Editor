import sys
from unittest.mock import MagicMock, AsyncMock

mock_fastapi_mail = MagicMock()
mock_fastapi_mail.FastMail.return_value.send_message = AsyncMock()

sys.modules["fastapi_mail"] = mock_fastapi_mail
sys.modules["fastapi_mail.config"] = mock_fastapi_mail

import pytest
from httpx import AsyncClient, ASGITransport


def _make_app(db_session, monkeypatch):
    from fastapi import FastAPI
    from app.api.auth.routes import router as auth_router, get_user_service
    from app.db.db_connect import get_db

    app = FastAPI()
    app.include_router(auth_router)

    app.dependency_overrides[get_db] = lambda: db_session

    mock_service = MagicMock()
    mock_service.authenticate_user = AsyncMock()
    mock_service.create_user = AsyncMock()

    mock_service.user_repo = MagicMock()
    mock_service.user_repo.exists_by_email = AsyncMock(return_value=False)
    mock_service.user_repo.get_by_email = AsyncMock()
    mock_service.user_repo.update_password = AsyncMock()

    def override_service():
        return mock_service

    app.dependency_overrides[get_user_service] = override_service
    
    monkeypatch.setattr(
        "app.api.auth.mail.send_confirmation_email",
        AsyncMock()
    )

    return app, mock_service

@pytest.mark.integration
@pytest.mark.asyncio
async def test_login_success(db_session, monkeypatch):
    app, mock_service = _make_app(db_session, monkeypatch)

    mock_service.authenticate_user.return_value = MagicMock(
        username="test_user"
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/auth/login",
            json={"email": "test@test.com", "password": "Password123"}
        )

    assert resp.status_code == 200

@pytest.mark.integration
@pytest.mark.asyncio
async def test_login_invalid_credentials(db_session, monkeypatch):
    app, mock_service = _make_app(db_session, monkeypatch)

    mock_service.authenticate_user.return_value = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/auth/login",
            json={"email": "bad@test.com", "password": "wrongpass"}
        )

    assert resp.status_code == 400

@pytest.mark.integration
@pytest.mark.asyncio
async def test_signup_sends_email(db_session, monkeypatch):
    app, mock_service = _make_app(db_session, monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/auth/signup",
            json={
                "email": "new@test.com",
                "password": "Password123",
                "username": "new_user"
            }
        )

    assert resp.status_code == 200

@pytest.mark.integration
@pytest.mark.asyncio
async def test_signup_existing_email(db_session, monkeypatch):
    app, mock_service = _make_app(db_session, monkeypatch)

    mock_service.user_repo.exists_by_email = AsyncMock(return_value=True)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/auth/signup",
            json={
                "email": "exist@test.com",
                "password": "Password123",
                "username": "user"
            }
        )

    assert resp.status_code == 400

@pytest.mark.integration
@pytest.mark.asyncio
async def test_signup_confirmation(db_session, monkeypatch):
    from app.api.auth.mail import generate_signup_confirmation_token
    from app.api.auth.schema import SignUpArgs

    app, mock_service = _make_app(db_session, monkeypatch)

    args = SignUpArgs(
        email="test@test.com",
        password="Password123",
        username="test_user"
    )

    token = generate_signup_confirmation_token(args)

    mock_service.create_user.return_value = MagicMock(username="test_user")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/auth/signup-confirmation",
            params={"token": token}
        )

    assert resp.status_code == 200

@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_password(db_session, monkeypatch):
    from app.api.auth.mail import generate_password_reset_token

    app, mock_service = _make_app(db_session, monkeypatch)

    email = "user@test.com"
    token = generate_password_reset_token(email)

    mock_service.user_repo.get_by_email = AsyncMock(return_value=MagicMock())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/auth/reset-password?new_password=Password123&token={token}"
        )

    assert resp.status_code == 200