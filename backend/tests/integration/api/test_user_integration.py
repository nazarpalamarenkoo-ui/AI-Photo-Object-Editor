import pytest
from passlib.context import CryptContext
from httpx import AsyncClient, ASGITransport

from app.api.auth.auth import create_access_token
from app.repository.user_repo import UserRepository

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _make_app(db_session):
    from fastapi import FastAPI
    from app.api.v1.user import router as user_router
    from app.api.auth.auth import get_current_user
    from app.db.db_connect import get_db

    app = FastAPI()
    app.include_router(user_router)
    app.dependency_overrides[get_db] = lambda: db_session
    return app


def _auth_headers(user):
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_me(db_session, sample_user):
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/users/me", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == sample_user.email
    assert data["username"] == sample_user.username


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_me_unauthorized(db_session):
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/users/me")

    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_me_username(db_session, sample_user):
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/users/me",
            json={"username": "updated_user", "email": sample_user.email},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json()["username"] == "updated_user"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_me_invalid_data(db_session, sample_user):
    repo = UserRepository(db_session)
    other = await repo.create(
        username="other", email="other@example.com", password_hash=pwd_context.hash("pw")
    )
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/users/me",
            json={"username": "x", "email": other.email},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 400


@pytest.mark.integration
@pytest.mark.asyncio
async def test_change_password_success(db_session):
    repo = UserRepository(db_session)
    user = await repo.create(
        username="pwuser", email="pw@example.com", password_hash=pwd_context.hash("oldpass")
    )
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/users/me/password",
            json={"old_password": "oldpass", "new_password": "Newpass123"},
            headers=_auth_headers(user)
        )

    assert resp.status_code == 200
    assert resp.json()["detail"] == "Password updated successfully"

    updated = await repo.get_by_id(user.id)
    assert pwd_context.verify("Newpass123", updated.password_hash)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_change_password_wrong_old(db_session):
    repo = UserRepository(db_session)
    user = await repo.create(
        username="pwuser2", email="pw2@example.com", password_hash=pwd_context.hash("correct")
    )
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/users/me/password",
            json={"old_password": "wrong", "new_password": "newpass"},
            headers=_auth_headers(user)
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_me(db_session):
    repo = UserRepository(db_session)
    user = await repo.create(
        username="todelete", email="del@example.com", password_hash=pwd_context.hash("pw")
    )
    app = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/users/me", headers=_auth_headers(user))

    assert resp.status_code == 200
    assert resp.json()["detail"] == "Account deleted successfully"

    deleted = await repo.get_by_id(user.id)
    assert deleted is None