import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.api.auth.auth import create_access_token
from app.repository.image_repo import ImageRepository


def _make_app(db_session, mock_s3, mock_redis):
    from fastapi import FastAPI
    from app.api.v1.image import router as image_router, get_image_service
    from app.db.db_connect import get_db
    from app.services.image_service import ImageService
    from app.repository.image_repo import ImageRepository

    app = FastAPI()
    app.include_router(image_router)
    app.dependency_overrides[get_db] = lambda: db_session

    def override_service():
        return ImageService(
            db=db_session,
            s3=mock_s3,
            redis_cache=mock_redis,
            image_repo=ImageRepository(db_session)
        )

    app.dependency_overrides[get_image_service] = override_service
    return app


def _auth_headers(user):
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_image(db_session, sample_user, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/images/upload",
            files={"file": ("test.jpg", BytesIO(b"fake image"), "image/jpeg")},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["filename"] == "test.jpg"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_image_unauthorized(db_session, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/images/upload",
            files={"file": ("test.jpg", BytesIO(b"data"), "image/jpeg")}
        )

    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_user_images(db_session, sample_user, multiple_images, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/images/", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert len(resp.json()) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_user_images_pagination(db_session, sample_user, multiple_images, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/images/?limit=2&offset=0", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_by_id(db_session, sample_user, sample_image, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/images/{sample_image.id}", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json()["id"] == sample_image.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_not_found(db_session, sample_user, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/images/99999", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_wrong_user(db_session, sample_image, mock_s3_storage, mock_redis_cache):
    from app.db.models.user import User
    from app.repository.user_repo import UserRepository
    from passlib.context import CryptContext

    other = await UserRepository(db_session).create(
        username="other", email="other@example.com",
        password_hash=CryptContext(schemes=["bcrypt"], deprecated="auto").hash("pw")
    )
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/images/{sample_image.id}",
            headers={"Authorization": f"Bearer {create_access_token({'sub': other.username})}"}
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_image(db_session, sample_user, sample_image, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/images/{sample_image.id}/download", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert "attachment" in resp.headers["content-disposition"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_presigned_url(db_session, sample_user, sample_image, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/images/{sample_image.id}/url?expiration=3600",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "url" in data
    assert data["expires_in"] == 3600


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image(db_session, sample_user, sample_image, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete(f"/images/{sample_image.id}", headers=_auth_headers(sample_user))

    assert resp.status_code == 204

    deleted = await ImageRepository(db_session).get_by_id(sample_image.id)
    assert deleted is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_not_found(db_session, sample_user, mock_s3_storage, mock_redis_cache):
    app = _make_app(db_session, mock_s3_storage, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/images/99999", headers=_auth_headers(sample_user))

    assert resp.status_code == 404