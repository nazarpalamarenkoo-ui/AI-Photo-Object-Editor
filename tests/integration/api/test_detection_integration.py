import pytest
from httpx import AsyncClient, ASGITransport

from app.api.auth.auth import create_access_token
from app.db.models.detection import Detection
from app.repository.detection_repo import DetectionRepository


def _make_app(db_session, mock_redis):
    from fastapi import FastAPI
    from app.api.v1.detection import router as detection_router, get_detection_service
    from app.db.db_connect import get_db
    from app.services.detection_service import DetectionService
    from app.repository.detection_repo import DetectionRepository
    from app.repository.image_repo import ImageRepository

    app = FastAPI()
    app.include_router(detection_router)
    app.dependency_overrides[get_db] = lambda: db_session

    def override_service():
        return DetectionService(
            db=db_session,
            redis_cache=mock_redis,
            detection_repo=DetectionRepository(db_session),
            image_repo=ImageRepository(db_session)
        )

    app.dependency_overrides[get_detection_service] = override_service
    return app


def _auth_headers(user):
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections(db_session, sample_user, sample_image, sample_detection, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/detections/images/{sample_image.id}",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["detected_class"] == "person"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_not_found(db_session, sample_user, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/detections/images/99999",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_image_detections_wrong_user(db_session, sample_image, sample_detection, mock_redis_cache):
    from app.repository.user_repo import UserRepository
    from passlib.context import CryptContext

    other = await UserRepository(db_session).create(
        username="other2", email="other2@example.com",
        password_hash=CryptContext(schemes=["bcrypt"], deprecated="auto").hash("pw")
    )
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/detections/images/{sample_image.id}",
            headers={"Authorization": f"Bearer {create_access_token({'sub': other.username})}"}
        )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_by_bbox(db_session, sample_user, sample_image, sample_detection, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/detections/images/{sample_image.id}/bbox/{sample_detection.bbox_id}",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json()["bbox_id"] == sample_detection.bbox_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_by_bbox_not_found(db_session, sample_user, sample_image, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/detections/images/{sample_image.id}/bbox/99",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_stats(db_session, sample_user, sample_image, sample_detection, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/detections/images/{sample_image.id}/stats",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_detections"] >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_detection_stats_not_found(db_session, sample_user, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/detections/images/99999/stats",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_detections(db_session, sample_user, sample_image, sample_detection, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete(
            f"/detections/images/{sample_image.id}",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1

    remaining = await DetectionRepository(db_session).get_by_image(sample_image.id)
    assert len(remaining) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_detections_not_found(db_session, sample_user, mock_redis_cache):
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete(
            "/detections/images/99999",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_image_detections_wrong_user(db_session, sample_image, sample_detection, mock_redis_cache):
    from app.repository.user_repo import UserRepository
    from passlib.context import CryptContext

    other = await UserRepository(db_session).create(
        username="other3", email="other3@example.com",
        password_hash=CryptContext(schemes=["bcrypt"], deprecated="auto").hash("pw")
    )
    app = _make_app(db_session, mock_redis_cache)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete(
            f"/detections/images/{sample_image.id}",
            headers={"Authorization": f"Bearer {create_access_token({'sub': other.username})}"}
        )

    assert resp.status_code == 403