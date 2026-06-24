import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.api.auth.auth import create_access_token
from app.db.models.user import User


def _make_app(db_session, pipeline=None, redis_storage=None, s3_storage=None):
    """Build a FastAPI app with MLService wired to a real db_session and mocked S3/Redis."""
    from fastapi import FastAPI
    from app.api.v1.ml import router as ml_router, get_ml_service
    from app.db.db_connect import get_db
    from app.services.ml_service import MLService
    from app.repository.image_repo import ImageRepository
    from app.repository.detection_repo import DetectionRepository

    app = FastAPI()
    app.include_router(ml_router)
    app.dependency_overrides[get_db] = lambda: db_session

    mock_pipeline = pipeline or MagicMock()

    mock_s3 = s3_storage or _default_mock_s3()
    mock_redis = redis_storage or _default_mock_redis()

    def override_service():
        return MLService(
            db=db_session,
            s3_storage=mock_s3,
            redis_storage=mock_redis,
            image_repo=ImageRepository(db_session),
            detection_repo=DetectionRepository(db_session),
            pipeline=mock_pipeline
        )

    app.dependency_overrides[get_ml_service] = override_service
    return app, mock_s3, mock_redis


def _default_mock_s3():
    s3 = MagicMock()
    s3.download = AsyncMock(return_value=b"fake image bytes")
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.url/result.jpg")
    return s3


def _default_mock_redis():
    redis = MagicMock()
    # generic state cache
    redis.get_cache_image = AsyncMock(return_value=None)
    redis.cache_image = AsyncMock()
    redis.delete = AsyncMock()
    redis.clear_history = AsyncMock()
    # undo/redo stacks
    redis.pop_undo_state = AsyncMock(return_value=None)
    redis.push_undo_state = AsyncMock()
    redis.pop_redo_state = AsyncMock(return_value=None)
    redis.push_redo_state = AsyncMock()
    redis.get_history_labels = AsyncMock(return_value=[])
    # detection cache (unused here but present on the real client)
    redis.cache_detections = AsyncMock()
    redis.get_cached_detections = AsyncMock(return_value=None)
    return redis


def _auth_headers(user):
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


async def _make_other_user(db_session):
    other = User(username="otheruser", email="other@example.com", password_hash="hashed")
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    return other


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_success(db_session, sample_user, sample_image):
    app, mock_s3, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=b"processed image bytes")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == f"edited_{sample_image.filename}"
    assert data["storage_path"] == "s3://bucket/result.jpg"
    assert data["cache_key"] is None
    assert "id" in data
    assert "uploaded_at" in data

    mock_redis.get_cache_image.assert_awaited_once_with(sample_image.id, suffix='current_state')
    mock_s3.upload_bytes.assert_awaited_once()
    _, kwargs = mock_s3.upload_bytes.await_args
    assert kwargs["content_type"] == "image/jpeg"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_no_processed_state(db_session, sample_user, sample_image):
    # default mock_redis.get_cache_image returns None -> nothing was processed yet
    app, mock_s3, mock_redis = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "No processed result to save. Run remove/replace first."
    mock_s3.upload_bytes.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_image_not_found(db_session, sample_user):
    app, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ml/images/99999/save",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_unauthorized(db_session, sample_user, sample_image):
    other_user = await _make_other_user(db_session)
    app, _, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=b"processed image bytes")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(other_user)
        )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_success(db_session, sample_user, sample_image):
    app, mock_s3, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=b"current state bytes")
    mock_redis.pop_undo_state = AsyncMock(return_value={"bytes": b"previous state bytes", "label": "remove bbox_id=0"})
    mock_redis.get_history_labels = AsyncMock(return_value=["remove bbox_id=0"])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["presigned_url"] == "https://presigned.url/result.jpg"
    assert data["label"] == "remove bbox_id=0"
    assert data["history"] == ["remove bbox_id=0"]

    # current state existed, so it must have been pushed onto the redo stack
    mock_redis.push_redo_state.assert_awaited_once()
    args, kwargs = mock_redis.push_redo_state.await_args
    assert args[0] == sample_image.id
    assert args[1] == b"current state bytes"

    # the popped state must become the new current state
    mock_redis.cache_image.assert_awaited_once()
    _, cache_kwargs = mock_redis.cache_image.await_args
    assert cache_kwargs["image_data"] == b"previous state bytes"
    assert cache_kwargs["suffix"] == "current_state"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_no_current_state_skips_redo_push(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=None)
    mock_redis.pop_undo_state = AsyncMock(return_value={"bytes": b"previous state bytes", "label": "remove bbox_id=0"})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    mock_redis.push_redo_state.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_nothing_to_undo(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.pop_undo_state = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Nothing to undo"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_image_not_found(db_session, sample_user):
    app, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ml/images/99999/undo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_success(db_session, sample_user, sample_image):
    app, mock_s3, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=b"current state bytes")
    mock_redis.pop_redo_state = AsyncMock(return_value={"bytes": b"next state bytes", "label": "redo"})
    mock_redis.get_history_labels = AsyncMock(return_value=["remove bbox_id=0", "redo"])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["presigned_url"] == "https://presigned.url/result.jpg"
    assert data["label"] == "redo"
    assert data["history"] == ["remove bbox_id=0", "redo"]

    # current state existed, so it must have been pushed onto the undo stack
    mock_redis.push_undo_state.assert_awaited_once()
    args, kwargs = mock_redis.push_undo_state.await_args
    assert args[0] == sample_image.id
    assert args[1] == b"current state bytes"
    assert kwargs.get("label") == "redo_checkpoint"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_no_current_state_skips_undo_push(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.get_cache_image = AsyncMock(return_value=None)
    mock_redis.pop_redo_state = AsyncMock(return_value={"bytes": b"next state bytes", "label": "redo"})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    mock_redis.push_undo_state.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_nothing_to_redo(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.pop_redo_state = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Nothing to redo"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_image_not_found(db_session, sample_user):
    app, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ml/images/99999/redo",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_success(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.get_history_labels = AsyncMock(return_value=["remove bbox_id=0", "replace bbox_id=1"])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json() == {"history": ["remove bbox_id=0", "replace bbox_id=1"]}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_empty(db_session, sample_user, sample_image):
    app, _, mock_redis = _make_app(db_session)
    mock_redis.get_history_labels = AsyncMock(return_value=[])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json() == {"history": []}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_image_not_found(db_session, sample_user):
    app, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/ml/images/99999/history",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_unauthorized(db_session, sample_user, sample_image):
    other_user = await _make_other_user(db_session)
    app, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(other_user)
        )

    assert resp.status_code == 403