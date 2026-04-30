import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from datetime import datetime

from app.api.auth.auth import create_access_token
from app.db.models.detection import Detection
from app.repository.detection_repo import DetectionRepository


def _make_app(db_session, mock_pipeline):
    from fastapi import FastAPI
    from app.api.v1.ml import router as ml_router, get_ml_service
    from app.db.db_connect import get_db
    from app.services.ml_service import MLService
    from app.repository.image_repo import ImageRepository
    from app.repository.detection_repo import DetectionRepository

    app = FastAPI()
    app.include_router(ml_router)
    app.dependency_overrides[get_db] = lambda: db_session

    mock_s3 = MagicMock()
    mock_s3.download = AsyncMock(return_value=b"fake image bytes")
    mock_s3.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
    mock_s3.get_presigned_url = AsyncMock(return_value="https://presigned.url/result.jpg")

    mock_redis = MagicMock()
    mock_redis.get_cache_image = AsyncMock(return_value=None)
    mock_redis.cache_image = AsyncMock()
    mock_redis.cache_detections = AsyncMock()
    mock_redis.get_cached_detections = AsyncMock(return_value=None)
    mock_redis.delete = AsyncMock()

    def override_service():
        svc = MLService(
            db=db_session,
            s3_storage=mock_s3,
            redis_storage=mock_redis,
            image_repo=ImageRepository(db_session),
            detection_repo=DetectionRepository(db_session),
            pipeline=mock_pipeline
        )
        return svc

    app.dependency_overrides[get_ml_service] = override_service
    return app


def _auth_headers(user):
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


def _mock_pipeline_detect(detections):
    pipeline = MagicMock()
    pipeline.detect_objects = AsyncMock(return_value={
        "detections": detections,
        "image_size": (640, 480),
        "metrics": {"num_detections": len(detections), "inference_time_ms": 50.0},
        "timestamp": datetime.now().isoformat()
    })
    return pipeline


def _mock_pipeline_remove():
    pipeline = MagicMock()
    pipeline.remove_object = AsyncMock(return_value={
        "result_bytes": b"processed image",
        "metrics": {"processing_time_ms": 200.0, "mode": "remove"},
        "timestamp": datetime.now().isoformat()
    })
    return pipeline


def _mock_pipeline_replace():
    pipeline = MagicMock()
    pipeline.replace_object = AsyncMock(return_value={
        "result_bytes": b"replaced image",
        "metrics": {"processing_time_ms": 300.0, "mode": "remove"},
        "timestamp": datetime.now().isoformat()
    })
    return pipeline


def _mock_pipeline_remove_multiple():
    pipeline = MagicMock()
    pipeline.remove_multiple_objects = AsyncMock(return_value={
        "result_bytes": b"multi removed",
        "metrics": {"processing_time_ms": 400.0, "mode": "remove"},
        "timestamp": datetime.now().isoformat()
    })
    return pipeline


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects(db_session, sample_user, sample_image):
    detections = [
        {"bbox_id": 0, "x1": 10, "y1": 10, "x2": 100, "y2": 100, "detected_class": "person", "confidence": 0.95}
    ]
    pipeline = _mock_pipeline_detect(detections)
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/detect",
            json={"conf_threshold": 0.5},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["detections"]) == 1
    assert data["detections"][0]["detected_class"] == "person"

    saved = await DetectionRepository(db_session).get_by_image(sample_image.id)
    assert len(saved) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_not_found(db_session, sample_user):
    pipeline = _mock_pipeline_detect([])
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ml/images/99999/detect",
            json={"conf_threshold": 0.5},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object(db_session, sample_user, sample_image, sample_detection):
    pipeline = _mock_pipeline_remove()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/{sample_detection.bbox_id}",
            json={"expand_mask_pixels": 5, "use_edge_blending": True},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "result_url" in data
    assert "presigned_url" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_bbox_not_found(db_session, sample_user, sample_image):
    pipeline = _mock_pipeline_remove()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/99",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object(db_session, sample_user, sample_image, sample_detection):
    pipeline = _mock_pipeline_replace()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/{sample_detection.bbox_id}",
            files={"replacement_file": ("dog.jpg", BytesIO(b"dog image bytes"), "image/jpeg")},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "result_url" in data
    assert "presigned_url" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_not_found(db_session, sample_user, sample_image):
    pipeline = _mock_pipeline_replace()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/99",
            files={"replacement_file": ("dog.jpg", BytesIO(b"data"), "image/jpeg")},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects(db_session, sample_user, sample_image, sample_detection):
    pipeline = _mock_pipeline_remove_multiple()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple",
            json={"bbox_ids": [sample_detection.bbox_id], "expand_mask_pixels": 5, "use_edge_blending": True},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert "result_url" in resp.json()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_invalid_bbox_ids(db_session, sample_user, sample_image):
    pipeline = _mock_pipeline_remove_multiple()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple",
            json={"bbox_ids": [99, 100], "expand_mask_pixels": 5, "use_edge_blending": True},
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state(db_session, sample_user, sample_image):
    pipeline = MagicMock()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/reset",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 200
    assert resp.json()["detail"] == "State reset to original image"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_not_found(db_session, sample_user):
    pipeline = MagicMock()
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ml/images/99999/reset",
            headers=_auth_headers(sample_user)
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_supported_classes(db_session, sample_user):
    pipeline = MagicMock()
    pipeline.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])
    app = _make_app(db_session, pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/classes", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert "person" in resp.json()