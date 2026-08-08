"""
Integration tests for the ML API routers.

Architecture notes:
- Sync routes (detect, remove, replace, segment, etc.) go through
  run_tracked(), which instantiates the service class from get_base_deps().
  We patch run_tracked at the router module level so the real service is
  never constructed.
- Session routes (reset, save, undo, redo, history, current) depend on
  get_version_history() from deps.py — overridden directly.
- Async routes depend on get_arq_pool() — overridden directly.
- Asset routes depend on get_asset() — overridden directly.
- Job-status route uses arq.jobs.Job — patched in ml_jobs_module.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from httpx import AsyncClient, ASGITransport
from arq.jobs import JobStatus

from app.api.auth.auth import create_access_token
from app.db.models.user import User

import app.api.v1.ml.jobs as ml_jobs_module

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

ML_RESULT = {
    "result_url": "s3://bucket/result.jpg",
    "presigned_url": "https://presigned.url/result.jpg",
    "metrics": {},
    "timestamp": "2025-01-01T00:00:00",
}

SEGMENT_RESULT = {
    "segments": [],
    "metrics": {},
    "image_size": [640, 480],
    "timestamp": "2025-01-01T00:00:00",
}

EXTRACT_RESULT = {
    "asset_id": "asset-1",
    "extracted_url": "s3://bucket/obj.png",
    "presigned_url": "https://presigned.url/obj.png",
    "storage_path": "s3://bucket/obj.png",
    "object_size": [50, 60],
    "area_pixels": 3000,
    "cropped_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 60},
    "timestamp": "2025-01-01T00:00:00",
}


def _auth_headers(user: User) -> dict:
    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}


def _mock_pool() -> MagicMock:
    pool = MagicMock()
    job = MagicMock()
    job.job_id = "job-123"
    pool.enqueue_job = AsyncMock(return_value=job)
    return pool


def _mock_asset_service() -> MagicMock:
    svc = MagicMock()
    svc.list_assets = AsyncMock(return_value=[
        {
            "asset_id": "a1", "width": 50, "height": 60, "area_pixels": 3000,
            "public_id": "pub-a1", "storage_path": "s3://bucket/a1.png",
            "content_type": "image/png", "created_at": "2025-01-01T00:00:00",
        },
        {
            "asset_id": "a2", "width": 50, "height": 60, "area_pixels": 3000,
            "public_id": "pub-a2", "storage_path": "s3://bucket/a2.png",
            "content_type": "image/png", "created_at": "2025-01-01T00:00:00",
        },
    ])
    svc.get_asset_thumbnail = AsyncMock(return_value=b"thumb")
    svc.get_asset_image = AsyncMock(return_value=b"asset-bytes")
    svc.rename_asset = AsyncMock(return_value={
        "asset_id": "asset-1",
        "source_image_id": 1,
        "width": 50, "height": 60, "area_pixels": 3000,
        "public_id": "pub-asset-1", "storage_path": "s3://bucket/asset-1.png",
        "content_type": "image/png",
        "label": "renamed", "s3_url": None,
        "created_at": "2025-01-01T00:00:00",
    })
    svc.delete_asset = AsyncMock()
    svc.paste_extracted_object = AsyncMock(return_value={
        "result_url": "s3://bucket/pasted.jpg",
        "presigned_url": "https://presigned.url/pasted.jpg",
        "paste_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        "object_size": [50, 50],
        "timestamp": "2025-01-01T00:00:00",
    })
    return svc


def _mock_version_history_service() -> MagicMock:
    svc = MagicMock()
    svc.get_current_state = AsyncMock(return_value={"presigned_url": "https://presigned.url/current.jpg"})
    svc.reset_current_state = AsyncMock()
    svc.save_result = AsyncMock(return_value={
        "id": 42,
        "filename": "edited.jpg",
        "storage_path": "s3://bucket/edited.jpg",
        "uploaded_at": "2025-01-01T00:00:00",
        "cache_key": None,
    })
    svc.undo = AsyncMock(return_value={"detail": "Undone"})
    svc.redo = AsyncMock(return_value={"detail": "Redone"})
    svc.get_history = AsyncMock(return_value={"history": []})
    return svc


def _make_app(
    db_session,
    *,
    pool: MagicMock | None = None,
    asset_svc: MagicMock | None = None,
    version_svc: MagicMock | None = None,
):
    from fastapi import FastAPI
    from app.api.v1.ml import router as ml_router
    from app.api.v1.ml.deps import get_arq_pool, get_asset, get_version_history, get_base_deps
    from app.db.db_connect import get_db

    _pool = pool or _mock_pool()
    _asset = asset_svc or _mock_asset_service()
    _version = version_svc or _mock_version_history_service()

    app = FastAPI()
    app.include_router(ml_router)

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_arq_pool] = lambda: _pool
    app.dependency_overrides[get_asset] = lambda: _asset
    app.dependency_overrides[get_version_history] = lambda: _version
    app.dependency_overrides[get_base_deps] = lambda: {}   # ← новий рядок

    return app, _pool, _asset, _version


async def _other_user(db_session) -> User:
    from passlib.context import CryptContext
    from app.repository.user_repo import UserRepository
    return await UserRepository(db_session).create(
        username="otheruser",
        email="other@example.com",
        password_hash=CryptContext(schemes=["bcrypt"], deprecated="auto").hash("pw"),
    )


# ──────────────────────────────────────────────
# detect.py — /ml/images/{id}/detect
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)
    result = {"detections": [{"class": "person"}], "count": 1}

    with patch("app.api.v1.ml.detect.run_tracked", new=AsyncMock(return_value=result)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/detect",
                json={"conf_threshold": 0.7, "classes": ["person"]},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    assert resp.json()["count"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_default_body(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.detect.run_tracked", new=AsyncMock(return_value={"detections": [], "count": 0})):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/detect",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_invalid_threshold_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/detect",
            json={"conf_threshold": 1.5},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_image_not_found_returns_404(db_session, sample_user):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.detect.run_tracked", side_effect=ValueError("Image not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ml/images/99999/detect", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_unauthorized_returns_403(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.detect.run_tracked", side_effect=ValueError("unauthorized")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/detect",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_no_auth_returns_401(db_session, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/ml/images/{sample_image.id}/detect")

    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_supported_classes(db_session, sample_user):
    app, *_ = _make_app(db_session)

    from app.api.v1.ml.deps import get_detector
    mock_det = MagicMock()
    mock_det.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])
    app.dependency_overrides[get_detector] = lambda: mock_det

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/classes", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json() == ["person", "car", "dog"]


# ──────────────────────────────────────────────
# editing.py — remove / remove-multiple / replace (sync)
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.editing.run_tracked", new=AsyncMock(return_value=ML_RESULT)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/remove/1",
                json={"expand_mask_pixels": 10, "use_edge_blending": False},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    assert resp.json()["result_url"] == ML_RESULT["result_url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_default_body(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.editing.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/remove/1",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["expand_mask_pixels"] == 5
    assert kwargs["use_edge_blending"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_bbox_not_found_returns_404(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.editing.run_tracked", side_effect=ValueError("bbox not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/remove/999",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_invalid_expand_mask_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1",
            json={"expand_mask_pixels": 999},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.editing.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/remove-multiple",
                json={"bbox_ids": [1, 2, 3]},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["bbox_ids"] == [1, 2, 3]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_missing_bbox_ids_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple",
            json={},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_unauthorized_returns_403(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.editing.run_tracked", side_effect=ValueError("unauthorized")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/remove-multiple",
                json={"bbox_ids": [1]},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.editing.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/replace/1",
                files={"replacement_file": ("img.png", b"fake-bytes", "image/png")},
                params={"color_match_method": "histogram", "expand_mask_pixels": 12},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["replace_image_bytes"] == b"fake-bytes"
    assert kwargs["color_match_method"] == "histogram"
    assert kwargs["expand_mask_pixels"] == 12


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_missing_file_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_invalid_color_method_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            files={"replacement_file": ("img.png", b"fake-bytes", "image/png")},
            params={"color_match_method": "bad_method"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_generic_error_returns_400(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.editing.run_tracked", side_effect=ValueError("invalid replacement image")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/replace/1",
                files={"replacement_file": ("img.png", b"fake-bytes", "image/png")},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 400


# ──────────────────────────────────────────────
# editing.py — diffusion replace (sync + async)
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.editing.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/replace/diffusion",
                files={
                    "mask_file": ("mask.png", b"mask-bytes", "image/png"),
                    "reference_file": ("ref.png", b"ref-bytes", "image/png"),
                },
                params={
                    "bbox_x1": 10,
                    "bbox_y1": 20,
                    "bbox_x2": 110,
                    "bbox_y2": 120,
                    "prompt": "replace object",
                    "use_color_matching": False,
                    "color_match_method": "color_transfer",
                    "seed": 0,
                },
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["mask_bytes"] == b"mask-bytes"
    assert kwargs["reference_image_bytes"] == b"ref-bytes"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_missing_reference_returns_400(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/diffusion",
            files={"mask_file": ("mask.png", b"mask-bytes", "image/png")},
            params={
                "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 110, "bbox_y2": 120,
                "prompt": "replace object",
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide reference_file or asset_id"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_with_asset_id(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=b"asset-ref-bytes")

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.editing.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/replace/diffusion",
                files={"mask_file": ("mask.png", b"mask-bytes", "image/png")},
                params={
                    "asset_id": "asset-1",
                    "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 110, "bbox_y2": 120,
                    "prompt": "replace object",
                },
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["reference_image_bytes"] == b"asset-ref-bytes"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_asset_not_found_returns_404(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/diffusion",
            files={"mask_file": ("mask.png", b"mask-bytes", "image/png")},
            params={
                "asset_id": "missing",
                "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 110, "bbox_y2": 120,
                "prompt": "replace object",
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_pipeline_error_returns_502(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.editing.run_tracked", side_effect=RuntimeError("NaN in output")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/replace/diffusion",
                files={
                    "mask_file": ("mask.png", b"mask-bytes", "image/png"),
                    "reference_file": ("ref.png", b"ref-bytes", "image/png"),
                },
                params={
                    "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 110, "bbox_y2": 120,
                    "prompt": "replace object",
                },
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 502


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_diffusion_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/diffusion/async",
            files={
                "mask_file": ("mask.png", b"mask-bytes", "image/png"),
                "reference_file": ("ref.png", b"ref-bytes", "image/png"),
            },
            params={
                "bbox_x1": 10, "bbox_y1": 20, "bbox_x2": 110, "bbox_y2": 120,
                "prompt": "replace object",
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["mask_bytes"] == b"mask-bytes"
    assert kwargs["reference_image_bytes"] == b"ref-bytes"
    assert kwargs["image_id"] == sample_image.id
    assert kwargs["user_id"] == sample_user.id

@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1/async",
            json={"expand_mask_pixels": 8, "use_edge_blending": True},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "remove_object_task",
        image_id=sample_image.id,
        bbox_id=1,
        user_id=sample_user.id,
        expand_mask_pixels=8,
        use_edge_blending=True,
        ldm_steps=ANY,
        ldm_sampler=ANY,
        hd_strategy=ANY,
        _trace_carrier=ANY,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple/async",
            json={"bbox_ids": [1, 2]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["bbox_ids"] == [1, 2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1/async",
            files={"replacement_file": ("img.png", b"replace-bytes", "image/png")},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["replace_image_bytes"] == b"replace-bytes"
    assert kwargs["bbox_id"] == 1


# ──────────────────────────────────────────────
# session.py — current / reset / save / undo / redo / history
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_current_state_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/current",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_version.get_current_state.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_current_state_not_found_returns_404(db_session, sample_user):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.get_current_state.side_effect = ValueError("not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/images/99999/current", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/reset",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "State reset to original image"}
    mock_version.reset_current_state.assert_awaited_once_with(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_not_found_returns_404(db_session, sample_user):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.reset_current_state.side_effect = ValueError("not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ml/images/99999/reset", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json()["id"] == 42
    mock_version.save_result.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_no_state_returns_400(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.save_result.side_effect = ValueError("No processed result to save. Run remove/replace first.")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert "No processed result" in resp.json()["detail"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_unauthorized_returns_403(db_session, sample_user, sample_image):
    other = await _other_user(db_session)
    app, _, _, mock_version = _make_app(db_session)
    mock_version.save_result.side_effect = ValueError("unauthorized")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(other),
        )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Undone"}
    mock_version.undo.assert_awaited_once_with(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_nothing_to_undo_returns_400(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.undo.side_effect = ValueError("Nothing to undo")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Nothing to undo"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Redone"}
    mock_version.redo.assert_awaited_once_with(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_nothing_to_redo_returns_400(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.redo.side_effect = ValueError("Nothing to redo")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Nothing to redo"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_success(db_session, sample_user, sample_image):
    app, _, _, mock_version = _make_app(db_session)
    mock_version.get_history = AsyncMock(return_value={"history": ["step1", "step2"]})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"history": ["step1", "step2"]}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_unauthorized_returns_403(db_session, sample_user, sample_image):
    other = await _other_user(db_session)
    app, _, _, mock_version = _make_app(db_session)
    mock_version.get_history.side_effect = ValueError("unauthorized")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(other),
        )

    assert resp.status_code == 403


# ──────────────────────────────────────────────
# segmentation.py — sync
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment",
                json={"min_area": 200, "max_segments": 20},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["min_area"] == 200
    assert kwargs["max_segments"] == 20


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_default_body(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["min_area"] == 500
    assert kwargs["max_segments"] == 50


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_not_found_returns_404(db_session, sample_user):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.segmentation.run_tracked", side_effect=ValueError("image not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ml/images/99999/segment", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/prompt",
                json={"point_coords": [[10, 20]], "point_labels": [1]},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["point_labels"] == [1]
    assert kwargs["bbox"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_with_bbox(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/prompt",
                json={"bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["bbox"] == {"x1": 0, "y1": 0, "x2": 50, "y2": 50}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_no_detections_returns_404(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.segmentation.run_tracked", side_effect=ValueError("no valid detections")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/prompt",
                json={},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/polygon",
                json={"points": [[0, 0], [10, 0], [5, 10]]},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["smooth"] is True
    assert kwargs["feather_px"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_too_few_points_returns_422(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon",
            json={"points": [[0, 0], [10, 0]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=SEGMENT_RESULT)
    with patch("app.api.v1.ml.segmentation.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/hybrid",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_not_found_returns_404(db_session, sample_user):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.segmentation.run_tracked", side_effect=ValueError("image not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ml/images/99999/segment/hybrid", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


# ──────────────────────────────────────────────
# segmentation.py — async
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/async",
            json={"min_area": 100, "max_segments": 10},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_objects_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        min_area=100,
        max_segments=10,
        _trace_carrier=ANY,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt/async",
            json={"point_coords": [[5, 5]], "point_labels": [1]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["point_labels"] == [1]
    assert kwargs["bbox"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon/async",
            json={"points": [[0, 0], [10, 0], [5, 10]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["smooth"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/hybrid/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once()
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["image_id"] == sample_image.id
    assert kwargs["user_id"] == sample_user.id


# ──────────────────────────────────────────────
# sam_ops.py — SAM remove / replace / extract (sync)
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.sam_ops.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/3/remove",
                json={"expand_mask_pixels": 12},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["mask_id"] == 3
    assert kwargs["expand_mask_pixels"] == 12


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_not_found_returns_404(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.sam_ops.run_tracked", side_effect=ValueError("mask not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/999/remove",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_success_with_file(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.sam_ops.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/3/replace",
                files={"replacement_file": ("rep.png", b"rep-bytes", "image/png")},
                params={"expand_mask_pixels": 15},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["replacement_image_bytes"] == b"rep-bytes"
    assert kwargs["expand_mask_pixels"] == 15
    assert kwargs["replacement_is_cutout"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_with_asset_id(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=b"asset-bytes")

    tracker = AsyncMock(return_value=ML_RESULT)
    with patch("app.api.v1.ml.sam_ops.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/3/replace",
                params={"asset_id": "asset-1"},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    _, kwargs = tracker.call_args
    assert kwargs["replacement_image_bytes"] == b"asset-bytes"
    assert kwargs["replacement_is_cutout"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_missing_source_returns_400(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide replacement_file or asset_id"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_asset_not_found_returns_404(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            params={"asset_id": "missing"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_success(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    tracker = AsyncMock(return_value=EXTRACT_RESULT)
    with patch("app.api.v1.ml.sam_ops.run_tracked", new=tracker):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/3/extract",
                json={"padding_pixels": 10, "label": "chair", "persist_to_s3": True},
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 200
    assert resp.json()["asset_id"] == "asset-1"
    _, kwargs = tracker.call_args
    assert kwargs["mask_id"] == 3
    assert kwargs["padding_pixels"] == 10
    assert kwargs["label"] == "chair"
    assert kwargs["persist_to_s3"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_not_found_returns_404(db_session, sample_user, sample_image):
    app, *_ = _make_app(db_session)

    with patch("app.api.v1.ml.sam_ops.run_tracked", side_effect=ValueError("mask not found")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/ml/images/{sample_image.id}/segment/999/extract",
                headers=_auth_headers(sample_user),
            )

    assert resp.status_code == 404


# ──────────────────────────────────────────────
# sam_ops.py — async
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/remove/async",
            json={"expand_mask_pixels": 8},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "sam_remove_object_task",
        image_id=sample_image.id,
        mask_id=3,
        user_id=sample_user.id,
        expand_mask_pixels=8,
        use_edge_blending=ANY,
        ldm_steps=ANY,
        ldm_sampler=ANY,
        hd_strategy=ANY,
        _trace_carrier=ANY,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_with_file(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            files={"replacement_file": ("rep.png", b"rep-bytes", "image/png")},
            params={"expand_mask_pixels": 15},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["replacement_image_bytes"] == b"rep-bytes"
    assert kwargs["expand_mask_pixels"] == 15
    assert kwargs["replacement_is_cutout"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_with_asset_id(db_session, sample_user, sample_image):
    app, mock_pool, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=b"asset-bytes")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            params={"asset_id": "asset-1"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_asset.get_asset_image.assert_awaited_once_with(sample_user.id, "asset-1")
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["replacement_image_bytes"] == b"asset-bytes"
    assert kwargs["replacement_is_cutout"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_missing_source_returns_400(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide replacement_file or asset_id"
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_asset_not_found_returns_404(db_session, sample_user, sample_image):
    app, mock_pool, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            params={"asset_id": "missing"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract/async",
            json={"padding_pixels": 20, "label": "my-object", "persist_to_s3": True},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "sam_extract_object_task",
        image_id=sample_image.id,
        mask_id=3,
        user_id=sample_user.id,
        padding_pixels=20,
        label="my-object",
        persist_to_s3=True,
        _trace_carrier=ANY,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_async_default_body(db_session, sample_user, sample_image):
    app, mock_pool, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_pool.enqueue_job.assert_awaited_once_with(
        "sam_extract_object_task",
        image_id=sample_image.id,
        mask_id=3,
        user_id=sample_user.id,
        padding_pixels=8,
        label=None,
        persist_to_s3=False,
        _trace_carrier=ANY,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_success(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert len(resp.json()) == 2
    mock_asset.list_assets.assert_awaited_once_with(sample_user.id, limit=50, offset=0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_pagination(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.list_assets = AsyncMock(return_value=[])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets?limit=10&offset=5", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    mock_asset.list_assets.assert_awaited_once_with(sample_user.id, limit=10, offset=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_no_auth_returns_401(db_session):
    app, *_ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets")

    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_thumbnail_success(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_thumbnail = AsyncMock(return_value=b"PNG-data")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/asset-1/thumbnail", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.content == b"PNG-data"
    assert resp.headers["content-type"] == "image/png"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_thumbnail_not_found_returns_404(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_thumbnail = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/missing/thumbnail", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_image_success(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=b"full-PNG")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/asset-1/image", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.content == b"full-PNG"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_image_not_found_returns_404(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/missing/image", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rename_asset_success(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/ml/assets/asset-1",
            json={"label": "new-name"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_asset.rename_asset.assert_awaited_once_with(sample_user.id, "asset-1", "new-name")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rename_asset_not_found_returns_404(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.rename_asset.side_effect = ValueError("asset not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/ml/assets/missing",
            json={"label": "x"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_asset_success(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/ml/assets/asset-1", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Asset deleted"}
    mock_asset.delete_asset.assert_awaited_once_with(sample_user.id, "asset-1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_asset_not_found_returns_404(db_session, sample_user):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.delete_asset.side_effect = ValueError("asset not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/ml/assets/missing", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_success(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)

    payload = {
        "asset_id": "asset-1",
        "target_bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        "scale": 1.0,
        "use_color_matching": True,
        "use_edge_blending": False,
        "color_match_method": "histogram",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json=payload,
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_asset.paste_extracted_object.assert_awaited_once()
    _, kwargs = mock_asset.paste_extracted_object.await_args
    assert kwargs["asset_id"] == "asset-1"
    assert kwargs["image_id"] == sample_image.id
    assert kwargs["user_id"] == sample_user.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_not_found_returns_404(db_session, sample_user, sample_image):
    app, _, mock_asset, _ = _make_app(db_session)
    mock_asset.paste_extracted_object.side_effect = ValueError("asset not found")

    payload = {
        "asset_id": "gone",
        "target_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        "scale": 1.0,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json=payload,
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


# ──────────────────────────────────────────────
# jobs.py — /ml/jobs/{job_id}
# ──────────────────────────────────────────────

def _patched_job(status, result_info=None):
    """Patch arq.jobs.Job inside ml_jobs_module namespace."""
    job_instance = MagicMock()
    job_instance.status = AsyncMock(return_value=status)
    job_instance.result_info = AsyncMock(return_value=result_info)
    return patch.object(ml_jobs_module, "Job", return_value=job_instance), job_instance


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_not_found_returns_404(db_session, sample_user):
    app, *_ = _make_app(db_session)
    patcher, _ = _patched_job(JobStatus.not_found)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/missing-job", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("status_name", ["deferred", "queued", "in_progress"])
async def test_get_job_status_pending_returns_status_only(db_session, sample_user, status_name):
    app, *_ = _make_app(db_session)
    job_status = getattr(JobStatus, status_name)
    patcher, job_instance = _patched_job(job_status)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-1", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    data = resp.json()
    assert data == {"job_id": "job-1", "status": job_status.value}
    job_instance.result_info.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_complete_success_includes_result(db_session, sample_user):
    app, *_ = _make_app(db_session)
    result_info = MagicMock(success=True, result={"result_url": "s3://out.jpg"})
    patcher, _ = _patched_job(JobStatus.complete, result_info=result_info)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-2", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == JobStatus.complete.value
    assert data["result"] == {"result_url": "s3://out.jpg"}
    assert "error" not in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_complete_failure_includes_error(db_session, sample_user):
    app, *_ = _make_app(db_session)
    result_info = MagicMock(success=False, result=ValueError("inpainting failed"))
    patcher, _ = _patched_job(JobStatus.complete, result_info=result_info)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-3", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == JobStatus.complete.value
    assert data["error"] == "inpainting failed"
    assert "result" not in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_complete_no_result_info(db_session, sample_user):
    app, *_ = _make_app(db_session)
    patcher, _ = _patched_job(JobStatus.complete, result_info=None)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-4", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-4", "status": JobStatus.complete.value}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_no_auth_returns_401(db_session):
    app, *_ = _make_app(db_session)
    patcher, _ = _patched_job(JobStatus.queued)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-5")

    assert resp.status_code == 401