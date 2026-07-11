import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from arq.jobs import JobStatus

from app.api.auth.auth import create_access_token
from app.db.models.user import User
import app.api.v1.ml as ml_module


def _default_mock_detector():
    service = MagicMock()
    service.detect_objects = AsyncMock(return_value={"detections": [], "count": 0})
    service.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])
    return service


def _default_mock_editor():
    service = MagicMock()
    service.remove_object = AsyncMock(return_value={
        "result_url": "s3://bucket/result.jpg",
        "presigned_url": "https://presigned.url/result.jpg",
        "metrics": {},
        "timestamp": "2025-01-01T00:00:00",
    })
    service.remove_multiple_objects = AsyncMock(return_value={
        "result_url": "s3://bucket/result.jpg",
        "presigned_url": "https://presigned.url/result.jpg",
        "metrics": {},
        "timestamp": "2025-01-01T00:00:00",
    })
    service.replace_object = AsyncMock(return_value={
        "result_url": "s3://bucket/result.jpg",
        "presigned_url": "https://presigned.url/result.jpg",
        "metrics": {},
        "timestamp": "2025-01-01T00:00:00",
    })
    service._get_image_authorized = AsyncMock()
    service.reset_current_state = AsyncMock()
    service.save_result = AsyncMock(return_value={
        "id": 42,
        "filename": "edited_test.jpg",
        "storage_path": "s3://bucket/result.jpg",
        "uploaded_at": "2025-01-01T00:00:00",
        "cache_key": None,
    })
    service.undo = AsyncMock(return_value={"detail": "Undone"})
    service.redo = AsyncMock(return_value={"detail": "Redone"})
    service.get_history = AsyncMock(return_value={"history": []})
    return service


def _default_mock_segmentation():
    service = MagicMock()
    service.segment_objects = AsyncMock(return_value={
        "segments": [],
        "metrics": {},
        "image_size": (640, 480),
        "timestamp": "2025-01-01T00:00:00",
    })
    service.segment_with_prompt = AsyncMock(return_value={
        "segments": [],
        "metrics": {},
        "image_size": (640, 480),
        "timestamp": "2025-01-01T00:00:00",
    })
    service.segment_by_polygon = AsyncMock(return_value={
        "segments": [],
        "metrics": {},
        "image_size": (640, 480),
        "timestamp": "2025-01-01T00:00:00",
    })
    service.segment_hybrid = AsyncMock(return_value={
        "segments": [],
        "metrics": {},
        "image_size": (640, 480),
        "timestamp": "2025-01-01T00:00:00",
    })
    service.sam_remove_object = AsyncMock(return_value={
        "result_url": "s3://bucket/result.jpg",
        "presigned_url": "https://presigned.url/result.jpg",
        "metrics": {},
        "timestamp": "2025-01-01T00:00:00",
    })
    service.sam_replace_object = AsyncMock(return_value={
        "result_url": "s3://bucket/result.jpg",
        "presigned_url": "https://presigned.url/result.jpg",
        "metrics": {},
        "timestamp": "2025-01-01T00:00:00",
    })
    return service


def _default_mock_asset():
    service = MagicMock()
    service.extract_object = AsyncMock(return_value={
        "asset_id": "asset-1",
        "extracted_url": "s3://bucket/obj.png",
        "presigned_url": "https://presigned.url/obj.png",
        "object_size": (50, 60),
        "area_pixels": 3000,
        "cropped_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 60},
        "timestamp": "2025-01-01T00:00:00",
    })
    service.paste_extracted_object = AsyncMock(return_value={
        "result_url": "s3://bucket/pasted.jpg",
        "presigned_url": "https://presigned.url/pasted.jpg",
        "paste_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        "object_size": (50, 50),
        "timestamp": "2025-01-01T00:00:00",
    })
    service.list_assets = AsyncMock(return_value=[])
    service.get_asset_thumbnail = AsyncMock(return_value=b"thumb-bytes")
    service.get_asset_image = AsyncMock(return_value=b"asset-bytes")
    service.rename_asset = AsyncMock(return_value={
        "asset_id": "asset-1",
        "source_image_id": 1,
        "object_size": (50, 60),
        "area_pixels": 3000,
        "label": "new-name",
        "s3_url": None,
        "created_at": "2025-01-01T00:00:00",
    })
    service.delete_asset = AsyncMock()
    return service


def _default_mock_pool():
    pool = MagicMock()
    job = MagicMock()
    job.job_id = "job-123"
    pool.enqueue_job = AsyncMock(return_value=job)
    return pool


def _make_app(db_session, detector=None, editor=None, segmentation=None, asset=None, pool=None):
    """Build a FastAPI app with the ml router wired to mocked services."""
    from fastapi import FastAPI
    from app.api.v1.ml import (
        router as ml_router,
        get_detector,
        get_editor,
        get_segmentation,
        get_asset,
        get_arq_pool,
    )
    from app.db.db_connect import get_db

    app = FastAPI()
    app.include_router(ml_router)
    app.dependency_overrides[get_db] = lambda: db_session

    mock_detector = detector or _default_mock_detector()
    mock_editor = editor or _default_mock_editor()
    mock_segmentation = segmentation or _default_mock_segmentation()
    mock_asset = asset or _default_mock_asset()
    mock_pool = pool or _default_mock_pool()

    app.dependency_overrides[get_detector] = lambda: mock_detector
    app.dependency_overrides[get_editor] = lambda: mock_editor
    app.dependency_overrides[get_segmentation] = lambda: mock_segmentation
    app.dependency_overrides[get_asset] = lambda: mock_asset
    app.dependency_overrides[get_arq_pool] = lambda: mock_pool

    return app, mock_detector, mock_editor, mock_segmentation, mock_asset, mock_pool


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
async def test_detect_objects_success(db_session, sample_user, sample_image):
    app, mock_detector, _, _, _, _ = _make_app(db_session)
    mock_detector.detect_objects = AsyncMock(return_value={"detections": [{"class": "person"}], "count": 1})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/detect",
            json={"conf_threshold": 0.7, "classes": ["person"]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    mock_detector.detect_objects.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id, conf_threshold=0.7, classes=["person"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_default_body(db_session, sample_user, sample_image):
    app, mock_detector, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/detect",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_detector.detect_objects.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id, conf_threshold=0.5, classes=None
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_invalid_threshold_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/detect",
            json={"conf_threshold": 1.5},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_not_found(db_session, sample_user):
    app, mock_detector, _, _, _, _ = _make_app(db_session)
    mock_detector.detect_objects.side_effect = ValueError("Image not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ml/images/99999/detect", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_unauthorized(db_session, sample_user, sample_image):
    app, mock_detector, _, _, _, _ = _make_app(db_session)
    mock_detector.detect_objects.side_effect = ValueError("unauthorized")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/ml/images/{sample_image.id}/detect", headers=_auth_headers(sample_user))

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_objects_no_auth_returns_401(db_session, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/ml/images/{sample_image.id}/detect")

    assert resp.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_supported_classes(db_session, sample_user):
    app, mock_detector, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/classes", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json() == ["person", "car", "dog"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_success(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1",
            json={"expand_mask_pixels": 10, "use_edge_blending": False},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["result_url"] == "s3://bucket/result.jpg"
    mock_editor.remove_object.assert_awaited_once_with(
        image_id=sample_image.id,
        bbox_id=1,
        user_id=sample_user.id,
        expand_mask_pixels=10,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_default_body(db_session, sample_user, sample_image):
    # RemoveRequest defaults: expand_mask_pixels=5, use_edge_blending=False
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_editor.remove_object.assert_awaited_once_with(
        image_id=sample_image.id,
        bbox_id=1,
        user_id=sample_user.id,
        expand_mask_pixels=5,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_bbox_not_found(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.remove_object.side_effect = ValueError("bbox not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/999",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_invalid_expand_mask_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
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
    # RemoveMultipleRequest defaults: expand_mask_pixels=5, use_edge_blending=False
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple",
            json={"bbox_ids": [1, 2, 3]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_editor.remove_multiple_objects.assert_awaited_once_with(
        image_id=sample_image.id,
        bbox_ids=[1, 2, 3],
        user_id=sample_user.id,
        expand_mask_pixels=5,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_missing_bbox_ids_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple",
            json={},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_unauthorized(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.remove_multiple_objects.side_effect = ValueError("unauthorized")

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
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            files={"replacement_file": ("replacement.png", b"fake-image-bytes", "image/png")},
            params={"color_match_method": "histogram", "expand_mask_pixels": 12},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_editor.replace_object.assert_awaited_once()
    _, kwargs = mock_editor.replace_object.await_args
    assert kwargs["replace_image_bytes"] == b"fake-image-bytes"
    assert kwargs["color_match_method"] == "histogram"
    assert kwargs["expand_mask_pixels"] == 12


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_missing_file_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_invalid_color_method_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            files={"replacement_file": ("replacement.png", b"fake-image-bytes", "image/png")},
            params={"color_match_method": "invalid"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_generic_error_returns_400(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.replace_object.side_effect = ValueError("invalid replacement image")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1",
            files={"replacement_file": ("replacement.png", b"fake-image-bytes", "image/png")},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_success(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/reset",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "State reset to original image"}
    mock_editor._get_image_authorized.assert_awaited_once_with(sample_image.id, sample_user.id)
    mock_editor.reset_current_state.assert_awaited_once_with(sample_image.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_current_state_not_found(db_session, sample_user):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor._get_image_authorized.side_effect = ValueError("not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ml/images/99999/reset", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_success(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 42
    assert data["storage_path"] == "s3://bucket/result.jpg"
    mock_editor.save_result.assert_awaited_once_with(image_id=sample_image.id, user_id=sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_no_processed_state(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.save_result.side_effect = ValueError("No processed result to save. Run remove/replace first.")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "No processed result to save. Run remove/replace first."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_result_unauthorized(db_session, sample_user, sample_image):
    other_user = await _make_other_user(db_session)
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.save_result.side_effect = ValueError("unauthorized")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/save",
            headers=_auth_headers(other_user),
        )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_success(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/undo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Undone"}
    mock_editor.undo.assert_awaited_once_with(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_nothing_to_undo(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.undo.side_effect = ValueError("Nothing to undo")

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
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/redo",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Redone"}
    mock_editor.redo.assert_awaited_once_with(sample_image.id, sample_user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_nothing_to_redo(db_session, sample_user, sample_image):
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.redo.side_effect = ValueError("Nothing to redo")

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
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.get_history = AsyncMock(return_value={"history": ["remove bbox_id=0", "replace bbox_id=1"]})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"history": ["remove bbox_id=0", "replace bbox_id=1"]}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_unauthorized(db_session, sample_user, sample_image):
    other_user = await _make_other_user(db_session)
    app, _, mock_editor, _, _, _ = _make_app(db_session)
    mock_editor.get_history.side_effect = ValueError("unauthorized")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/ml/images/{sample_image.id}/history",
            headers=_auth_headers(other_user),
        )

    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_success(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment",
            json={"min_area": 200, "max_segments": 20},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_objects.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id, min_area=200, max_segments=20
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_default_body(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_objects.assert_awaited_once_with(
        image_id=sample_image.id, user_id=sample_user.id, min_area=500, max_segments=50
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_not_found(db_session, sample_user):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    mock_segmentation.segment_objects.side_effect = ValueError("image not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ml/images/99999/segment", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_success(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt",
            json={"point_coords": [[10, 20]], "point_labels": [1]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_with_prompt.assert_awaited_once_with(
        image_id=sample_image.id,
        user_id=sample_user.id,
        point_coords=[(10, 20)],
        point_labels=[1],
        bbox=None,
        multimask_output=None,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_with_bbox(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt",
            json={"bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    _, kwargs = mock_segmentation.segment_with_prompt.await_args
    assert kwargs["bbox"] == {"x1": 0, "y1": 0, "x2": 50, "y2": 50}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_multimask_output(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt",
            json={"multimask_output": True},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    _, kwargs = mock_segmentation.segment_with_prompt.await_args
    assert kwargs["multimask_output"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_mask_not_found(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    mock_segmentation.segment_with_prompt.side_effect = ValueError("no valid detections")

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
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon",
            json={"points": [[0, 0], [10, 0], [5, 10]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_by_polygon.assert_awaited_once_with(
        image_id=sample_image.id,
        user_id=sample_user.id,
        points=[(0, 0), (10, 0), (5, 10)],
        smooth=True,
        smoothing_factor=0.0,
        feather_px=0,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_too_few_points_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon",
            json={"points": [[0, 0], [10, 0]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_not_found(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    mock_segmentation.segment_by_polygon.side_effect = ValueError("mask not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon",
            json={"points": [[0, 0], [10, 0], [5, 10]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_success(db_session, sample_user, sample_image):
    # SamRemoveRequest defaults: expand_mask_pixels=12, use_edge_blending=False
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/remove",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.sam_remove_object.assert_awaited_once_with(
        image_id=sample_image.id,
        mask_id=3,
        user_id=sample_user.id,
        expand_mask_pixels=12,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_mask_not_found(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    mock_segmentation.sam_remove_object.side_effect = ValueError("mask not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/999/remove",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_success_with_file(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            files={"replacement_file": ("replacement.png", b"fake-bytes", "image/png")},
            params={"expand_mask_pixels": 15},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.sam_replace_object.assert_awaited_once()
    _, kwargs = mock_segmentation.sam_replace_object.await_args
    assert kwargs["replacement_image_bytes"] == b"fake-bytes"
    assert kwargs["expand_mask_pixels"] == 15
    assert kwargs["replacement_is_cutout"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_success_with_asset_id(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=b"asset-bytes")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            params={"asset_id": "asset-1"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    _, kwargs = mock_segmentation.sam_replace_object.await_args
    assert kwargs["replacement_image_bytes"] == b"asset-bytes"
    assert kwargs["replacement_is_cutout"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_asset_not_found_returns_404(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            params={"asset_id": "missing-asset"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_missing_file_and_asset_id_returns_400(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide replacement_file or asset_id"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_success(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract",
            json={"padding_pixels": 20},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json()["extracted_url"] == "s3://bucket/obj.png"
    mock_asset.extract_object.assert_awaited_once_with(
        image_id=sample_image.id, mask_id=3, user_id=sample_user.id,
        padding_pixels=20, label=None, persist_to_s3=False,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_default_body(db_session, sample_user, sample_image):
    # ExtractRequest.padding_pixels has no ge/le constraints in the schema
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_asset.extract_object.assert_awaited_once_with(
        image_id=sample_image.id, mask_id=3, user_id=sample_user.id,
        padding_pixels=8, label=None, persist_to_s3=False,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_large_padding_is_accepted(db_session, sample_user, sample_image):
    # No upper bound is defined on padding_pixels, so a large value is valid
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract",
            json={"padding_pixels": 100},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    _, kwargs = mock_asset.extract_object.await_args
    assert kwargs["padding_pixels"] == 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_non_integer_padding_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/extract",
            json={"padding_pixels": "not-a-number"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_mask_not_found(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.extract_object.side_effect = ValueError("mask not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/999/extract",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_success(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.list_assets = AsyncMock(return_value=[{
        "asset_id": "asset-1", "source_image_id": 1, "object_size": (50, 60),
        "area_pixels": 3000, "label": None, "s3_url": None,
        "created_at": "2025-01-01T00:00:00",
    }])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert len(resp.json()) == 1
    mock_asset.list_assets.assert_awaited_once_with(sample_user.id, limit=50, offset=0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_custom_pagination(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets?limit=10&offset=5", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    mock_asset.list_assets.assert_awaited_once_with(sample_user.id, limit=10, offset=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_assets_limit_above_max_returns_422(db_session, sample_user):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets?limit=500", headers=_auth_headers(sample_user))

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_thumbnail_success(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/asset-1/thumbnail", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.content == b"thumb-bytes"
    assert resp.headers["content-type"] == "image/png"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_thumbnail_not_found(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_thumbnail = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/missing/thumbnail", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_image_success(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/asset-1/image", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.content == b"asset-bytes"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_asset_image_not_found(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ml/assets/missing/image", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rename_asset_success(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/ml/assets/asset-1",
            json={"label": "new-name"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json()["label"] == "new-name"
    mock_asset.rename_asset.assert_awaited_once_with(sample_user.id, "asset-1", "new-name")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rename_asset_missing_label_returns_422(db_session, sample_user):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            "/ml/assets/asset-1",
            json={},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rename_asset_not_found(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
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
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/ml/assets/asset-1", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    assert resp.json() == {"detail": "Asset deleted"}
    mock_asset.delete_asset.assert_awaited_once_with(sample_user.id, "asset-1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_asset_not_found(db_session, sample_user):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.delete_asset.side_effect = ValueError("asset not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/ml/assets/missing", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_success(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json={
                "extracted_url": "s3://bucket/obj.png",
                "target_bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
                "scale": 1.5,
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json()["result_url"] == "s3://bucket/pasted.jpg"
    # PasteRequest defaults: use_color_matching=False, use_edge_blending=False
    mock_asset.paste_extracted_object.assert_awaited_once_with(
        image_id=sample_image.id,
        user_id=sample_user.id,
        asset_id=None,
        extracted_url="s3://bucket/obj.png",
        target_bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        scale=1.5,
        use_color_matching=False,
        use_edge_blending=False,
        color_match_method='color_transfer',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_missing_required_field_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json={"extracted_url": "s3://bucket/obj.png"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_missing_source_returns_422(db_session, sample_user, sample_image):
    # PasteRequest requires either asset_id or extracted_url via model_validator
    app, _, _, _, _, _ = _make_app(db_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json={"target_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paste_extracted_object_not_found(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, _ = _make_app(db_session)
    mock_asset.paste_extracted_object.side_effect = ValueError("extracted object not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/paste",
            json={
                "extracted_url": "s3://bucket/missing.png",
                "target_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404

@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_success(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/hybrid",
            json={
                "yolo_conf_threshold": 0.4,
                "yolo_classes": ["person", "car"],
                "fallback_min_area": 200,
                "fallback_max_segments": 20,
                "overlap_iou_thresh": 0.3,
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_hybrid.assert_awaited_once_with(
        image_id=sample_image.id,
        user_id=sample_user.id,
        yolo_conf_threshold=0.4,
        yolo_classes=["person", "car"],
        fallback_min_area=200,
        fallback_max_segments=20,
        overlap_iou_thresh=0.3,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_default_body(db_session, sample_user, sample_image):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/hybrid",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_segmentation.segment_hybrid.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_not_found(db_session, sample_user):
    app, _, _, mock_segmentation, _, _ = _make_app(db_session)
    mock_segmentation.segment_hybrid.side_effect = ValueError("image not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ml/images/99999/segment/hybrid", headers=_auth_headers(sample_user))

    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1/async",
            json={"expand_mask_pixels": 10, "use_edge_blending": False},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "remove_object_task",
        image_id=sample_image.id,
        bbox_id=1,
        user_id=sample_user.id,
        expand_mask_pixels=10,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_object_async_default_body(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove/1/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_pool.enqueue_job.assert_awaited_once_with(
        "remove_object_task",
        image_id=sample_image.id,
        bbox_id=1,
        user_id=sample_user.id,
        expand_mask_pixels=5,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple/async",
            json={"bbox_ids": [1, 2, 3]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "remove_multiple_objects_task",
        image_id=sample_image.id,
        bbox_ids=[1, 2, 3],
        user_id=sample_user.id,
        expand_mask_pixels=5,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_multiple_objects_async_missing_bbox_ids_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/remove-multiple/async",
            json={},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_async_reads_file_and_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1/async",
            files={"replacement_file": ("replacement.png", b"fake-image-bytes", "image/png")},
            params={"color_match_method": "histogram", "expand_mask_pixels": 12},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once()
    args, kwargs = mock_pool.enqueue_job.call_args
    assert args[0] == "replace_object_task"
    assert kwargs["replace_image_bytes"] == b"fake-image-bytes"
    assert kwargs["color_match_method"] == "histogram"
    assert kwargs["expand_mask_pixels"] == 12


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_object_async_missing_file_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/replace/1/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/async",
            json={"min_area": 200, "max_segments": 20},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_objects_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        min_area=200,
        max_segments=20,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_objects_async_default_body(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_objects_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        min_area=500,
        max_segments=50,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt/async",
            json={"point_coords": [[10, 20]], "point_labels": [1]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_with_prompt_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        point_coords=[(10, 20)],
        point_labels=[1],
        bbox=None,
        multimask_output=None,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_with_prompt_async_with_bbox(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/prompt/async",
            json={"bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["bbox"] == {"x1": 0, "y1": 0, "x2": 50, "y2": 50}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon/async",
            json={"points": [[0, 0], [10, 0], [5, 10]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_by_polygon_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        points=[(0, 0), (10, 0), (5, 10)],
        smooth=True,
        smoothing_factor=0.0,
        feather_px=0,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_by_polygon_async_too_few_points_returns_422(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/polygon/async",
            json={"points": [[0, 0], [10, 0]]},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 422
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_segment_hybrid_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/hybrid/async",
            json={
                "yolo_conf_threshold": 0.4,
                "yolo_classes": ["person"],
                "fallback_min_area": 200,
                "fallback_max_segments": 20,
                "overlap_iou_thresh": 0.3,
            },
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "segment_hybrid_task",
        image_id=sample_image.id,
        user_id=sample_user.id,
        yolo_conf_threshold=0.4,
        yolo_classes=["person"],
        fallback_min_area=200,
        fallback_max_segments=20,
        overlap_iou_thresh=0.3,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_remove_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/remove/async",
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    mock_pool.enqueue_job.assert_awaited_once_with(
        "sam_remove_object_task",
        image_id=sample_image.id,
        mask_id=3,
        user_id=sample_user.id,
        expand_mask_pixels=12,
        use_edge_blending=False,
        ldm_steps=25,
        ldm_sampler='plms',
        hd_strategy='CROP',
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_with_file(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            files={"replacement_file": ("replacement.png", b"fake-bytes", "image/png")},
            params={"expand_mask_pixels": 15},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-123"}
    _, kwargs = mock_pool.enqueue_job.call_args
    assert kwargs["replacement_image_bytes"] == b"fake-bytes"
    assert kwargs["expand_mask_pixels"] == 15
    assert kwargs["replacement_is_cutout"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_with_asset_id(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, mock_pool = _make_app(db_session)
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
async def test_sam_replace_object_async_asset_not_found_returns_404(db_session, sample_user, sample_image):
    app, _, _, _, mock_asset, mock_pool = _make_app(db_session)
    mock_asset.get_asset_image = AsyncMock(return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/ml/images/{sample_image.id}/segment/3/replace/async",
            params={"asset_id": "missing-asset"},
            headers=_auth_headers(sample_user),
        )

    assert resp.status_code == 404
    mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sam_replace_object_async_missing_file_and_asset_id_returns_400(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

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
async def test_extract_object_async_enqueues_job(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

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
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_object_async_default_body(db_session, sample_user, sample_image):
    app, _, _, _, _, mock_pool = _make_app(db_session)

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
    )


def _patched_job(status, result_info=None):
    job_instance = MagicMock()
    job_instance.status = AsyncMock(return_value=status)
    job_instance.result_info = AsyncMock(return_value=result_info)
    return patch.object(ml_module, "Job", return_value=job_instance), job_instance


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
    assert resp.json() == {"job_id": "job-1", "status": job_status.value}
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
async def test_get_job_status_complete_without_result_info(db_session, sample_user):
    app, *_ = _make_app(db_session)
    patcher, _ = _patched_job(JobStatus.complete, result_info=None)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-4", headers=_auth_headers(sample_user))

    assert resp.status_code == 200
    data = resp.json()
    assert data == {"job_id": "job-4", "status": JobStatus.complete.value}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_job_status_no_auth_returns_401(db_session):
    app, *_ = _make_app(db_session)
    patcher, _ = _patched_job(JobStatus.queued)

    with patcher:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ml/jobs/job-5")

    assert resp.status_code == 401