import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException, UploadFile

from app.api.v1.ml import (
    detect_objects,
    remove_object,
    replace_object,
    remove_multiple_objects,
    reset_current_state,
    get_supported_classes,
    save_result,
    undo,
    redo,
    get_history,
)
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
)

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.detect_objects = AsyncMock()
    service.remove_object = AsyncMock()
    service.replace_object = AsyncMock()
    service.remove_multiple_objects = AsyncMock()
    service._get_image_authorized = AsyncMock()
    service.reset_current_state = AsyncMock()
    service.get_supported_classes = MagicMock()
    service.save_result = AsyncMock()
    service.undo = AsyncMock()
    service.redo = AsyncMock()
    service.get_history = AsyncMock()
    return service


@pytest.fixture
def mock_file():
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=b"image-bytes")
    return file


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_success_full_params(mock_user, mock_service, mock_file):
    mock_service.replace_object.return_value = {"result": "ok"}

    result = await replace_object(
        image_id=1,
        bbox_id=2,
        replacement_file=mock_file,
        expand_mask_pixels=15,
        use_color_matching=True,
        use_edge_blending=True,
        color_match_method="histogram",
        ldm_steps=40,
        ldm_sampler="ddim",
        hd_strategy="RESIZE",
        current_user=mock_user,
        service=mock_service
    )

    mock_service.replace_object.assert_called_once_with(
        image_id=1,
        bbox_id=2,
        replace_image_bytes=b"image-bytes",
        user_id=1,
        expand_mask_pixels=15,
        use_color_matching=True,
        use_edge_blending=True,
        color_match_method="histogram",
        ldm_steps=40,
        ldm_sampler="ddim",
        hd_strategy="RESIZE"
    )
    assert result["result"] == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_unauthorized(mock_user, mock_service, mock_file):
    mock_service.replace_object.side_effect = ValueError("unauthorized access")

    with pytest.raises(HTTPException) as exc:
        await replace_object(
            image_id=1,
            bbox_id=2,
            replacement_file=mock_file,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_generic_value_error_returns_400(mock_user, mock_service, mock_file):
    mock_service.replace_object.side_effect = ValueError("invalid color match method")

    with pytest.raises(HTTPException) as exc:
        await replace_object(
            image_id=1,
            bbox_id=2,
            replacement_file=mock_file,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_unauthorized(mock_user, mock_service):
    mock_service.remove_multiple_objects.side_effect = ValueError("unauthorized")

    body = RemoveMultipleRequest(bbox_ids=[1, 2], expand_mask_pixels=5, use_edge_blending=True)

    with pytest.raises(HTTPException) as exc:
        await remove_multiple_objects(image_id=1, body=body, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_generic_value_error_returns_400(mock_user, mock_service):
    mock_service.remove_multiple_objects.side_effect = ValueError("no overlapping masks")

    body = RemoveMultipleRequest(bbox_ids=[1, 2], expand_mask_pixels=5, use_edge_blending=True)

    with pytest.raises(HTTPException) as exc:
        await remove_multiple_objects(image_id=1, body=body, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_generic_value_error_returns_400(mock_user, mock_service):
    mock_service.detect_objects.side_effect = ValueError("bad confidence threshold")

    with pytest.raises(HTTPException) as exc:
        await detect_objects(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_generic_value_error_returns_400(mock_user, mock_service):
    mock_service.remove_object.side_effect = ValueError("mask generation failed")

    with pytest.raises(HTTPException) as exc:
        await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_generic_value_error_returns_400(mock_user, mock_service):
    mock_service._get_image_authorized.side_effect = ValueError("corrupted state")

    with pytest.raises(HTTPException) as exc:
        await reset_current_state(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_success(mock_user, mock_service):
    mock_service.save_result.return_value = {"id": 42, "filename": "result.jpg"}

    result = await save_result(image_id=1, current_user=mock_user, service=mock_service)

    mock_service.save_result.assert_called_once_with(image_id=1, user_id=1)
    assert result["id"] == 42


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_not_found(mock_user, mock_service):
    mock_service.save_result.side_effect = ValueError("image not found")

    with pytest.raises(HTTPException) as exc:
        await save_result(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_unauthorized(mock_user, mock_service):
    mock_service.save_result.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await save_result(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_result_generic_value_error_returns_400(mock_user, mock_service):
    mock_service.save_result.side_effect = ValueError("no processed state to save")

    with pytest.raises(HTTPException) as exc:
        await save_result(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_success(mock_user, mock_service):
    mock_service.undo.return_value = {"detail": "Undone"}

    result = await undo(image_id=1, current_user=mock_user, service=mock_service)

    mock_service.undo.assert_called_once_with(1, 1)
    assert result["detail"] == "Undone"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_not_found(mock_user, mock_service):
    mock_service.undo.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await undo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_unauthorized(mock_user, mock_service):
    mock_service.undo.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await undo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undo_nothing_to_undo_returns_400(mock_user, mock_service):
    mock_service.undo.side_effect = ValueError("nothing to undo")

    with pytest.raises(HTTPException) as exc:
        await undo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_success(mock_user, mock_service):
    mock_service.redo.return_value = {"detail": "Redone"}

    result = await redo(image_id=1, current_user=mock_user, service=mock_service)

    mock_service.redo.assert_called_once_with(1, 1)
    assert result["detail"] == "Redone"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_not_found(mock_user, mock_service):
    mock_service.redo.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await redo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_unauthorized(mock_user, mock_service):
    mock_service.redo.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await redo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redo_nothing_to_redo_returns_400(mock_user, mock_service):
    mock_service.redo.side_effect = ValueError("nothing to redo")

    with pytest.raises(HTTPException) as exc:
        await redo(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_success(mock_user, mock_service):
    mock_service.get_history.return_value = {
        "current_index": 1,
        "states": ["original.jpg", "edited.jpg"]
    }

    result = await get_history(image_id=1, current_user=mock_user, service=mock_service)

    mock_service.get_history.assert_called_once_with(1, 1)
    assert result["current_index"] == 1
    assert len(result["states"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_not_found(mock_user, mock_service):
    mock_service.get_history.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await get_history(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_unauthorized(mock_user, mock_service):
    mock_service.get_history.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await get_history(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403