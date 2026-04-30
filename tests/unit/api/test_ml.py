import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException, UploadFile

from app.api.v1.ml import (
    detect_objects,
    remove_object,
    replace_object,
    remove_multiple_objects,
    reset_current_state,
    get_supported_classes
)
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest
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
    return service


@pytest.fixture
def mock_file():
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=b"image-bytes")
    return file


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_success(mock_user, mock_service):
    mock_service.detect_objects.return_value = {"detections": [1]}

    body = DetectRequest(conf_threshold=0.5, classes=None)

    result = await detect_objects(
        image_id=1,
        body=body,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.detect_objects.assert_called_once_with(
        image_id=1,
        user_id=1,
        conf_threshold=0.5,
        classes=None
    )
    assert len(result["detections"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_not_found(mock_user, mock_service):
    mock_service.detect_objects.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await detect_objects(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_unauthorized(mock_user, mock_service):
    mock_service.detect_objects.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await detect_objects(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_success(mock_user, mock_service):
    mock_service.remove_object.return_value = {"result": "ok"}

    body = RemoveRequest(expand_mask_pixels=5, use_edge_blending=True)

    result = await remove_object(
        image_id=1,
        bbox_id=2,
        body=body,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.remove_object.assert_called_once_with(
        image_id=1,
        bbox_id=2,
        user_id=1,
        expand_mask_pixels=5,
        use_edge_blending=True
    )
    assert result["result"] == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_not_found(mock_user, mock_service):
    mock_service.remove_object.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_unauthorized(mock_user, mock_service):
    mock_service.remove_object.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await remove_object(image_id=1, bbox_id=2, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_success(mock_user, mock_service, mock_file):
    mock_service.replace_object.return_value = {"result": "ok"}

    result = await replace_object(
        image_id=1,
        bbox_id=2,
        replacement_file=mock_file,
        expand_mask_pixels=0,
        use_color_matching=True,
        use_edge_blending=False,
        color_match_method="mean_std",
        current_user=mock_user,
        service=mock_service
    )

    mock_service.replace_object.assert_called_once_with(
        image_id=1,
        bbox_id=2,
        replace_image_bytes=b"image-bytes",
        user_id=1,
        expand_mask_pixels=0,
        use_color_matching=True,
        use_edge_blending=False,
        color_match_method="mean_std"
    )
    assert result["result"] == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_not_found(mock_user, mock_service, mock_file):
    mock_service.replace_object.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await replace_object(
            image_id=1,
            bbox_id=2,
            replacement_file=mock_file,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_unauthorized(mock_user, mock_service, mock_file):
    mock_service.replace_object.side_effect = ValueError("unauthorized")

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
async def test_remove_multiple_success(mock_user, mock_service):
    mock_service.remove_multiple_objects.return_value = {"result": "ok"}

    body = RemoveMultipleRequest(bbox_ids=[1, 2], expand_mask_pixels=5, use_edge_blending=True)

    result = await remove_multiple_objects(
        image_id=1,
        body=body,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.remove_multiple_objects.assert_called_once_with(
        image_id=1,
        bbox_ids=[1, 2],
        user_id=1,
        expand_mask_pixels=5,
        use_edge_blending=True
    )
    assert result["result"] == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remove_multiple_not_found(mock_user, mock_service):
    mock_service.remove_multiple_objects.side_effect = ValueError("not found")

    body = RemoveMultipleRequest(bbox_ids=[1, 2], expand_mask_pixels=5, use_edge_blending=True)

    with pytest.raises(HTTPException) as exc:
        await remove_multiple_objects(image_id=1, body=body, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_success(mock_user, mock_service):
    result = await reset_current_state(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    mock_service._get_image_authorized.assert_called_once_with(1, 1)
    mock_service.reset_current_state.assert_called_once_with(1)
    assert result["detail"] == "State reset to original image"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_not_found(mock_user, mock_service):
    mock_service._get_image_authorized.side_effect = ValueError("not found")

    with pytest.raises(HTTPException) as exc:
        await reset_current_state(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_unauthorized(mock_user, mock_service):
    mock_service._get_image_authorized.side_effect = ValueError("unauthorized")

    with pytest.raises(HTTPException) as exc:
        await reset_current_state(image_id=1, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_supported_classes(mock_user, mock_service):
    mock_service.get_supported_classes.return_value = ["person", "car"]

    result = await get_supported_classes(current_user=mock_user, service=mock_service)

    mock_service.get_supported_classes.assert_called_once()
    assert "person" in result