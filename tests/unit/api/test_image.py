import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException, UploadFile

from app.api.v1.image import (
    upload_image,
    get_user_images,
    get_image,
    download_image,
    get_presigned_url,
    delete_image
)

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_service():
    service = MagicMock()

    service.upload_image = AsyncMock()
    service.get_user_image = AsyncMock()
    service.get_image = AsyncMock()
    service.download_image = AsyncMock()
    service.get_presigned_url = AsyncMock()
    service.delete_image = AsyncMock()

    return service


@pytest.fixture
def mock_file():
    file = MagicMock(spec=UploadFile)
    file.filename = "test.jpg"
    return file


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_success(mock_user, mock_service, mock_file):
    mock_service.upload_image.return_value = {"id": 1}

    result = await upload_image(
        file=mock_file,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.upload_image.assert_called_once_with(
        file=mock_file,
        user_id=1
    )

    assert result["id"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_image_error(mock_user, mock_service, mock_file):
    mock_service.upload_image.side_effect = ValueError("Invalid file")

    with pytest.raises(HTTPException) as exc:
        await upload_image(
            file=mock_file,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 400

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_images(mock_user, mock_service):
    mock_service.get_user_image.return_value = [{"id": 1}, {"id": 2}]

    result = await get_user_images(
        limit=10,
        offset=0,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_user_image.assert_called_once_with(
        user_id=1,
        limit=10,
        offset=0
    )

    assert len(result) == 2

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_success(mock_user, mock_service):
    mock_service.get_image.return_value = {"id": 1}

    result = await get_image(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_image.assert_called_once_with(
        image_id=1,
        user_id=1
    )

    assert result["id"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_image_not_found(mock_user, mock_service):
    mock_service.get_image.side_effect = ValueError("Not found")

    with pytest.raises(HTTPException) as exc:
        await get_image(
            image_id=1,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 404

@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_success(mock_user, mock_service):
    mock_service.download_image.return_value = b"image-bytes"

    response = await download_image(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    assert response.body == b"image-bytes"
    assert response.media_type == "image/jpeg"
    assert "attachment" in response.headers["Content-Disposition"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_image_not_found(mock_user, mock_service):
    mock_service.download_image.side_effect = ValueError("Not found")

    with pytest.raises(HTTPException) as exc:
        await download_image(
            image_id=1,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 404

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_success(mock_user, mock_service):
    mock_service.get_presigned_url.return_value = "http://test-url"

    result = await get_presigned_url(
        image_id=1,
        expiration=3600,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.get_presigned_url.assert_called_once_with(
        image_id=1,
        user_id=1,
        expiration=3600
    )

    assert result["url"] == "http://test-url"
    assert result["expires_in"] == 3600


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_presigned_url_not_found(mock_user, mock_service):
    mock_service.get_presigned_url.side_effect = ValueError("Not found")

    with pytest.raises(HTTPException) as exc:
        await get_presigned_url(
            image_id=1,
            expiration=3600,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_success(mock_user, mock_service):
    result = await delete_image(
        image_id=1,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.delete_image.assert_called_once_with(
        image_id=1,
        user_id=1
    )

    assert result is None  # 204 No Content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_not_found(mock_user, mock_service):
    mock_service.delete_image.side_effect = ValueError("Not found")

    with pytest.raises(HTTPException) as exc:
        await delete_image(
            image_id=1,
            current_user=mock_user,
            service=mock_service
        )

    assert exc.value.status_code == 404