import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException

from app.api.v1.user import (
    get_me,
    update_me,
    change_password,
    delete_me
)


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    user.username = "test_user"
    user.email = "test@example.com"
    return user


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.update_user = AsyncMock()
    service.change_password = AsyncMock()
    service.delete_user = AsyncMock()
    return service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_me_returns_current_user(mock_user):
    result = await get_me(current_user=mock_user)
    assert result.id == mock_user.id
    assert result.username == mock_user.username
    assert result.email == mock_user.email

@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_me_success(mock_user, mock_service):
    updated_user = MagicMock()
    updated_user.id = 1
    updated_user.username = "new_name"
    updated_user.email = "new@email.com"

    mock_service.update_user.return_value = updated_user

    body = MagicMock()
    body.username = "new_name"
    body.email = "new@email.com"

    result = await update_me(
        body=body,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.update_user.assert_called_once_with(
        user_id=1,
        username="new_name",
        email="new@email.com"
    )
    assert result.username == "new_name"
    assert result.email == "new@email.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_me_validation_error(mock_user, mock_service):
    mock_service.update_user.side_effect = ValueError("Invalid data")

    body = MagicMock()
    body.username = "bad"
    body.email = "bad@email.com"

    with pytest.raises(HTTPException) as exc:
        await update_me(body=body, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400
    assert "Invalid data" in exc.value.detail

@pytest.mark.unit
@pytest.mark.asyncio
async def test_change_password_success(mock_user, mock_service):
    body = MagicMock()
    body.old_password = "old"
    body.new_password = "new"

    result = await change_password(
        body=body,
        current_user=mock_user,
        service=mock_service
    )

    mock_service.change_password.assert_called_once_with(
        user_id=1,
        old_password="old",
        new_password="new"
    )
    assert result["detail"] == "Password updated successfully"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_change_password_wrong_old_password(mock_user, mock_service):
    mock_service.change_password.side_effect = ValueError("Wrong password")

    body = MagicMock()
    body.old_password = "wrong"
    body.new_password = "new"

    with pytest.raises(HTTPException) as exc:
        await change_password(body=body, current_user=mock_user, service=mock_service)

    assert exc.value.status_code == 400
    assert "Wrong password" in exc.value.detail



@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_me_calls_service(mock_user, mock_service):
    """delete_me must call service.delete_user with the correct user_id."""
    await delete_me(current_user=mock_user, service=mock_service)
    mock_service.delete_user.assert_called_once_with(1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_me_returns_success_detail(mock_user, mock_service):
    """delete_me returns detail message on success."""
    result = await delete_me(current_user=mock_user, service=mock_service)
    assert result["detail"] == "Account deleted successfully"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_me_not_found(mock_user, mock_service):
    mock_service.delete_user.side_effect = ValueError("not found")

    with pytest.raises(ValueError, match="not found"):
        await delete_me(current_user=mock_user, service=mock_service)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_me_unauthorized(mock_user, mock_service):
    mock_service.delete_user.side_effect = ValueError("unauthorized")

    with pytest.raises(ValueError, match="unauthorized"):
        await delete_me(current_user=mock_user, service=mock_service)