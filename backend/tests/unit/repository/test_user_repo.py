import pytest
from unittest.mock import AsyncMock, MagicMock
from app.repository.user_repo import UserRepository
from app.db.models.user import User


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user():
    db = AsyncMock()
    repo = UserRepository(db)

    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    user = await repo.create("john", "john@test.com", "hash")

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(user)

    assert user.username == "john"
    assert user.email == "john@test.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_id():
    db = AsyncMock()
    repo = UserRepository(db)

    mock_user = User(id=1, username="u", email="e", password_hash="h")

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = mock_user
    db.execute = AsyncMock(return_value=result_mock)

    user = await repo.get_by_id(1)

    assert user == mock_user


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_by_email_true():
    db = AsyncMock()
    repo = UserRepository(db)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = 1
    db.execute = AsyncMock(return_value=result_mock)

    exists = await repo.exists_by_email("test@test.com")

    assert exists is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_by_email_false():
    db = AsyncMock()
    repo = UserRepository(db)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result_mock)

    exists = await repo.exists_by_email("test@test.com")

    assert exists is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_password():
    db = AsyncMock()
    repo = UserRepository(db)

    user = User(id=1, username="u", email="e", password_hash="old")

    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    updated = await repo.update_password(user, "new_hash")

    assert updated.password_hash == "new_hash"
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(user)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_user_found():
    db = AsyncMock()
    repo = UserRepository(db)

    user = User(id=1, username="u", email="e", password_hash="h")

    repo.get_by_id = AsyncMock(return_value=user)
    db.delete = AsyncMock()
    db.commit = AsyncMock()

    result = await repo.delete(1)

    assert result is True
    db.delete.assert_called_once_with(user)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_user_not_found():
    db = AsyncMock()
    repo = UserRepository(db)

    repo.get_by_id = AsyncMock(return_value=None)

    result = await repo.delete(1)

    assert result is False