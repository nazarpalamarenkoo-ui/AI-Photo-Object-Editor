import pytest
from app.repository.user_repo import UserRepository


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_user(db_session):
    repo = UserRepository(db_session)
    user = await repo.create("john", "john@test.com", "hash")
    assert user.id is not None
    assert user.username == "john"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_user_by_id(db_session, sample_user):
    repo = UserRepository(db_session)
    user = await repo.get_by_id(sample_user.id)
    assert user is not None
    assert user.id == sample_user.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_user_by_email(db_session, sample_user):
    repo = UserRepository(db_session)
    user = await repo.get_by_email(sample_user.email)
    assert user is not None
    assert user.email == sample_user.email


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_update_user(db_session, sample_user):
    repo = UserRepository(db_session)
    sample_user.username = "updated"
    updated = await repo.update(sample_user)
    assert updated.username == "updated"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_user(db_session, sample_user):
    repo = UserRepository(db_session)
    user_id = sample_user.id
    result = await repo.delete(user_id)
    assert result is True
    user = await repo.get_by_id(user_id)
    assert user is None