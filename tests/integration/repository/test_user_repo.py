import pytest
from app.repository.user_repo import UserRepository
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_exists_by_email_true(db_session, sample_user):
    repo = UserRepository(db_session)

    exists = await repo.exists_by_email(sample_user.email)

    assert exists is True


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_exists_by_email_false(db_session):
    repo = UserRepository(db_session)

    exists = await repo.exists_by_email("not_exists@test.com")

    assert exists is False


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_update_password(db_session):
    repo = UserRepository(db_session)

    # create user
    user = await repo.create(
        username="pw_user",
        email="pw@test.com",
        password_hash=pwd_context.hash("old_password")
    )

    # update password
    new_password = "new_password_123"
    new_hash = pwd_context.hash(new_password)

    updated_user = await repo.update_password(user, new_hash)

    assert pwd_context.verify("new_password_123", updated_user.password_hash)