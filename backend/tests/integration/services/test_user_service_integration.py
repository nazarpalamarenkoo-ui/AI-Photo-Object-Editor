import pytest
from app.repository.user_repo import UserRepository
from app.services.user_service import UserService


def _make_service(db_session) -> UserService:
    """UserService.__init__ only takes user_repo (session-per-operation
    repos, no db session on the service itself)."""
    return UserService(user_repo=UserRepository(db_session))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user_success(db_session):
    service = _make_service(db_session)
    user = await service.create_user("alice", "alice@test.com", "Password1")
    assert user.id is not None
    assert user.username == "alice"
    assert user.email == "alice@test.com"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user_duplicate_username(db_session):
    service = _make_service(db_session)
    await service.create_user("bob", "bob@test.com", "Password1")
    with pytest.raises(ValueError, match="already exists"):
        await service.create_user("bob", "bob2@test.com", "Password1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user_duplicate_email(db_session):
    service = _make_service(db_session)
    await service.create_user("charlie", "charlie@test.com", "Password1")
    with pytest.raises(ValueError, match="already registered"):
        await service.create_user("charlie2", "charlie@test.com", "Password1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_authenticate_user_success(db_session):
    service = _make_service(db_session)
    await service.create_user("dave", "dave@test.com", "Password1")
    user = await service.authenticate_user("dave@test.com", "Password1")
    assert user is not None
    assert user.username == "dave"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_authenticate_user_wrong_password(db_session):
    service = _make_service(db_session)
    await service.create_user("eve", "eve@test.com", "Password1")
    user = await service.authenticate_user("eve@test.com", "wrongpass")
    assert user is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_authenticate_user_not_found(db_session):
    service = _make_service(db_session)
    user = await service.authenticate_user("nobody@test.com", "Password1")
    assert user is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_user_success(db_session):
    service = _make_service(db_session)
    created = await service.create_user("frank", "frank@test.com", "Password1")
    fetched = await service.get_user(created.id)
    assert fetched.id == created.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_user_not_found(db_session):
    service = _make_service(db_session)
    with pytest.raises(ValueError, match="not found"):
        await service.get_user(99999)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_user_username(db_session):
    service = _make_service(db_session)
    user = await service.create_user("iris", "iris@test.com", "Password1")
    updated = await service.update_user(user.id, username="iris2")
    assert updated.username == "iris2"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_user_username_conflict(db_session):
    service = _make_service(db_session)
    await service.create_user("owner", "owner@test.com", "Password1")
    other = await service.create_user("other", "other@test.com", "Password1")
    with pytest.raises(ValueError, match="already exists"):
        await service.update_user(other.id, username="owner")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_change_password_success(db_session):
    service = _make_service(db_session)
    user = await service.create_user("grace", "grace@test.com", "OldPass1")
    result = await service.change_password(user.id, "OldPass1", "NewPass1")
    assert result is True
    authenticated = await service.authenticate_user("grace@test.com", "NewPass1")
    assert authenticated is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_change_password_wrong_old(db_session):
    service = _make_service(db_session)
    user = await service.create_user("henry", "henry@test.com", "OldPass1")
    with pytest.raises(ValueError, match="Incorrect"):
        await service.change_password(user.id, "wrongold", "NewPass1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_user(db_session):
    service = _make_service(db_session)
    user = await service.create_user("ivan", "ivan@test.com", "Password1")
    result = await service.delete_user(user.id)
    assert result is True
    with pytest.raises(ValueError, match="not found"):
        await service.get_user(user.id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_password_too_short(db_session):
    service = _make_service(db_session)
    with pytest.raises(ValueError, match="8 characters"):
        await service.create_user("judy", "judy@test.com", "Ab1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_password_no_uppercase(db_session):
    service = _make_service(db_session)
    with pytest.raises(ValueError, match="uppercase"):
        await service.create_user("kate", "kate@test.com", "password1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_password_no_digit(db_session):
    service = _make_service(db_session)
    with pytest.raises(ValueError, match="digit"):
        await service.create_user("leo", "leo@test.com", "Password")