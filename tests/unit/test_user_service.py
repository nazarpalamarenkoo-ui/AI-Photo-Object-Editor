import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.user_service import UserService
from app.db.models.user import User


@pytest.fixture
def mock_db():
    """Mock database session"""
    return MagicMock()


@pytest.fixture
def mock_user_repo():
    """Mock User repository"""
    repo = MagicMock()
    
    # Mock user object with proper bcrypt hash for password "TestPass123"
    # This is a real bcrypt hash generated for testing
    mock_user = MagicMock(spec=User)
    mock_user.id = 123
    mock_user.username = 'testuser'
    mock_user.email = 'test@example.com'
    mock_user.password_hash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYfZl.eui8m'  # Hash for "TestPass123"
    
    repo.create = AsyncMock(return_value=mock_user)
    repo.get_by_id = AsyncMock(return_value=mock_user)
    repo.get_by_username = AsyncMock(return_value=None)
    repo.get_by_email = AsyncMock(return_value=None)
    repo.update = AsyncMock(return_value=mock_user)
    repo.delete = AsyncMock(return_value=True)
    
    return repo


@pytest.fixture
def user_service(mock_db, mock_user_repo):
    """User Service instance with mocked dependencies"""
    return UserService(
        db=mock_db,
        user_repo=mock_user_repo
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_success(user_service, mock_user_repo):
    """Test successful user creation"""
    result = await user_service.create_user(
        username='newuser',
        email='new@example.com',
        password='SecPass123'
    )
    
    # Verify user created
    mock_user_repo.create.assert_called_once()
    
    # Check password was hashed (not plain text)
    call_args = mock_user_repo.create.call_args
    assert call_args.kwargs['password_hash'] != 'SecPass123'
    assert call_args.kwargs['password_hash'].startswith('$2b$')  # bcrypt hash
    
    assert result.username == 'testuser'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_username_exists(user_service, mock_user_repo):
    """Test create user with existing username"""
    mock_user_repo.get_by_username = AsyncMock(return_value=MagicMock())
    
    with pytest.raises(ValueError, match="Username 'existinguser' already exists"):
        await user_service.create_user(
            username='existinguser',
            email='new@example.com',
            password='SecPass123'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_email_exists(user_service, mock_user_repo):
    """Test create user with existing email"""
    mock_user_repo.get_by_email = AsyncMock(return_value=MagicMock())
    
    with pytest.raises(ValueError, match="Email 'existing@example.com' already registered"):
        await user_service.create_user(
            username='newuser',
            email='existing@example.com',
            password='SecPass123'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_weak_password(user_service):
    """Test create user with weak password"""
    with pytest.raises(ValueError, match="Password must be at least 8 characters"):
        await user_service.create_user(
            username='newuser',
            email='new@example.com',
            password='short'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_password_no_uppercase(user_service):
    """Test password validation - no uppercase"""
    with pytest.raises(ValueError, match="must contain at least one uppercase letter"):
        await user_service.create_user(
            username='newuser',
            email='new@example.com',
            password='lowercase123'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_password_no_lowercase(user_service):
    """Test password validation - no lowercase"""
    with pytest.raises(ValueError, match="must contain at least one lowercase letter"):
        await user_service.create_user(
            username='newuser',
            email='new@example.com',
            password='UPPERCASE123'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_password_no_digit(user_service):
    """Test password validation - no digit"""
    with pytest.raises(ValueError, match="must contain at least one digit"):
        await user_service.create_user(
            username='newuser',
            email='new@example.com',
            password='NoDigitsHere'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authenticate_user_success(user_service, mock_user_repo):
    """Test successful authentication"""
    # Create a user with known password
    created_user = await user_service.create_user(
        username='authuser',
        email='auth@example.com',
        password='SecPass123'
    )
    
    # Get the hashed password
    password_hash = mock_user_repo.create.call_args.kwargs['password_hash']
    
    # Setup mock to return user with this hash
    mock_user = MagicMock(spec=User)
    mock_user.email = 'auth@example.com'
    mock_user.password_hash = password_hash
    mock_user_repo.get_by_email = AsyncMock(return_value=mock_user)
    
    # Try to authenticate
    result = await user_service.authenticate_user(
        email='auth@example.com',
        password='SecPass123'
    )
    
    assert result is not None
    assert result.email == 'auth@example.com'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authenticate_user_wrong_password(user_service, mock_user_repo):
    """Test authentication with wrong password"""
    # Create a real hash for "CorrectPass1"
    with patch.object(user_service.pwd_context, 'hash') as mock_hash:
        mock_hash.return_value = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYfZl.eui8m'
        
        mock_user = MagicMock(spec=User)
        mock_user.email = 'test@example.com'
        mock_user.password_hash = mock_hash.return_value
        mock_user_repo.get_by_email = AsyncMock(return_value=mock_user)
        
        result = await user_service.authenticate_user(
            email='test@example.com',
            password='WrongPass123'
        )
        
        assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authenticate_user_email_not_found(user_service, mock_user_repo):
    """Test authentication with non-existent email"""
    mock_user_repo.get_by_email = AsyncMock(return_value=None)
    
    result = await user_service.authenticate_user(
        email='nonexistent@example.com',
        password='SecPass123'
    )
    
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_success(user_service, mock_user_repo):
    """Test get user by ID"""
    result = await user_service.get_user(user_id=123)
    
    mock_user_repo.get_by_id.assert_called_once_with(123)
    assert result.id == 123


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_not_found(user_service, mock_user_repo):
    """Test get non-existent user"""
    mock_user_repo.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(ValueError, match="User 999 not found"):
        await user_service.get_user(user_id=999)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_username(user_service, mock_user_repo):
    """Test update username"""
    result = await user_service.update_user(
        user_id=123,
        username='newusername'
    )
    
    # Verify update was called
    mock_user_repo.update.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_email(user_service, mock_user_repo):
    """Test update email"""
    result = await user_service.update_user(
        user_id=123,
        email='newemail@example.com'
    )
    
    mock_user_repo.update.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_username_already_exists(user_service, mock_user_repo):
    """Test update to existing username"""
    # Mock existing user with same username
    existing_user = MagicMock()
    existing_user.id = 999  # Different user
    mock_user_repo.get_by_username = AsyncMock(return_value=existing_user)
    
    with pytest.raises(ValueError, match="Username 'existingname' already exists"):
        await user_service.update_user(
            user_id=123,
            username='existingname'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_email_already_exists(user_service, mock_user_repo):
    """Test update to existing email"""
    existing_user = MagicMock()
    existing_user.id = 999
    mock_user_repo.get_by_email = AsyncMock(return_value=existing_user)
    
    with pytest.raises(ValueError, match="Email 'existing@example.com' already registered"):
        await user_service.update_user(
            user_id=123,
            email='existing@example.com'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_change_password_success(user_service, mock_user_repo):
    """Test successful password change"""
    # Create user with known password
    await user_service.create_user(
        username='passuser',
        email='pass@example.com',
        password='OldPass123'
    )
    
    old_hash = mock_user_repo.create.call_args.kwargs['password_hash']
    
    # Setup mock user with old password
    mock_user = MagicMock(spec=User)
    mock_user.id = 123
    mock_user.password_hash = old_hash
    mock_user_repo.get_by_id = AsyncMock(return_value=mock_user)
    
    # Change password
    result = await user_service.change_password(
        user_id=123,
        old_password='OldPass123',
        new_password='NewPass456'
    )
    
    assert result is True
    mock_user_repo.update.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_change_password_wrong_old_password(user_service, mock_user_repo):
    """Test password change with wrong old password"""
    # Use a real bcrypt hash
    mock_user = MagicMock(spec=User)
    mock_user.id = 123
    mock_user.password_hash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYfZl.eui8m'  # Hash for "TestPass123"
    mock_user_repo.get_by_id = AsyncMock(return_value=mock_user)
    
    with pytest.raises(ValueError, match="Incorrect current password"):
        await user_service.change_password(
            user_id=123,
            old_password='WrongPass1',
            new_password='NewPass456'
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_change_password_weak_new_password(user_service, mock_user_repo):
    """Test password change with weak new password"""
    await user_service.create_user(
        username='passuser',
        email='pass@example.com',
        password='OldPass123'
    )
    
    old_hash = mock_user_repo.create.call_args.kwargs['password_hash']
    mock_user = MagicMock(spec=User)
    mock_user.password_hash = old_hash
    mock_user_repo.get_by_id = AsyncMock(return_value=mock_user)
    
    with pytest.raises(ValueError, match="Password must be at least 8 characters"):
        await user_service.change_password(
            user_id=123,
            old_password='OldPass123',
            new_password='weak'
        )

@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_user_success(user_service, mock_user_repo):
    """Test successful user deletion"""
    result = await user_service.delete_user(user_id=123)
    
    mock_user_repo.delete.assert_called_once_with(123)
    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_user_not_found(user_service, mock_user_repo):
    """Test delete non-existent user"""
    mock_user_repo.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(ValueError, match="User 999 not found"):
        await user_service.delete_user(user_id=999)