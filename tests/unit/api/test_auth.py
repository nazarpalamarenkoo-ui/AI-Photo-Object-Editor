import sys
from unittest.mock import MagicMock

mock_fastapi_mail = MagicMock()
sys.modules['fastapi_mail'] = mock_fastapi_mail
sys.modules['fastapi_mail.config'] = mock_fastapi_mail


import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import timedelta, datetime, timezone

from fastapi import HTTPException
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.api.auth.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    authenticate_user,
    get_current_user,
    SECRET_KEY,
    ALGORITHM,
)
from app.api.auth.mail import (
    generate_signup_confirmation_token,
    validate_signup_confirmation_token,
    generate_password_reset_token,
    validate_password_reset_token,
)
from app.api.auth.schema import SignUpArgs
from app.db.models.user import User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _make_user(username="testuser", email="test@example.com", password="secret"):
    user = User()
    user.id = 1
    user.username = username
    user.email = email
    user.password_hash = pwd_context.hash(password)
    return user

@pytest.mark.unit
class TestPasswordUtils:

    def test_hash_and_verify(self):
        hashed = get_password_hash("pw")
        assert verify_password("pw", hashed)

    def test_wrong_password(self):
        hashed = get_password_hash("pw")
        assert not verify_password("wrong", hashed)

    def test_hash_not_plain(self):
        assert get_password_hash("pw") != "pw"

    def test_hash_salt(self):
        assert get_password_hash("pw") != get_password_hash("pw")

@pytest.mark.unit
class TestAccessToken:

    def test_contains_sub(self):
        token = create_access_token({"sub": "alice"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "alice"

    def test_has_exp(self):
        token = create_access_token({"sub": "alice"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_custom_expiry(self):
        before = datetime.now(timezone.utc)

        token = create_access_token(
            {"sub": "alice"},
            expires_delta=timedelta(minutes=5)
        )

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)

        assert before < exp < before + timedelta(minutes=6)

    def test_expired_token(self):
        token = create_access_token(
            {"sub": "alice"},
            expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(JWTError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

@pytest.mark.unit
class TestSignupTokens:

    def test_signup_token_roundtrip(self):
        args = SignUpArgs(
            username="alice",
            email="alice@example.com",
            password="pw"
        )

        token = generate_signup_confirmation_token(args)
        result = validate_signup_confirmation_token(token)

        assert result.username == args.username
        assert result.email == args.email

    def test_invalid_signup_token(self):
        with pytest.raises(HTTPException):
            validate_signup_confirmation_token("bad.token")

    def test_reset_token_roundtrip(self):
        token = generate_password_reset_token("alice@example.com")
        email = validate_password_reset_token(token)

        assert email == "alice@example.com"

    def test_invalid_reset_token(self):
        with pytest.raises(HTTPException):
            validate_password_reset_token("bad.token")

@pytest.mark.unit
class TestAuthenticateUser:

    @pytest.mark.asyncio
    async def test_success(self):
        user = _make_user(password="correct")

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = user
        mock_db.execute.return_value = mock_result

        result = await authenticate_user(mock_db, user.email, "correct")

        assert result is not None
        assert result["user"] == user

    @pytest.mark.asyncio
    async def test_wrong_password(self):
        user = _make_user(password="correct")

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = user
        mock_db.execute.return_value = mock_result

        result = await authenticate_user(mock_db, user.email, "wrong")

        assert result is None

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute.return_value = mock_result

        result = await authenticate_user(mock_db, "none@test.com", "pw")

        assert result is None

@pytest.mark.unit
class TestGetCurrentUser:

    @pytest.mark.asyncio
    async def test_valid_token(self):
        user = _make_user()
        token = create_access_token({"sub": user.username})

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = user
        mock_db.execute.return_value = mock_result

        result = await get_current_user(token=token, db=mock_db)

        assert result.username == user.username

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        with pytest.raises(HTTPException):
            await get_current_user(token="bad.token", db=AsyncMock())

    @pytest.mark.asyncio
    async def test_missing_sub(self):
        token = create_access_token({})

        with pytest.raises(HTTPException):
            await get_current_user(token=token, db=AsyncMock())

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        token = create_access_token({"sub": "ghost"})

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException):
            await get_current_user(token=token, db=mock_db)

    @pytest.mark.asyncio
    async def test_expired_token(self):
        token = create_access_token(
            {"sub": "alice"},
            expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(HTTPException):
            await get_current_user(token=token, db=AsyncMock())