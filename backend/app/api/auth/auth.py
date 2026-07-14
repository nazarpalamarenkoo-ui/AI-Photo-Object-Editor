from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config.settings import settings
from app.db.db_connect import get_db
from app.db.models.user import User
from app.api.auth.schema import TokenData
from app.core.logging import get_logger, bind_user

logger = get_logger(__name__)

SECRET_KEY = settings.SECRET_KEY_AUTH
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub") # type: ignore
        if username is None:
            logger.warning("auth_token_missing_subject")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        logger.warning("auth_token_invalid")
        raise credentials_exception
    
    result = await db.execute(select(User).filter(User.username == token_data.username))
    user = result.scalars().first()

    if user is None:
        logger.warning("auth_token_user_not_found", username=token_data.username)
        raise credentials_exception

    # Bind user_id into contextvars so every downstream log line in this
    # request (services, ML pipeline, repositories) carries it automatically.
    bind_user(user.id)

    return user

async def authenticate_user(db: AsyncSession, email: str, password: str):
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()

    if user and verify_password(password, user.password_hash):
        logger.info("login_succeeded", user_id=user.id)
        return {"user": user}

    logger.warning("login_failed", email=email)
    return None