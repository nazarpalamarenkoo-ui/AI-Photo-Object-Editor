from datetime import datetime, timedelta, timezone
from json import dumps, loads
from pathlib import Path
from fastapi import HTTPException
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt

from app.config.settings import settings
from app.api.auth.schema import SignUpArgs
from app.repository.user_repo import UserRepository
from app.core.logging import get_logger

logger = get_logger(__name__)

mail_config = ConnectionConfig(
    MAIL_USERNAME = settings.MAIL_USERNAME,
    MAIL_PASSWORD= settings.MAIL_PASSWORD,
    MAIL_FROM= settings.MAIL_FROM,
    MAIL_PORT= settings.MAIL_PORT,
    MAIL_SERVER= settings.MAIL_SERVER,
    MAIL_STARTTLS= settings.MAIL_STARTTLS,
    MAIL_SSL_TLS= settings.MAIL_SSL_TLS,
    USE_CREDENTIALS= settings.USE_CREDENTIALS,
    TEMPLATE_FOLDER=Path(__file__).parent / 'email-templates' / 'build'
)

fm = FastMail(mail_config)
SECRET_KEY = settings.SECRET_KEY_AUTH
ALGORITHM = "HS256"


async def send_confirmation_email(token: str, signup_args: SignUpArgs):
    subject = f'Image editor service - email confirmation for user {signup_args.username}'
    link = f'http://localhost:5173/confirm-email/{token}'
    message = MessageSchema(
        subject=subject,
        recipients=[signup_args.email],
        template_body={
            'project_name': 'Image editor service',
            'username': signup_args.username,
            'valid_hours': 14,
            'link': link
        },
        subtype=MessageType.html
    )
    try:
        await fm.send_message(message, template_name='signup_confirmation.html')
        logger.info("confirmation_email_sent", email=signup_args.email)
    except Exception as e:
        logger.error("confirmation_email_failed", email=signup_args.email, exc_info=e)
        raise


async def send_reset_password_email(db: AsyncSession, email: EmailStr, token: str):
    user = await UserRepository(db).get_by_email(email)
    if not user:
        logger.warning("reset_password_email_user_not_found", email=email)
        raise HTTPException(status_code=404, detail="User not found")

    subject = f'Image editor service - password recovery for user {user.username}'
    link = f'http://localhost:5173/reset-password/{token}'
    message = MessageSchema(
        subject=subject,
        recipients=[email],
        template_body={
            'project_name': 'Image editor service',
            'username': user.username,
            'valid_hours': 14,
            'link': link
        },
        subtype=MessageType.html
    )
    try:
        await fm.send_message(message, template_name='reset_password.html')
        logger.info("reset_password_email_sent", email=email, user_id=user.id)
    except Exception as e:
        logger.error("reset_password_email_failed", email=email, user_id=user.id, exc_info=e)
        raise


def generate_password_reset_token(email: EmailStr) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=14)
    return jwt.encode({"exp": expires, "sub": email}, SECRET_KEY, algorithm=ALGORITHM)


def validate_password_reset_token(token: str) -> EmailStr:
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        return str(decoded_token["sub"])
    except JWTError:
        logger.warning("password_reset_token_invalid")
        raise HTTPException(status_code=400, detail="Invalid token")


def generate_signup_confirmation_token(signup_args: SignUpArgs) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=14)
    return jwt.encode(
        {"exp": expires, "sub": dumps(signup_args.model_dump())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def validate_signup_confirmation_token(token: str) -> SignUpArgs:
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        return SignUpArgs(**loads(decoded_token["sub"]))
    except JWTError as e:
        logger.warning("signup_confirmation_token_invalid", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))