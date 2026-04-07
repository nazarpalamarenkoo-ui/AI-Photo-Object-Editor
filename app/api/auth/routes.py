from fastapi import APIRouter, Depends, HTTPException
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.mail import (
    generate_password_reset_token,
    generate_signup_confirmation_token,
    send_confirmation_email,
    send_reset_password_email,
    validate_password_reset_token,
    validate_signup_confirmation_token,
)
from app.api.auth.schema import SignInArgs, SignUpArgs
from app.api.auth.auth import create_access_token
from app.db.db_connect import get_db
from app.repository.user_repo import UserRepository
from app.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["Auth"])


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db=db, user_repo=UserRepository(db))


@router.post("/login")
async def login(
    args: SignInArgs,
    service: UserService = Depends(get_user_service)
):
    user = await service.authenticate_user(args.email, args.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/signup")
async def signup(
    signup_args: SignUpArgs,
    service: UserService = Depends(get_user_service)
):
    if await service.user_repo.exists_by_email(signup_args.email):
        raise HTTPException(status_code=400, detail="This user already exists")

    token = generate_signup_confirmation_token(signup_args)
    await send_confirmation_email(token, signup_args)

    raise HTTPException(status_code=200, detail="Email has been sent")


@router.post("/signup-confirmation")
async def signup_confirmation(
    token: str,
    service: UserService = Depends(get_user_service)
):
    args = validate_signup_confirmation_token(token)

    if await service.user_repo.exists_by_email(args.email):
        raise HTTPException(status_code=200, detail="Email has been already confirmed")

    user = await service.create_user(
        username=args.username,
        email=args.email,
        password=args.password
    )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/password-recovery")
async def recover_password(
    email: EmailStr,
    db: AsyncSession = Depends(get_db)
):
    token = generate_password_reset_token(email)
    await send_reset_password_email(db, email, token)
    raise HTTPException(status_code=200, detail="Email has been sent")


@router.patch("/reset-password")
async def reset_password(
    new_password: str,
    email: EmailStr = Depends(validate_password_reset_token),
    service: UserService = Depends(get_user_service)
):
    user = await service.user_repo.get_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    await service.user_repo.update_password(user, pwd_context.hash(new_password))

    raise HTTPException(status_code=200, detail="Password updated successfully")