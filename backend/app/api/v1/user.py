from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.user import UserResponse, UserUpdate, ChangePassword
from app.repository.user_repo import UserRepository
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db=db, user_repo=UserRepository(db))


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    """Update username or email."""
    try:
        return await service.update_user(
            user_id=current_user.id,
            username=body.username,
            email=body.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/me/password")
async def change_password(
    body: ChangePassword,
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    """Change password (requires current password)."""
    try:
        await service.change_password(
            user_id=current_user.id,
            old_password=body.old_password,
            new_password=body.new_password,
        )
        return {"detail": "Password updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/me")
async def delete_me(
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    """Delete current user account."""
    await service.delete_user(current_user.id)
    return {"detail": "Account deleted successfully"}