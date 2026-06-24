from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext

from app.repository.user_repo import UserRepository
from app.db.models.user import User

class UserService:
    
    def __init__(
        self,
        db: AsyncSession,
        user_repo: UserRepository
    ):
         
        self.db = db
        self.user_repo = user_repo
        
        self.pwd_context = CryptContext(schemes = ['bcrypt'], deprecated = ['auto'])
        
    async def create_user(
        self,
        username: str,
        email: str,
        password: str
    ) -> User:
        
        # Check username uniqueness
        existing_user = await self.user_repo.get_by_username(username)
        if existing_user:
            raise ValueError(f"Username '{username}' already exists")
        
        # Check email uniqueness
        existing_email = await self.user_repo.get_by_email(email)
        if existing_email:
            raise ValueError(f"Email '{email}' already registered")
        
        self._validate_password(password)
        
        password_hash = self.pwd_context.hash(password)
        
        user = await self.user_repo.create(
            username = username,
            email = email,
            password_hash = password_hash
        )
        
        return user
    
    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        
        # Get user by email
        user = await self.user_repo.get_by_email(email)
        
        if not user:
            return None
        
        # Verify password
        if not self.pwd_context.verify(password, user.password_hash):
            return None
        
        return user
    
    async def get_user(self, user_id: int) -> User:
         
        user = await self.user_repo.get_by_id(user_id)
        
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        return user
    
    async def update_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        email: Optional[str] = None
    ) -> User:
        
        user = await self.get_user(user_id)
        
         # Update username if provided
        if username and username != user.username:
            # Check uniqueness
            existing = await self.user_repo.get_by_username(username)
            if existing and existing.id != user_id:
                raise ValueError(f"Username '{username}' already exists")
            user.username = username
            
        # Update email if provided
        if email and email != user.email:
            # Check uniqueness
            existing = await self.user_repo.get_by_email(email)
            if existing and existing.id != user_id:
                raise ValueError(f"Email '{email}' already registered")
            user.email = email
        
        user = await self.user_repo.update(user)
        
        return user
    
    async def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> bool:
        
        user = await self.get_user(user_id)
        
        if not self.pwd_context.verify(old_password, user.password_hash):
            raise ValueError("Incorrect current password")
        
        self._validate_password(new_password) 
        
        # Hash new password
        user.password_hash = self.pwd_context.hash(new_password)
        
        await self.user_repo.update(user)
        
        return True
    
    async def delete_user(self, user_id: int) -> bool:
        
        # Check user exists
        await self.get_user(user_id)
        
        # Delete user (cascade deletes images and detections)
        success = await self.user_repo.delete(user_id)
        
        return success
    
    def _validate_password(self, password: str) -> None:
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")