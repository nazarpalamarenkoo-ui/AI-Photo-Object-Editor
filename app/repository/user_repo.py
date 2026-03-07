from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from app.db.models.user import User

class UserRepository:
    
    def __init__(self, db: AsyncSession):
        
        self.db = db
        
    async def create(self, username: str, email: str, password_hash: str) -> User:
        
        user = User(username = username, email = email, password_hash = password_hash)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        
        return result.scalar_one_or_none()
    
    async def update(self, user: User) -> User:
        
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def delete(self, user_id: int) -> bool:
        
        user = await self.get_by_id(user_id)
        if user is None:
            return False
        await self.db.delete(user)
        await self.db.commit()
        return True
    
    
        