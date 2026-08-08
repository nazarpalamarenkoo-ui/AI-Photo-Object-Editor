from typing import Optional

from sqlalchemy import select

from app.db.models.user import User
from app.repository.base_repo import BaseRepository


class UserRepository(BaseRepository):

    async def create(self, username: str, email: str, password_hash: str) -> User:
        async with self.session_factory() as db:
            user = User(username=username, email=email, password_hash=password_hash)
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user

    async def get_by_id(self, user_id: int) -> Optional[User]:
        async with self.session_factory() as db:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        async with self.session_factory() as db:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        async with self.session_factory() as db:
            result = await db.execute(select(User).where(User.username == username))
            return result.scalar_one_or_none()

    async def exists_by_email(self, email: str) -> bool:
        async with self.session_factory() as db:
            result = await db.execute(select(User.id).where(User.email == email))
            return result.scalar_one_or_none() is not None

    async def update(self, user: User) -> User:
        """`user` may be a detached instance whose attributes were changed
        outside this session (e.g. a route handler mutated user.email
        after fetching it earlier). merge() is required to pick those
        changes up — a fresh session has nothing pending to flush
        otherwise."""
        async with self.session_factory() as db:
            merged = await db.merge(user)
            await db.commit()
            await db.refresh(merged)
            return merged

    async def update_password(self, user: User, new_password_hash: str) -> User:
        async with self.session_factory() as db:
            merged = await db.merge(user)
            merged.password_hash = new_password_hash
            await db.commit()
            await db.refresh(merged)
            return merged

    async def delete(self, user_id: int) -> bool:
        async with self.session_factory() as db:
            user = await db.get(User, user_id)
            if user is None:
                return False
            await db.delete(user)
            await db.commit()
            return True