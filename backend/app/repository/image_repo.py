from typing import Optional, List

from sqlalchemy import select

from app.db.models.image import Image
from app.repository.base_repo import BaseRepository


class ImageRepository(BaseRepository):

    async def create(
        self,
        filename,
        storage_path,
        user_id,
        mime_type,
        width,
        height,
        file_size,
        cache_key=None,
    ) -> Image:
        async with self.session_factory() as db:
            image = Image(
                filename=filename,
                storage_path=storage_path,
                user_id=user_id,
                mime_type=mime_type,
                width=width,
                height=height,
                file_size=file_size,
                cache_key=cache_key,
            )
            db.add(image)
            await db.commit()
            await db.refresh(image)
            return image

    async def get_by_id(self, image_id: int) -> Optional[Image]:
        async with self.session_factory() as db:
            result = await db.execute(select(Image).where(Image.id == image_id))
            return result.scalar_one_or_none()

    async def get_user_images(self, user_id: int) -> List[Image]:
        async with self.session_factory() as db:
            result = await db.execute(
                select(Image).where(Image.user_id == user_id).order_by(Image.uploaded_at.desc())
            )
            return result.scalars().all()  # type: ignore

    async def update(self, image: Image) -> Image:
        """`image` is typically a detached instance carried over from a
        previous short session (e.g. mutated by a service layer between
        calls). It must be merged into *this* session rather than
        add()/commit()-ed directly — a plain commit() on a fresh session
        has nothing pending to flush and silently drops the changes."""
        async with self.session_factory() as db:
            merged = await db.merge(image)
            await db.commit()
            await db.refresh(merged)
            return merged

    async def delete(self, image_id: int) -> bool:
        async with self.session_factory() as db:
            image = await db.get(Image, image_id)
            if not image:
                return False
            await db.delete(image)
            await db.commit()
            return True