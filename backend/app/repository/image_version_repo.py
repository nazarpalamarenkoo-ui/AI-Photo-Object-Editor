from typing import Optional, List

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.image import Image
from app.db.models.image_version import ImageVersion
from app.repository.base_repo import BaseRepository


class ImageVersionRepository(BaseRepository):

    async def create_original(self, image: Image, content_id: int) -> ImageVersion:
        async with self.session_factory() as db:
            image = await db.merge(image)
            version = ImageVersion(
                image_id=image.id,
                version_number=0,
                storage_path=image.storage_path,
                parent_version_id=None,
                content_id=content_id,
            )
            db.add(version)
            await db.flush()

            image.current_version_id = version.id
            await db.commit()
            await db.refresh(version)
            return version

    async def create_next(self, image: Image, storage_path: str, content_id: int) -> ImageVersion:
        async with self.session_factory() as db:
            image = await db.merge(image)
            current = await self._get_current(db, image)
            if current is None:
                raise ValueError(f"image {image.id} has no current version to fork from")

            # Lock the image row for the rest of this transaction. Two
            # forks racing on the same image (e.g. a retried worker task
            # racing the original attempt) must serialize here, or they
            # can both read the same "next" number below and collide on
            # the unique constraint anyway.
            await db.execute(select(Image.id).where(Image.id == image.id).with_for_update())

            next_number = await self._next_version_number(db, image.id)

            next_version = ImageVersion(
                image_id=image.id,
                version_number=next_number,
                storage_path=storage_path,
                parent_version_id=current.id,
                content_id=content_id,
            )
            db.add(next_version)
            try:
                await db.flush()
            except IntegrityError:
                # Defense in depth: if we still collided despite the lock
                # above (shouldn't happen, but don't leave the session
                # unusable for whoever catches this).
                await db.rollback()
                raise

            image.current_version_id = next_version.id
            await db.commit()
            await db.refresh(next_version)
            return next_version

    async def _get_current(self, db: AsyncSession, image: Image) -> Optional[ImageVersion]:
        """Internal helper that operates on an already-open session.
        Used both by create_next() (must stay in the same transaction as
        the row lock + insert above) and by the public get_current()."""
        if image.current_version_id is None:
            return None
        result = await db.execute(
            select(ImageVersion).where(ImageVersion.id == image.current_version_id)
        )
        return result.scalar_one_or_none()

    async def _next_version_number(self, db: AsyncSession, image_id: int) -> int:
        """
        The next version_number for this image, globally — NOT
        current.version_number + 1.

        History branches: undo moves current_version_id back to an older
        version without deleting anything, and a subsequent edit forks a
        new sibling from that older version. If we numbered children as
        "parent + 1", two siblings forked from the same parent (one before
        an undo, one after) would both compute the same number and violate
        uq_image_version_number. The number must always be higher than
        every version_number this image has ever had, regardless of which
        branch produced it.
        """
        result = await db.execute(
            select(func.max(ImageVersion.version_number)).where(ImageVersion.image_id == image_id)
        )
        current_max = result.scalar_one()
        return (current_max or 0) + 1

    async def get_current(self, image: Image) -> Optional[ImageVersion]:
        async with self.session_factory() as db:
            image = await db.merge(image)
            return await self._get_current(db, image)

    async def get_by_id(self, version_id: int) -> Optional[ImageVersion]:
        async with self.session_factory() as db:
            result = await db.execute(select(ImageVersion).where(ImageVersion.id == version_id))
            return result.scalar_one_or_none()

    async def list_by_image(self, image_id: int) -> List[ImageVersion]:
        async with self.session_factory() as db:
            result = await db.execute(
                select(ImageVersion)
                .where(ImageVersion.image_id == image_id)
                .order_by(ImageVersion.version_number.asc())
            )
            return result.scalars().all()  # type: ignore

    async def set_current(self, image: Image, version_id: int) -> ImageVersion:
        """Undo/redo — just moves the pointer, no rows change."""
        async with self.session_factory() as db:
            image = await db.merge(image)
            result = await db.execute(select(ImageVersion).where(ImageVersion.id == version_id))
            version = result.scalar_one_or_none()
            if version is None or version.image_id != image.id:
                raise ValueError(f"version {version_id} does not belong to image {image.id}")
            image.current_version_id = version.id
            await db.commit()
            await db.refresh(image)
            return version