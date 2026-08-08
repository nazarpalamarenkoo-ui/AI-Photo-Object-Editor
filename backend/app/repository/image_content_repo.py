from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models.image_content import ImageContent
from app.repository.base_repo import BaseRepository


class ImageContentRepository(BaseRepository):

    async def get_by_id(self, content_id: int) -> Optional[ImageContent]:
        async with self.session_factory() as db:
            result = await db.execute(select(ImageContent).where(ImageContent.id == content_id))
            return result.scalar_one_or_none()

    async def get_by_hash(self, content_hash: str) -> Optional[ImageContent]:
        async with self.session_factory() as db:
            result = await db.execute(select(ImageContent).where(ImageContent.content_hash == content_hash))
            return result.scalar_one_or_none()

    async def create(
        self,
        content_hash: str,
        storage_path: str,
        width: int,
        height: int,
        file_size: int,
    ) -> ImageContent:
        async with self.session_factory() as db:
            content = ImageContent(
                content_hash=content_hash,
                storage_path=storage_path,
                width=width,
                height=height,
                file_size=file_size,
            )
            db.add(content)
            await db.commit()
            await db.refresh(content)
            return content

    async def get_or_create(
        self,
        content_hash: str,
        storage_path: str,
        width: int,
        height: int,
        file_size: int,
    ) -> Tuple[ImageContent, bool]:
        """
        Основна точка входу для дедуплікації. Повертає (content, created):
        created=False означає, що такий вміст уже був у базі — виклик,
        що прив'язує нову ImageVersion, може одразу переюзати готові
        detections/segmentation_masks замість запуску нового ML job.

        Гонка між SELECT і INSERT (два паралельних запити з однаковим
        content_hash) лишається можлива, як і раніше. Але тепер create()
        відкриває власну коротку сесію — якщо вона впаде на
        IntegrityError, ця сесія сама відкотиться і закриється при виході
        з `async with` усередині create(). Ручний `db.rollback()`, який
        був потрібен на спільній довгоживучій сесії, тут більше не
        потрібен: наступний get_by_hash() уже працює на новій, чистій
        сесії незалежно від того, що сталось у попередній.
        """
        existing = await self.get_by_hash(content_hash)
        if existing is not None:
            return existing, False

        try:
            content = await self.create(content_hash, storage_path, width, height, file_size)
            return content, True
        except IntegrityError:
            existing = await self.get_by_hash(content_hash)
            if existing is None:
                raise
            return existing, False