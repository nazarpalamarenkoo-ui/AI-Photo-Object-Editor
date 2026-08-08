from typing import Optional, List

from sqlalchemy import select

from app.db.models.image_edit_history import ImageEditHistory
from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.repository.base_repo import BaseRepository


class ImageEditHistoryRepository(BaseRepository):

    async def create(
        self,
        image_version_id: int,
        operation: EditOperation,
        engine: EngineType,
        parameters: Optional[dict] = None,
        processing_time_ms: Optional[int] = None,
    ) -> ImageEditHistory:
        async with self.session_factory() as db:
            entry = ImageEditHistory(
                image_version_id=image_version_id,
                operation=operation,
                engine=engine,
                parameters=parameters,
                processing_time_ms=processing_time_ms,
            )
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
            return entry

    async def get_by_id(self, entry_id: int) -> Optional[ImageEditHistory]:
        async with self.session_factory() as db:
            result = await db.execute(select(ImageEditHistory).where(ImageEditHistory.id == entry_id))
            return result.scalar_one_or_none()

    async def get_by_version(self, image_version_id: int) -> List[ImageEditHistory]:
        async with self.session_factory() as db:
            result = await db.execute(
                select(ImageEditHistory)
                .where(ImageEditHistory.image_version_id == image_version_id)
                .order_by(ImageEditHistory.created_at.asc())
            )
            return result.scalars().all()  # type: ignore