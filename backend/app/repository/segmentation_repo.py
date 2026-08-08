from typing import Optional, List

from sqlalchemy import select, update

from app.db.models.segmentation import SegmentationMask
from app.repository.base_repo import BaseRepository


class SegmentationRepository(BaseRepository):

    async def create_many(self, masks: List[SegmentationMask]) -> List[SegmentationMask]:
        async with self.session_factory() as db:
            db.add_all(masks)
            await db.commit()
            return masks

    async def get_by_content(self, content_id: int, active_only: bool = True) -> List[SegmentationMask]:
        async with self.session_factory() as db:
            stmt = select(SegmentationMask).where(SegmentationMask.content_id == content_id)
            if active_only:
                stmt = stmt.where(SegmentationMask.is_active.is_(True))
            result = await db.execute(stmt)
            return result.scalars().all()  # type: ignore

    async def get_by_id(self, mask_id: int) -> Optional[SegmentationMask]:
        async with self.session_factory() as db:
            result = await db.execute(select(SegmentationMask).where(SegmentationMask.id == mask_id))
            return result.scalar_one_or_none()

    async def max_mask_id(self, content_id: int) -> int:
        masks = await self.get_by_content(content_id, active_only=False)
        return max((m.mask_id for m in masks), default=-1)

    async def soft_delete(self, mask_id: int) -> Optional[SegmentationMask]:
        async with self.session_factory() as db:
            await db.execute(
                update(SegmentationMask).where(SegmentationMask.id == mask_id).values(is_active=False)
            )
            await db.commit()
            result = await db.execute(select(SegmentationMask).where(SegmentationMask.id == mask_id))
            return result.scalar_one_or_none()

    async def delete_by_content(self, content_id: int) -> int:
        async with self.session_factory() as db:
            result = await db.execute(
                select(SegmentationMask).where(SegmentationMask.content_id == content_id)
            )
            masks = result.scalars().all()
            count = len(masks)
            for mask in masks:
                await db.delete(mask)
            await db.commit()
            return count