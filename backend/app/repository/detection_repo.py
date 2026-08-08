from typing import Optional, List

from sqlalchemy import select, update

from app.db.models.detection import Detection
from app.repository.base_repo import BaseRepository


class DetectionRepository(BaseRepository):

    async def create_many(self, detections: List[Detection]) -> List[Detection]:
        async with self.session_factory() as db:
            db.add_all(detections)
            await db.commit()
            return detections

    async def get_by_content(self, content_id: int, active_only: bool = True) -> List[Detection]:
        async with self.session_factory() as db:
            stmt = select(Detection).where(Detection.content_id == content_id)
            if active_only:
                stmt = stmt.where(Detection.is_active.is_(True))
            result = await db.execute(stmt)
            return result.scalars().all()  # type: ignore

    async def get_by_id(self, detection_id: int) -> Optional[Detection]:
        async with self.session_factory() as db:
            result = await db.execute(select(Detection).where(Detection.id == detection_id))
            return result.scalar_one_or_none()

    async def max_bbox_id(self, content_id: int) -> int:
        detections = await self.get_by_content(content_id, active_only=False)
        return max((d.bbox_id for d in detections), default=-1)

    async def soft_delete(self, detection_id: int) -> Optional[Detection]:
        async with self.session_factory() as db:
            await db.execute(
                update(Detection).where(Detection.id == detection_id).values(is_active=False)
            )
            await db.commit()
            result = await db.execute(select(Detection).where(Detection.id == detection_id))
            return result.scalar_one_or_none()

    async def delete_by_content(self, content_id: int) -> int:
        async with self.session_factory() as db:
            result = await db.execute(
                select(Detection).where(Detection.content_id == content_id)
            )
            detections = result.scalars().all()
            count = len(detections)
            for det in detections:
                await db.delete(det)
            await db.commit()
            return count