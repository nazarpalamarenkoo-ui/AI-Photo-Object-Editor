from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from app.db.models.detection import Detection

class DetectionRepository:
    
    def __init__(self, db: AsyncSession):
        
        self.db = db
        
    async def create_many(self, detections: List[Detection]) -> List[Detection]:
        
        self.db.add_all(detections)
        await self.db.commit()
        return detections
    
    async def get_by_image(self, image_id: int) -> List[Detection]:
        result = await self.db.execute(
            select(Detection).where(Detection.image_id == image_id)
        )
        return result.scalars().all()