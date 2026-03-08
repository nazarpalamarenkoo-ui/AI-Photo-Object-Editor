from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from app.db.models.image import Image

class ImageRepository:
    
    def __init__(self, db: AsyncSession):
        
        self.db = db
        
    async def create(self, filename: str, storage_path: str, user_id: int, cache_key: str = None) -> Image:
        
        image = Image(
            filename = filename,
            storage_path = storage_path,
            user_id = user_id,
            cache_key = cache_key
        )
        
        self.db.add(image)
        await self.db.commit()
        await self.db.refresh(image)
        return image
    
    async def get_by_id(self, image_id: str) -> Optional[Image]:
        
        result = await self.db.execute(
            select(Image).where(Image.id == image_id)
        )
        
        return result.scalar_one_or_none()
    
    async def get_user_images(self, user_id: int) -> List[Image]:
        result = await self.db.execute(
            select(Image).where(Image.user_id == user_id).order_by(Image.uploaded_at.desc())
        )
        return result.scalars().all()
    
    async def update(self, image: Image) -> Image:
        await self.db.commit()
        await self.db.refresh(image)
        return image
    
    async def delete(self, image_id: int) -> bool:
        image = await self.get_by_id(image_id)
        if not image:
            return False
        await self.db.delete(image)
        await self.db.commit()
        return True