from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ImageVersionBase(BaseModel):
    image_id: int
    version_number: int
    storage_path: str
    parent_version_id: Optional[int] = None


class ImageVersionCreate(ImageVersionBase):
    pass


class ImageVersionResponse(ImageVersionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
        
