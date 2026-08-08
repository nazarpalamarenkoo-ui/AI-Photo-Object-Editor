from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AssetBase(BaseModel):
    width: int
    height: int
    area_pixels: int
    label: Optional[str] = None


class AssetCreate(AssetBase):
    user_id: int
    storage_path: str
    thumbnail_path: Optional[str] = None
    content_type: str = "image/png"
    file_size: Optional[int] = None
    source_image_version_id: Optional[int] = None
    source_segmentation_mask_id: Optional[int] = None


class AssetUpdate(BaseModel):
    label: str


class AssetResponse(AssetBase):
    public_id: str
    storage_path: str
    thumbnail_path: Optional[str] = None
    content_type: str
    file_size: Optional[int] = None
    source_image_version_id: Optional[int] = None
    source_segmentation_mask_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True