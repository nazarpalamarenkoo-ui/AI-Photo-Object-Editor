from pydantic import BaseModel
from datetime import datetime


class ImageContentBase(BaseModel):
    width: int
    height: int
    file_size: int


class ImageContentCreate(ImageContentBase):
    content_hash: str
    storage_path: str


class ImageContentResponse(ImageContentBase):
    id: int
    content_hash: str
    storage_path: str
    created_at: datetime

    class Config:
        from_attributes = True