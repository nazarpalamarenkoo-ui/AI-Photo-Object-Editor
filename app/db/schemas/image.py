from pydantic import BaseModel, field_serializer
from datetime import datetime
class ImageBase(BaseModel):
    filename: str
    storage_path: str

class ImageCreate(ImageBase):
    user_id:int

class ImageResponse(ImageBase):
    id: int
    uploaded_at: datetime
    cache_key: str | None = None
    
    @field_serializer('uploaded_at')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()
    
    class Config:
        from_attributes = True

