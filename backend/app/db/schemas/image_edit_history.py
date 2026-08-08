from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType


class ImageEditHistoryBase(BaseModel):
    image_version_id: int
    operation: EditOperation
    engine: EngineType
    parameters: Optional[dict] = None
    processing_time_ms: Optional[int] = None


class ImageEditHistoryCreate(ImageEditHistoryBase):
    pass


class ImageEditHistoryResponse(ImageEditHistoryBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True