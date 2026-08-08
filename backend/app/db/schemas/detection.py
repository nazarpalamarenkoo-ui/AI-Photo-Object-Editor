from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DetectionBase(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    detected_class: Optional[str] = "unknown"
    confidence: float


class DetectionCreate(DetectionBase):
    content_id: int
    bbox_id: int
    model_name: str
    model_version: str
    inference_time_ms: float


class DetectionUpdate(BaseModel):
    is_active: Optional[bool] = None


class DetectionResponse(DetectionBase):
    id: int
    content_id: int
    bbox_id: int
    is_active: bool
    model_name: str
    model_version: str
    inference_time_ms: float
    created_at: datetime

    class Config:
        from_attributes = True