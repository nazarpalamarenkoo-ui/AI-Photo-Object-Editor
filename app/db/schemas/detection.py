from pydantic import BaseModel
from typing import Optional

class DetectionBase(BaseModel):
    image_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    detected_class: Optional[str] = "unknown"
    confidence: float

class DetectionCreate(DetectionBase):
    pass

class DetectionUpdate(DetectionBase):
    pass

class DetectionResponse(DetectionBase):
    id: int
    class Config:
        from_attributes = True
