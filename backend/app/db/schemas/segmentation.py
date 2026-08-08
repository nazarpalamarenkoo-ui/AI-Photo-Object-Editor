from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

from app.db.schemas.common import BboxSchema
from app.db.enums.segmentation_mode import SegmentationMode


class SegmentRequest(BaseModel):
    min_area: int = Field(500, ge=0)
    max_segments: int = Field(50, ge=1, le=200)


class SegmentWithPromptRequest(BaseModel):
    point_coords: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None   # 1=fg, 0=bg
    bbox: Optional[BboxSchema] = None
    multimask_output: Optional[bool] = None


class SegmentByPolygonRequest(BaseModel):
    points: List[Tuple[int, int]] = Field(..., min_length=3)
    smooth: bool = True
    smoothing_factor: float = 0.0
    feather_px: int = 0


class SegmentHybridRequest(BaseModel):
    yolo_conf_threshold: float = 0.35
    yolo_classes: Optional[List[str]] = None
    fallback_min_area: int = 800
    fallback_max_segments: int = 50
    overlap_iou_thresh: float = 0.5


class SegmentInfo(BaseModel):
    mask_id: int
    bbox_id: int
    bbox: BboxSchema
    area: int
    stability_score: Optional[float] = None
    mask_url: Optional[str] = None  # base64 PNG data URL of the raster mask


class SegmentResponse(BaseModel):
    segments: List[SegmentInfo]
    metrics: dict
    image_size: Tuple[int, int]
    timestamp: datetime



class SegmentationMaskBase(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    area: float
    score: float


class SegmentationMaskCreate(SegmentationMaskBase):
    content_id: int
    mask_id: int
    mask_storage_path: str
    preview_storage_path: str
    segmentation_mode: SegmentationMode
    model_name: str
    model_version: str
    inference_time_ms: float


class SegmentationMaskUpdate(BaseModel):
    is_active: Optional[bool] = None


class SegmentationMaskResponse(SegmentationMaskBase):
    id: int
    content_id: int
    mask_id: int
    is_active: bool
    segmentation_mode: SegmentationMode
    model_name: str
    model_version: str
    inference_time_ms: float
    created_at: datetime

    class Config:
        from_attributes = True