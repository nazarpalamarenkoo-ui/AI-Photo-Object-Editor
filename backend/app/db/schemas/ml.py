from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

class BboxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class LdmConfig(BaseModel):
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'


class DetectRequest(BaseModel):
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    classes: Optional[List[str]] = None


class RemoveRequest(BaseModel):
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = True
    ldm: LdmConfig = Field(default_factory=LdmConfig)


class RemoveMultipleRequest(BaseModel):
    bbox_ids: List[int] = Field(..., min_length=1)
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = True
    ldm: LdmConfig = Field(default_factory=LdmConfig)


class ReplaceRequest(BaseModel):
    expand_mask_pixels: int = Field(0, ge=0, le=50)
    use_color_matching: bool = True
    use_edge_blending: bool = True
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std'
    ldm: LdmConfig = Field(default_factory=LdmConfig)


class SegmentRequest(BaseModel):
    min_area: int = Field(500, ge=0)
    max_segments: int = Field(50, ge=1, le=200)


class SegmentWithPromptRequest(BaseModel):
    point_coords: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None   # 1=fg, 0=bg
    bbox: Optional[BboxSchema] = None


class SamRemoveRequest(BaseModel):
    expand_mask_pixels: int = Field(12, ge=0, le=50)
    use_edge_blending: bool = True
    ldm: LdmConfig = Field(default_factory=LdmConfig)


class SamReplaceRequest(BaseModel):
    expand_mask_pixels: int = Field(8, ge=0, le=50)
    use_color_matching: bool = True
    use_edge_blending: bool = False
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'color_transfer'
    ldm: LdmConfig = Field(default_factory=LdmConfig)

class ExtractRequest(BaseModel):
    padding_pixels: int = Field(8, ge=0, le=64)


class PasteRequest(BaseModel):
    extracted_url: str
    target_bbox: BboxSchema
    scale: float = Field(1.0, ge=0.1, le=3.0)
    use_color_matching: bool = True
    use_edge_blending: bool = True
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'color_transfer'


class MLResultResponse(BaseModel):
    result_url: str
    presigned_url: str
    metrics: dict
    timestamp: datetime


class SegmentInfo(BaseModel):
    mask_id: int
    bbox_id: int
    bbox: BboxSchema
    area: int
    stability_score: float


class SegmentResponse(BaseModel):
    segments: List[SegmentInfo]
    metrics: dict
    image_size: Tuple[int, int]
    timestamp: datetime


class ExtractResponse(BaseModel):
    extracted_url: str
    presigned_url: str
    object_size: Tuple[int, int]
    area_pixels: int
    cropped_bbox: BboxSchema
    timestamp: datetime


class PasteResponse(BaseModel):
    result_url: str
    presigned_url: str
    paste_bbox: BboxSchema
    object_size: Tuple[int, int]
    timestamp: datetime