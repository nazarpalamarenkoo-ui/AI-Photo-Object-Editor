from typing import List, Optional, Literal
from pydantic import BaseModel, Field

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
    scene_bboxes: Optional[List[BboxSchema]] = None
    ldm: LdmConfig = Field(default_factory=LdmConfig)

class RemoveMultipleRequest(BaseModel):
    bbox_ids: List[int] = Field(..., min_length=1)
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = True
    scene_bboxes: Optional[List[BboxSchema]] = None
    ldm: LdmConfig = Field(default_factory=LdmConfig)

class ReplaceRequest(BaseModel):
    expand_mask_pixels: int = Field(0, ge=0, le=50)
    use_color_matching: bool = True
    use_edge_blending: bool = True
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std'
    scene_bboxes: Optional[List[BboxSchema]] = None
    ldm: LdmConfig = Field(default_factory=LdmConfig)

class MLResultResponse(BaseModel):
    result_url: str
    presigned_url: str
    metrics: dict
    timestamp: str