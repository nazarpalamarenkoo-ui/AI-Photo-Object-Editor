from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    classes: Optional[List[str]] = None

class RemoveRequest(BaseModel):
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = True


class RemoveMultipleRequest(BaseModel):
    bbox_ids: List[int] = Field(..., min_length=1)
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = True


class ReplaceRequest(BaseModel):
    expand_mask_pixels: int = Field(0, ge=0, le=50)
    use_color_matching: bool = True
    use_edge_blending: bool = True
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std'


class MLResultResponse(BaseModel):
    result_url: str
    presigned_url: str
    metrics: dict
    timestamp: str