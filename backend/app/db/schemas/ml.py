from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field, model_validator
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
    use_edge_blending: bool = False
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'

    @property
    def ldm(self) -> LdmConfig:
        return LdmConfig(
            ldm_steps=self.ldm_steps,
            ldm_sampler=self.ldm_sampler,
            hd_strategy=self.hd_strategy,
        )


class RemoveMultipleRequest(BaseModel):
    bbox_ids: List[int] = Field(..., min_length=1)
    expand_mask_pixels: int = Field(5, ge=0, le=50)
    use_edge_blending: bool = False
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'

    @property
    def ldm(self) -> LdmConfig:
        return LdmConfig(
            ldm_steps=self.ldm_steps,
            ldm_sampler=self.ldm_sampler,
            hd_strategy=self.hd_strategy,
        )


class ReplaceRequest(BaseModel):
    expand_mask_pixels: int = Field(0, ge=0, le=50)
    use_color_matching: bool = False
    use_edge_blending: bool = False
    color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std'
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'

    @property
    def ldm(self) -> LdmConfig:
        return LdmConfig(
            ldm_steps=self.ldm_steps,
            ldm_sampler=self.ldm_sampler,
            hd_strategy=self.hd_strategy,
        )


class SegmentRequest(BaseModel):
    min_area: int = Field(500, ge=0)
    max_segments: int = Field(50, ge=1, le=200)


class SegmentWithPromptRequest(BaseModel):
    point_coords: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None   # 1=fg, 0=bg
    bbox: Optional[BboxSchema] = None
    multimask_output: Optional[bool] = None

class SamRemoveRequest(BaseModel):
    expand_mask_pixels: int = Field(12, ge=0, le=50)
    use_edge_blending: bool = False
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'

    @property
    def ldm(self) -> LdmConfig:
        return LdmConfig(
            ldm_steps=self.ldm_steps,
            ldm_sampler=self.ldm_sampler,
            hd_strategy=self.hd_strategy,
        )


class SamReplaceRequest(BaseModel):
    expand_mask_pixels: int = 8
    use_color_matching: bool = False
    use_edge_blending: bool = False
    color_match_method: str = "color_transfer"
    ldm_steps: int = Field(25, ge=5, le=50)
    ldm_sampler: Literal['plms', 'ddim'] = 'plms'
    hd_strategy: Literal['CROP', 'RESIZE', 'ORIGINAL'] = 'CROP'

    @property
    def ldm(self) -> LdmConfig:
        return LdmConfig(
            ldm_steps=self.ldm_steps,
            ldm_sampler=self.ldm_sampler,
            hd_strategy=self.hd_strategy,
        )

class ExtractRequest(BaseModel):
    padding_pixels: int = 8
    label: Optional[str] = None
    persist_to_s3: bool = False


class PasteRequest(BaseModel):
    target_bbox: "BboxSchema"          
    asset_id: Optional[str] = None
    extracted_url: Optional[str] = None
    scale: float = 1.0
    use_color_matching: bool = False
    use_edge_blending: bool = False
    color_match_method: str = "color_transfer"

    @model_validator(mode="after")
    def _check_source(self):
        if not self.asset_id and not self.extracted_url:
            raise ValueError("Provide either asset_id or extracted_url")
        return self


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
    stability_score: Optional[float] = None


class SegmentResponse(BaseModel):
    segments: List[SegmentInfo]
    metrics: dict
    image_size: Tuple[int, int]
    timestamp: datetime

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
    
class ExtractResponse(BaseModel):
    asset_id: str
    extracted_url: Optional[str] = None
    presigned_url: Optional[str] = None
    object_size: tuple
    area_pixels: int
    cropped_bbox: dict
    timestamp: str


class PasteResponse(BaseModel):
    result_url: str
    presigned_url: str
    paste_bbox: BboxSchema
    object_size: Tuple[int, int]
    timestamp: datetime
    
class AssetResponse(BaseModel):
    asset_id: str
    source_image_id: int
    object_size: tuple
    area_pixels: int
    label: Optional[str] = None
    s3_url: Optional[str] = None
    created_at: str


class RenameAssetRequest(BaseModel):
    label: str
    