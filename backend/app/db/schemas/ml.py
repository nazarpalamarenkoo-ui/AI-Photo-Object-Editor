from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field, model_validator
from datetime import datetime

from app.db.schemas.common import BboxSchema
from app.db.schemas.assets import AssetResponse  


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

class SamReplaceDiffusionRequest(BaseModel):
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int

    prompt: str = ""
    negative_prompt: Optional[str] = None

    use_color_matching: bool = False
    color_match_method: str = 'color_transfer'

    num_inference_steps: Optional[int] = Field(None, ge=5, le=100)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    ip_adapter_scale: Optional[float] = Field(None, ge=0.0, le=1.0)
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    seed: int = 0

    @property
    def bbox(self) -> dict:
        return {
            "x1": self.bbox_x1, "y1": self.bbox_y1,
            "x2": self.bbox_x2, "y2": self.bbox_y2,
        }

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
    target_bbox: BboxSchema
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


class ExtractResponse(BaseModel):
    asset_id: str
    storage_path: str
    thumbnail_path: Optional[str] = None
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


class RenameAssetRequest(BaseModel):
    label: str