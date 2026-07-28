import time
from typing import Dict, List, Literal, Optional
from datetime import datetime

from app.ml.modes.yolo_lama_mode import YoloLamaMode
from app.ml.modes.sam_lama_mode import SAMLamaMode
from app.ml.pipeline.validator import Validator
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class ReplacementMixin:
    yolo_lama_mode: YoloLamaMode
    sam_lama_mode: SAMLamaMode
    validator: Validator

    async def replace_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        replacement_image_bytes: bytes,
        expand_mask_pixels: int = 25,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: Literal["mean_std", "histogram", "color_transfer"] = "mean_std",
        scene_bboxes: Optional[List[Dict[str, int]]] = None,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Replace object in image (YOLO bbox → LaMa → composite → ColorMatch → EdgeBlend).

        Args:
            image_bytes:               Input image bytes
            selected_bbox:             Bounding box to replace
            replacement_image_bytes:   Replacement image bytes
            expand_mask_pixels:        Pixels to expand mask (default: 25)
            use_color_matching:        Apply color matching (default: True)
            use_edge_blending:         Apply edge blending (default: True)
            color_match_method:        'mean_std' | 'histogram' | 'color_transfer'
            scene_bboxes:              Other bboxes in the scene
            ldm_steps:                 LaMa inference steps (default: 25)
            ldm_sampler:               LaMa sampler (default: 'plms')
            hd_strategy:               HD strategy (default: 'CROP')

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_replace_object",
            logger=logger,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            use_edge_blending=use_edge_blending,
            color_match_method=color_match_method,
        ):
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_bbox(selected_bbox)
            self.validator.validate_image_bytes(replacement_image_bytes)

            result = await self.yolo_lama_mode.replace_object(
                image_bytes=image_bytes,
                selected_bbox=selected_bbox,
                replacement_image_bytes=replacement_image_bytes,
                expand_mask_pixels=expand_mask_pixels,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                scene_bboxes=scene_bboxes,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result["timestamp"] = datetime.now().isoformat()



        return result

    async def sam_replace_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        replacement_image_bytes: bytes,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: Literal["mean_std", "histogram", "color_transfer"] = "color_transfer",
        expand_mask_pixels: int = 8,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
        replacement_is_cutout: bool = False,
    ) -> Dict:
        """
        Replace object using a SAM mask + LaMa + compositing.

        Pipeline:
            prepare replacement as RGBA cutout (rembg for a plain photo, or
            a plain resize when it's already a cutout — see
            replacement_is_cutout) → dilate mask → LaMa REMOVE →
            composite → ColorMatch (optional) → EdgeBlend (optional)

        Args:
            image_bytes:               Input image bytes
            mask_bytes:                Binary mask from SAM (PNG, L mode)
            bbox:                      Segment bbox {'x1','y1','x2','y2'}
            replacement_image_bytes:   Replacement image bytes. A plain
                                       photo (any format) when
                                       replacement_is_cutout is False, or an
                                       already-transparent RGBA PNG (e.g.
                                       from the asset library) when True.
            use_color_matching:        Apply color correction (default: True)
            use_edge_blending:         Smooth boundaries (default: False)
            color_match_method:        'mean_std' | 'histogram' | 'color_transfer'
            expand_mask_pixels:        Mask dilation in pixels (default: 8)
            ldm_steps:                 LaMa inference steps (default: 25)
            ldm_sampler:               LaMa sampler (default: 'plms')
            hd_strategy:               HD strategy (default: 'CROP')
            replacement_is_cutout:     True when replacement_image_bytes is
                                       an already-transparent RGBA asset from
                                       the asset library rather than an
                                       uploaded photo — skips rembg background
                                       removal (default: False)

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_replace_object",
            logger=logger,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            use_edge_blending=use_edge_blending,
            color_match_method=color_match_method,
            replacement_is_cutout=replacement_is_cutout,
        ):
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_mask_bytes(mask_bytes)
            self.validator.validate_bbox(bbox)
            self.validator.validate_image_bytes(replacement_image_bytes)

            result = await self.sam_lama_mode.replace_object(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                bbox=bbox,
                replacement_image_bytes=replacement_image_bytes,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                expand_mask_pixels=expand_mask_pixels,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
                replacement_is_cutout=replacement_is_cutout,
            )

            result["timestamp"] = datetime.now().isoformat()


        return result

    async def sam_replace_object_diffusion(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        reference_image_bytes: bytes,
        prompt: str = "",
        use_color_matching: bool = False,
        color_match_method: Literal["mean_std", "histogram", "color_transfer"] = "color_transfer",
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: int = 0,
    ) -> Dict:
        """
        Replace object using a SAM mask + diffusion (SD-inpainting + IP-Adapter).
        Pipeline:
            DiffusionReplacer: crop around the mask with context padding →
            hi-res SD-inpaint + IP-Adapter generation → feathered composite
            back into the frame → ColorMatch (optional)

        Args:
            image_bytes:               Input image bytes
            mask_bytes:                Binary mask from SAM (PNG, L mode)
            bbox:                      Segment bbox {'x1','y1','x2','y2'}
                                       (used for color matching)
            reference_image_bytes:     Reference/asset image for IP-Adapter
                                       conditioning
            prompt:                    Text prompt describing the desired
                                       result (strongly recommended — falls
                                       back to a generic quality prompt if
                                       empty)
            use_color_matching:        Apply color correction (default: False)
            color_match_method:        'mean_std' | 'histogram' | 'color_transfer'
            negative_prompt:           Overrides the configured default
            num_inference_steps:       SD inference steps override
            guidance_scale:            SD guidance scale override
            ip_adapter_scale:          IP-Adapter conditioning strength override
            strength:                  SD inpainting strength override
            seed:                      Generator seed (default: 0, deterministic)

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_replace_object_diffusion",
            logger=logger,
            use_color_matching=use_color_matching,
            color_match_method=color_match_method,
        ):
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_mask_bytes(mask_bytes)
            self.validator.validate_bbox(bbox)
            self.validator.validate_image_bytes(reference_image_bytes)

            result = await self.sam_lama_mode.replace_object_diffusion(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                bbox=bbox,
                reference_image_bytes=reference_image_bytes,
                prompt=prompt,
                use_color_matching=use_color_matching,
                color_match_method=color_match_method,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_scale=ip_adapter_scale,
                strength=strength,
                seed=seed,
            )

            result["timestamp"] = datetime.now().isoformat()

        return result