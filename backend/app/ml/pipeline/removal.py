import time
from typing import Dict, List, Optional
from datetime import datetime

from app.ml.modes.yolo_lama_mode import YoloLamaMode
from app.ml.modes.sam_lama_mode import SAMLamaMode
from app.ml.experiment_tracker import ExperimentTracker
from app.ml.pipeline.validator import Validator
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class RemovalMixin:
    yolo_lama_mode: YoloLamaMode
    sam_lama_mode: SAMLamaMode
    tracker: ExperimentTracker
    validator: Validator

    async def remove_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        expand_mask_pixels: int = 10,
        use_edge_blending: bool = False,
        track_metrics: bool = True,
        scene_bboxes: Optional[List[Dict[str, int]]] = None,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Remove object from image (YOLO bbox → LaMa → EdgeBlend).

        Args:
            image_bytes:         Input image bytes
            selected_bbox:       Bounding box to remove {'x1','y1','x2','y2'}
            expand_mask_pixels:  Pixels to expand mask (default: 10)
            use_edge_blending:   Apply edge blending (default: True)
            track_metrics:       Track metrics to MLflow (default: True)
            scene_bboxes:        Other bboxes in the scene (context for LaMa)
            ldm_steps:           LaMa inference steps (default: 25)
            ldm_sampler:         LaMa sampler (default: 'plms')
            hd_strategy:         HD strategy (default: 'CROP')

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_remove_object",
            logger=logger,
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
        ):
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_bbox(selected_bbox)

            result = await self.yolo_lama_mode.remove_object(
                image_bytes=image_bytes,
                selected_bbox=selected_bbox,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                scene_bboxes=scene_bboxes,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics:
                self.tracker.log_metrics({
                    "operation": "remove_object",
                    "processing_time": time.time() - start_time,
                    "expand_mask_pixels": expand_mask_pixels,
                    "edge_blending": use_edge_blending,
                })

        return result

    async def remove_multiple_objects(
        self,
        image_bytes: bytes,
        selected_bboxes: List[Dict[str, int]],
        expand_mask_pixels: int = 10,
        use_edge_blending: bool = False,
        scene_bboxes: Optional[List[Dict[str, int]]] = None,
        track_metrics: bool = True,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Remove multiple objects from image in one operation.

        Args:
            image_bytes:         Input image bytes
            selected_bboxes:     List of bounding boxes to remove
            expand_mask_pixels:  Pixels to expand mask (default: 10)
            use_edge_blending:   Apply edge blending (default: True)
            scene_bboxes:        Other bboxes in the scene
            track_metrics:       Track metrics to MLflow (default: True)
            ldm_steps:           LaMa inference steps (default: 25)
            ldm_sampler:         LaMa sampler (default: 'plms')
            hd_strategy:         HD strategy (default: 'CROP')

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_remove_multiple_objects",
            logger=logger,
            num_objects=len(selected_bboxes) if selected_bboxes else 0,
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
        ):
            self.validator.validate_image_bytes(image_bytes)

            if not selected_bboxes:
                raise ValueError("selected_bboxes cannot be empty")

            for bbox in selected_bboxes:
                self.validator.validate_bbox(bbox)

            result = await self.yolo_lama_mode.remove_multiple_objects(
                image_bytes=image_bytes,
                selected_bboxes=selected_bboxes,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics:
                self.tracker.log_metrics({
                    "operation": "remove_multiple_objects",
                    "num_objects": len(selected_bboxes),
                    "processing_time": time.time() - start_time,
                    "edge_blending": use_edge_blending,
                })

        return result

    async def sam_remove_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        use_edge_blending: bool = False,
        expand_mask_pixels: int = 12,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
        track_metrics: bool = True,
    ) -> Dict:
        """
        Remove object using a SAM mask + LaMa inpainting.

        Pipeline: dilate mask → LaMa REMOVE → EdgeBlend (optional)

        Args:
            image_bytes:         Input image bytes
            mask_bytes:          Binary mask from SAM (PNG, L mode)
            use_edge_blending:   Smooth mask boundaries (default: True)
            expand_mask_pixels:  Mask dilation in pixels (default: 12)
            ldm_steps:           LaMa inference steps (default: 25)
            ldm_sampler:         LaMa sampler (default: 'plms')
            hd_strategy:         HD strategy (default: 'CROP')
            track_metrics:       Track metrics to MLflow (default: True)

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - metrics:       Dict
                - timestamp:     str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_remove_object",
            logger=logger,
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
        ):
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_mask_bytes(mask_bytes)

            result = await self.sam_lama_mode.remove_object(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                use_edge_blending=use_edge_blending,
                expand_mask_pixels=expand_mask_pixels,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics:
                self.tracker.log_metrics({
                    "operation": "sam_remove_object",
                    "processing_time": time.time() - start_time,
                    "expand_mask_pixels": expand_mask_pixels,
                    "edge_blending": use_edge_blending,
                })

        return result