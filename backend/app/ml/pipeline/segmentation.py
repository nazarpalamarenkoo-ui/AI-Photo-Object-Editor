import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.ml.modes.sam_lama_mode import SAMLamaMode
from app.ml.pipeline.validator import Validator
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class SegmentationMixin:
    sam_lama_mode: SAMLamaMode
    validator: Validator

    async def sam_segment_objects(
        self,
        image_bytes: bytes,
        min_area: int = 500,
        max_segments: int = 50,
    ) -> Dict:
        """
        Auto-segmentation of the entire image using MobileSAM (no prompts).

        Args:
            image_bytes:    Input image bytes
            min_area:       Minimum segment area in pixels (noise filter, default: 500)
            max_segments:   Maximum number of segments in response (default: 50)
            
        Returns:
            Dict:
                - segments:    List[Dict] — bbox, area, mask_bytes, mask_id, bbox_id,
                                           stability_score, predicted_iou
                - metrics:     Dict
                - image_size:  Tuple[int, int] — (W, H)
                - timestamp:   str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_segment_auto",
            logger=logger,
            min_area=min_area,
            max_segments=max_segments,
        ):
            self.validator.validate_image_bytes(image_bytes)

            result = await self.sam_lama_mode.segment_objects(
                image_bytes=image_bytes,
                min_area=min_area,
                max_segments=max_segments,
            )

            result["timestamp"] = datetime.now().isoformat()


        return result

    async def sam_segment_with_prompt(
        self,
        image_bytes: bytes,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        bbox: Optional[Dict[str, int]] = None,
        multimask_output: Optional[bool] = None
    ) -> Dict:
        """
        Prompt-based segmentation using MobileSAM (points and/or bounding box).

        At least one of point_coords or bbox must be provided.

        Args:
            image_bytes:    Input image bytes
            point_coords:   List of points [(x, y), ...]
            point_labels:   1 = foreground, 0 = background for each point
            bbox:           {'x1','y1','x2','y2'} — used as a spatial prompt

        Returns:
            Dict:
                - segments:    List[Dict] — sorted by stability_score descending;
                               each entry: bbox, area, mask_bytes, mask_id, bbox_id,
                               stability_score, predicted_iou
                - metrics:     Dict
                - image_size:  Tuple[int, int] — (W, H)
                - timestamp:   str
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_segment_prompt",
            logger=logger,
            has_points=point_coords is not None,
            has_bbox=bbox is not None,
        ):
            self.validator.validate_image_bytes(image_bytes)

            if point_coords is None and bbox is None:
                raise ValueError("Provide at least one of: point_coords or bbox")

            if point_coords is not None and point_labels is not None:
                if len(point_coords) != len(point_labels):
                    raise ValueError("point_coords and point_labels must have the same length")

            if bbox is not None:
                self.validator.validate_bbox(bbox)

            result = await self.sam_lama_mode.segment_with_prompt(
                image_bytes=image_bytes,
                point_coords=point_coords,
                point_labels=point_labels,
                bbox=bbox,
                multimask_output=multimask_output
            )

            result["timestamp"] = datetime.now().isoformat()


        return result

    async def sam_segment_with_prompts_batch(
        self,
        image_bytes: bytes,
        bboxes: List[Dict[str, int]],
    ) -> Dict:
        """
        Batched box-prompt segmentation using MobileSAM — one image-encoder
        pass shared across all bboxes, one cheap decoder call each.

        Args:
            image_bytes:    Input image bytes
            bboxes:         List of {'x1','y1','x2','y2'} MobileSAM box prompt

        Returns:
            Dict:
                - segments:    List[Dict] — one top mask per bbox, each with
                               bbox, area, mask_bytes, mask_id, bbox_id,
                               stability_score, predicted_iou, prompt_bbox
                - metrics:     Dict
                - image_size:  Tuple[int, int] — (W, H)
                - timestamp:   str

        Raises:
            ValueError: If bboxes is empty.
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_segment_prompts_batch",
            logger=logger,
            num_bboxes=len(bboxes) if bboxes else 0,
        ):
            self.validator.validate_image_bytes(image_bytes)

            if not bboxes:
                raise ValueError("Provide at least one bbox")

            for bbox in bboxes:
                self.validator.validate_bbox(bbox)

            result = await self.sam_lama_mode.segment_with_prompts_batch(
                image_bytes=image_bytes,
                bboxes=bboxes,
            )

            result["timestamp"] = datetime.now().isoformat()


        return result

    async def sam_segment_by_polygon(
        self,
        image_bytes: bytes,
        points: List[Tuple[int, int]],
        smooth: bool = True,
        smoothing_factor: float = 0.0,
        feather_px: int = 0,
    ) -> Dict:
        """
        Exact segmentation by polygon points (lasso), without MobileSAM —
        the mask exactly repeats the user's polygon.

        Args:
            points:            ordered (x, y) points along the contour, min. 3
            smooth:            smooth the contour with a spline (default: True)
            smoothing_factor:  strength of smoothing (default: 0.0)
            feather_px:        softness of the mask edges (default: 0)

        Returns:
            Dict: segments (1 element), metrics, image_size, timestamp
        """
        start_time = time.time()

        with log_execution(
            "pipeline_sam_segment_polygon",
            logger=logger,
            num_points=len(points) if points else 0,
        ):
            self.validator.validate_image_bytes(image_bytes)

            if points is None or len(points) < 3:
                raise ValueError("Provide at least 3 points to form a polygon")

            result = await self.sam_lama_mode.segment_by_polygon(
                image_bytes=image_bytes,
                points=points,
                smooth=smooth,
                smoothing_factor=smoothing_factor,
                feather_px=feather_px,
            )

            result["timestamp"] = datetime.now().isoformat()


        return result