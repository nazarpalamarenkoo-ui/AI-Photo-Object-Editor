from datetime import datetime
from typing import Dict, List, Optional

from app.services.ml.base_ml_service import BaseMLService


class SegmentationService(BaseMLService):
    """
    Handles SAM 2.1 segmentation and SAM-based editing.

    Workflow:
        Upload image
            -> segment_objects / segment_with_prompt  (cache segments in Redis)
            -> sam_remove_object / sam_replace_object
    """
    async def _next_mask_offset(self, image_id: int) -> int:
        """Compute the next free mask_id for this image based on what's cached."""
        cached = await self.redis_storage.get_cached_segments(image_id)
        if not cached:
            return 0
        return max(seg["mask_id"] for seg in cached) + 1
    
    async def segment_objects(
        self,
        image_id: int,
        user_id: int,
        min_area: int = 500,
        max_segments: int = 50,
    ) -> Dict:
        """
        Auto-segment all objects using SAM 2.1 (no prompts).

        Args:
            image_id:     ID of image to process
            user_id:      ID of requesting user
            min_area:     Minimum segment area in pixels (default: 500)
            max_segments: Maximum segments returned (default: 50)

        Returns:
            Dict:
                - segments:   List[Dict] — mask_id, bbox_id, bbox, area, stability_score
                - metrics:    Dict
                - image_size: Tuple[int, int]
                - timestamp:  str ISO

        Raises:
            ValueError: If image not found or unauthorized.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        result = await self.pipeline.sam_segment_objects(
            image_bytes=image_bytes,
            min_area=min_area,
            max_segments=max_segments,
        )
        for idx, seg in enumerate(result["segments"]):
            seg["mask_id"] = idx
            seg["bbox_id"] = idx

        await self.redis_storage.cache_segments(
            image_id=image_id,
            segments=result["segments"],
            ttl=7200,
        )

        segments_for_response = [
            {k: v for k, v in seg.items() if k != "mask_bytes"}
            for seg in result["segments"]
        ]

        return {
            "segments": segments_for_response,
            "metrics": result["metrics"],
            "image_size": result["image_size"],
            "timestamp": datetime.now().isoformat(),
        }

    async def segment_with_prompt(
        self,
        image_id: int,
        user_id: int,
        point_coords: Optional[List[tuple]] = None,
        point_labels: Optional[List[int]] = None,
        bbox: Optional[Dict[str, int]] = None,
        multimask_output: Optional[bool] = None
    ) -> Dict:
        """
        Prompt-based SAM segmentation — points or bbox as input.

        Args:
            image_id:     ID of image to process
            user_id:      ID of requesting user
            point_coords: List of (x, y) points
            point_labels: 1=foreground, 0=background per point
            bbox:         {'x1','y1','x2','y2'} as SAM prompt

        Returns:
            Dict:
                - segments:   List[Dict] — sorted by stability_score desc
                - metrics:    Dict
                - image_size: Tuple[int, int]
                - timestamp:  str ISO

        Raises:
            ValueError: If image not found or unauthorized.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        result = await self.pipeline.sam_segment_with_prompt(
            image_bytes=image_bytes,
            point_coords=point_coords,
            point_labels=point_labels,
            bbox=bbox,
            multimask_output=multimask_output
        )

        offset = await self._next_mask_offset(image_id)
        for i, seg in enumerate(result["segments"]):
            seg["mask_id"] = offset + i
            seg["bbox_id"] = offset + i

        existing = await self.redis_storage.get_cached_segments(image_id) or []
        await self.redis_storage.cache_segments(
            image_id=image_id,
            segments=existing + result["segments"],
            ttl=7200,
        )

        segments_for_response = [
            {k: v for k, v in seg.items() if k != "mask_bytes"}
            for seg in result["segments"]
        ]

        return {
            "segments": segments_for_response,
            "metrics": result["metrics"],
            "image_size": result["image_size"],
            "timestamp": datetime.now().isoformat(),
        }

    async def segment_by_polygon(
        self,
        image_id: int,
        user_id: int,
        points: List[tuple],
        smooth: bool = True,
        smoothing_factor: float = 0.0,
        feather_px: int = 0,
    ) -> Dict:
        """
        Exact segmentation by polygon points (lasso), without SAM2.

        Returns:
            Dict: segments, metrics, image_size, timestamp

        Raises:
            ValueError: If image not found or unauthorized.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        result = await self.pipeline.sam_segment_by_polygon(
            image_bytes=image_bytes,
            points=points,
            smooth=smooth,
            smoothing_factor=smoothing_factor,
            feather_px=feather_px,
        )

        offset = await self._next_mask_offset(image_id)
        for seg in result["segments"]:
            seg["mask_id"] = offset
            seg["bbox_id"] = offset

        existing = await self.redis_storage.get_cached_segments(image_id) or []
        await self.redis_storage.cache_segments(
            image_id=image_id,
            segments=existing + result["segments"],
            ttl=7200,
        )

        segments_for_response = [
            {k: v for k, v in seg.items() if k != "mask_bytes"}
            for seg in result["segments"]
        ]

        return {
            "segments": segments_for_response,
            "metrics": result["metrics"],
            "image_size": result["image_size"],
            "timestamp": datetime.now().isoformat(),
        }
    
    async def sam_remove_object(
        self,
        image_id: int,
        mask_id: int,
        user_id: int,
        expand_mask_pixels: int = 12,
        use_edge_blending: bool = False,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Remove object selected by SAM mask_id using LaMa inpainting.

        Args:
            image_id:           ID of image to process
            mask_id:            Segment mask_id from segment_objects
            user_id:            ID of requesting user
            expand_mask_pixels: Mask dilation in pixels (default: 12)
            use_edge_blending:  Apply edge blending (default: True)
            ldm_steps:          LaMa diffusion steps (default: 25)
            ldm_sampler:        LaMa sampler (default: 'plms')
            hd_strategy:        LaMa HD strategy (default: 'CROP')

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or segment not cached.
        """
        image = await self._get_image_authorized(image_id, user_id)
        segment = await self._get_segment_or_raise(image_id, mask_id)

        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        await self.redis_history.push_undo_state(
            image_id, image_bytes, label=f"sam_remove mask_id={mask_id}"
        )

        result = await self.pipeline.sam_remove_object(
            image_bytes=image_bytes,
            mask_bytes=segment["mask_bytes"],
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
            track_metrics=True,
        )

        await self._save_current_state(image_id, result["result_bytes"])

        result_path = (
            f"results/{user_id}/{image_id}/"
            f"sam_remove_{mask_id}_{int(datetime.utcnow().timestamp())}.jpg"
        )
        result_url, presigned_url = await self._upload_result(
            result["result_bytes"], result_path
        )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
        }

    async def sam_replace_object(
        self,
        image_id: int,
        mask_id: int,
        replacement_image_bytes: bytes,
        user_id: int,
        expand_mask_pixels: int = 8,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = "color_transfer",
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
        replacement_is_cutout: bool = False,
    ) -> Dict:
        """
        Replace object selected by SAM mask_id with a provided image.

        Args:
            image_id:                 ID of image to process
            mask_id:                  Segment mask_id from segment_objects
            replacement_image_bytes:  Replacement image bytes — a plain photo
                                       when replacement_is_cutout is False, or
                                       an already-transparent RGBA cutout
                                       (e.g. from the asset library) when True
            user_id:                  ID of requesting user
            expand_mask_pixels:       Mask dilation in pixels (default: 8)
            use_color_matching:       Apply color matching (default: True)
            use_edge_blending:        Apply edge blending (default: False)
            color_match_method:       Color match method (default: 'color_transfer')
            ldm_steps:                LaMa diffusion steps (default: 25)
            ldm_sampler:              LaMa sampler (default: 'plms')
            hd_strategy:              LaMa HD strategy (default: 'CROP')
            replacement_is_cutout:    True when replacement_image_bytes comes
                                       from the asset library (already a
                                       transparent RGBA cutout) instead of an
                                       uploaded photo — skips rembg background
                                       removal in the pipeline (default: False)

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or segment not cached.
        """
        image = await self._get_image_authorized(image_id, user_id)
        segment = await self._get_segment_or_raise(image_id, mask_id)

        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        await self.redis_history.push_undo_state(
            image_id, image_bytes, label=f"sam_replace mask_id={mask_id}"
        )

        result = await self.pipeline.sam_replace_object(
            image_bytes=image_bytes,
            mask_bytes=segment["mask_bytes"],
            bbox=segment["bbox"],
            replacement_image_bytes=replacement_image_bytes,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            use_edge_blending=use_edge_blending,
            color_match_method=color_match_method, # type: ignore
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
            track_metrics=True,
            replacement_is_cutout=replacement_is_cutout,
        )

        await self._save_current_state(image_id, result["result_bytes"])

        result_path = (
            f"results/{user_id}/{image_id}/"
            f"sam_replace_{mask_id}_{int(datetime.utcnow().timestamp())}.jpg"
        )
        result_url, presigned_url = await self._upload_result(
            result["result_bytes"], result_path
        )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
        }
    
    async def segment_hybrid(
        self,
        image_id: int,
        user_id: int,
        yolo_conf_threshold: float = 0.35,
        yolo_classes: Optional[List[str]] = None,
        fallback_min_area: int = 800,
        fallback_max_segments: int = 50,
        overlap_iou_thresh: float = 0.5,
    ) -> Dict:
        """
        Hybrid segmentation: YOLO finds common objects first (cheap, fast),
        then each YOLO bbox is segmented with SAM2 as a prompt (cheap decoder
        calls).

        Args:
            image_id:              ID of image to process
            user_id:                ID of requesting user
            yolo_conf_threshold:    YOLO confidence threshold (default: 0.35)
            yolo_classes:           Optional YOLO class name filter
            fallback_min_area:      Minimum area (px) for fallback SAM2 auto
                                     segments (default: 800)
            fallback_max_segments:  Max segments returned by the fallback
                                     SAM2 auto pass (default: 50)
            overlap_iou_thresh:     IoU threshold above which a fallback
                                     segment is considered a duplicate of an
                                     already-covered YOLO bbox (default: 0.5)

        Returns:
            Dict:
                - segments:   List[Dict] — mask_id, bbox_id, bbox, area,
                              stability_score, source ('yolo' | 'sam_auto')
                - image_size: Tuple[int, int]
                - timestamp:  str ISO

        Raises:
            ValueError: If image not found or unauthorized.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        # 1. YOLO — fast pass (~50-100ms on CPU), not persisted to the
        #    Detection table — this is an internal prompt source
        detection_result = await self.pipeline.detect_objects(
            image_bytes=image_bytes,
            conf_threshold=yolo_conf_threshold,
            classes=yolo_classes,
            track_metrics=True,
        )
        
        yolo_bboxes = [
            {"x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"]}
            for det in detection_result["detections"]
        ]
        
        all_segments: List[Dict] = []
        covered_bboxes: List[Dict] = []

        #  2. SAM2 for all YOLO bboxes in a single encoder pass — see. SAM2Segmentor.segment_with_prompts_batch.
        if yolo_bboxes:
            batch_result = await self.pipeline.sam_segment_with_prompts_batch(
                image_bytes=image_bytes,
                bboxes=yolo_bboxes,
                track_metrics=True,
            )
            for seg in batch_result["segments"]:
                seg["source"] = "yolo"
                all_segments.append(seg)
                covered_bboxes.append(seg["bbox"])

        # 3. Sparse SAM2 auto pass to catch whatever YOLO missed.
        fallback = await self.pipeline.sam_segment_objects(
            image_bytes=image_bytes,
            min_area=fallback_min_area,
            max_segments=fallback_max_segments,
        )
        for seg in fallback["segments"]:
            if not self._overlaps_any(seg["bbox"], covered_bboxes, overlap_iou_thresh):
                seg["source"] = "sam_auto"
                all_segments.append(seg)

        for idx, seg in enumerate(all_segments):
            seg["mask_id"] = idx
            seg["bbox_id"] = idx

        await self.redis_storage.cache_segments(
            image_id=image_id,
            segments=all_segments,
            ttl=7200,
        )

        segments_for_response = [
            {k: v for k, v in seg.items() if k != "mask_bytes"}
            for seg in all_segments
        ]

        return {
            "segments": segments_for_response,
            "image_size": fallback["image_size"],
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _overlaps_any(bbox: Dict, existing_bboxes: List[Dict], iou_thresh: float) -> bool:
        """Check whether bbox overlaps any of existing_bboxes above iou_thresh."""
        return any(
            SegmentationService._iou(bbox, eb) > iou_thresh
            for eb in existing_bboxes
        )

    @staticmethod
    def _iou(a: Dict, b: Dict) -> float:
        """Intersection-over-union of two {x1,y1,x2,y2} bounding boxes."""
        x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0