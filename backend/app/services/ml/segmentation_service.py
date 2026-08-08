import base64
from datetime import datetime
from typing import Dict, List, Optional

from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.db.enums.segmentation_mode import SegmentationMode
from app.db.models.image_version import ImageVersion
from app.db.models.segmentation import SegmentationMask
from app.services.ml.base_ml_service import BaseMLService
from app.services.ml.version_carry_forward import VersionCarryForwardMixin
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class SegmentationService(VersionCarryForwardMixin, BaseMLService):
    """
    Handles MobileSAM segmentation and SAM-based editing.

    Workflow:
        Upload image
            -> segment_objects / segment_with_prompt / segment_by_polygon / segment_hybrid
            -> sam_remove_object / sam_replace_object
    """

    DEFAULT_MODEL_NAME = "mobile_sam"
    DEFAULT_MODEL_VERSION = "unknown"

    async def _next_mask_offset(self, content_id: int) -> int:
        """Next free mask_id for this content, based on what's persisted in DB —
        scoped by content_id so two versions sharing the same pixel content
        also share the same mask_id sequence instead of colliding/duplicating."""
        return await self.segmentation_repo.max_mask_id(content_id) + 1

    async def _segments_from_cached_masks(
        self, image_id: int, content_id: int, masks: List[SegmentationMask]
    ) -> List[Dict]:
        """
        Rehydrate already-persisted SegmentationMask rows into the same
        dict shape the pipeline produces (bbox/area/score/mask_bytes), so
        a cache hit can go through the exact same _segments_for_response
        formatting a fresh pipeline run would.

        """
        segments = []
        for m in masks:
            cache_suffix = f"mask:{content_id}:{m.mask_id}"
            mask_bytes = await self.redis_storage.get_cache_image(image_id, suffix=cache_suffix)
            if not mask_bytes:
                mask_bytes = await self.s3.download(m.mask_storage_path)
                await self.redis_storage.cache_image(
                    image_id=image_id,
                    image_data=mask_bytes,
                    suffix=cache_suffix,
                    ttl=7200,
                )
            segments.append({
                "mask_id": m.mask_id,
                "bbox_id": m.mask_id,
                "bbox": {"x1": m.x1, "y1": m.y1, "x2": m.x2, "y2": m.y2},
                "area": m.area,
                "stability_score": m.score,
                "mask_bytes": mask_bytes,
            })
        return segments

    async def _persist_segments(
        self,
        image_id: int,
        user_id: int,
        version: ImageVersion,
        segments: List[Dict],
        mode: SegmentationMode,
        metrics: Optional[dict] = None,
    ) -> List[SegmentationMask]:
        """
        Upload each segment's raster mask to S3, write SegmentationMask rows,
        and warm the Redis mask-bytes cache.

        """
        content_id = version.content_id
        db_masks = []
        for seg in segments:
            seg_metrics = seg.pop("_meta_source_metrics", None) or metrics 
            meta = self._extract_model_meta(seg_metrics, self.DEFAULT_MODEL_NAME, self.DEFAULT_MODEL_VERSION)
            mask_bytes = seg["mask_bytes"]
            path = f"masks/content_{content_id}/{seg['mask_id']}.png"
            mask_url = await self.s3.upload_bytes(mask_bytes, path, content_type="image/png")

            bbox = seg["bbox"]
            db_masks.append(
                SegmentationMask(
                    content_id=content_id,
                    mask_id=seg["mask_id"],
                    mask_storage_path=mask_url,
                    preview_storage_path=mask_url,
                    x1=bbox["x1"],
                    y1=bbox["y1"],
                    x2=bbox["x2"],
                    y2=bbox["y2"],
                    area=seg.get("area", 0.0),
                    score=seg.get("stability_score", seg.get("score", 0.0)) or 0.0,
                    segmentation_mode=mode,
                    model_name=meta.model_name,
                    model_version=meta.model_version,
                    inference_time_ms=meta.inference_time_ms,
                )
            )

            # Cache key must match base_ml_service._get_segment_or_raise's
            # read key exactly: f"mask:{content_id}:{mask_id}".
            await self.redis_storage.cache_image(
                image_id=image_id,
                image_data=mask_bytes,
                suffix=f"mask:{content_id}:{seg['mask_id']}",
                ttl=7200,
            )

        persisted = await self.segmentation_repo.create_many(db_masks)
        logger.info(
            "segments_persisted",
            image_id=image_id,
            image_version_id=version.id,
            content_id=content_id,
            mode=mode.value,
            count=len(persisted),
        )
        return persisted

    async def segment_objects(
        self,
        image_id: int,
        user_id: int,
        min_area: int = 500,
        max_segments: int = 50,
    ) -> Dict:
        """
        Auto-segment all objects using MobileSAM (no prompts).

        Returns:
            Dict: segments, metrics, image_size, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or has no current version.
        """
        with log_execution(
            "service_segment_objects",
            logger=logger,
            image_id=image_id,
            min_area=min_area,
            max_segments=max_segments,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)

            # Dedup point, same shape as DetectorService.detect_objects:
            # this content_id may already have been auto-segmented (same
            # upload twice, a redo, a no-op edit). If so, skip MobileSAM
            # entirely and rehydrate the persisted masks instead.
            existing = await self.segmentation_repo.get_by_content(
                version.content_id, active_only=True
            )
            if existing:
                segments_for_response = _segments_for_response(
                    await self._segments_from_cached_masks(image_id, version.content_id, existing)
                )
                logger.debug(
                    "segments_served_from_cache",
                    image_id=image_id,
                    image_version_id=version.id,
                    content_id=version.content_id,
                    count=len(existing),
                )
                return {
                    "segments": segments_for_response,
                    "metrics": {"cache_hit": True},
                    "image_size": (image.width, image.height),
                    "timestamp": datetime.now().isoformat(),
                }

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            result = await self.pipeline.sam_segment_objects(
                image_bytes=image_bytes,
                min_area=min_area,
                max_segments=max_segments,
            )

            offset = await self._next_mask_offset(version.content_id)
            for i, seg in enumerate(result["segments"]):
                seg["mask_id"] = offset + i
                seg["bbox_id"] = offset + i

            await self._persist_segments(
                image_id, user_id, version, result["segments"],
                mode=SegmentationMode.SAM, metrics=result.get("metrics"),
            )

            segments_for_response = _segments_for_response(result["segments"])

            logger.info(
                "segments_persisted_auto",
                image_id=image_id,
                image_version_id=version.id,
                num_segments=len(segments_for_response),
            )

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
        multimask_output: Optional[bool] = None,
    ) -> Dict:
        """
        Prompt-based MobileSAM segmentation — points or bbox as input.

        Returns:
            Dict: segments, metrics, image_size, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or has no current version.
        """
        with log_execution(
            "service_segment_with_prompt",
            logger=logger,
            image_id=image_id,
            num_points=len(point_coords) if point_coords else 0,
            has_bbox=bbox is not None,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            result = await self.pipeline.sam_segment_with_prompt(
                image_bytes=image_bytes,
                point_coords=point_coords,
                point_labels=point_labels,
                bbox=bbox,
                multimask_output=multimask_output,
            )

            offset = await self._next_mask_offset(version.content_id)
            for i, seg in enumerate(result["segments"]):
                seg["mask_id"] = offset + i
                seg["bbox_id"] = offset + i

            await self._persist_segments(
                image_id, user_id, version, result["segments"],
                mode=SegmentationMode.SAM, metrics=result.get("metrics"),
            )

            segments_for_response = _segments_for_response(result["segments"])

            logger.info(
                "segments_persisted_prompt",
                image_id=image_id,
                image_version_id=version.id,
                num_segments=len(segments_for_response),
            )

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
        Exact segmentation by polygon points (lasso), without MobileSAM.

        Returns:
            Dict: segments, metrics, image_size, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or has no current version.
        """
        with log_execution(
            "service_segment_by_polygon",
            logger=logger,
            image_id=image_id,
            num_points=len(points),
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            result = await self.pipeline.sam_segment_by_polygon(
                image_bytes=image_bytes,
                points=points,
                smooth=smooth,
                smoothing_factor=smoothing_factor,
                feather_px=feather_px,
            )

            offset = await self._next_mask_offset(version.content_id)
            for seg in result["segments"]:
                seg["mask_id"] = offset
                seg["bbox_id"] = offset

            await self._persist_segments(
                image_id, user_id, version, result["segments"],
                mode=SegmentationMode.POLYGON, metrics=result.get("metrics"),
            )

            segments_for_response = _segments_for_response(result["segments"])

            logger.info(
                "segments_persisted_polygon",
                image_id=image_id,
                image_version_id=version.id,
                num_segments=len(segments_for_response),
            )

        return {
            "segments": segments_for_response,
            "metrics": result["metrics"],
            "image_size": result["image_size"],
            "timestamp": datetime.now().isoformat(),
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
        Hybrid segmentation: YOLO finds common objects first,
        then each YOLO bbox is segmented with MobileSAM as a prompt.

        Returns:
            Dict: segments, image_size, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or has no current version.
        """
        with log_execution(
            "service_segment_hybrid",
            logger=logger,
            image_id=image_id,
            yolo_conf_threshold=yolo_conf_threshold,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            # 1. YOLO — fast pass, internal prompt source only (not persisted).
            detection_result = await self.pipeline.detect_objects(
                image_bytes=image_bytes,
                conf_threshold=yolo_conf_threshold,
                classes=yolo_classes,
            )

            yolo_bboxes = [
                {"x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"]}
                for det in detection_result["detections"]
            ]

            all_segments: List[Dict] = []
            covered_bboxes: List[Dict] = []

            # 2. MobileSAM for all YOLO bboxes in a single encoder pass.
            if yolo_bboxes:
                batch_result = await self.pipeline.sam_segment_with_prompts_batch(
                    image_bytes=image_bytes,
                    bboxes=yolo_bboxes,
                )
                for seg in batch_result["segments"]:
                    seg["source"] = "yolo"
                    seg["_meta_source_metrics"] = batch_result.get("metrics")
                    all_segments.append(seg)
                    covered_bboxes.append(seg["bbox"])

            fallback = await self.pipeline.sam_segment_objects(
                image_bytes=image_bytes,
                min_area=fallback_min_area,
                max_segments=fallback_max_segments,
            )
            for seg in fallback["segments"]:
                if not self._overlaps_any(seg["bbox"], covered_bboxes, overlap_iou_thresh):
                    seg["source"] = "sam_auto"
                    seg["_meta_source_metrics"] = fallback.get("metrics")
                    all_segments.append(seg)

            offset = await self._next_mask_offset(version.content_id)
            for i, seg in enumerate(all_segments):
                seg["mask_id"] = offset + i
                seg["bbox_id"] = offset + i

            await self._persist_segments(
                image_id, user_id, version, all_segments,
                mode=SegmentationMode.HYBRID,
            )

            segments_for_response = _segments_for_response(all_segments)

            logger.info(
                "hybrid_segments_persisted",
                image_id=image_id,
                image_version_id=version.id,
                num_yolo=len(covered_bboxes),
                num_sam_auto=len(all_segments) - len(covered_bboxes),
                total=len(all_segments),
            )

        return {
            "segments": segments_for_response,
            "image_size": fallback["image_size"],
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
        Remove object selected by MobileSAM mask_id using LaMa inpainting.

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image not found, unauthorized, or mask not found.
        """
        with log_execution(
            "service_sam_remove_object",
            logger=logger,
            image_id=image_id,
            mask_id=mask_id,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            segment = await self._get_segment_or_raise(version.content_id, mask_id, image_id)

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
            )

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"sam_remove_{mask_id}_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            affected_boxes = [segment["bbox"]]
            await self._carry_forward_detections_by_overlap(version.content_id, new_version.content_id, affected_boxes)
            await self._carry_forward_masks(version.content_id, new_version.content_id, affected_boxes)

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REMOVE,
                engine=EngineType.LAMA,
                parameters={
                    "mask_id": mask_id,
                    "expand_mask_pixels": expand_mask_pixels,
                    "use_edge_blending": use_edge_blending,
                    "ldm_steps": ldm_steps,
                    "ldm_sampler": ldm_sampler,
                    "hd_strategy": hd_strategy,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "sam_object_removed",
                image_id=image_id,
                old_version_id=version.id,
                new_version_id=new_version.id,
            )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
            "image_version_id": new_version.id,
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
        Replace object selected by MobileSAM mask_id with a provided image.

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image not found, unauthorized, or mask not found.
        """
        with log_execution(
            "service_sam_replace_object",
            logger=logger,
            image_id=image_id,
            mask_id=mask_id,
            color_match_method=color_match_method,
            replacement_is_cutout=replacement_is_cutout,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            segment = await self._get_segment_or_raise(version.content_id, mask_id, image_id)

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
                color_match_method=color_match_method,  # type: ignore
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
                replacement_is_cutout=replacement_is_cutout,
            )

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"sam_replace_{mask_id}_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            affected_boxes = [segment["bbox"]]
            await self._carry_forward_detections_by_overlap(version.content_id, new_version.content_id, affected_boxes)
            await self._carry_forward_masks(version.content_id, new_version.content_id, affected_boxes)

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REPLACE,
                engine=EngineType.LAMA,
                parameters={
                    "mask_id": mask_id,
                    "expand_mask_pixels": expand_mask_pixels,
                    "use_color_matching": use_color_matching,
                    "use_edge_blending": use_edge_blending,
                    "color_match_method": color_match_method,
                    "ldm_steps": ldm_steps,
                    "ldm_sampler": ldm_sampler,
                    "hd_strategy": hd_strategy,
                    "replacement_is_cutout": replacement_is_cutout,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "sam_object_replaced",
                image_id=image_id,
                old_version_id=version.id,
                new_version_id=new_version.id,
            )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
            "image_version_id": new_version.id,
        }

    def get_supported_classes(self) -> List[str]:
        """Passthrough to the internal YOLO pass used by segment_hybrid."""
        return self.pipeline.get_supported_classes()

    @staticmethod
    def _overlaps_any(bbox: Dict, existing_bboxes: List[Dict], iou_thresh: float) -> bool:
        return any(
            SegmentationService._iou(bbox, eb) > iou_thresh
            for eb in existing_bboxes
        )

    @staticmethod
    def _iou(a: Dict, b: Dict) -> float:
        x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


def _mask_to_data_url(mask_bytes: bytes) -> str:
    """PNG mask bytes -> base64 data URL the frontend can drop straight
    into an <image href="..."> / SVG <mask>, no separate fetch needed."""
    b64 = base64.b64encode(mask_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _segments_for_response(segments: List[Dict]) -> List[Dict]:
    """
    Strip the raw mask_bytes from each segment before sending it to the client,
    but keep a `mask_url` data-URL in its place so the frontend can render the real mask
    contour instead of falling back to a bbox rectangle.
    """
    result = []
    for seg in segments:
        mask_bytes = seg.get("mask_bytes")
        seg_out = {k: v for k, v in seg.items() if k != "mask_bytes"}
        if mask_bytes:
            seg_out["mask_url"] = _mask_to_data_url(mask_bytes)
        result.append(seg_out)
    return result