from datetime import datetime
from typing import Dict, List, Optional

from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.services.ml.base_ml_service import BaseMLService
from app.services.ml.version_carry_forward import VersionCarryForwardMixin
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class EditingService(VersionCarryForwardMixin, BaseMLService):
    """
    Handles YOLO-based destructive pixel edits: remove / replace /
    remove_multiple, plus SAM-mask diffusion replace. Owns version
    forking for these ops; carry-forward mechanics live in
    VersionCarryForwardMixin 

    Versioning model:
      - Every destructive op forks a new ImageVersion from the current one,
        storage_path = the freshly uploaded result image in S3.
      - Detection/SegmentationMask rows for the OLD version that are still
        valid get cloned onto the NEW version (VersionCarryForwardMixin).
      - undo/redo stay on the Redis byte-stack (VersionHistoryService) —
        that's a fast per-edit UI convenience layer, separate from
        ImageVersion, which represents the data-level checkpoint.

    Workflow:
        detect_objects (DetectorService)
            -> User selects bbox
            -> remove_object / replace_object / remove_multiple_objects
            -> undo / redo / save_result / reset_current_state (VersionHistoryService)
    """

    async def remove_object(
        self,
        image_id: int,
        bbox_id: int,
        user_id: int,
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Remove a single YOLO-detected object using LaMa inpainting.
        Forks a new ImageVersion; carries forward everything except the
        removed detection.

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image/detection not found or unauthorized.
        """
        with log_execution(
            "service_remove_object",
            logger=logger,
            image_id=image_id,
            bbox_id=bbox_id,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)

            detections = await self.detection_repo.get_by_content(version.content_id, active_only=True)
            detection = next((d for d in detections if d.bbox_id == bbox_id), None)
            if not detection:
                logger.warning("detection_not_found", image_id=image_id, bbox_id=bbox_id)
                raise ValueError(f"Detection with bbox_id={bbox_id} not found")

            image_bytes = await self._get_current_image_bytes(image_id, version.storage_path)
            await self.redis_history.push_undo_state(
                image_id, image_bytes, label=f"remove bbox_id={bbox_id}"
            )

            selected_bbox = {
                "x1": detection.x1, "y1": detection.y1,
                "x2": detection.x2, "y2": detection.y2,
            }
            scene_bboxes = [
                {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                for d in detections
            ]

            result = await self.pipeline.remove_object(
                image_bytes=image_bytes,
                selected_bbox=selected_bbox,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                scene_bboxes=scene_bboxes,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"remove_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            removed_boxes = await self._carry_forward_detections(
                version.content_id, new_version.content_id, excluded_bbox_ids=frozenset({bbox_id})
            )
            await self._carry_forward_masks(
                version.content_id, new_version.content_id, affected_boxes=removed_boxes or [selected_bbox]
            )

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REMOVE,
                engine=EngineType.LAMA,
                parameters={
                    "bbox_id": bbox_id,
                    "expand_mask_pixels": expand_mask_pixels,
                    "use_edge_blending": use_edge_blending,
                    "ldm_steps": ldm_steps,
                    "ldm_sampler": ldm_sampler,
                    "hd_strategy": hd_strategy,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "object_removed",
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

    async def replace_object(
        self,
        image_id: int,
        bbox_id: int,
        replace_image_bytes: bytes,
        user_id: int,
        expand_mask_pixels: int = 25,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = "mean_std",
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Replace a single YOLO-detected object with a provided image.
        Forks a new ImageVersion; carries forward everything except the
        replaced detection (and any mask overlapping its region).

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image/detection not found or unauthorized.
        """
        with log_execution(
            "service_replace_object",
            logger=logger,
            image_id=image_id,
            bbox_id=bbox_id,
            color_match_method=color_match_method,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)

            detections = await self.detection_repo.get_by_content(version.content_id, active_only=True)
            detection = next((d for d in detections if d.bbox_id == bbox_id), None)
            if not detection:
                logger.warning("detection_not_found", image_id=image_id, bbox_id=bbox_id)
                raise ValueError(f"Detection with bbox_id={bbox_id} not found")

            image_bytes = await self._get_current_image_bytes(image_id, version.storage_path)
            await self.redis_history.push_undo_state(
                image_id, image_bytes, label=f"replace bbox_id={bbox_id}"
            )

            selected_bbox = {
                "x1": detection.x1, "y1": detection.y1,
                "x2": detection.x2, "y2": detection.y2,
            }
            scene_bboxes = [
                {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                for d in detections
            ]

            result = await self.pipeline.replace_object(
                image_bytes=image_bytes,
                selected_bbox=selected_bbox,
                replacement_image_bytes=replace_image_bytes,
                expand_mask_pixels=expand_mask_pixels,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                scene_bboxes=scene_bboxes,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"replace_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            removed_boxes = await self._carry_forward_detections(
                version.content_id, new_version.content_id, excluded_bbox_ids=frozenset({bbox_id})
            )
            await self._carry_forward_masks(
                version.content_id, new_version.content_id, affected_boxes=removed_boxes or [selected_bbox]
            )

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REPLACE,
                engine=EngineType.LAMA,
                parameters={
                    "bbox_id": bbox_id,
                    "expand_mask_pixels": expand_mask_pixels,
                    "use_color_matching": use_color_matching,
                    "use_edge_blending": use_edge_blending,
                    "color_match_method": color_match_method,
                    "ldm_steps": ldm_steps,
                    "ldm_sampler": ldm_sampler,
                    "hd_strategy": hd_strategy,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "object_replaced",
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

    async def sam_replace_object_diffusion(
        self,
        image_id: int,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        reference_image_bytes: bytes,
        user_id: int,
        prompt: str = "",
        use_color_matching: bool = False,
        color_match_method: str = "color_transfer",
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: int = 0,
    ) -> Dict:
        """
        Replace a SAM-segmented object using diffusion (SD-inpainting +
        IP-Adapter) instead of LaMa + paste.

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image not found or unauthorized.
        """
        with log_execution(
            "service_sam_replace_object_diffusion",
            logger=logger,
            image_id=image_id,
            use_color_matching=use_color_matching,
            color_match_method=color_match_method,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)

            image_bytes = await self._get_current_image_bytes(image_id, version.storage_path)
            await self.redis_history.push_undo_state(
                image_id, image_bytes, label="sam replace (diffusion)"
            )

            result = await self.pipeline.sam_replace_object_diffusion(
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

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"sam_replace_diffusion_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            await self._carry_forward_detections_by_overlap(
                version.content_id, new_version.content_id, affected_boxes=[bbox]
            )
            await self._carry_forward_masks(
                version.content_id, new_version.content_id, affected_boxes=[bbox]
            )

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REPLACE,
                engine=EngineType.DIFFUSION,
                parameters={
                    "bbox": bbox,
                    "prompt": prompt,
                    "use_color_matching": use_color_matching,
                    "color_match_method": color_match_method,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "ip_adapter_scale": ip_adapter_scale,
                    "strength": strength,
                    "seed": seed,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "sam_object_replaced_diffusion",
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

    async def remove_multiple_objects(
        self,
        image_id: int,
        bbox_ids: List[int],
        user_id: int,
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Remove multiple YOLO-detected objects in a single LaMa inpainting pass.
        Forks a new ImageVersion; carries forward everything except the
        removed detections.

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp, image_version_id

        Raises:
            ValueError: If image not found, unauthorized, or no valid detections.
        """
        with log_execution(
            "service_remove_multiple_objects",
            logger=logger,
            image_id=image_id,
            num_requested=len(bbox_ids),
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)

            all_detections = await self.detection_repo.get_by_content(version.content_id, active_only=True)
            selected_detections = [d for d in all_detections if d.bbox_id in bbox_ids]

            if not selected_detections:
                logger.warning(
                    "no_valid_detections_for_removal", image_id=image_id, bbox_ids=bbox_ids
                )
                raise ValueError(f"No valid detections found for bbox_ids: {bbox_ids}")

            image_bytes = await self._get_current_image_bytes(image_id, version.storage_path)
            await self.redis_history.push_undo_state(
                image_id, image_bytes, label=f"remove {len(bbox_ids)} objects"
            )

            selected_bboxes = [
                {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                for d in selected_detections
            ]
            scene_bboxes = [
                {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
                for d in all_detections
                if d.bbox_id not in bbox_ids
            ]

            result = await self.pipeline.remove_multiple_objects(
                image_bytes=image_bytes,
                selected_bboxes=selected_bboxes,
                expand_mask_pixels=expand_mask_pixels,
                use_edge_blending=use_edge_blending,
                scene_bboxes=scene_bboxes or None,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            bbox_ids_str = "_".join(map(str, bbox_ids))
            result_path = (
                f"results/{user_id}/{image_id}/"
                f"remove_multi_{bbox_ids_str}_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

            new_version = await self._fork_version(image, result["result_bytes"], result_url)

            actually_removed_ids = frozenset(d.bbox_id for d in selected_detections)
            removed_boxes = await self._carry_forward_detections(
                version.content_id, new_version.content_id, excluded_bbox_ids=actually_removed_ids
            )
            await self._carry_forward_masks(
                version.content_id, new_version.content_id, affected_boxes=removed_boxes or selected_bboxes
            )

            await self._save_current_state(image_id, result["result_bytes"])

            await self.edit_history_repo.create(
                image_version_id=new_version.id,
                operation=EditOperation.REMOVE,
                engine=EngineType.LAMA,
                parameters={
                    "bbox_ids": bbox_ids,
                    "expand_mask_pixels": expand_mask_pixels,
                    "use_edge_blending": use_edge_blending,
                    "ldm_steps": ldm_steps,
                    "ldm_sampler": ldm_sampler,
                    "hd_strategy": hd_strategy,
                },
                processing_time_ms=self._extract_processing_time_ms(result.get("metrics")),
            )

            logger.info(
                "multiple_objects_removed",
                image_id=image_id,
                old_version_id=version.id,
                new_version_id=new_version.id,
                num_removed=len(selected_detections),
            )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
            "image_version_id": new_version.id,
        }