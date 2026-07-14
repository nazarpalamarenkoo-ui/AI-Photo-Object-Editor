from datetime import datetime
from typing import Dict, List

from app.db.models.image import Image
from app.services.ml.base_ml_service import BaseMLService
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class EditingService(BaseMLService):
    """
    Handles YOLO-based image editing: remove / replace / remove_multiple.
    Owns undo / redo / history and save / reset lifecycle.

    Workflow:
        detect_objects (DetectorService)
            -> User selects bbox
            -> remove_object / replace_object / remove_multiple_objects
            -> undo / redo / save_result / reset_current_state
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

        Args:
            image_id:           ID of image to process
            bbox_id:            Detection bbox_id to remove
            user_id:            ID of requesting user
            expand_mask_pixels: Mask expansion in pixels (default: 5)
            use_edge_blending:  Apply edge blending (default: True)
            ldm_steps:          LaMa diffusion steps (default: 25)
            ldm_sampler:        LaMa sampler (default: 'plms')
            hd_strategy:        LaMa HD strategy (default: 'CROP')

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp

        Raises:
            ValueError: If image/detection not found or unauthorized.
        """
        with log_execution(
            "service_remove_object",
            logger=logger,
            image_id=image_id,
            bbox_id=bbox_id,
        ):
            image = await self._get_image_authorized(image_id, user_id)

            detections = await self.detection_repo.get_by_image(image_id)
            detection = next((d for d in detections if d.bbox_id == bbox_id), None)
            if not detection:
                logger.warning("detection_not_found", image_id=image_id, bbox_id=bbox_id)
                raise ValueError(f"Detection with bbox_id={bbox_id} not found")

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
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
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            await self._save_current_state(image_id, result["result_bytes"])
            await self.detection_repo.delete_by_image(image_id)
            await self.redis_storage.delete(f"image:{image_id}:detections")

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"remove_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
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

        Args:
            image_id:             ID of image to process
            bbox_id:              Detection bbox_id to replace
            replace_image_bytes:  Replacement image bytes
            user_id:              ID of requesting user
            expand_mask_pixels:   Mask expansion in pixels (default: 25)
            use_color_matching:   Apply color matching (default: True)
            use_edge_blending:    Apply edge blending (default: True)
            color_match_method:   Color match method (default: 'mean_std')
            ldm_steps:            LaMa diffusion steps (default: 25)
            ldm_sampler:          LaMa sampler (default: 'plms')
            hd_strategy:          LaMa HD strategy (default: 'CROP')

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp

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
            image = await self._get_image_authorized(image_id, user_id)

            detections = await self.detection_repo.get_by_image(image_id)
            detection = next((d for d in detections if d.bbox_id == bbox_id), None)
            if not detection:
                logger.warning("detection_not_found", image_id=image_id, bbox_id=bbox_id)
                raise ValueError(f"Detection with bbox_id={bbox_id} not found")

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
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
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            await self._save_current_state(image_id, result["result_bytes"])
            await self.detection_repo.delete_by_image(image_id)
            await self.redis_storage.delete(f"image:{image_id}:detections")

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"replace_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
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

        Args:
            image_id:           ID of image to process
            bbox_ids:           List of detection bbox_ids to remove
            user_id:            ID of requesting user
            expand_mask_pixels: Mask expansion per bbox in pixels (default: 5)
            use_edge_blending:  Apply edge blending (default: True)
            ldm_steps:          LaMa diffusion steps (default: 25)
            ldm_sampler:        LaMa sampler (default: 'plms')
            hd_strategy:        LaMa HD strategy (default: 'CROP')

        Returns:
            Dict: result_url, presigned_url, metrics, timestamp

        Raises:
            ValueError: If image not found, unauthorized, or no valid detections.
        """
        with log_execution(
            "service_remove_multiple_objects",
            logger=logger,
            image_id=image_id,
            num_requested=len(bbox_ids),
        ):
            image = await self._get_image_authorized(image_id, user_id)

            all_detections = await self.detection_repo.get_by_image(image_id)
            selected_detections = [d for d in all_detections if d.bbox_id in bbox_ids]

            if not selected_detections:
                logger.warning(
                    "no_valid_detections_for_removal", image_id=image_id, bbox_ids=bbox_ids
                )
                raise ValueError(f"No valid detections found for bbox_ids: {bbox_ids}")

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
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
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy,
            )

            await self._save_current_state(image_id, result["result_bytes"])

            for det in selected_detections:
                await self.db.delete(det)
            await self.db.commit()
            await self.redis_storage.delete(f"image:{image_id}:detections")

            bbox_ids_str = "_".join(map(str, bbox_ids))
            result_path = (
                f"results/{user_id}/{image_id}/"
                f"remove_multi_{bbox_ids_str}_{int(datetime.utcnow().timestamp())}.jpg"
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

    async def undo(self, image_id: int, user_id: int) -> Dict:
        """
        Undo last operation — pop from undo stack, push current to redo.

        Returns:
            Dict: presigned_url, label, history

        Raises:
            ValueError: If nothing to undo.
        """
        await self._get_image_authorized(image_id, user_id)

        current = await self.redis_storage.get_cache_image(image_id, suffix="current_state")
        prev_state = await self.redis_history.pop_undo_state(image_id)

        if not prev_state:
            logger.info("undo_nothing_to_undo", image_id=image_id)
            raise ValueError("Nothing to undo")

        if current:
            await self.redis_history.push_redo_state(image_id, current, label="redo")

        await self._save_current_state(image_id, prev_state["bytes"])
        presigned_url = await self._get_temp_url_from_bytes(
            image_id, user_id, prev_state["bytes"], "undo"
        )
        logger.info("undo_applied", image_id=image_id, label=prev_state["label"])

        return {
            "presigned_url": presigned_url,
            "label": prev_state["label"],
            "history": await self.redis_history.get_history_labels(image_id),
        }

    async def redo(self, image_id: int, user_id: int) -> Dict:
        """
        Redo last undone operation.

        Returns:
            Dict: presigned_url, label, history

        Raises:
            ValueError: If nothing to redo.
        """
        await self._get_image_authorized(image_id, user_id)

        current = await self.redis_storage.get_cache_image(image_id, suffix="current_state")
        next_state = await self.redis_history.pop_redo_state(image_id)

        if not next_state:
            logger.info("redo_nothing_to_redo", image_id=image_id)
            raise ValueError("Nothing to redo")

        if current:
            await self.redis_history.push_undo_state(
                image_id, current, label="redo_checkpoint"
            )

        await self._save_current_state(image_id, next_state["bytes"])
        presigned_url = await self._get_temp_url_from_bytes(
            image_id, user_id, next_state["bytes"], "redo"
        )
        logger.info("redo_applied", image_id=image_id, label=next_state["label"])

        return {
            "presigned_url": presigned_url,
            "label": next_state["label"],
            "history": await self.redis_history.get_history_labels(image_id),
        }

    async def get_history(self, image_id: int, user_id: int) -> Dict:
        """Return undo stack labels for UI display."""
        await self._get_image_authorized(image_id, user_id)
        labels = await self.redis_history.get_history_labels(image_id)
        return {"history": labels}

    async def get_current_state(self, image_id: int, user_id: int) -> Dict:
        """
        Return the presigned URL the editor should actually display: the Redis
        current_state if the user has made edits, otherwise the original S3 image.

        This must be called every time the editor page (re)loads — on first open,
        on refresh, and after a dropped connection — so the UI always shows what
        the backend will actually keep editing on top of, instead of silently
        falling back to the untouched original.

        Returns:
            Dict: presigned_url, is_edited, history
        """
        image = await self._get_image_authorized(image_id, user_id)
        presigned_url, is_edited = await self._get_current_state_url(
            image_id, user_id, image.storage_path
        )
        return {
            "presigned_url": presigned_url,
            "is_edited": is_edited,
            "history": await self.redis_history.get_history_labels(image_id),
        }

    async def save_result(self, image_id: int, user_id: int) -> Image:
        """
        Persist current Redis state as a new Image record in DB + S3.

        Returns:
            Newly created Image record with status='processed'.

        Raises:
            ValueError: If no processed result exists in Redis.
        """
        image = await self._get_image_authorized(image_id, user_id)

        result_bytes = await self.redis_storage.get_cache_image(
            image_id, suffix="current_state"
        )
        if not result_bytes:
            logger.warning("save_result_nothing_to_save", image_id=image_id)
            raise ValueError("No processed result to save. Run an operation first.")

        result_path = (
            f"saved/{user_id}/{image_id}/"
            f"result_{int(datetime.utcnow().timestamp())}.jpg"
        )
        result_s3_uri = await self.s3.upload_bytes(
            data=result_bytes, path=result_path, content_type="image/jpeg"
        )

        saved = await self.image_repo.create(
            filename=f"edited_{image.filename}",
            storage_path=result_s3_uri,
            user_id=user_id,
            cache_key=None,
        )
        saved.status = "processed"
        await self.image_repo.update(saved)
        logger.info("result_saved", source_image_id=image_id, new_image_id=saved.id)
        return saved

    async def reset_current_state(self, image_id: int) -> None:
        """Reset current state — next operation will use original S3 image."""
        await self.redis_storage.delete(f"image:{image_id}:current_state")
        await self.redis_history.clear_history(image_id)
        logger.info("current_state_reset", image_id=image_id)