import io
from datetime import datetime
from typing import Dict

from PIL import Image as PILImage

from app.db.models.image import Image
from app.db.enums.image_status import ImageStatus
from app.services.ml.base_ml_service import BaseMLService
from app.core.logging import get_logger

logger = get_logger(__name__)


class VersionHistoryService(BaseMLService):
    """
    Owns the editing SESSION layer: undo/redo (Redis byte-stack) and
    session save/reset.
    """

    async def undo(self, image_id: int, user_id: int) -> Dict:
        """
        Undo last operation — pop from undo stack, push current to redo.
        Pure Redis byte-stack; does NOT move ImageVersion pointer or touch
        Detection/SegmentationMask.

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
        Redo last undone operation. Same Redis-only caveat as undo.

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
        """Return undo stack labels for UI display (Redis-level, not ImageVersion history)."""
        await self._get_image_authorized(image_id, user_id)
        labels = await self.redis_history.get_history_labels(image_id)
        return {"history": labels}

    async def get_current_state(self, image_id: int, user_id: int) -> Dict:
        """
        Return the presigned URL the editor should display: Redis
        current_state if present, otherwise the CURRENT VERSION's stored
        image.

        Returns:
            Dict: presigned_url, is_edited, history, image_version_id
        """
        image, version = await self._get_current_version_authorized(image_id, user_id)
        presigned_url, is_edited = await self._get_current_state_url(
            image_id, user_id, version.storage_path
        )
        return {
            "presigned_url": presigned_url,
            "is_edited": is_edited,
            "history": await self.redis_history.get_history_labels(image_id),
            "image_version_id": version.id,
        }

    async def save_result(self, image_id: int, user_id: int) -> Image:
        """
        Persist current Redis state as a new Image record in DB + S3.
        Unrelated to ImageVersion — this creates an entirely separate
        Image, left as-is.

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

        pil_img = PILImage.open(io.BytesIO(result_bytes))
        width, height = pil_img.size

        saved = await self.image_repo.create(
            filename=f"edited_{image.filename}",
            storage_path=result_s3_uri,
            user_id=user_id,
            cache_key=None,
            mime_type="image/jpeg",
            width=width,
            height=height,
            file_size=len(result_bytes),
        )
        saved.status = ImageStatus.READY
        await self.image_repo.update(saved)
        logger.info("result_saved", source_image_id=image_id, new_image_id=saved.id)
        return saved

    async def reset_current_state(self, image_id: int, user_id: int) -> None:
        """
        Reset editing session: clear Redis undo/redo state AND move
        image.current_version_id back to version 0.
        """
        image = await self._get_image_authorized(image_id, user_id)

        versions = await self.image_version_repo.list_by_image(image_id)
        original = next((v for v in versions if v.version_number == 0), None)
        if original is not None:
            await self.image_version_repo.set_current(image, original.id)

        await self.redis_storage.delete(f"image:{image_id}:current_state")
        await self.redis_history.clear_history(image_id)
        logger.info("current_state_reset", image_id=image_id)