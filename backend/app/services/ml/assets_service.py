from datetime import datetime
from typing import Dict, List, Optional

from app.services.ml.base_ml_service import BaseMLService
from app.storage.redis import redis_assets
from app.storage.redis.redis_assets import RedisAssetsStorage
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)


class AssetService(BaseMLService):
    """
    Handles extracted object assets: extract, paste, and the asset library
    (Redis is the primary store, S3 is an optional long-term backup).
    """

    def __init__(self, redis_assets: RedisAssetsStorage, **kwargs):
        super().__init__(redis_assets=redis_assets, **kwargs)
        self.redis_assets = redis_assets

    async def extract_object(
        self,
        image_id: int,
        mask_id: int,
        user_id: int,
        padding_pixels: int = 8,
        label: Optional[str] = None,
        persist_to_s3: bool = False,
    ) -> Dict:
        """
        Extract a SAM-segmented object from an image as an RGBA PNG cutout
        and save it into the user's asset library (Redis, optionally S3).

        Args:
            image_id: ID of the source image the segment belongs to.
            mask_id: ID of the SAM segment/mask to extract.
            user_id: ID of the requesting user (used for authorization and
                as the owner of the resulting asset).
            padding_pixels: Extra padding (in pixels) added around the
                segment's bounding box before cropping.
            label: Optional human-readable label to store with the asset.
            persist_to_s3: If True, also upload the extracted PNG to S3 and
                return an S3 URL / presigned URL alongside the Redis copy.

        Returns:
            A dict with the new asset's ID, optional S3/presigned URLs,
            object size, area in pixels, cropped bbox, and timestamp.
        """
        with log_execution(
            "service_extract_object",
            logger=logger,
            image_id=image_id,
            mask_id=mask_id,
            persist_to_s3=persist_to_s3,
        ):
            image = await self._get_image_authorized(image_id, user_id)
            segment = await self._get_segment_or_raise(image_id, mask_id)

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            result = await self.pipeline.sam_extract_object(
                image_bytes=image_bytes,
                mask_bytes=segment["mask_bytes"],
                bbox=segment["bbox"],
                padding_pixels=padding_pixels,
                track_metrics=True,
            )

            s3_url, presigned_url = None, None
            if persist_to_s3:
                extract_path = (
                    f"extracted/{user_id}/{image_id}/"
                    f"mask_{mask_id}_{int(datetime.utcnow().timestamp())}.png"
                )
                s3_url, presigned_url = await self._upload_result(
                    result["extracted_bytes"], extract_path, content_type="image/png"
                )

            asset_meta = await self.redis_assets.save_asset(
                user_id=user_id,
                extracted_bytes=result["extracted_bytes"],
                source_image_id=image_id,
                object_size=result["object_size"],
                area_pixels=result["area_pixels"],
                label=label,
                s3_url=s3_url,
            )

            logger.info(
                "asset_extracted",
                image_id=image_id,
                mask_id=mask_id,
                asset_id=asset_meta["asset_id"],
                user_id=user_id,
            )

        return {
            "asset_id": asset_meta["asset_id"],
            "extracted_url": s3_url,
            "presigned_url": presigned_url,
            "object_size": result["object_size"],
            "area_pixels": result["area_pixels"],
            "cropped_bbox": result["cropped_bbox"],
            "timestamp": result["timestamp"],
        }

    async def list_assets(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List the user's saved assets.

        Args:
            user_id: ID of the asset owner.
            limit: Maximum number of assets to return.
            offset: Number of most-recent assets to skip.

        Returns:
            A list of asset metadata dicts.
        """
        assets = await self.redis_assets.list_assets(user_id, limit=limit, offset=offset)
        logger.debug("assets_listed", user_id=user_id, count=len(assets))
        return assets

    async def get_asset_thumbnail(self, user_id: int, asset_id: str) -> Optional[bytes]:
        """
        Get the thumbnail PNG bytes for a single asset, for use in a
        library panel/dropdown preview.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset.

        Returns:
            Thumbnail PNG bytes, or None if the asset/thumbnail doesn't exist.
        """
        return await self.redis_assets.get_thumbnail(user_id, asset_id)

    async def get_asset_image(self, user_id: int, asset_id: str) -> Optional[bytes]:
        """
        Get the full-resolution extracted object bytes (RGBA PNG) for an asset.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset.

        Returns:
            Raw PNG bytes of the extracted object, or None if not found.
        """
        asset = await self.redis_assets.get_asset(user_id, asset_id, with_bytes=True)
        return asset["extracted_bytes"] if asset else None

    async def rename_asset(self, user_id: int, asset_id: str, label: str) -> Dict:
        """
        Update the label of an existing asset.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset to rename.
            label: New label to assign to the asset.

        Returns:
            The updated asset metadata dict.

        Raises:
            ValueError: If the asset does not exist.
        """
        meta = await self.redis_assets.rename_asset(user_id, asset_id, label)
        if not meta:
            logger.warning("asset_not_found", user_id=user_id, asset_id=asset_id)
            raise ValueError("Asset not found")
        logger.info("asset_renamed", user_id=user_id, asset_id=asset_id, label=label)
        return meta

    async def delete_asset(self, user_id: int, asset_id: str) -> None:
        """
        Delete an asset (metadata, extracted bytes, and thumbnail) from
        the library.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset to delete.

        Raises:
            ValueError: If the asset does not exist.
        """
        deleted = await self.redis_assets.delete_asset(user_id, asset_id)
        if not deleted:
            logger.warning("asset_not_found", user_id=user_id, asset_id=asset_id)
            raise ValueError("Asset not found")
        logger.info("asset_deleted", user_id=user_id, asset_id=asset_id)

    async def paste_extracted_object(
        self,
        image_id: int,
        user_id: int,
        target_bbox: Dict[str, int],
        asset_id: Optional[str] = None,
        extracted_url: Optional[str] = None,
        scale: float = 1.0,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = "color_transfer",
    ) -> Dict:
        """
        Paste a previously extracted object (from the asset library or an
        S3 URL) onto the current working state of an image.

        Args:
            image_id: ID of the target image to paste onto.
            user_id: ID of the requesting user (used for authorization).
            target_bbox: Destination bounding box (x1, y1, x2, y2) where
                the object should be placed.
            asset_id: ID of a saved asset in the user's library to paste.
            extracted_url: S3 URL of a previously extracted object to
                download and paste, used if `asset_id` is not provided.
            scale: Scale factor applied to the extracted object before pasting.

        Returns:
            A dict with the result image URL, presigned URL, the bbox the
            object was pasted into, its size, and a timestamp.

        """
        if not asset_id and not extracted_url:
            logger.warning(
                "paste_missing_source", image_id=image_id, user_id=user_id
            )
            raise ValueError("Provide either asset_id or extracted_url")

        with log_execution(
            "service_paste_extracted_object",
            logger=logger,
            image_id=image_id,
            asset_id=asset_id,
            scale=scale,
        ):
            image = await self._get_image_authorized(image_id, user_id)
            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            if asset_id:
                asset = await self.redis_assets.get_asset(user_id, asset_id, with_bytes=True)
                if not asset or not asset.get("extracted_bytes"):
                    logger.warning(
                        "asset_not_found_or_expired", user_id=user_id, asset_id=asset_id
                    )
                    raise ValueError("Asset not found or expired")
                extracted_bytes = asset["extracted_bytes"]
            else:
                try:
                    extracted_bytes = await self.s3.download(extracted_url)
                except Exception as e:
                    logger.error(
                        "extracted_object_download_failed",
                        extracted_url=extracted_url,
                        exc_info=e,
                    )
                    raise ValueError(f"Failed to download extracted object from S3: {e}")

            await self.redis_history.push_undo_state(
                image_id, image_bytes, label=f"paste extracted (scale={scale})"
            )

            result = await self.pipeline.sam_paste_extracted_object(
                image_bytes=image_bytes,
                extracted_bytes=extracted_bytes,
                target_bbox=target_bbox,
                scale=scale,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
                track_metrics=True,
            )

            await self._save_current_state(image_id, result["result_bytes"])

            result_path = (
                f"results/{user_id}/{image_id}/"
                f"paste_{int(datetime.utcnow().timestamp())}.jpg"
            )
            result_url, presigned_url = await self._upload_result(
                result["result_bytes"], result_path
            )

        return {
            "result_url": result_url,
            "presigned_url": presigned_url,
            "paste_bbox": result["paste_bbox"],
            "object_size": result["object_size"],
            "timestamp": result["timestamp"],
        }