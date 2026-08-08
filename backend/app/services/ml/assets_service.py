import uuid
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image as PILImage

from app.services.ml.base_ml_service import BaseMLService
from app.db.models.assets import Asset
from app.db.schemas.assets import AssetCreate
from app.core.logging import get_logger, log_execution

logger = get_logger(__name__)

THUMBNAIL_MAX_SIZE = 256


class AssetService(BaseMLService):
    """
    Handles extracted object assets: extract, paste, and the asset library.
    Postgres is now the source of truth for metadata;
    S3 holds the actual PNG bytes. No Redis involved
    anymore — assets are permanent until deleted or evicted by the
    per-user cap in AssetRepository.
    """

    def _make_thumbnail(self, extracted_bytes: bytes, max_size: int = THUMBNAIL_MAX_SIZE) -> bytes:
        img = PILImage.open(BytesIO(extracted_bytes)).convert("RGBA")
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def _delete_asset_files(self, asset: Asset) -> None:
        """Best-effort cleanup of S3 objects — logs and swallows failures so
        a dangling S3 file never blocks a DB delete the user asked for."""
        try:
            await self.s3.delete(asset.storage_path)
            if asset.thumbnail_path:
                await self.s3.delete(asset.thumbnail_path)
        except Exception as e:
            logger.warning(
                "asset_s3_delete_failed",
                asset_id=asset.public_id,
                storage_path=asset.storage_path,
                exc_info=e,
            )

    async def extract_object(
        self,
        image_id: int,
        mask_id: int,
        user_id: int,
        padding_pixels: int = 8,
        label: Optional[str] = None,
        persist_to_s3: bool = False,  # kept for API compat, no longer optional in practice
    ) -> Dict:
        """
        Extract a MobileSAM-segmented object from an image as an RGBA PNG
        cutout, upload it (+ thumbnail) to S3, and persist metadata as an
        Asset row.
        """
        with log_execution(
            "service_extract_object",
            logger=logger,
            image_id=image_id,
            mask_id=mask_id,
        ):
            image, version = await self._get_current_version_authorized(image_id, user_id)
            segment = await self._get_segment_or_raise(
                content_id=version.content_id, mask_id=mask_id, image_id=image_id
            )

            image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

            result = await self.pipeline.sam_extract_object(
                image_bytes=image_bytes,
                mask_bytes=segment["mask_bytes"],
                bbox=segment["bbox"],
                padding_pixels=padding_pixels,
            )

            extracted_bytes = result["extracted_bytes"]
            width, height = result["object_size"]
            thumbnail_bytes = self._make_thumbnail(extracted_bytes)

            key_prefix = f"assets/{user_id}/{image_id}/{uuid.uuid4().hex}"
            storage_path = f"{key_prefix}.png"
            thumbnail_path = f"{key_prefix}_thumb.png"

            await self.s3.upload_bytes(
                data=extracted_bytes, path=storage_path, content_type="image/png"
            )
            await self.s3.upload_bytes(
                data=thumbnail_bytes, path=thumbnail_path, content_type="image/png"
            )

            asset_create = AssetCreate(
                user_id=user_id,
                storage_path=storage_path,
                thumbnail_path=thumbnail_path,
                content_type="image/png",
                file_size=len(extracted_bytes),
                width=width,
                height=height,
                area_pixels=result["area_pixels"],
                label=label,
                source_image_version_id=version.id,
                source_segmentation_mask_id=segment["id"],
            )
            asset = await self.assets_repo.create(Asset(**asset_create.model_dump()))

            # Evict oldest assets over the per-user cap — repo only tells us
            # what to remove, service owns the S3 side effect.
            evicted = await self.assets_repo.get_overflow(user_id)
            for old_asset in evicted:
                await self._delete_asset_files(old_asset)
            if evicted:
                await self.assets_repo.delete_many(evicted)

            logger.info(
                "asset_extracted",
                image_id=image_id,
                mask_id=mask_id,
                asset_id=asset.public_id,
                user_id=user_id,
                evicted_count=len(evicted),
            )

        return {
            "asset_id": asset.public_id,
            "storage_path": asset.storage_path,
            "thumbnail_path": asset.thumbnail_path,
            "object_size": result["object_size"],
            "area_pixels": result["area_pixels"],
            "cropped_bbox": result["cropped_bbox"],
            "timestamp": result["timestamp"],
        }

    async def list_assets(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Asset]:
        assets = await self.assets_repo.list_by_user(user_id, limit=limit, offset=offset)
        logger.debug("assets_listed", user_id=user_id, count=len(assets))
        return assets

    async def get_asset_thumbnail(self, user_id: int, asset_id: str) -> Optional[bytes]:
        asset = await self.assets_repo.get_by_public_id(user_id, asset_id)
        if not asset or not asset.thumbnail_path:
            return None
        return await self.s3.download(asset.thumbnail_path)

    async def get_asset_image(self, user_id: int, asset_id: str) -> Optional[bytes]:
        asset = await self.assets_repo.get_by_public_id(user_id, asset_id)
        if not asset:
            return None
        return await self.s3.download(asset.storage_path)

    async def rename_asset(self, user_id: int, asset_id: str, label: str) -> Asset:
        asset = await self.assets_repo.get_by_public_id(user_id, asset_id)
        if not asset:
            logger.warning("asset_not_found", user_id=user_id, asset_id=asset_id)
            raise ValueError("Asset not found")
        asset = await self.assets_repo.rename(asset, label)
        logger.info("asset_renamed", user_id=user_id, asset_id=asset_id, label=label)
        return asset

    async def delete_asset(self, user_id: int, asset_id: str) -> None:
        asset = await self.assets_repo.get_by_public_id(user_id, asset_id)
        if not asset:
            logger.warning("asset_not_found", user_id=user_id, asset_id=asset_id)
            raise ValueError("Asset not found")
        await self._delete_asset_files(asset)
        await self.assets_repo.delete(asset)
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
        Paste a previously extracted object (from the asset library, by
        public_id, or a raw S3 URL) onto the current working state of an
        image.
        """
        if not asset_id and not extracted_url:
            logger.warning("paste_missing_source", image_id=image_id, user_id=user_id)
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
                asset = await self.assets_repo.get_by_public_id(user_id, asset_id)
                if not asset:
                    logger.warning("asset_not_found", user_id=user_id, asset_id=asset_id)
                    raise ValueError("Asset not found")
                extracted_bytes = await self.s3.download(asset.storage_path)
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