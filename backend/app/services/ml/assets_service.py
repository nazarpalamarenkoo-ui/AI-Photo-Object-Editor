from datetime import datetime
from typing import Dict

from app.services.ml.base_ml_service import BaseMLService


class AssetService(BaseMLService):
    """
    Handles extracted object assets: extract and paste.

    Future extensions (list_assets, delete_asset, rename_asset, categories)
    belong here — this service owns the extracted-object lifecycle.

    Workflow:
        segment_objects (SegmentationService)
            -> extract_object        (SAM mask -> RGBA PNG -> S3)
            -> paste_extracted_object (RGBA PNG -> composite -> result)
    """

    async def extract_object(
        self,
        image_id: int,
        mask_id: int,
        user_id: int,
        padding_pixels: int = 8,
    ) -> Dict:
        """
        Extract object by SAM mask_id as RGBA PNG and store in S3.

        Args:
            image_id:       ID of image to process
            mask_id:        Segment mask_id from segment_objects
            user_id:        ID of requesting user
            padding_pixels: Padding around bbox crop (default: 8)

        Returns:
            Dict:
                - extracted_url: str — S3 URI (PNG)
                - presigned_url: str — temporary download URL
                - object_size:   Tuple[int, int] — (W, H)
                - area_pixels:   int
                - cropped_bbox:  Dict
                - timestamp:     str ISO

        Raises:
            ValueError: If image not found, unauthorized, or segment not cached.
        """
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

        extract_path = (
            f"extracted/{user_id}/{image_id}/"
            f"mask_{mask_id}_{int(datetime.utcnow().timestamp())}.png"
        )
        extracted_url, presigned_url = await self._upload_result(
            result["extracted_bytes"], extract_path, content_type="image/png"
        )

        return {
            "extracted_url": extracted_url,
            "presigned_url": presigned_url,
            "object_size": result["object_size"],
            "area_pixels": result["area_pixels"],
            "cropped_bbox": result["cropped_bbox"],
            "timestamp": result["timestamp"],
        }

    async def paste_extracted_object(
        self,
        image_id: int,
        user_id: int,
        extracted_url: str,
        target_bbox: Dict[str, int],
        scale: float = 1.0,
        use_color_matching: bool = True,
        use_edge_blending: bool = True,
        color_match_method: str = "color_transfer",
    ) -> Dict:
        """
        Paste a previously extracted RGBA object onto the current image.

        Args:
            image_id:           ID of target image
            user_id:            ID of requesting user
            extracted_url:      S3 URI of extracted RGBA PNG
            target_bbox:        {'x1','y1','x2','y2'} — placement bbox
            scale:              Scale relative to bbox (0.1-3.0, default: 1.0)
            use_color_matching: Apply color matching (default: True)
            use_edge_blending:  Apply edge blending (default: True)
            color_match_method: Color match method (default: 'color_transfer')

        Returns:
            Dict:
                - result_url:    str — S3 URI
                - presigned_url: str — temporary download URL
                - paste_bbox:    Dict — actual bbox after scale+center
                - object_size:   Tuple[int, int]
                - timestamp:     str ISO

        Raises:
            ValueError: If image not found, unauthorized, or extracted download fails.
        """
        image = await self._get_image_authorized(image_id, user_id)
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)

        try:
            extracted_bytes = await self.s3.download(extracted_url)
        except Exception as e:
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