import time
from typing import Dict, Literal, Optional, Tuple
from datetime import datetime

from app.ml.modes.sam_lama_mode import SAMLamaMode
from app.ml.experiment_tracker import ExperimentTracker
from app.ml.pipeline.validator import Validator


class ExtractionMixin:
    sam_lama_mode: SAMLamaMode
    tracker: ExperimentTracker
    validator: Validator

    async def sam_extract_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        padding_pixels: int = 8,
        output_format: str = "PNG",
        track_metrics: bool = True,
    ) -> Dict:
        """
        Extract an object as an RGBA image using a SAM mask.

        Pipeline: crop by bbox+padding → mask as alpha → RGBA PNG

        Args:
            image_bytes:     Input image bytes
            mask_bytes:      Binary mask from SAM (PNG, L mode)
            bbox:            Segment bbox {'x1','y1','x2','y2'}
            padding_pixels:  Padding around bbox when cropping (default: 8)
            output_format:   Output format: 'PNG' or 'WEBP' (default: 'PNG')
            track_metrics:   Track metrics to MLflow (default: True)

        Returns:
            Dict:
                - extracted_bytes:  bytes — RGBA image of extracted object
                - cropped_bbox:     Dict  — actual bbox after padding
                - original_size:    Tuple[int, int] — (W, H) of original image
                - object_size:      Tuple[int, int] — (W, H) of extracted object
                - area_pixels:      int   — number of non-transparent pixels
                - timestamp:        str
        """
        start_time = time.time()

        try:
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_mask_bytes(mask_bytes)
            self.validator.validate_bbox(bbox)

            if output_format not in ("PNG", "WEBP"):
                raise ValueError("output_format must be 'PNG' or 'WEBP'")

            result = await self.sam_lama_mode.extract_object(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                bbox=bbox,
                padding_pixels=padding_pixels,
                output_format=output_format,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics:
                self.tracker.log_metrics({
                    "operation": "sam_extract_object",
                    "processing_time": time.time() - start_time,
                    "area_pixels": result.get("area_pixels", 0),
                    "padding_pixels": padding_pixels,
                })

            return result

        except Exception as e:
            print(f"SAM object extraction failed: {e}")
            raise

    async def sam_paste_extracted_object(
        self,
        image_bytes: bytes,
        extracted_bytes: bytes,
        target_bbox: Dict[str, int],
        scale: float = 1.0,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: Literal["mean_std", "histogram", "color_transfer"] = "color_transfer",
        track_metrics: bool = True,
    ) -> Dict:
        """
        Paste a previously extracted RGBA object into a target image.

        Pipeline:
            scale to target_bbox → center → alpha-composite →
            ColorMatch (optional) → EdgeBlend (optional)

        Args:
            image_bytes:         Target image bytes
            extracted_bytes:     RGBA PNG of the extracted object
            target_bbox:         Destination bbox {'x1','y1','x2','y2'}
            scale:               Scale factor relative to bbox (0.1–3.0, default: 1.0)
            use_color_matching:  Apply color correction (default: True)
            use_edge_blending:   Smooth alpha boundaries (default: True)
            color_match_method:  'mean_std' | 'histogram' | 'color_transfer'
            track_metrics:       Track metrics to MLflow (default: True)

        Returns:
            Dict:
                - result_bytes:  bytes — JPEG
                - paste_bbox:    Dict  — actual bbox after scaling and centering
                - object_size:   Tuple[int, int] — (W, H) after scaling
                - timestamp:     str
        """
        start_time = time.time()

        try:
            self.validator.validate_image_bytes(image_bytes)
            self.validator.validate_image_bytes(extracted_bytes)
            self.validator.validate_bbox(target_bbox)

            if not (0.1 <= scale <= 3.0):
                raise ValueError("scale must be between 0.1 and 3.0")

            result = await self.sam_lama_mode.paste_extracted_object(
                image_bytes=image_bytes,
                extracted_bytes=extracted_bytes,
                target_bbox=target_bbox,
                scale=scale,
                use_color_matching=use_color_matching,
                use_edge_blending=use_edge_blending,
                color_match_method=color_match_method,
            )

            result["timestamp"] = datetime.now().isoformat()

            if track_metrics:
                self.tracker.log_metrics({
                    "operation": "sam_paste_extracted_object",
                    "processing_time": time.time() - start_time,
                    "scale": scale,
                    "color_matching": use_color_matching,
                    "edge_blending": use_edge_blending,
                })

            return result

        except Exception as e:
            print(f"SAM paste extracted object failed: {e}")
            raise