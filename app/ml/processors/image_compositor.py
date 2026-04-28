from typing import Dict
from io import BytesIO

import numpy as np
import cv2
from PIL import Image


class ImageCompositor:
    """
    Image compositing utilities for object replacement.

    Responsibilities:
        1. Alpha-based object pasting
        2. Soft edge blending (localized, not full bbox blur)
        3. Final composition of replacement over cleaned background

    This module is intentionally separate from:
        - Color matching (color_matcher.py)
        - Inpainting (LaMa)

    Because compositing is a different stage of the pipeline.
    """

    def compose(
        self,
        clean_bg_bytes: bytes,
        replacement_rgba_bytes: bytes,
        bbox: Dict[str, int],
        edge_softness: int = 0
    ) -> bytes:
        """
        Paste replacement using pure alpha compositing.

        Args:
            clean_bg_bytes: Background after LaMa
            replacement_rgba_bytes: Object with alpha (RGBA)
            bbox: {'x1','y1','x2','y2'}

        Returns:
            JPEG bytes
        """

        def sync():
            clean_bg = Image.open(BytesIO(clean_bg_bytes)).convert('RGB')
            replacement = Image.open(BytesIO(replacement_rgba_bytes)).convert('RGBA')

            x1, y1 = bbox['x1'], bbox['y1']

            alpha = replacement.split()[3]
            bbox_alpha = alpha.getbbox()

            if bbox_alpha:
                replacement = replacement.crop(bbox_alpha)
                x1 += bbox_alpha[0]
                y1 += bbox_alpha[1]

            clean_bg.paste(replacement, (x1, y1), replacement)

            buf = BytesIO()
            clean_bg.save(buf, format='JPEG', quality=95)
            return buf.getvalue()


        return sync()

    def _soft_edge_blend(
        self,
        clean_bg: np.ndarray,
        result: np.ndarray,
        bbox: Dict[str, int],
        edge: int = 4
    ) -> np.ndarray:
        """
        Apply localized edge blending around bbox borders.

        Unlike heavy edge blending:
            - Only affects a small boundary region
            - Does NOT blur the entire bbox
            - Prevents visible rectangular artifacts

        Args:
            clean_bg: Original background (before paste)
            result: Image after paste
            bbox: Placement bbox
            edge: Edge softness radius

        Returns:
            Blended image array
        """
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        roi_clean = clean_bg[y1:y2, x1:x2]
        roi_result = result[y1:y2, x1:x2]

        h, w = roi_clean.shape[:2]
         
        mask = np.ones((h, w), dtype=np.float32)
        border = edge
        mask[:border, :] *= np.linspace(0, 1, border)[:, None]
        mask[-border:, :] *= np.linspace(1, 0, border)[:, None]
        mask[:, :border] *= np.linspace(0, 1, border)[None, :]
        mask[:, -border:] *= np.linspace(1, 0, border)[None, :]

        mask = mask[:, :, None]
        # Small blur → only edges affected
        blended_roi = roi_result * mask + roi_clean * (1 - mask)
        result[y1:y2, x1:x2] = blended_roi
        
        return result


_compositor_instance = None


def get_compositor() -> ImageCompositor:
    """
    Singleton getter for ImageCompositor.
    """
    global _compositor_instance
    if _compositor_instance is None:
        _compositor_instance = ImageCompositor()
    return _compositor_instance