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
        2. Alpha edge cleanup (fringe/halo removal)
        3. Final composition of replacement over cleaned background
    """

    def compose(
        self,
        clean_bg_bytes: bytes,
        replacement_rgba_bytes: bytes,
        bbox: Dict[str, int],
        edge_softness: int = 0
    ) -> bytes:
        """
        Paste replacement using pure alpha compositing with fringe cleanup.

        Args:
            clean_bg_bytes: Background after LaMa
            replacement_rgba_bytes: Object with alpha (RGBA)
            bbox: {'x1','y1','x2','y2'}
            edge_softness: unused, kept for API compat

        Returns:
            JPEG bytes
        """

        def sync():
            clean_bg = Image.open(BytesIO(clean_bg_bytes)).convert('RGB')
            replacement = Image.open(BytesIO(replacement_rgba_bytes)).convert('RGBA')

            replacement = self._clean_alpha_fringe(replacement)

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

    def _clean_alpha_fringe(self, img_rgba: Image.Image, threshold: int = 10) -> Image.Image:
        """
        Remove color fringing/halo from RGBA image edges.

        The problem: rembg leaves semi-transparent edge pixels that carry
        the original background color (e.g. blue/grey sky). When composited,
        these bleed into the target background causing a visible halo.

        Fix pipeline:
            1. Erode alpha slightly to cut off contaminated edge pixels
            2. For remaining semi-transparent pixels, replace their RGB
               with the nearest fully-opaque neighbor color (color bleeding fix)
            3. Re-apply cleaned alpha

        Args:
            img_rgba: RGBA PIL image from background remover
            threshold: Alpha threshold below which pixels are treated as background

        Returns:
            RGBA PIL image with cleaned edges
        """
        arr = np.array(img_rgba, dtype=np.uint8)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        #Erode alpha mask by 1px to cut outermost fringe pixels
        kernel = np.ones((3, 3), np.uint8)
        alpha_eroded = cv2.erode(alpha, kernel, iterations=1)

        #replace RGB with nearest fully-opaque pixel color.
        solid_mask = (alpha_eroded > 200).astype(np.uint8)  # fully opaque region

        if solid_mask.sum() > 0:
            # Use inpainting to fill fringe pixels with neighbor solid colors
            fringe_mask = ((alpha_eroded > threshold) & (solid_mask == 0)).astype(np.uint8) * 255
            if fringe_mask.sum() > 0:
                rgb_fixed = cv2.inpaint(rgb, fringe_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            else:
                rgb_fixed = rgb
        else:
            rgb_fixed = rgb

        #Rebuild RGBA with eroded alpha and fixed RGB
        result = np.dstack([rgb_fixed, alpha_eroded])
        return Image.fromarray(result, mode='RGBA')

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