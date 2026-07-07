import asyncio
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep

class PolygonMasker:
    """
    Generate a binary mask from a polygon defined by a list of (x, y) points.

    Use this when the UI expects the mask to exactly repeat what the user
    drew with points, rather than a semantic "guess" from SAM
    point-prompt.

    Points must be ordered along the contour (clockwise or counterclockwise —
    it doesn't matter as long as they are sequential, in the order the user placed them).
    """

    async def generate_mask(
        self,
        image_size: Tuple[int, int],
        points: List[Tuple[int, int]],
        smooth: bool = True,
        smoothing_factor: float = 0.0,
        num_smooth_points: int = 200,
        feather_px: int = 0,
    ) -> bytes:
        """
        Args:
            image_size:        (W, H) target image size
            points:             ordered (x, y) points along the contour
            smooth:             whether to fit a closed spline through the points for
                                 a smooth curve instead of a polyline (recommended
                                 when there are few points, e.g., 15-20 clicks on hair)
            smoothing_factor:   smoothing of the spline (0 = passes exactly through
                                 the points, more = smoother curve)
            num_smooth_points:  resolution of the smoothed contour
            feather_px:         feathering of the mask edges in N px (0 = sharp edge)

        Returns:
            PNG bytes, mode 'L', 255 = inside the polygon, 0 = outside
        """
        return await asyncio.to_thread(
            self._generate_mask_sync,
            image_size,
            points,
            smooth,
            smoothing_factor,
            num_smooth_points,
            feather_px,
        )

    def _generate_mask_sync(
        self,
        image_size: Tuple[int, int],
        points: List[Tuple[int, int]],
        smooth: bool,
        smoothing_factor: float,
        num_smooth_points: int,
        feather_px: int,
    ) -> bytes:
        if len(points) < 3:
            raise ValueError("Need at least 3 points to form a polygon")

        W, H = image_size
        pts = np.array(points, dtype=np.float32)

        # clip points to image bounds to avoid cv2.fillPoly errors
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        if smooth and len(pts) >= 4:
            pts_closed = np.vstack([pts, pts[0]])
            # per=True makes the spline periodic (closed)
            tck, _ = splprep(
                [pts_closed[:, 0], pts_closed[:, 1]],
                s=smoothing_factor,
                per=True,
            )
            u_new = np.linspace(0, 1, num_smooth_points)
            x_new, y_new = splev(u_new, tck)
            poly = np.stack([x_new, y_new], axis=1).astype(np.int32)
        else:
            poly = pts.astype(np.int32)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        if feather_px > 0:
            k = feather_px * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        buf = BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        return buf.getvalue()


_polygon_masker_instance: Optional[PolygonMasker] = None


def get_polygon_masker() -> PolygonMasker:
    global _polygon_masker_instance
    if _polygon_masker_instance is None:
        _polygon_masker_instance = PolygonMasker()
    return _polygon_masker_instance