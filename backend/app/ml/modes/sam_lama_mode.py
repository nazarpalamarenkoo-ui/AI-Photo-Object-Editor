import asyncio
from typing import Optional, List, Dict, Tuple
from io import BytesIO
import numpy as np
from PIL import Image

from app.ml.segmentor import SAM2Segmentor, get_segmentor
from app.ml.inpainter import LaMaInpainter, InpaintMode, get_inpainter
from app.ml.processors.edge_blender import EdgeBlender, get_edge_blender
from app.ml.processors.color_matcher import ColorMatcher, get_color_matcher
from app.ml.processors.background_remover import BackgroundRemover, get_background_remover
from app.ml.processors.image_compositor import get_compositor
from app.ml.processors.polygon_mask import PolygonMasker, get_polygon_masker


class SAMLamaMode:
    """
    SAM 2.1 + LaMa combined processing mode.

    Methods:
        1. segment_objects          — auto-segmentation (without prompts)
        2. segment_with_prompt      — point/bbox prompt segmentation
        3. remove_object            — SAM mask → LaMa remove → EdgeBlend
        4. replace_object           — SAM mask → LaMa remove → composite → ColorMatch
        5. extract_object           — SAM mask → RGBA PNG crop
        6. paste_extracted_object   — RGBA PNG → scale → composite → ColorMatch → EdgeBlend
    """

    def __init__(
        self,
        segmentor: Optional[SAM2Segmentor] = None,
        inpainter: Optional[LaMaInpainter] = None,
        edge_blender: Optional[EdgeBlender] = None,
        color_matcher: Optional[ColorMatcher] = None,
        background_remover: Optional[BackgroundRemover] = None,
        polygon_masker: Optional[PolygonMasker] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.segmentor = segmentor or get_segmentor(device=device)
        self.inpainter = inpainter or get_inpainter(device=device)
        self.edge_blender = edge_blender or get_edge_blender()
        self.color_matcher = color_matcher or get_color_matcher()
        self.background_remover = background_remover or get_background_remover(rembg_available=True)
        self.compositor = get_compositor()
        self.polygon_masker = polygon_masker or get_polygon_masker()
        print(f"SAMMode initialized (device: {device})")


    async def segment_objects(
        self,
        image_bytes: bytes,
        min_area: int = 500,
        max_segments: int = 50,
    ) -> Dict:
        """
        Auto-segmentation of the entire image without prompts.

        Args:
            image_bytes:    Input image
            min_area:       Minimum segment area in pixels (noise filter)
            max_segments:   Maximum number of segments in the response

        Returns:
            Dict:
                - segments:   List[Dict] — bbox, area, mask_bytes, mask_id, bbox_id
                - metrics:    Dict
                - image_size: (W, H)
        """
        result = await self.segmentor.segment_auto(image_bytes)

        filtered = [
            s for s in result["segments"]
            if s["area"] >= min_area
        ][:max_segments]

        for idx, seg in enumerate(filtered):
            seg["bbox_id"] = idx

        img = Image.open(BytesIO(image_bytes))
        return {
            "segments": filtered,
            "metrics": result["metrics"],
            "image_size": img.size,
        }

    async def segment_with_prompt(
        self,
        image_bytes: bytes,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        bbox: Optional[Dict[str, int]] = None,
        multimask_output: Optional[bool] = None
    ) -> Dict:
        """
        Prompt-based segmentation using points or a bounding box.

        Args:
            image_bytes:    Input image
            point_coords:   List of points [(x, y), ...]
            point_labels:   1 = foreground, 0 = background for each point
            bbox:           {'x1', 'y1', 'x2', 'y2'} used as a prompt

        Returns:
            Dict:
                - segments:   List[Dict] — sorted by stability_score
                - metrics:    Dict
                - image_size: tuple[int, int] — (W, H)
        """
        result = await self.segmentor.segment_with_prompt(
            image_bytes,
            point_coords=point_coords,
            point_labels=point_labels,
            bbox=bbox,
            multimask_output=multimask_output
        )

        for idx, seg in enumerate(result["segments"]):
            seg["bbox_id"] = idx

        img = Image.open(BytesIO(image_bytes))
        return {
            "segments": result["segments"],
            "metrics": result["metrics"],
            "image_size": img.size,
        }

    async def segment_by_polygon(
        self,
        image_bytes: bytes,
        points: List[Tuple[int, int]],
        smooth: bool = True,
        smoothing_factor: float = 0.0,
        feather_px: int = 0,
    ) -> Dict:
        """
        Exact segmentation by polygon points (lasso), without SAM2.

        Returns:
            Dict: segments (1 element: bbox, area, mask_bytes, mask_id=0,
                bbox_id=0, source='polygon'), metrics, image_size
        """
        img = Image.open(BytesIO(image_bytes))
        W, H = img.size

        mask_bytes = await self.polygon_masker.generate_mask(
            image_size=(W, H),
            points=points,
            smooth=smooth,
            smoothing_factor=smoothing_factor,
            feather_px=feather_px,
        )

        mask_arr = np.array(Image.open(BytesIO(mask_bytes)))
        ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            raise ValueError("Polygon produced an empty mask — check point coordinates")

        bbox = {"x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()), "y2": int(ys.max())}
        area = int((mask_arr > 0).sum())

        segment = {
            "mask_id": 0,
            "bbox_id": 0,
            "bbox": bbox,
            "area": area,
            "source": "polygon",
            "mask_bytes": mask_bytes,
        }

        return {
            "segments": [segment],
            "metrics": {"num_segments": 1, "total_area_px": area},
            "image_size": (W, H),
        }
    
    async def remove_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        use_edge_blending: bool = False,
        expand_mask_pixels: int = 12,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
    ) -> Dict:
        """
        Object removal using a SAM mask and LaMa inpainting.

        Pipeline:
            1. Dilate the mask (expand_mask_pixels)
            2. Use LaMa REMOVE to generate the background
            3. Apply Edge Blending to smooth transitions (optional)

        Args:
            image_bytes:        Input image
            mask_bytes:         Binary mask from SAM (PNG, L mode)
            use_edge_blending:  Apply edge blending (default: True)
            expand_mask_pixels: Mask expansion in pixels (default: 12)
            ldm_steps:          Number of LaMa inference steps (default: 25)
            ldm_sampler:        LaMa sampler (default: 'plms')
            hd_strategy:        High-resolution processing strategy (default: 'CROP')

        Returns:
            Dict:
                - result_bytes: bytes — resulting JPEG image
                - metrics:      Dict
        """
        if expand_mask_pixels > 0:
            mask_bytes = await self._dilate_mask(mask_bytes, expand_mask_pixels)

        inpaint_result = await self.inpainter.inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            mode=InpaintMode.REMOVE,
            track_metrics=True,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
        )

        result_bytes = inpaint_result["result_bytes"]
        result_bytes = await _normalize_size(result_bytes, image_bytes)

        if use_edge_blending:
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=expand_mask_pixels,
            )

        return {
            "result_bytes": result_bytes,
            "metrics": inpaint_result["metrics"],
        }

    async def replace_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        replacement_image_bytes: bytes,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = "color_transfer",
        expand_mask_pixels: int = 8,
        ldm_steps: int = 25,
        ldm_sampler: str = "plms",
        hd_strategy: str = "CROP",
        replacement_is_cutout: bool = False,
    ) -> Dict:
        """
        Object replacement using LaMa + compositing.

        Pipeline:
            1. Prepare the replacement as an RGBA cutout sized to the bbox —
               either by running background removal (rembg) on a plain
               photo, or, when the replacement is already a pre-cut RGBA
               asset from the asset library, by simply resizing it
               (see replacement_is_cutout).
            2. Dilate the SAM mask
            3. Use LaMa REMOVE to generate a clean background
            4. Composite the replacement image using its alpha channel
            5. Apply Color Matching (optional)
            6. Apply Edge Blending (optional)

        Args:
            image_bytes:               Input image
            mask_bytes:                Binary mask from SAM
            bbox:                      Segment bounding box {'x1', 'y1', 'x2', 'y2'}
            replacement_image_bytes:   Replacement image bytes. A plain photo
                                       when replacement_is_cutout
                                       is False, or an already-transparent
                                       RGBA PNG when replacement_is_cutout is True.
            use_color_matching:        Apply color correction (default: True)
            use_edge_blending:         Apply edge blending (default: False)
            color_match_method:        Color matching method (default: 'color_transfer')
            expand_mask_pixels:        Mask expansion in pixels (default: 8)
            ldm_steps:                 Number of LaMa inference steps (default: 25)
            ldm_sampler:               LaMa sampler (default: 'plms')
            hd_strategy:               High-resolution processing strategy (default: 'CROP')
            replacement_is_cutout:     If True, skip rembg background removal
                                       and treat replacement_image_bytes as an
                                       already-transparent RGBA cutout that
                                       only needs resizing to the bbox 
                                       (default: False)

        Returns:
            Dict:
                - result_bytes: bytes — resulting JPEG image
                - metrics:      Dict
        """
        bbox_w = bbox["x2"] - bbox["x1"]
        bbox_h = bbox["y2"] - bbox["y1"]

        if replacement_is_cutout:
            replacement_rgba = await asyncio.to_thread(
                self._resize_rgba_to_bbox, replacement_image_bytes, (bbox_w, bbox_h)
            )
        else:
            replacement_rgba = await self.background_remover.remove_and_resize(
                replacement_image_bytes, (bbox_w, bbox_h)
            )

        if expand_mask_pixels > 0:
            mask_bytes = await self._dilate_mask(mask_bytes, expand_mask_pixels)

        inpaint_result = await self.inpainter.inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            mode=InpaintMode.REMOVE,
            track_metrics=True,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
        )

        clean_bytes = await _normalize_size(inpaint_result["result_bytes"], image_bytes)

        result_bytes = self.compositor.compose(
            clean_bg_bytes=clean_bytes,
            replacement_rgba_bytes=replacement_rgba,
            bbox=bbox,
            edge_softness=0,
        )

        if use_color_matching:
            result_bytes = self.color_matcher.match_against_original(
                result_bytes=result_bytes,
                original_image_bytes=image_bytes,
                bbox=bbox,
                method=color_match_method, # type: ignore
            )

        if use_edge_blending:
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=expand_mask_pixels,
            )

        return {
            "result_bytes": result_bytes,
            "metrics": inpaint_result["metrics"],
        }


    async def extract_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        padding_pixels: int = 8,
        output_format: str = "PNG",
    ) -> Dict:
        """
        Extracts an object using a SAM mask and returns an RGBA image with a transparent background.

        Pipeline:
            1. Crop by bounding box with padding
            2. Convert the mask to an alpha channel
            3. Return RGBA image bytes

        Args:
            image_bytes:     Original image
            mask_bytes:      Binary mask from SAM (PNG, L mode)
            bbox:            {'x1', 'y1', 'x2', 'y2'} — segment bounding box
            padding_pixels:  Padding around the bounding box during cropping (default: 8)
            output_format:   'PNG' or 'WEBP' (default: 'PNG')

        Returns:
            Dict:
                - extracted_bytes: bytes  — RGBA image of the extracted object
                - cropped_bbox:    Dict   — actual bounding box after padding
                - original_size:   tuple  — (W, H) of the original image
                - object_size:     tuple  — (W, H) of the extracted object
                - area_pixels:     int    — number of non-transparent pixels
        """
        return await asyncio.to_thread(
            self._extract_object_sync,
            image_bytes, mask_bytes, bbox, padding_pixels, output_format
        )

    def _extract_object_sync(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        bbox: Dict[str, int],
        padding_pixels: int,
        output_format: str,
    ) -> Dict:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        mask = Image.open(BytesIO(mask_bytes)).convert("L")

        W, H = img.size

        x1 = max(0, bbox["x1"] - padding_pixels)
        y1 = max(0, bbox["y1"] - padding_pixels)
        x2 = min(W, bbox["x2"] + padding_pixels)
        y2 = min(H, bbox["y2"] + padding_pixels)

        img_crop = img.crop((x1, y1, x2, y2))
        mask_crop = mask.crop((x1, y1, x2, y2))

        r, g, b, _ = img_crop.split()
        result_img = Image.merge("RGBA", (r, g, b, mask_crop))

        alpha_array = np.array(mask_crop)
        area_pixels = int((alpha_array > 128).sum())

        buf = BytesIO()
        result_img.save(buf, format=output_format)

        return {
            "extracted_bytes": buf.getvalue(),
            "cropped_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "original_size": (W, H),
            "object_size": result_img.size,
            "area_pixels": area_pixels,
        }


    async def paste_extracted_object(
        self,
        image_bytes: bytes,
        extracted_bytes: bytes,
        target_bbox: Dict[str, int],
        scale: float = 1.0,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = "color_transfer",
    ) -> Dict:
        """
        Inserts a cut-out RGBA object into an image.

        Pipeline:
            1. Scale the extracted object relative to the target_bbox
            2. Center it within the target_bbox
            3. Alpha-composite it onto the original image
            4. Apply Color Matching (optional)
            5. Apply Edge Blending along alpha-mask boundaries (optional)

        Args:
            image_bytes:         Target image
            extracted_bytes:     RGBA PNG of the extracted object (from S3)
            target_bbox:         {'x1', 'y1', 'x2', 'y2'} — user-defined bounding box
            scale:               Scale factor relative to the bounding box (0.1–3.0, default: 1.0)
            use_color_matching:  Apply color correction (default: True)
            use_edge_blending:   Apply edge blending along alpha boundaries (default: True)
            color_match_method:  Color matching method (default: 'color_transfer')

        Returns:
            Dict:
                - result_bytes: bytes — resulting JPEG image
                - paste_bbox:   Dict  — actual bounding box after scaling and centering
                - object_size:  tuple — (W, H) after scaling
        """
        result = await asyncio.to_thread(
            self._paste_extracted_sync,
            image_bytes, extracted_bytes, target_bbox, scale
        )

        result_bytes = result["result_bytes"]

        if use_color_matching:
            result_bytes = self.color_matcher.match_against_original(
                result_bytes=result_bytes,
                original_image_bytes=image_bytes,
                bbox=result["paste_bbox"],
                method=color_match_method, # type: ignore
            )

        if use_edge_blending:
            mask_bytes = await self._alpha_to_mask(
                extracted_bytes=extracted_bytes,
                paste_bbox=result["paste_bbox"],
                object_size=result["object_size"],
                canvas_size=Image.open(BytesIO(image_bytes)).size,
            )
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=6,
            )

        result["result_bytes"] = result_bytes
        return result

    def _paste_extracted_sync(
        self,
        image_bytes: bytes,
        extracted_bytes: bytes,
        target_bbox: Dict[str, int],
        scale: float,
    ) -> Dict:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        extracted = Image.open(BytesIO(extracted_bytes)).convert("RGBA")

        bbox_w = target_bbox["x2"] - target_bbox["x1"]
        bbox_h = target_bbox["y2"] - target_bbox["y1"]

        orig_w, orig_h = extracted.size
        fit_ratio = min(bbox_w / orig_w, bbox_h / orig_h)
        final_ratio = fit_ratio * scale

        new_w = max(1, int(orig_w * final_ratio))
        new_h = max(1, int(orig_h * final_ratio))
        extracted_scaled = extracted.resize((new_w, new_h), Image.Resampling.LANCZOS)

        cx = target_bbox["x1"] + bbox_w // 2
        cy = target_bbox["y1"] + bbox_h // 2
        paste_x = cx - new_w // 2
        paste_y = cy - new_h // 2

        img_w, img_h = img.size
        paste_x = max(0, min(paste_x, img_w - new_w))
        paste_y = max(0, min(paste_y, img_h - new_h))

        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
        canvas.paste(extracted_scaled, (paste_x, paste_y))
        result_rgba = Image.alpha_composite(img, canvas)

        result_rgb = result_rgba.convert("RGB")
        buf = BytesIO()
        result_rgb.save(buf, format="JPEG", quality=95)

        paste_bbox = {
            "x1": paste_x,
            "y1": paste_y,
            "x2": paste_x + new_w,
            "y2": paste_y + new_h,
        }

        return {
            "result_bytes": buf.getvalue(),
            "paste_bbox": paste_bbox,
            "object_size": (new_w, new_h),
        }

    def _resize_rgba_to_bbox(self, image_bytes: bytes, size: Tuple[int, int]) -> bytes:
        """
        Resize an already-transparent RGBA cutout (e.g. a saved asset) to
        exactly fit the target bbox, without touching its alpha channel —
        used instead of background removal when replacement_is_cutout=True.

        Args:
            image_bytes: RGBA PNG bytes of the cutout.
            size:        Target (width, height) matching the bbox.

        Returns:
            RGBA PNG bytes resized to `size`.
        """
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def _dilate_mask(self, mask_bytes: bytes, pixels: int) -> bytes:
        """Expands the mask by N pixels using morphological dilation."""
        def sync():
            import cv2
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
            kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            buf = BytesIO()
            Image.fromarray(dilated, mode="L").save(buf, format="PNG")
            return buf.getvalue()

        return await asyncio.to_thread(sync)

    async def _alpha_to_mask(
        self,
        extracted_bytes: bytes,
        paste_bbox: Dict[str, int],
        object_size: tuple,
        canvas_size: tuple,
    ) -> bytes:
        """Generates a binary mask from alpha channel for EdgeBlender."""
        def sync():
            extracted = Image.open(BytesIO(extracted_bytes)).convert("RGBA")
            scaled = extracted.resize(object_size, Image.Resampling.LANCZOS)
            alpha = scaled.split()[3]

            canvas = Image.new("L", canvas_size, 0)
            canvas.paste(alpha, (paste_bbox["x1"], paste_bbox["y1"]))

            buf = BytesIO()
            canvas.save(buf, format="PNG")
            return buf.getvalue()

        return await asyncio.to_thread(sync)


async def _normalize_size(processed_bytes: bytes, reference_bytes: bytes) -> bytes:
    """Ensure processed image has same size as reference. Fixes LaMa padding artifacts."""
    def sync():
        ref = Image.open(BytesIO(reference_bytes))
        proc = Image.open(BytesIO(processed_bytes)).convert("RGB")
        if proc.size != ref.size:
            proc = proc.resize(ref.size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        proc.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    return await asyncio.to_thread(sync)


_sam_mode_instance = None


def get_sam_mode(device: str = "cpu") -> SAMLamaMode:
    global _sam_mode_instance
    if _sam_mode_instance is None:
        _sam_mode_instance = SAMLamaMode(device=device)
    return _sam_mode_instance