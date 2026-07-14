import asyncio
from typing import Optional, List, Dict
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image

from app.config.settings import settings
from app.config.device_manager import DeviceManager
from app.ml.detector import YOLODetector, get_detector
from app.ml.inpainter import LaMaInpainter, InpaintMode, get_inpainter
from app.ml.processors.edge_blender import EdgeBlender, get_edge_blender
from app.ml.processors.color_matcher import ColorMatcher, get_color_matcher
from app.ml.processors.background_remover import BackgroundRemover, get_background_remover
from app.ml.processors.image_compositor import get_compositor

from app.core.logging import get_logger, log_execution, log_ml_operation

logger = get_logger(__name__)


class YoloLamaMode:
    """
    YOLO + LaMa combined processing mode.

    Provides high-level interface for:
        1. Object detection (YOLO)
        2. Object removal (YOLO + LaMa + EdgeBlender)
        3. Object replacement (YOLO + LaMa + ColorMatcher + EdgeBlender)
        4. Multiple object removal (YOLO + LaMa + EdgeBlender)

    Handles:
        1. Temp image management for YOLO file-based API
        2. Mask creation from bbox (single and combined)
        3. Pipeline orchestration across components
        4. Result assembly
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        inpainter: Optional[LaMaInpainter] = None,
        edge_blender: Optional[EdgeBlender] = None,
        color_matcher: Optional[ColorMatcher] = None,
        background_remover: Optional[BackgroundRemover] = None,
    ):
        """
        Initialize YoloLamaMode with detector, inpainter and processors.

        Args:
            1. detector: YOLO detector instance (default: auto-created)
            2. inpainter: LaMa inpainter instance (default: auto-created)
            3. edge_blender: Edge blending processor (default: auto-created)
            4. color_matcher: Color matching processor (default: auto-created)
            5. backgroud_remover: Background remover processor (default: auto-created)
            6. compositor: Image compositor processor (default: auto-created)
            7. device: Device to use ('cuda' or 'cpu', default: 'cuda')
        """
        self.device = DeviceManager.get(settings.YOLO_DEVICE)
        self.detector = detector or get_detector()
        self.inpainter = inpainter or get_inpainter()
        self.edge_blender = edge_blender or get_edge_blender()
        self.color_matcher = color_matcher or get_color_matcher()
        self.background_remover = background_remover or get_background_remover(rembg_available=True)
        self.compositor = get_compositor()
        logger.info("yolo_lama_mode_initialized", device=str(self.device))

    async def _save_temp_image(self, image_bytes: bytes) -> Path:
        """
        Save image bytes to temp file for YOLO inference.

        Args:
        image_bytes: Input image bytes

        Returns: Path to saved temp image file
        """
        temp_dir = Path('/tmp/yolo_lama_mode')
        temp_dir.mkdir(exist_ok=True)

        # Use microsecond timestamp for unique filename
        temp_path = temp_dir / f"temp_{int(time.time() * 1000000)}.jpg"

        def write_sync():
            img = Image.open(BytesIO(image_bytes))
            img.save(temp_path, format='JPEG')

        await asyncio.to_thread(write_sync)
        return temp_path

    async def _create_remove_mask(
    self,
    image_bytes: bytes,
    bbox: Dict[str, int],
    expand_pixels: int = 10,
    all_bboxes: Optional[List[Dict[str, int]]] = None
    ) -> bytes:
        """
        Create a binary mask for object removal based on bbox.

        Passes all other detected bboxes to the inpainter so expansion
        is safe and never overlaps neighboring objects.

        Args:
            image_bytes: Input image
            bbox: Target region {'x1','y1','x2','y2'}
            expand_pixels: Max pixels to expand bbox on each side
            all_bboxes: All YOLO detections on this frame (including target).
                        Used to compute safe expansion per side.

        Returns:
            PNG bytes of binary mask (255 = remove, 0 = keep)
        """
        def create_mask_sync():
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size

            # Exclude the current target from the neighbors list
            other_bboxes = [
                b for b in (all_bboxes or [])
                if not (
                    b['x1'] == bbox['x1'] and b['y1'] == bbox['y1'] and
                    b['x2'] == bbox['x2'] and b['y2'] == bbox['y2']
                )
            ] if all_bboxes else None

            mask = self.inpainter.create_remove_mask(
                (height, width),
                bbox,
                expand_pixels=expand_pixels,
                other_bboxes=other_bboxes
            )

            buf = BytesIO()
            Image.fromarray(mask).save(buf, format='PNG')
            return buf.getvalue()

        return await asyncio.to_thread(create_mask_sync)

    async def _create_combined_mask(
        self,
        image_bytes: bytes,
        bboxes: List[Dict[str, int]],
        expand_pixels: int = 8,
        scene_bboxes: Optional[List[Dict[str, int]]] = None
    ) -> bytes:
        """
        Create combined binary mask from multiple bounding boxes.

        Each bbox is expanded safely: expansion on each side is limited by
        the distance to the nearest *non-selected* bbox on the scene.
        Note: selected bboxes don't constrain each other — they merge freely.

        Args:
            image_bytes: Input image bytes (for size reference)
            bboxes: List of target bboxes {'x1','y1','x2','y2'}
            expand_pixels: Max expansion per side for each bbox (default: 8)

        Returns:
            Combined mask image bytes (PNG, grayscale)
        """
        def create_mask_sync():
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            mask = np.zeros((height, width), dtype=np.uint8)

            for bbox in bboxes:
                external_obstacles = [
                    b for b in (scene_bboxes or [])
                    if not any(
                        b['x1'] == s['x1'] and b['y1'] == s['y1']
                        for s in bboxes
                    )
                ]

                single_mask = self.inpainter.create_remove_mask(
                    (height, width),
                    bbox,
                    expand_pixels=expand_pixels,
                    other_bboxes=external_obstacles or None
                )
                mask = np.maximum(mask, single_mask)

            mask_img = Image.fromarray(mask, mode='L')
            buffer = BytesIO()
            mask_img.save(buffer, format='PNG')
            return buffer.getvalue()

        return await asyncio.to_thread(create_mask_sync)

    async def detect_objects(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect objects in image using YOLO.

        Pipeline:
            1. Save image to temp file (YOLO requires file path)
            2. Run YOLO detection
            3. Assign bbox_id to each detection
            4. Clean up temp file

        Args:
            1. image_bytes: Input image bytes
            2. conf_threshold: Confidence threshold (0.0-1.0, default: 0.5)
            3. classes: Optional list of class names to filter (default: None)

        Returns:
            Dict {
                - detections: List[Dict] - detected objects with bbox_id
                    {
                        'bbox_id': int,
                        'detected_class': str,
                        'confidence': float,
                        'x1', 'y1', 'x2', 'y2': int
                    }
                - metrics: Dict - detection metrics
                - image_size: Tuple[int, int] - (width, height)
            }
        """
        temp_path = await self._save_temp_image(image_bytes)
        img = Image.open(BytesIO(image_bytes))

        try:
            async with log_ml_operation(
                "yolo_detect",
                model="yolo",
                device=str(self.device),
                image_size=img.size,
                conf_threshold=conf_threshold,
                classes=classes,
            ) as op:
                detection_result = await self.detector.detect(
                    str(temp_path),
                    conf_threshold=conf_threshold,
                    classes=classes,
                    track_metrics=True
                )

                detections = detection_result['detections']

                # Assign sequential bbox_id to each detection
                for idx, det in enumerate(detections):
                    det['bbox_id'] = idx

                op.set_output(num_detections=len(detections))

            return {
                'detections': detections,
                'metrics': detection_result['metrics'],
                'image_size': img.size
            }

        finally:
            # Always clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    async def remove_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        expand_mask_pixels: int = 30,
        use_edge_blending: bool = True,
        scene_bboxes: Optional[List[Dict[str, int]]] = None,
        ldm_steps: int = 25,
        ldm_sampler: str = 'plms',
        hd_strategy: str = 'CROP'
    ) -> Dict:
        """
        Remove object from image using LaMa inpainting.

        Pipeline:
            1. Create mask from bbox
            2. LaMa inpainting (REMOVE mode - generates background)
            3. Edge blending (smooth transition, optional)

        Args:
            1. image_bytes: Input image bytes
            2. selected_bbox: Bounding box to remove {'x1', 'y1', 'x2', 'y2'}
            3. expand_mask_pixels: Pixels to expand mask beyond bbox (default: 5)
            4. use_edge_blending: Apply edge blending (default: True, recommended)

        Returns:
            Dict {
                - result_bytes: Processed image bytes (JPEG)
                - metrics: Dict - inpainting metrics
            }
        """
        async with log_ml_operation(
            "yolo_lama_remove_object",
            model="lama",
            device=str(self.device),
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
        ) as op:
            # Create mask from selected bbox
            mask_bytes = await self._create_remove_mask(
                image_bytes,
                selected_bbox,
                expand_pixels=expand_mask_pixels,
                all_bboxes=scene_bboxes
            )

            # Run LaMa inpainting in REMOVE mode
            inpaint_result = await self.inpainter.inpaint(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                mode=InpaintMode.REMOVE,
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy
            )

            result_bytes = inpaint_result['result_bytes']

            result_bytes = await _normalize_size(result_bytes, image_bytes)
            # Apply edge blending for smooth transition
            if use_edge_blending:
                result_bytes = await self.edge_blender.auto_blend(
                    original_image_bytes=image_bytes,
                    processed_image_bytes=result_bytes,
                    mask_bytes=mask_bytes,
                    expand_mask_pixels=expand_mask_pixels
                )

            op.set_output(**inpaint_result['metrics'])

        return {
            'result_bytes': result_bytes,
            'metrics': inpaint_result['metrics']
        }

    async def replace_object(
        self,
        image_bytes: bytes,
        selected_bbox: Optional[Dict[str, int]],
        replacement_image_bytes: bytes,
        expand_mask_pixels: int = 10,
        use_color_matching: bool = False,
        use_edge_blending: bool = False,
        color_match_method: str = 'color_transfer',
        scene_bboxes: Optional[List[Dict[str, int]]] = None,
        ldm_steps: int = 25,
        ldm_sampler: str = 'plms',
        hd_strategy: str = 'CROP'
    ) -> Dict:
        """
        Replace object in image with replacement image.

        Pipeline:
            1. Remove background from replacement image (GrabCut)
            2. Create mask from bbox
            3. LaMa inpainting — clean background where old object was
            4. Paste replacement (with alpha) on clean background
            5. Color matching (optional)
            6. Edge blending (optional)

        Args:
            1. image_bytes: Input image bytes
            2. selected_bbox: Bounding box to replace {'x1', 'y1', 'x2', 'y2'}
            3. replacement_image_bytes: Replacement object image bytes
            4. expand_mask_pixels: Pixels to expand mask beyond bbox (default: 0)
            5. use_color_matching: Apply color matching (default: True, recommended)
            6. use_edge_blending: Apply edge blending (default: True, recommended)
            7. color_match_method: Color matching method (default: 'mean_std')
                - 'mean_std': Fast, good quality
                - 'histogram': Slower, more accurate
                - 'color_transfer': Slowest, best quality

        Returns:
            Dict {
                - result_bytes: Processed image bytes (JPEG)
                - metrics: Dict - inpainting metrics
            }
        """
        # Remove background + resize
        if selected_bbox is None:
            logger.error("yolo_lama_replace_object_missing_bbox")
            raise ValueError("selected_bbox is required")

        bbox_w = selected_bbox['x2'] - selected_bbox['x1']
        bbox_h = selected_bbox['y2'] - selected_bbox['y1']

        expand_mask_pixels = int(min(bbox_w, bbox_h) * 0.25)

        async with log_ml_operation(
            "yolo_lama_replace_object",
            model="lama",
            device=str(self.device),
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            color_match_method=color_match_method,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
        ) as op:
            replacement_rgba_bytes = await self.background_remover.remove_and_resize(
                replacement_image_bytes,
                (bbox_w, bbox_h)
            )

            # Create mask for remove
            mask_bytes = await self._create_remove_mask(
                image_bytes,
                selected_bbox,
                expand_pixels=expand_mask_pixels,
                all_bboxes=scene_bboxes
            )

            # LaMa — clean background (remove old object)
            inpaint_result = await self.inpainter.inpaint(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                mode=InpaintMode.REMOVE,
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy
            )
            clean_bytes = inpaint_result['result_bytes']
            clean_bytes = await _normalize_size(clean_bytes, image_bytes)

            #  Compose (paste + edge fix)
            result_bytes = self.compositor.compose(
                clean_bg_bytes=clean_bytes,
                replacement_rgba_bytes=replacement_rgba_bytes,
                bbox=selected_bbox,
                edge_softness=0,
            )

            if use_color_matching:
                result_bytes = self.color_matcher.match_against_original(
                    result_bytes=result_bytes,
                    original_image_bytes=image_bytes,
                    bbox=selected_bbox,
                    method=color_match_method  # type: ignore
                )

            op.set_output(**inpaint_result['metrics'])

        return {
            'result_bytes': result_bytes,
            'metrics': inpaint_result['metrics']
        }

    async def remove_multiple_objects(
        self,
        image_bytes: bytes,
        selected_bboxes: List[Dict[str, int]],
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        ldm_steps: int = 25,
        ldm_sampler: str = 'plms',
        hd_strategy: str = 'CROP'
    ) -> Dict:
        """
        Remove multiple objects from image in one inpainting pass.

        Pipeline:
            1. Create combined mask from all bboxes
            2. LaMa inpainting (REMOVE mode - generates background for all regions)
            3. Edge blending (smooth transition, optional)

        Args:
            1. image_bytes: Input image bytes
            2. selected_bboxes: List of bounding boxes to remove
            3. expand_mask_pixels: Pixels to expand each mask beyond bbox (default: 5)
            4. use_edge_blending: Apply edge blending (default: True, recommended)

        Returns:
            Dict {
                - result_bytes: Processed image bytes (JPEG)
                - metrics: Dict - inpainting metrics
            }
        """
        async with log_ml_operation(
            "yolo_lama_remove_multiple_objects",
            model="lama",
            device=str(self.device),
            num_objects=len(selected_bboxes),
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy,
        ) as op:
            # Create combined mask covering all selected bboxes
            mask_bytes = await self._create_combined_mask(
                image_bytes,
                selected_bboxes,
                expand_pixels=expand_mask_pixels
            )

            # Run LaMa inpainting in REMOVE mode (single pass for all objects)
            inpaint_result = await self.inpainter.inpaint(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                mode=InpaintMode.REMOVE,
                track_metrics=True,
                ldm_steps=ldm_steps,
                ldm_sampler=ldm_sampler,
                hd_strategy=hd_strategy
            )

            result_bytes = inpaint_result['result_bytes']
            result_bytes = await _normalize_size(result_bytes, image_bytes)

            # Apply edge blending for smooth transitions
            if use_edge_blending:
                result_bytes = await self.edge_blender.auto_blend(
                    original_image_bytes=image_bytes,
                    processed_image_bytes=result_bytes,
                    mask_bytes=mask_bytes,
                    expand_mask_pixels=expand_mask_pixels
                )

            op.set_output(**inpaint_result['metrics'])

        return {
            'result_bytes': result_bytes,
            'metrics': inpaint_result['metrics']
        }

    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported YOLO detection classes.

        Returns: List of class names (80 COCO classes)
        """
        return self.detector.get_class_names()


async def _normalize_size(processed_bytes: bytes, reference_bytes: bytes) -> bytes:
    """
    Ensure processed image has same size as reference image.
    Fixes LaMa padding / resizing artifacts.
    """
    def sync():
        ref = Image.open(BytesIO(reference_bytes))
        proc = Image.open(BytesIO(processed_bytes)).convert('RGB')
        if proc.size != ref.size:
            proc = proc.resize(ref.size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        proc.save(buf, format='JPEG', quality=95)
        return buf.getvalue()

    return await asyncio.to_thread(sync)


import threading
_yolo_lama_mode_instance = None
_yolo_lama_mode_lock = threading.Lock()

def get_yolo_lama_mode() -> YoloLamaMode:
    """
    Singleton getter for YoloLamaMode.

    Args:
    device: Device to use ('cuda' or 'cpu', default: 'cuda')

    Returns: YoloLamaMode instance
    """
    global _yolo_lama_mode_instance
    if _yolo_lama_mode_instance is None:
        with _yolo_lama_mode_lock:
            if _yolo_lama_mode_instance is None:
                _yolo_lama_mode_instance = YoloLamaMode()
    return _yolo_lama_mode_instance