import asyncio
import time
from typing import Dict, Optional, Literal
from enum import Enum
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from app.ml.experiment_tracker import ExperimentTracker, get_tracker
 
 
class InpaintMode(str, Enum):
    REMOVE = "remove"
    REPLACE = "replace"
 
 
class LaMaInpainter:
    """
    LaMa-based image inpainter.
    
    Provides:
        1. Object removal (inpaint with background generation)
        2. Object replacement (paste replacement into bbox)
    
    Handles:
        1. LaMa model inference
        2. Mask creation from bbox
        3. Metrics calculation
        4. MLflow tracking
    """

    def __init__(
        self,
        device: str = 'cpu',
        tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize LaMa Inpainter.
        
        Args:
            device: Device to use ('cuda' or 'cpu', default: 'cuda')
            tracker: ExperimentTracker for MLflow (default: auto-created)
        """
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config, HDStrategy
 
            self.device = device
            self.tracker = tracker or get_tracker()
 
            self.model_manager = ModelManager(
                name='lama',
                device=self.device # type: ignore
            )
            self.default_config = Config(
                ldm_steps=25,
                ldm_sampler='plms',
                hd_strategy=HDStrategy.CROP,
                hd_strategy_crop_margin=32,
                hd_strategy_crop_trigger_size=800,
                hd_strategy_resize_limit=2048,
            )
            print(f"LaMa Inpainter initialized (device: {device})")
 
        except ImportError as e:
            raise RuntimeError(
                f"lama-cleaner not installed or incompatible: {e}. "
                "Set ML_ENABLED=false to run without ML."
            )
    
    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        bbox: Optional[Dict[str, int]] = None,
        mode: InpaintMode = InpaintMode.REMOVE,
        replacement_image_bytes: Optional[bytes] = None,
        track_metrics: bool = True,
        ldm_steps: int = 25,
        ldm_sampler: str = 'plms',
        hd_strategy: str = 'CROP'
    ) -> Dict:
        """
        Run inpainting on image.
        
        Pipeline:
            1. Run inpainting (REMOVE or REPLACE mode)
            2. Calculate metrics
            3. Track metrics to MLflow (optional)
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (white = inpaint area, default: None)
            3. bbox: Bounding box dict to auto-create mask (default: None)
            4. mode: Inpaint mode - REMOVE or REPLACE (default: REMOVE)
            5. replacement_image_bytes: Replacement image bytes (required for REPLACE mode)
            6. track_metrics: Track metrics to MLflow (default: True)
        
        Returns:
            Dict {
                - result_bytes: Processed image bytes (JPEG)
                - metrics: Dict - processing metrics
            }
        
        Raises:
            ValueError: If neither mask_bytes nor bbox provided
            ValueError: If REPLACE mode and no replacement_image_bytes
        """
        if not mask_bytes and not bbox:
            raise ValueError("Either mask_bytes or bbox must be provided")
        
        if mode == InpaintMode.REPLACE and not replacement_image_bytes:
            raise ValueError("replacement_image_bytes required for REPLACE mode")
        
        from lama_cleaner.schema import Config, HDStrategy
        config = Config(
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=getattr(HDStrategy, hd_strategy),
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=2048,
        )

        start_time = time.time()
        
        # Run inpainting based on mode
        if mode == InpaintMode.REMOVE:
            result_bytes = await self._inpaint_remove(
                image_bytes,
                mask_bytes,
                bbox,
                config
            )
        else:
            result_bytes = await self._inpaint_replace(
                image_bytes,
                mask_bytes,
                bbox,
                replacement_image_bytes,
                config
            )
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate processing metrics
        metrics = await self._calculate_metrics(
            image_bytes,
            mask_bytes,
            bbox,
            processing_time,
            mode
        )
        
        # Track metrics to MLflow
        if track_metrics:
            await self._track_metrics(metrics)
        
        return {
            'result_bytes': result_bytes,
            'metrics': metrics
        }
    
    async def _inpaint_remove(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        bbox: Optional[Dict[str, int]] = None,
        config=None
    ) -> bytes:
        """
        Run REMOVE inpainting asynchronously.
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (default: None)
            3. bbox: Bounding box to auto-create mask (default: None)
        
        Returns: Result image bytes (JPEG)
        """
        result_bytes = await asyncio.to_thread(
            self._inpaint_remove_sync,
            image_bytes,
            mask_bytes,
            bbox,
            config
        )
        return result_bytes
    
    def _inpaint_remove_sync(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        bbox: Optional[Dict[str, int]] = None,
        config = None
    ) -> bytes:
        """
        Run REMOVE inpainting synchronously (blocking).
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (default: None)
            3. bbox: Bounding box to auto-create mask (default: None)
        
        Returns: Result image bytes (JPEG)
        
        Raises:
            ValueError: If neither mask_bytes nor bbox provided
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        
        # Create mask from bytes or bbox
        if mask_bytes:
            mask = Image.open(BytesIO(mask_bytes)).convert('L')
            mask_array = np.array(mask)
        elif bbox:
            mask_array = self.create_remove_mask(
                img_array.shape[:2],
                bbox
            )
        else:
            raise ValueError("Either mask_bytes or bbox must be provided")
        
        # Run LaMa inpainting (generates background)
        result_array = self.model_manager(
            image=img_array,
            mask=mask_array,
            config=config or self.default_config
        )
        
        if result_array.dtype != np.uint8:
            result_array = result_array.clip(0, 255).astype(np.uint8)

        result_array = result_array[:, :, ::-1]
        # Convert result to bytes
        result_img = Image.fromarray(result_array)
        # Convert result to bytes
        result_buffer = BytesIO()
        result_img.save(result_buffer, format='JPEG', quality=95)
        
        return result_buffer.getvalue()
    
    async def _inpaint_replace(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes],
        bbox: Optional[Dict[str, int]],
        replacement_image_bytes: bytes,
        config = None
    ) -> bytes:
        """
        Run REPLACE inpainting asynchronously.
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (default: None)
            3. bbox: Bounding box for replacement area
            4. replacement_image_bytes: Replacement object image bytes
        
        Returns: Result image bytes (JPEG)
        """
        result_bytes = await asyncio.to_thread(
            self._inpaint_replace_sync,
            image_bytes,
            mask_bytes,
            bbox,
            replacement_image_bytes,
            config
        )
        return result_bytes
    
    def _inpaint_replace_sync(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes],
        bbox: Optional[Dict[str, int]],
        replacement_image_bytes: bytes,
        config=None
    ) -> bytes:
        """
        Run REPLACE inpainting synchronously (blocking).
        
        Pipeline:
            1. Get bbox from mask if not provided
            2. Resize replacement to bbox size
            3. Paste replacement into bbox region
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (used to extract bbox if bbox not provided)
            3. bbox: Bounding box for replacement area (default: None)
            4. replacement_image_bytes: Replacement object image bytes
        
        Returns: Result image bytes (JPEG)
        
        Raises:
            ValueError: If neither bbox nor mask_bytes provided
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        replacement_img = Image.open(BytesIO(replacement_image_bytes)).convert('RGB')

        if not bbox:
            raise ValueError("bbox required")

        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        replacement_img = replacement_img.resize(
            (bbox_width, bbox_height),
            Image.Resampling.LANCZOS
        )

        replacement_array = np.array(replacement_img)

        result_array = img_array.copy()
        result_array[y1:y2, x1:x2] = replacement_array

        result_img = Image.fromarray(result_array)
        buffer = BytesIO()
        result_img.save(buffer, format='JPEG', quality=95)

        return buffer.getvalue()
    
    def create_remove_mask(
        self,
        image_shape,
        bbox,
        expand_pixels=12,
        other_bboxes=None
    ):
        """
        Create a binary mask for object removal (inpainting).

        Expands bbox asymmetrically based on available free space on each side,
        taking into account other detected bboxes to avoid overlapping them.

        Process:
            - For each side, compute max safe expansion = distance to nearest
            neighboring bbox on that side (with a small safety margin).
            - Clamp expansion to [0, expand_pixels] on each side.
            - Fill mask region with 255 (remove area).

        Args:
            image_shape: Tuple (H, W)
            bbox: {'x1','y1','x2','y2'} — target object bbox
            expand_pixels: Desired expansion on each side (hard upper limit)
            other_bboxes: List of other detected bbox dicts to avoid overlapping.
                        Each: {'x1','y1','x2','y2'}. Pass all YOLO detections
                        except the current target.

        Returns:
            np.ndarray mask (H, W), uint8:
                255 → region to remove
                0   → keep
        """
        H, W = image_shape
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # Safety margin — keep this gap between expanded mask and neighbor bbox
        SAFETY_MARGIN = 3

        # Default: expand freely up to expand_pixels on each side
        exp_left  = expand_pixels
        exp_right = expand_pixels
        exp_top   = expand_pixels
        exp_bottom = expand_pixels

        if other_bboxes:
            for ob in other_bboxes:
                ox1, oy1, ox2, oy2 = ob['x1'], ob['y1'], ob['x2'], ob['y2']

                # Check horizontal overlap (neighbor is vertically adjacent to target)
                h_overlap = oy1 < y2 and oy2 > y1

                # Check vertical overlap (neighbor is horizontally adjacent to target)
                v_overlap = ox1 < x2 and ox2 > x1

                if h_overlap:
                    # Neighbor to the right
                    if ox1 >= x2:
                        gap = ox1 - x2
                        safe = max(0, gap - SAFETY_MARGIN)
                        exp_right = min(exp_right, safe)

                    # Neighbor to the left
                    if ox2 <= x1:
                        gap = x1 - ox2
                        safe = max(0, gap - SAFETY_MARGIN)
                        exp_left = min(exp_left, safe)

                if v_overlap:
                    # Neighbor below
                    if oy1 >= y2:
                        gap = oy1 - y2
                        safe = max(0, gap - SAFETY_MARGIN)
                        exp_bottom = min(exp_bottom, safe)

                    # Neighbor above
                    if oy2 <= y1:
                        gap = y1 - oy2
                        safe = max(0, gap - SAFETY_MARGIN)
                        exp_top = min(exp_top, safe)

        # Apply asymmetric expansion, clamped to image boundaries
        x1_exp = max(0, x1 - exp_left)
        y1_exp = max(0, y1 - exp_top)
        x2_exp = min(W, x2 + exp_right)
        y2_exp = min(H, y2 + exp_bottom)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255

        return mask

    def create_replace_mask(self, image_shape, bbox):
        """
        Create a binary mask for object replacement.

        Unlike remove mask, this uses the exact bbox without expansion.
        Used when precise placement is required.

        Process:
            - Use bbox directly (no padding)
            - Fill mask region with 255

        Args:
            image_shape: Tuple (H, W)
            bbox: {'x1','y1','x2','y2'}

        Returns:
            np.ndarray mask (H, W), uint8:
                255 → target region
                0   → background
        """
        H, W = image_shape
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        return mask
    
    def _get_bbox_from_mask(self, mask_array: np.ndarray) -> Dict[str, int]:
        """
        Extract bounding box from binary mask.
        
        Args:
        mask_array: np.ndarray mask (uint8)
        
        Returns:
            Dict {'x1', 'y1', 'x2', 'y2'} - tight bbox around white pixels
        
        Raises:
            ValueError: If mask is empty (no white pixels)
        """
        white_pixels = np.where(mask_array > 128)
        
        if len(white_pixels[0]) == 0:
            raise ValueError("Empty mask")
        
        y1 = int(np.min(white_pixels[0]))
        y2 = int(np.max(white_pixels[0]))
        x1 = int(np.min(white_pixels[1]))
        x2 = int(np.max(white_pixels[1]))
        
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    
    async def _calculate_metrics(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes],
        bbox: Optional[Dict[str, int]],
        processing_time_ms: float,
        mode: InpaintMode
    ) -> Dict:
        """
        Calculate inpainting metrics asynchronously.
        
        Args:
            1. image_bytes: Input image bytes (for size calculation)
            2. mask_bytes: Mask bytes (for pixel count, default: None)
            3. bbox: Bounding box (for pixel count if no mask, default: None)
            4. processing_time_ms: Processing time in milliseconds
            5. mode: Inpaint mode (REMOVE or REPLACE)
            
        Returns:
            Dict {
                - processing_time_ms: float - Processing time in ms
                - mask_size_pixels: int - Number of pixels in mask/bbox
                - image_size: Tuple[int, int] - (width, height)
                - mode: str - Inpaint mode value
            }
        """
        def calc_sync():
            # Get image dimensions
            img = Image.open(BytesIO(image_bytes))
            image_size = img.size  # (width, height)
            
            # Calculate mask size in pixels
            if bbox:
                mask_size_pixels = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
            elif mask_bytes:
                mask = Image.open(BytesIO(mask_bytes)).convert('L')
                mask_array = np.array(mask)
                mask_size_pixels = int(np.sum(mask_array > 128))
            else:
                mask_size_pixels = 0
            
            return {
                'processing_time_ms': processing_time_ms,
                'mask_size_pixels': mask_size_pixels,
                'image_size': image_size,
                'mode': mode.value
            }
        
        metrics = await asyncio.to_thread(calc_sync)
        return metrics
    
    async def _track_metrics(self, metrics: Dict) -> None:
        """
        Track inpainting metrics to MLflow asynchronously.
        
        Args:
        metrics: Dict with inpainting metrics (from _calculate_metrics)
        """
        def log_sync():
            self.tracker.log_inpaint_metrics(
                processing_time_ms=metrics['processing_time_ms'],
                mask_size_pixels=metrics['mask_size_pixels'],
                image_size=metrics['image_size'],
                model_name=f"lama_{metrics['mode']}"
            )
        
        await asyncio.to_thread(log_sync)
 
import threading
_inpainter_instance = None
_inpainter_lock = threading.Lock()
 
def get_inpainter(
    device: str = 'cuda',
    tracker: Optional[ExperimentTracker] = None
) -> LaMaInpainter:
    """
    Singleton getter for LaMaInpainter.
    
    Args:
        1. device: Device to use ('cuda' or 'cpu', default: 'cuda')
        2. tracker: ExperimentTracker for MLflow (default: auto-created)
    
    Returns: LaMaInpainter instance
    """
    global _inpainter_instance
    if _inpainter_instance is None:
        with _inpainter_lock:
            if _inpainter_instance is None:
                _inpainter_instance = LaMaInpainter(device, tracker=tracker)
    return _inpainter_instance