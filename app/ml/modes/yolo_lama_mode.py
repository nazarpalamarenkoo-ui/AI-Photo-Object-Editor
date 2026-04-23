import asyncio
from typing import Optional, List, Dict
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image

from app.ml.detector import YOLODetector, get_detector
from app.ml.inpainter import LaMaInpainter, InpaintMode, get_inpainter
from app.ml.processors.edge_blender import EdgeBlender, get_edge_blender
from app.ml.processors.color_matcher import ColorMatcher, get_color_matcher
from app.ml.processors.background_remover import BackgroundRemover, get_background_remover

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
        device: str = 'cpu'
    ):
        """
        Initialize YoloLamaMode with detector, inpainter and processors.
        
        Args:
            1. detector: YOLO detector instance (default: auto-created)
            2. inpainter: LaMa inpainter instance (default: auto-created)
            3. edge_blender: Edge blending processor (default: auto-created)
            4. color_matcher: Color matching processor (default: auto-created)
            5. device: Device to use ('cuda' or 'cpu', default: 'cuda')
        """
        self.device = device
        self.detector = detector or get_detector(device=device)
        self.inpainter = inpainter or get_inpainter(device=device)
        self.edge_blender = edge_blender or get_edge_blender()
        self.color_matcher = color_matcher or get_color_matcher()
        self.background_remover = background_remover or get_background_remover(rembg_available=False)
        print(f'YoloLamaMode initialized (device: {device})')
    
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
    
    async def _create_single_bbox_mask(
        self,
        image_bytes: bytes,
        bbox: Dict[str, int],
        expand_pixels: int = 5
    ) -> bytes:
        """
        Create binary mask from single bounding box.
        
        Args:
            1. image_bytes: Input image bytes (for size reference)
            2. bbox: Bounding box dict {'x1', 'y1', 'x2', 'y2'}
            3. expand_pixels: Pixels to expand mask beyond bbox edges (default: 5)
        
        Returns: Mask image bytes (PNG, grayscale - white = mask region)
        """
        def create_mask_sync():
            
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Expand bbox and clamp to image bounds
            x1 = max(0, bbox['x1'] - expand_pixels)
            y1 = max(0, bbox['y1'] - expand_pixels)
            x2 = min(width, bbox['x2'] + expand_pixels)
            y2 = min(height, bbox['y2'] + expand_pixels)
            
            mask[y1:y2, x1:x2] = 255
            
            mask_img = Image.fromarray(mask, mode='L')
            buffer = BytesIO()
            mask_img.save(buffer, format='PNG')
            
            return buffer.getvalue()
        
        mask_bytes = await asyncio.to_thread(create_mask_sync)
        return mask_bytes
    
    async def _create_combined_mask(
        self,
        image_bytes: bytes,
        bboxes: List[Dict[str, int]],
        expand_pixels: int = 5
    ) -> bytes:
        """
        Create combined binary mask from multiple bounding boxes.
        
        Args:
            1. image_bytes: Input image bytes (for size reference)
            2. bboxes: List of bounding box dicts {'x1', 'y1', 'x2', 'y2'}
            3. expand_pixels: Pixels to expand each mask beyond bbox edges (default: 5)
        
        Returns: Combined mask image bytes (PNG, grayscale - white = all masked regions)
        """
        def create_mask_sync():
            
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Union of all expanded bboxes
            for bbox in bboxes:
                x1 = max(0, bbox['x1'] - expand_pixels)
                y1 = max(0, bbox['y1'] - expand_pixels)
                x2 = min(width, bbox['x2'] + expand_pixels)
                y2 = min(height, bbox['y2'] + expand_pixels)
                
                mask[y1:y2, x1:x2] = 255
            
            mask_img = Image.fromarray(mask, mode='L')
            buffer = BytesIO()
            mask_img.save(buffer, format='PNG')
            
            return buffer.getvalue()
        
        mask_bytes = await asyncio.to_thread(create_mask_sync)
        return mask_bytes
    
    async def _remove_replacement_background(
        self,
        replacement_image_bytes: bytes,
        bbox: Dict[str, int]
    ) -> bytes:
        """
        Remove background from replacement image and crop to bbox size.
 
        Uses GrabCut (no extra deps) to remove background.
        Returns RGBA PNG with transparent background.
        """
        bbox_width = bbox['x2'] - bbox['x1']
        bbox_height = bbox['y2'] - bbox['y1']
 
        # Remove background using GrabCut
        no_bg_bytes = await self.background_remover.remove_background(
            image_bytes=replacement_image_bytes,
            method='grabcut',
            return_format='png'
        )
 
        # Resize to bbox dimensions keeping transparency
        def resize_sync():
            img = Image.open(BytesIO(no_bg_bytes)).convert('RGBA')
            resized = img.resize((bbox_width, bbox_height), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            resized.save(buffer, format='PNG')
            return buffer.getvalue()
 
        return await asyncio.to_thread(resize_sync)
    
    async def _paste_with_alpha(
        self,
        base_image_bytes: bytes,
        replacement_rgba_bytes: bytes,
        bbox: Dict[str, int]
    ) -> bytes:
        """
        Paste RGBA replacement into base image at bbox position.
        Uses alpha channel as mask so transparent pixels are not pasted.
        """
        def paste_sync():
            base = Image.open(BytesIO(base_image_bytes)).convert('RGB')
            replacement = Image.open(BytesIO(replacement_rgba_bytes)).convert('RGBA')
 
            # Use alpha channel as paste mask
            r, g, b, alpha = replacement.split()
            replacement_rgb = Image.merge('RGB', (r, g, b))
 
            base.paste(replacement_rgb, (bbox['x1'], bbox['y1']), mask=alpha)
 
            buffer = BytesIO()
            base.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
 
        return await asyncio.to_thread(paste_sync)
    
    
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
        
        try:
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
                
            img = Image.open(BytesIO(image_bytes))
            image_size = img.size
            
            return {
                'detections': detections,
                'metrics': detection_result['metrics'],
                'image_size': image_size
            }
            
        finally:
            # Always clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    async def remove_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True
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
        # Create mask from selected bbox
        mask_bytes = await self._create_single_bbox_mask(
            image_bytes,
            selected_bbox,
            expand_pixels=expand_mask_pixels
        )
        
        # Run LaMa inpainting in REMOVE mode
        inpaint_result = await self.inpainter.inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            mode=InpaintMode.REMOVE,
            track_metrics=True
        )
        
        result_bytes = inpaint_result['result_bytes']
        
        # Apply edge blending for smooth transition
        if use_edge_blending:
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=expand_mask_pixels
            )
        
        return {
            'result_bytes': result_bytes,
            'metrics': inpaint_result['metrics']
        }
        
    async def replace_object(
        self,
        image_bytes: bytes,
        selected_bbox: Optional[Dict[str, int]],
        replacement_image_bytes: bytes,
        expand_mask_pixels: int = 0,
        use_color_matching: bool = True,
        use_edge_blending: bool = True,
        color_match_method: str = 'mean_std'
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
        # Remove background from replacement image
        replacement_rgba_bytes = await self._remove_replacement_background(
            replacement_image_bytes, selected_bbox
        )
 
        # Create mask
        mask_bytes = await self._create_single_bbox_mask(
            image_bytes, selected_bbox, expand_pixels=expand_mask_pixels
        )
 
        # LaMa — clean background (remove old object)
        inpaint_result = await self.inpainter.inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            mode=InpaintMode.REMOVE,
            track_metrics=True
        )
        clean_bytes = inpaint_result['result_bytes']
 
        # Step 4: Paste replacement with alpha on clean background
        result_bytes = await self._paste_with_alpha(
            base_image_bytes=clean_bytes,
            replacement_rgba_bytes=replacement_rgba_bytes,
            bbox=selected_bbox
        )
 
        # Color matching
        if use_color_matching:
            result_bytes = await self.color_matcher.match_colors(
                image_with_replacement_bytes=result_bytes,
                bbox=selected_bbox,
                method=color_match_method,
                context_margin=20
            )
 
        # Edge blending
        if use_edge_blending:
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=expand_mask_pixels
            )
 
        return {'result_bytes': result_bytes, 'metrics': inpaint_result['metrics']}
        
    async def remove_multiple_objects(
        self,
        image_bytes: bytes,
        selected_bboxes: List[Dict[str, int]],
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True
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
            track_metrics=True
        )
        
        result_bytes = inpaint_result['result_bytes']
        
        # Apply edge blending for smooth transitions
        if use_edge_blending:
            result_bytes = await self.edge_blender.auto_blend(
                original_image_bytes=image_bytes,
                processed_image_bytes=result_bytes,
                mask_bytes=mask_bytes,
                expand_mask_pixels=expand_mask_pixels
            )
            
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
    
    
_yolo_lama_mode_instance = None
 
 
def get_yolo_lama_mode(device: str = 'cuda') -> YoloLamaMode:
    """
    Singleton getter for YoloLamaMode.
    
    Args:
    device: Device to use ('cuda' or 'cpu', default: 'cuda')
    
    Returns: YoloLamaMode instance
    """
    global _yolo_lama_mode_instance
    if _yolo_lama_mode_instance is None:
        _yolo_lama_mode_instance = YoloLamaMode(device=device)
    return _yolo_lama_mode_instance