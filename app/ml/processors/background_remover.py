import asyncio
from typing import List, Dict, Literal, Optional
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
from rembg import remove


class BackgroundRemover:
    """
    Background removal processor.
    
    Provides:
        1. rembg AI-based removal (best quality, requires rembg)
        2. GrabCut segmentation (no extra deps, needs bbox hint)
        3. Otsu threshold removal (fast, simple backgrounds)
        4. Mask-based removal (manual mask input)
    
    Handles:
        1. Method selection and dispatch
        2. RGBA output with transparency
        3. JPEG fallback with white background compositing
    """
    
    def __init__(self, rembg_available: bool = True):
        """
        Initialize Background Remover.
        
        Args:
        rembg_available: Enable rembg AI removal (default: False)
            - True: Loads rembg model (requires rembg installed)
            - False: Only GrabCut and threshold methods available
        """
        self.rembg_available = rembg_available
        
        if rembg_available:
            try:
                self.rembg_remove = remove
                print("BackgroundRemover initialized")
            except ImportError:
                self.rembg_available = False
                print("BackgroundRemover not initialized")
        else:
            print("BackgroundRemover initialized (rembg disabled)")
                    
    async def remove_background(
        self,
        image_bytes: bytes,
        method: Literal['rembg', 'grabcut', 'threshold'] = 'rembg',
        return_format: Literal['png', 'jpeg'] = 'png',
        bbox: Optional[dict] = None
    ) -> bytes:
        """
        Remove background from image.
        
        Args:
            1. image_bytes: Input image bytes
            2. method: Removal method (default: 'rembg')
                - 'rembg': AI-based, best quality (requires rembg)
                - 'grabcut': OpenCV GrabCut segmentation
                - 'threshold': Otsu threshold, fast for simple backgrounds
            3. return_format: Output format (default: 'png')
                - 'png': RGBA with transparency
                - 'jpeg': RGB with white background
            4. bbox: Optional bounding box hint for grabcut method (default: None)
        
        Returns: Result image bytes in specified format
        
        Raises:
            ValueError: If rembg method requested but rembg not available
            ValueError: If unknown method specified
        """
        result_bytes = await asyncio.to_thread(
            self._remove_background_sync,
            image_bytes,
            method,
            return_format,
            bbox
        )
        
        return result_bytes
    
    def _remove_background_sync(
        self,
        image_bytes: bytes,
        method: str,
        return_format: str,
        bbox: Optional[dict]
    ) -> bytes:
        """
        Remove background synchronously (blocking).
        
        Args:
            1. image_bytes: Input image bytes
            2. method: Removal method ('rembg', 'grabcut', 'threshold')
            3. return_format: Output format ('png' or 'jpeg')
            4. bbox: Optional bounding box hint for grabcut
        
        Returns: Result image bytes
        
        Raises:
            ValueError: If rembg not available when method='rembg'
            ValueError: If unknown method specified
        """
        if method == 'rembg':
            if not self.rembg_available:
                raise ValueError('rembg not available')
           
            result_bytes = self.rembg_remove(image_bytes)
           
            # Convert to JPEG with white bg if requested
            if return_format == 'jpeg':
                result_img = Image.open(BytesIO(image_bytes)).convert('RGBA')
                rgb_img = self._rgba_to_rgb_white_bg(result_img)
                
                buffer = BytesIO()
                rgb_img.save(buffer, format='JPEG', quality=95)
                return buffer.getvalue()
            
            else:
                return result_bytes
            
        elif method == 'grabcut':
            return self._remove_with_grabcut(image_bytes, return_format, bbox)
        
        elif method == 'threshold':
            return self._remove_with_threshold(image_bytes, return_format)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
    async def remove_with_mask(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        return_format: Literal['png', 'jpeg'] = 'png',
        invert_mask: bool = False
    ) -> bytes:
        """
        Remove background using provided binary mask.
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (white = keep, black = remove)
            3. return_format: Output format (default: 'png')
                - 'png': RGBA with transparency
                - 'jpeg': RGB with white background
            4. invert_mask: Invert mask before applying (default: False)
        
        Returns: Result image bytes in specified format
        """
        result_bytes = await asyncio.to_thread(
            self._remove_with_mask_sync,
            image_bytes,
            mask_bytes,
            return_format,
            invert_mask
        )
        
        return result_bytes
    
    def _remove_with_mask_sync(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        return_format: str,
        invert_mask: bool
    ) -> bytes:
        """
        Remove background using mask synchronously (blocking).
        
        Args:
            1. image_bytes: Input image bytes
            2. mask_bytes: Mask bytes (white = keep)
            3. return_format: Output format ('png' or 'jpeg')
            4. invert_mask: Invert mask before applying
        
        Returns: Result image bytes
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        mask = Image.open(BytesIO(mask_bytes)).convert('L')
        
        # Invert mask if requested
        if invert_mask:
            mask_array = np.array(mask)
            mask_array = 255 - mask_array  # flip pixel values: 0 -> 255 (black -> white) and vice versa
            mask = Image.fromarray(mask_array, mode='L')
            
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Combine image with mask as alpha channel
        rgba_array = np.dstack([img_array, mask_array])  # (H, W, 3) + (H, W) -> (H, W, 4): appends mask as alpha
        
        result_img = Image.fromarray(rgba_array, mode='RGBA')
        
        if return_format == 'png':
            buffer = BytesIO()
            result_img.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            rgb_img = self._rgba_to_rgb_white_bg(result_img)
            buffer = BytesIO()
            rgb_img.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
        
    def _remove_with_grabcut(
        self,
        image_bytes: bytes,
        return_format: str,
        bbox: Optional[dict]
    ) -> bytes:
        """
        Remove background using OpenCV GrabCut algorithm.
        
        Args:
            1. image_bytes: Input image bytes
            2. return_format: Output format ('png' or 'jpeg')
            3. bbox: Optional bbox hint for foreground {'x1', 'y1', 'x2', 'y2'}
                - If None: uses 10% margin from image edges
        
        Returns: Result image bytes
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        
        # Build GrabCut rect from bbox or default margin
        if bbox:
            rect = (bbox['x1'], bbox['y1'],
                    bbox['x2'] - bbox['x1'],   # width = x2 - x1
                    bbox['y2'] - bbox['y1'])    # height = y2 - y1
        else:
            # Auto-detect with 10% margin
            h, w = img_array.shape[:2]
            margin_h = int(h * 0.1)  # 10% of height
            margin_w = int(w * 0.1)  # 10% of width
            # rect format for GrabCut: (x, y, width, height) — inset from all edges
            rect = (margin_h, margin_w, w - 2 * margin_w, h - 2 * margin_h)
        
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Run GrabCut segmentation
        cv2.grabCut(
            img_array,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_RECT
        )
        
        # Convert GrabCut mask to binary (foreground vs background)
        # GrabCut values: 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD
        # values 0 and 2 are (probable) background -> set to 0 (transparent)
        # values 1 and 3 are (probable) foreground -> set to 255 (opaque)
        mask_binary = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        
        # Stack RGB channels with binary mask as alpha: (H, W, 3) + (H, W) -> (H, W, 4)
        rgba_array = np.dstack([img_array, mask_binary])
        result_img = Image.fromarray(rgba_array, mode='RGBA')
        
        if return_format == 'png':
            buffer = BytesIO()
            result_img.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            rgb_img = self._rgba_to_rgb_white_bg(result_img)
            buffer = BytesIO()
            rgb_img.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
      
    def _remove_with_threshold(
        self,
        image_bytes: bytes,
        return_format: str
    ) -> bytes:
        """
        Remove background using Otsu thresholding.
        
        Fast method for simple/uniform backgrounds.
        
        Args:
            1. image_bytes: Input image bytes
            2. return_format: Output format ('png' or 'jpeg')
        
        Returns: Result image bytes
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale and apply Otsu threshold
        # Otsu automatically finds the optimal threshold value that separates
        # foreground from background by minimizing intra-class intensity variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Stack RGB channels with Otsu mask as alpha: (H, W, 3) + (H, W) -> (H, W, 4)
        rgba_array = np.dstack([img_array, mask])
        
        result_img = Image.fromarray(rgba_array, mode='RGBA')
        
        if return_format == 'png':
            buffer = BytesIO()
            result_img.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            rgb_img = self._rgba_to_rgb_white_bg(result_img)
            buffer = BytesIO()
            rgb_img.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
        
    def _rgba_to_rgb_white_bg(
        self,
        rgba_img: Image.Image
    ) -> Image.Image:
        """
        Composite RGBA image onto white RGB background.
        
        Args:
        rgba_img: RGBA PIL image with transparency
        
        Returns: RGB PIL image with white background
        """
        rgb_img = Image.new('RGB', rgba_img.size, (255, 255, 255))
        
        # Use alpha channel as compositing mask
        rgb_img.paste(rgba_img, mask=rgba_img.split()[3])
        
        return rgb_img

    async def remove_and_resize(
        self,
        image_bytes: bytes,
        target_size: tuple
    ) -> bytes:
        """
        Remove background from an image and resize it to target dimensions.

        Pipeline:
            1. Calls background removal model (e.g. rembg / grabcut)
            2. Cleans alpha channel (removes noise, smooths edges)
            3. Resizes image using high-quality resampling
            4. Returns RGBA PNG bytes

        Notes:
            - Alpha channel is post-processed to reduce artifacts
            - Designed for compositing (not perfect segmentation)

        Args:
            image_bytes: Input image bytes
            target_size: (width, height)

        Returns:
            RGBA PNG bytes
        """
        img = Image.open(BytesIO(image_bytes)).convert('RGBA')
        orig_w, orig_h = img.size

        no_bg_bytes = await self.remove_background(
            image_bytes=image_bytes,
            method='rembg',
            return_format='png',
        )
        
        def fix_alpha_sync(img_rgba):
            arr = np.array(img_rgba)

            alpha = arr[:, :, 3]

            # remove garbage but KEEP gradients
            alpha = cv2.medianBlur(alpha, 5)

            # slightly sharpen mask edges
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.erode(alpha, kernel, iterations=1)

            arr[:, :, 3] = alpha

            return Image.fromarray(arr, mode='RGBA')
        
        def resize_sync():
            img = Image.open(BytesIO(no_bg_bytes)).convert('RGBA')
            arr = np.array(img)
            print("ALPHA MIN/MAX:", arr[:, :, 3].min(), arr[:, :, 3].max())
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img = fix_alpha_sync(img)
            buf = BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        return await asyncio.to_thread(resize_sync)
    
_background_remover_instance = None
 
 
def get_background_remover(rembg_available: bool = True) -> BackgroundRemover:
    """
    Singleton getter for BackgroundRemover.
    
    Args:
    rembg_available: Enable rembg AI removal (default: False)
    
    Returns: BackgroundRemover instance
    """
    global _background_remover_instance
    if _background_remover_instance is None:
        _background_remover_instance = BackgroundRemover(rembg_available)
    return _background_remover_instance