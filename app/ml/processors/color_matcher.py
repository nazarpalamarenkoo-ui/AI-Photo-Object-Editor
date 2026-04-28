import asyncio 
from typing import Dict, Literal
from io import BytesIO
import numpy as np
from PIL import Image
import cv2


class ColorMatcher:
    """
    Color matching processor for object replacement.
    
    Provides:
        1. Mean/std normalization matching (fast)
        2. Histogram matching (accurate)
        3. Color transfer in LAB space (best quality)
    
    Handles:
        1. Context region extraction (surrounding background)
        2. Per-channel color statistics
        3. Result compositing
    """
    
    def __init__(self):
        """
        Initialize Color Matcher.
        """
        print("Color Matcher initialized")
     
    async def match_colors(
        self,
        image_with_replacement_bytes: bytes,
        bbox: Dict[str, int],
        method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std',
        context_margin: int = 20
    ) -> bytes:
        """
        Match colors of replacement object to surrounding background.
        
        Args:
            1. image_with_replacement_bytes: Image with replacement already pasted in
            2. bbox: Bounding box of replacement region {'x1', 'y1', 'x2', 'y2'}
            3. method: Color matching method (default: 'mean_std')
                - 'mean_std': Fast, good quality
                - 'histogram': Slower, more accurate
                - 'color_transfer': Slowest, best quality
            4. context_margin: Pixels around bbox to sample background from (default: 20)
        
        Returns: Result image bytes (JPEG) with color-matched replacement
        """
        result_bytes = await asyncio.to_thread(
            self._match_colors_sync,
            image_with_replacement_bytes,
            bbox,
            method,
            context_margin
        )
        
        return result_bytes
    
    def _match_colors_sync(
        self,
        image_with_replacement_bytes: bytes,
        bbox: Dict[str, int],
        method: str,
        context_margin: int
    ) -> bytes:
        """
        Match colors synchronously (blocking).
        
        Args:
            1. image_with_replacement_bytes: Image bytes with replacement pasted in
            2. bbox: Bounding box dict {'x1', 'y1', 'x2', 'y2'}
            3. method: Color matching method ('mean_std', 'histogram', 'color_transfer')
            4. context_margin: Pixels around bbox to sample for context
        
        Returns: Result image bytes (JPEG)
        
        Raises:
            ValueError: If unknown method specified
        """
        img = Image.open(BytesIO(image_with_replacement_bytes)).convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        
        height, width = img_array.shape[:2]
        
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Extract replacement region and surrounding context
        object_region = img_array[y1:y2, x1:x2]
        context_region = self._extract_context(
            img_array,
            bbox,
            context_margin,
            width,
            height
        )
        
        # Apply selected color matching method
        if method == 'mean_std':
            matched_object = self._match_mean_std(object_region, context_region)
        elif method == 'histogram':
            matched_object = self._match_histogram(object_region, context_region)
        elif method == 'color_transfer':
            matched_object = self._color_transfer(object_region, context_region)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Paste matched object back into image
        result_array = img_array.copy()
        result_array[y1:y2, x1:x2] = matched_object
        
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result_array, mode='RGB')
        
        result_buffer = BytesIO()
        result_img.save(result_buffer, format='JPEG', quality=95)
        
        return result_buffer.getvalue()
    
    def _extract_context(
        self,
        img_array: np.ndarray,
        bbox: Dict[str, int],
        margin: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Extract background context pixels surrounding bbox.
        
        Args:
            1. img_array: Full image as float32 array (H, W, 3)
            2. bbox: Bounding box dict {'x1', 'y1', 'x2', 'y2'}
            3. margin: Pixels to extend region beyond bbox edges
            4. width: Image width (for clipping)
            5. height: Image height (for clipping)
        
        Returns: np.ndarray (N, 3) - background pixels only (bbox excluded)
        """
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Expand context region with margin (clamped to image bounds)
        ctx_x1 = max(0, x1 - margin)   # don't go left of image edge
        ctx_y1 = max(0, y1 - margin)   # don't go above image edge
        ctx_x2 = min(width, x2 + margin)   # don't go right of image edge
        ctx_y2 = min(height, y2 + margin)  # don't go below image edge
        
        extended_region = img_array[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        
        # Create mask excluding the bbox (replacement) area
        # start with all True (include everything), then punch out the bbox hole
        mask = np.ones(extended_region.shape[:2], dtype=bool)
        
        # re-express bbox coordinates relative to the cropped context region origin
        bbox_in_context_x1 = x1 - ctx_x1
        bbox_in_context_y1 = y1 - ctx_y1
        bbox_in_context_x2 = x2 - ctx_x1
        bbox_in_context_y2 = y2 - ctx_y1
        
        # mark bbox pixels as False so they are excluded from background sample
        mask[bbox_in_context_y1:bbox_in_context_y2, bbox_in_context_x1:bbox_in_context_x2] = False
        
        context_pixels = extended_region[mask]  # boolean indexing -> (N, 3) flat list of background pixels
        
        return context_pixels
    
    def _match_mean_std(
        self,
        source: np.ndarray,
        target_context: np.ndarray
    ) -> np.ndarray:
        """
        Match colors using mean/std normalization.
        
        Normalizes source statistics to match target context statistics.
        Fast and good quality for most cases.
        
        Args:
            1. source: Source region to adjust (H, W, 3) float32
            2. target_context: Target background pixels (N, 3) float32
        
        Returns: np.ndarray (H, W, 3) - color-matched region
        """
        source_mean = np.mean(source, axis=(0, 1), keepdims=True)  # (1, 1, 3) mean per channel across H,W
        source_std = np.std(source, axis=(0, 1), keepdims=True) + 1e-6  # +1e-6 avoids division by zero
        
        target_mean = np.mean(target_context, axis=0, keepdims=True)  # (1, 3) mean per channel across N pixels
        target_std = np.std(target_context, axis=0, keepdims=True) + 1e-6
        
        # Step 1: standardize source to zero mean, unit std
        # Step 2: rescale to target distribution (mean and spread)
        # result pixels have the same statistical distribution as the background
        normalized = (source - source_mean) / source_std
        matched = normalized * target_std + target_mean
        
        return matched
    
    def _match_histogram(
        self,
        source: np.ndarray,
        target_context: np.ndarray
    ) -> np.ndarray:
        """
        Match colors using histogram equalization per channel.
        
        Uses CDF-based lookup table for accurate color mapping.
        Slower than mean_std but more accurate for complex distributions.
        
        Args:
            1. source: Source region to adjust (H, W, 3) float32
            2. target_context: Target background pixels (N, 3) float32
        
        Returns: np.ndarray (H, W, 3) - color-matched region
        """
        matched = np.zeros_like(source)
        
        for channel in range(3):  # R, G, B
            
            source_channel = source[:, :, channel].flatten()   # (H*W,) — all pixels for this channel
            target_channel = target_context[:, channel].flatten()  # (N,) — background pixels for this channel
            
            # cast to uint8 so histogram bins align to integer [0..255] values
            source_uint8 = source_channel.astype(np.uint8)
            target_uint8 = target_channel.astype(np.uint8)
            
            # Calculate CDFs for both distributions
            source_hist, _ = np.histogram(source_uint8, bins=256, range=(0, 256))
            target_hist, _ = np.histogram(target_uint8, bins=256, range=(0, 256))
            
            # cumulative sum gives the CDF: how many pixels fall at or below each value
            source_cdf = np.cumsum(source_hist).astype(np.float32)
            target_cdf = np.cumsum(target_hist).astype(np.float32)
            
            # normalize CDFs to [0, 1] so they represent probabilities
            source_cdf /= source_cdf[-1]
            target_cdf /= target_cdf[-1]
            
            # Build lookup table mapping source -> target values
            # for each source intensity i, find the target intensity j whose CDF
            # probability is closest — this remaps the source distribution to target
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                diff = np.abs(target_cdf - source_cdf[i])
                lookup_table[i] = np.argmin(diff)  # target bin with closest CDF value
                
            # apply lookup: replace each source pixel value with its mapped target value
            matched_channel = lookup_table[source_uint8].astype(np.float32)
            matched[:, :, channel] = matched_channel.reshape(source.shape[:2])  # restore (H, W) shape
            
        return matched
    
    def _color_transfer(
        self,
        source: np.ndarray,
        target_context: np.ndarray
    ) -> np.ndarray:
        """
        Match colors using LAB color space transfer.
        
        Performs mean/std matching in LAB space for perceptually
        uniform color transfer. Best quality, slowest method.
        
        Args:
            1. source: Source region to adjust (H, W, 3) float32
            2. target_context: Target background pixels (N, 3) float32
        
        Returns: np.ndarray (H, W, 3) - color-matched region
        """
        # Convert source to LAB
        # LAB separates lightness (L) from color (A, B), making color transfer
        # more perceptually uniform than operating in RGB space
        source_uint8 = np.clip(source, 0, 255).astype(np.uint8)
        source_lab = cv2.cvtColor(source_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Convert target context pixels to LAB
        # reshape to (N, 1, 3) because cvtColor requires at least 2D image input
        target_reshaped = target_context.reshape(-1, 1, 3).astype(np.uint8)
        target_lab_pixels = cv2.cvtColor(target_reshaped, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = target_lab_pixels.reshape(-1, 3)  # flatten back to (N, 3)
        
        # Match statistics in LAB space — same mean/std approach as _match_mean_std
        # but operating on perceptually uniform LAB channels instead of RGB
        source_mean = np.mean(source_lab, axis=(0, 1), keepdims=True)
        source_std = np.std(source_lab, axis=(0, 1), keepdims=True) + 1e-6
        
        target_mean = np.mean(target_lab, axis=0, keepdims=True)
        target_std = np.std(target_lab, axis=0, keepdims=True) + 1e-6
        
        # Standardize source LAB, then rescale to target LAB distribution
        normalized = (source_lab - source_mean) / source_std
        matched_lab = normalized * target_std + target_mean
        
        # Convert back to RGB
        matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
        matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        return matched_rgb
    
    def match_against_original(
    self,
    result_bytes: bytes,
    original_image_bytes: bytes,
    bbox: Dict[str, int],
    method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std',
    context_margin: int = 25
    ) -> bytes:
        """
        Apply color matching to the object region using ORIGINAL image context.

        Args:
            result_bytes: Image with inserted object
            original_image_bytes: Original image (before inpainting)
            bbox: Target region {'x1','y1','x2','y2'}
            method: Matching method ('mean_std', 'histogram', 'color_transfer')
            context_margin: Pixels around bbox used for sampling background

        Returns:
            JPEG bytes with adjusted colors
        """
        
        return self._match_against_original_sync(
            result_bytes,
            original_image_bytes,
            bbox,
            method,
            context_margin
        )
    
    def _match_against_original_sync(
        self,
        result_bytes: bytes,
        original_image_bytes: bytes,
        bbox: Dict[str, int],
        method: str,
        context_margin: int
    ) -> bytes:
        """
        Perform color matching using background pixels from the original image.

        Steps:
            1. Extract region around bbox (expanded by context_margin)
            2. Remove bbox area → keep only background pixels
            3. Compute color statistics from context
            4. Adjust object region to match context
            5. Insert corrected region back into result image

        Args:
            result_bytes: Image after object insertion
            original_image_bytes: Original image
            bbox: Object region
            method: Matching algorithm
            context_margin: Context expansion size

        Returns:
            JPEG bytes
        """
        result = np.array(Image.open(BytesIO(result_bytes)).convert('RGB'), dtype=np.float32)
        original = np.array(Image.open(BytesIO(original_image_bytes)).convert('RGB'), dtype=np.float32)

        height, width = original.shape[:2]

        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        #context from ORIGINAL
        ctx_x1 = max(0, x1 - context_margin)
        ctx_y1 = max(0, y1 - context_margin)
        ctx_x2 = min(width, x2 + context_margin)
        ctx_y2 = min(height, y2 + context_margin)

        context_region = original[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

        mask = np.ones(context_region.shape[:2], dtype=bool)

        bx1 = x1 - ctx_x1
        by1 = y1 - ctx_y1
        bx2 = x2 - ctx_x1
        by2 = y2 - ctx_y1

        mask[by1:by2, bx1:bx2] = False

        context_pixels = context_region[mask]

        if len(context_pixels) == 0:
            return result_bytes

        object_region = result[y1:y2, x1:x2].copy()

        if method == 'mean_std':
            matched = self._match_mean_std(object_region, context_pixels)
        elif method == 'histogram':
            matched = self._match_histogram(object_region, context_pixels)
        elif method == 'color_transfer':
            matched = self._color_transfer(object_region, context_pixels)
        else:
            raise ValueError(f"Unknown method: {method}")

        result[y1:y2, x1:x2] = matched

        result = np.clip(result, 0, 255).astype(np.uint8)

        buf = BytesIO()
        Image.fromarray(result).save(buf, format='JPEG', quality=95)
        return buf.getvalue()
    
_color_matcher_instance = None
 
 
def get_color_matcher() -> ColorMatcher:
    """
    Singleton getter for ColorMatcher.
    
    Returns: ColorMatcher instance
    """
    global _color_matcher_instance
    if _color_matcher_instance is None:
        _color_matcher_instance = ColorMatcher()
    return _color_matcher_instance