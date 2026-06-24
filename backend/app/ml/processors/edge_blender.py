import asyncio
from typing import Optional
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


class EdgeBlender:
    """
    Edge blending processor for smooth compositing.
    
    Provides:
        1. Feathered mask blending (gaussian / box blur)
        2. Auto blend with adaptive feather radius
    
    Handles:
        1. Mask feathering (soft edges)
        2. Alpha blending between original and processed images
        3. Feather radius selection based on mask expansion
    """
    
    def __init__(self):
        """
        Initialize Edge Blender.
        """
        print('Edge Blender initialized')
    
    def _feather_mask(
        self,
        mask: np.ndarray,
        feather_radius: int,
        blur_method: str
    ) -> np.ndarray:
        """
        Apply feathering (soft edge) to binary mask.
        
        Args:
            1. mask: Float32 mask array (H, W), values 0.0 - 1.0
            2. feather_radius: Blur kernel half-size in pixels
            3. blur_method: Blur type ('gaussian' or 'box')
        
        Returns: np.ndarray (H, W) float32 - feathered mask
        
        Raises:
            ValueError: If unknown blur_method specified
        """
        # scale float [0.0, 1.0] mask to uint8 [0, 255] for cv2 blur functions
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if blur_method == 'gaussian':
            kernel_size = feather_radius * 2 + 1  # must be odd: radius 5 -> kernel 11x11
            sigma = feather_radius / 3  # rule of thumb: sigma ≈ radius/3 gives natural falloff
            
            # Apply gaussian blur for smooth falloff
            feathered_mask_uint8 = cv2.GaussianBlur(
                mask_uint8,
                (kernel_size, kernel_size),
                sigma
            )
            
        elif blur_method == 'box':
            kernel_size = feather_radius * 2 + 1  # must be odd: same formula as gaussian
            
            # Apply box blur for uniform falloff
            feathered_mask_uint8 = cv2.blur(
                mask_uint8,
                (kernel_size, kernel_size)
            )
        
        else:
            raise ValueError(f'Unknown blur_method: {blur_method}. Use gaussian or box')
        
        # scale back to float [0.0, 1.0] for use in alpha blending arithmetic
        feathered_mask = feathered_mask_uint8.astype(np.float32) / 255.0
        
        return feathered_mask
    
    def _blend_edges_sync(
        self,
        original_image_bytes: bytes,
        processed_image_bytes: bytes,
        mask_bytes: bytes,
        feather_radius: int,
        blur_method: str
    ) -> bytes:
        """
        Blend edges between original and processed image synchronously (blocking).
        
        Pipeline:
            1. Load original, processed images and mask
            2. Feather mask edges
            3. Alpha blend original and processed using feathered mask
        
        Args:
            1. original_image_bytes: Original unmodified image bytes
            2. processed_image_bytes: Processed image bytes (with inpainting/replacement)
            3. mask_bytes: Mask bytes (white = processed region)
            4. feather_radius: Blur kernel half-size in pixels
            5. blur_method: Blur type ('gaussian' or 'box')
        
        Returns: Blended result image bytes (JPEG)
        """
        original_img = Image.open(BytesIO(original_image_bytes)).convert('RGB')
        processed_img = Image.open(BytesIO(processed_image_bytes)).convert('RGB')
        mask_img = Image.open(BytesIO(mask_bytes)).convert('L')
        
        original_array = np.array(original_img, dtype=np.float32)
        processed_array = np.array(processed_img, dtype=np.float32)
        mask_array = np.array(mask_img, dtype=np.float32)
        
        # Normalize mask to 0.0 - 1.0 so it can be used as blend weight
        mask_normalized = mask_array / 255.0
        
        # Apply feathering to mask edges
        feathered_mask = self._feather_mask(
            mask_normalized,
            feather_radius,
            blur_method
        )
        
        # Expand mask to 3 channels for broadcasting
        # (H, W) -> (H, W, 1) so it multiplies against (H, W, 3) image arrays
        feathered_mask_3d = feathered_mask[:, :, np.newaxis]
        
        # Alpha blend: where mask=0 keep original pixel, where mask=1 use processed pixel
        # at soft edges (0 < mask < 1) the two images mix proportionally
        blended_array = (
            original_array * (1.0 - feathered_mask_3d) +
            processed_array * feathered_mask_3d
        )
        
        # clamp to valid uint8 range before converting, prevents wrap-around artifacts
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(blended_array, mode='RGB')
        
        result_buffer = BytesIO()
        result_img.save(result_buffer, format='JPEG', quality=95)
    
        return result_buffer.getvalue()
    
    async def blend_edges(
        self,
        original_image_bytes: bytes,
        processed_image_bytes: bytes,
        mask_bytes: bytes,
        feather_radius: int = 10,
        blur_method: str = 'gaussian'
    ) -> bytes:
        """
        Blend edges between original and processed image asynchronously.
        
        Args:
            1. original_image_bytes: Original unmodified image bytes
            2. processed_image_bytes: Processed image bytes (with inpainting/replacement)
            3. mask_bytes: Mask bytes (white = processed region)
            4. feather_radius: Blur kernel half-size in pixels (default: 10)
            5. blur_method: Blur type ('gaussian' or 'box', default: 'gaussian')
        
        Returns: Blended result image bytes (JPEG)
        """
        result_bytes = await asyncio.to_thread(
            self._blend_edges_sync,
            original_image_bytes,
            processed_image_bytes,
            mask_bytes,
            feather_radius,
            blur_method
        )
        
        return result_bytes
    
    async def auto_blend(
        self,
        original_image_bytes: bytes,
        processed_image_bytes: bytes,
        mask_bytes: bytes,
        expand_mask_pixels: int = 0
    ) -> bytes:
        """
        Blend edges with automatically selected feather radius.
        
        Feather radius is adapted based on expand_mask_pixels:
        - 0 pixels expanded -> feather_radius = 15 (large feather)
        - 1-5 pixels expanded -> feather_radius = 10 (medium feather)
        - 5+ pixels expanded -> feather_radius = 5 (small feather)
        
        Args:
            1. original_image_bytes: Original unmodified image bytes
            2. processed_image_bytes: Processed image bytes (with inpainting/replacement)
            3. mask_bytes: Mask bytes (white = processed region)
            4. expand_mask_pixels: How many pixels the mask was expanded (default: 0)
        
        Returns: Blended result image bytes (JPEG)
        """
        # Select feather radius based on mask expansion
        if expand_mask_pixels == 0:
            feather_radius = 15
        elif expand_mask_pixels <= 5:
            feather_radius = 10
        else:
            feather_radius = 5
            
        return await self.blend_edges(
            original_image_bytes,
            processed_image_bytes,
            mask_bytes,
            feather_radius=feather_radius,
            blur_method='gaussian'
        )
        
        
_edge_blender_instance = None
 
 
def get_edge_blender() -> EdgeBlender:
    """
    Singleton getter for EdgeBlender.
    
    Returns: EdgeBlender instance
    """
    global _edge_blender_instance
    if _edge_blender_instance is None:
        _edge_blender_instance = EdgeBlender()
    return _edge_blender_instance