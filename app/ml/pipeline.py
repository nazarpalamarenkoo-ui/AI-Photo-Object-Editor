import asyncio
from tracemalloc import start
from typing import Optional, List, Dict, Literal
from io import BytesIO
from datetime import datetime
import time

from PIL import Image

from app.ml.modes import yolo_lama_mode
from app.ml.modes.yolo_lama_mode import YoloLamaMode, get_yolo_lama_mode
from app.ml.experiment_tracker import ExperimentTracker, get_tracker


class MLPipeline:
    
    """
    Main ML pipeline orchestrator
    
    Provides high-level interface for:
    
    1. Object Detection (YOLO)
    2. Object Removal (YOLO + LaMa + processors)
    3. Object Replacement (YOLO + LaMa + processors)
    4. Multiple Object Removal (YOLO + LaMa + processors)
    In future will added segmentation mode and hybrid mode
    
    Handles:
    1. ML Operations
    2. Metrics Tracking (MLflow)
    3. Error Handing
    4. Input Validation
    """
    
    def __init__(
        self,
        mode: Optional[YoloLamaMode] = None,
        tracker: Optional[ExperimentTracker] = None,
        device: str = 'cuda'
    ):
        
        """
        Initialize ML Pipeline.
        mode: YoloLamaMode instance (default: auto-created)
        tracker: ExperimentTracker for MLflow (default: auto-created)
        device: Device to use ('cuda' or 'cpu')
        """
        
        self.device = device
        self.mode = mode or get_yolo_lama_mode(device = device)
        self.tracker = tracker or get_tracker()
        
    async def detect_objects(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        track_metrics: bool = True
    ) -> Dict:
        
        """
        Detect Objects in image using YOLO.
        
        Args:
        1. image_bytes: Input image bytes
        2. conf_threshold: Cofidence threshold (0.0 - 1.0, default: 0.5)
        3. classes: Optional list of class name to filter
        4. track_metrics: Track metrics to MLflow (default: True)
        
        Returns:
            Dict:
            detections: List[Dict] - detected objects with bbox_id
            {
                        'bbox_id': int,
                        'class': str,
                        'confidence': float,
                        'bbox': {'x1', 'y1', 'x2', 'y2'}
                    }
                    
            image_size: Tuple[int, int] - (width, height)
            metrics: Dict - detection metrics
            timestamp: str - ISO timestamp
        """
        
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_image_bytes(image_bytes)
            
            # Run detection
            result = await self.mode.detect_objects(
                image_bytes = image_bytes,
                conf_threshold = conf_threshold,
                classes = classes
            )
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            
            # Tracking metrics
            if track_metrics and result.get('metrics'):
                inference_time = time.time() - start_time
                # Calculate average confidence
                detections = result['detections']
                avg_confidence = None
                if detections:
                    avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                    
                self.tracker.log_detection_metrics(
                    num_detections=len(detections),
                    inference_time=inference_time,  # seconds, will convert to ms
                    avg_confidence=avg_confidence,
                    conf_threshold=conf_threshold
                )
                
            return result
        
        except Exception as e:
            print(f'Detection failed: {e}')
            raise
    
    async def remove_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        track_metrics: bool = True
    ) -> Dict:
        
        """
        Remove object from image
        
        PipeLine:
        1. Create mask from bbox
        2. LaMa inpainting (REMOVE mode)
        3. Edge blending (smooth transition)
        
        Args:
        1. image_bytes: Input image bytes
        2. selected_bbox: Bounding box to remove
        3. expand_mask_pixels: Pixels to expand mask (default: 5)
        4. use_edge_blending: Apply edge blending (default: True, recommended)
        5. tracking_metrics: Track metrics to MLflow (default: True)
        
        Return:
        Dict {
            - result_bytes: Processed image bytes (JPEG)
            - metrics: Dict - processing metrics
            - timestamp: str - ISO timestamp 
        }
        """
        
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_image_bytes(image_bytes)
            self._validate_bbox(selected_bbox)
            
            # Run removal
            result = await self.mode.remove_object(
                image_bytes = image_bytes,
                selected_bbox = selected_bbox,
                expand_mask_pixels = expand_mask_pixels,
                use_edge_blending = use_edge_blending
            )

            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            
            if track_metrics:
                
                processing_time = time.time() - start_time
                self.tracker.log_metrics({
                    'operation': 'remove_object',
                    'processing_time': processing_time,
                    'expand_mask_pixels': expand_mask_pixels,
                    'edge_blending': use_edge_blending
                })
                
            return result
        
        except Exception as e:
            print(f'Object removal failed: {e}')
            raise
        
    async def replace_object(
        self,
        image_bytes: bytes,
        selected_bbox: Dict[str, int],
        replacement_image_bytes: bytes,
        expand_mask_pixels: int = 0,
        use_color_matching: bool = True,
        use_edge_blending: bool = True,
        color_match_method: Literal['mean_std', 'histogram', 'color_transfer'] = 'mean_std',
        track_metrics: bool = True
    ) -> Dict:
        
        """
        Replace object in image
        
        PipeLine:
        1. Create mask from bbox
        2. LaMa inpainting (REPLACE mode)
        3. Colo matching (match background colors)
        4. Edge Blending (smooth transition)
        
        Args:
        1. image_bytes: Input image bytes
        2. selected_bbox: Bounding box to replace
        3. replacement_image_bytes: Replacement object image bytes
        4. expand_mask_pixels: Pixels to expand mask (default: 0)
        5. use_color_matching: Apply color matching (default: True, recommended)
        6. use_edge_blending: Apply edge blending (default: True, recommended)
        7. color_match_method: Color matching method (default: 'mean_std')
            - 'mean_str': Fast, good quality
            - 'histogram': Slower, more accurate
            - 'color_transfer': Slowest, best quality
        8. tracking_metrics: Track metrics to MLflow (default: True)
        
        Returns:
        1. Dict {
            - result_bytes: Processed image bytes (JPEG)
            - metrics: Dict - processed metrics
            - timestamp: str - ISO timestamp
        }
        """
        
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_image_bytes(image_bytes)
            self._validate_bbox(selected_bbox)
            self._validate_image_bytes(replacement_image_bytes)
            
            # Run replacement
            result = await self.mode.replace_object(
                image_bytes = image_bytes,
                selected_bbox = selected_bbox,
                replacement_image_bytes = replacement_image_bytes,
                expand_mask_pixels = expand_mask_pixels,
                use_color_matching = use_color_matching,
                use_edge_blending = use_edge_blending,
                color_match_method = color_match_method
            )
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            
            # Tracking metircs
            if track_metrics:
                
                processing_time = time.time() - start_time
                self.tracker.log_metrics({
                    'operation': 'replace_object',
                    'processing_time': processing_time,
                    'color_matching': use_color_matching,
                    'edge_blending': use_edge_blending,
                    'color_match_method': color_match_method
                })
                
            return result
        
        except Exception as e:
            print(f'Object replacemnt failed: {e}')
            raise
        
    async def remove_multiple_objects(
        self,
        image_bytes: bytes,
        selected_bboxes: List[Dict[str, int]],
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        track_metrics: bool = True
    ) -> Dict:
    
        """
        Remove multiple objects from image in one operation
        
        Pipeline:
        1. Create combined mask from all bboxes
        2. LaMa inpainting (REMOVE mode)
        3. Edge blending (smooth transition)
        
        Args:
        1. image_bytes: Input image bytes
        2. selected_bbox: List of bounding boxes to remove
        3. expand_mask_pixels: Pixels to expand mask (default: 5)
        4. use_edge_blending: Apply edge blending (default: True, recommended)
        5. track_metrics: Track metrics to MLflow (default: True)
        
        Return:
        Dict {
            - result_bytes: Processed image bytes (JPEG)
            - metrics: Dict - processed metrics
            - timestamp: str - ISO timestamp
        }
        """
        
        start_time = time.time()
        
        try:    
            # Validate input
            self._validate_image_bytes(image_bytes)
            
            if not selected_bboxes:
                raise ValueError('selected_bboxes cannot be empty')
            
            for bbox in selected_bboxes:
                self._validate_bbox(bbox)
                
            # Run removal
            result = await self.mode.remove_multiple_objects(
                image_bytes = image_bytes,
                selected_bboxes = selected_bboxes,
                expand_mask_pixels = expand_mask_pixels,
                use_edge_blending = use_edge_blending
            )
            
            result['timestamp'] = datetime.now().isoformat()
            
            # Track metrics
            if track_metrics:
                processing_time = time.time() - start_time
                self.tracker.log_metrics({
                    'operation': 'remove_multiple_objects',
                    'num_objects': len(selected_bboxes),
                    'processing_time': processing_time,
                    'edge_blending': use_edge_blending
                })
            
            return result
        
        except Exception as e:
            print(f'Multiple objects removal failed: {e}')
            raise

    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported YOLO classes.
        
        Return: List of class name (80 COCO classes)
        """
        
        return self.mode.get_supported_classes()
    
    def _validate_image_bytes(self, image_bytes: bytes) -> None:
        
        """
        Validate image bytes
        
        Args: image_bytes: Image bytes to validate
        
        Raises: ValueError: If image bytes are invalid
        """
        
        if not image_bytes:
            raise ValueError('image_bytes cannot be empty')
        
        if not isinstance(image_bytes, bytes):
            raise ValueError('image_bytes must be bytes')
        
        # Try to open image
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify() # Verify its a valid image
            
        except Exception as e:
            raise ValueError(f'Invalid image bytes: {e}')
        
    def _validate_bbox(self, bbox: Dict[str, int]) -> None:
        
        """
        Validate bounding box
        
        Args: bbox: Bounding box dict
        
        Raise: ValueError: if bbox is invalid
        """
        
        required_keys = ['x1', 'y1', 'x2', 'y2']
        
        if not isinstance(bbox, dict):
            raise ValueError('bbox must be a dict')
        
        for key in required_keys:
            if key not in bbox:
                raise ValueError(f'bbox missing required key: {key}')

        # Validate coordinates
        if bbox['x1'] >= bbox['x2']:
            raise ValueError("bbox x1 must be < x2")
        
        if bbox['y1'] >= bbox['y2']:
            raise ValueError("bbox y1 must be < y2")
        
        if any(bbox[key] < 0 for key in required_keys):
            raise ValueError("bbox coordinates must be >= 0")
            
_pipeline_instance = None
 
 
def get_pipeline(device: str = 'cuda') -> MLPipeline:
    """
    Singleton getter for MLPipeline.
    
    Args: device: Device to use ('cuda' or 'cpu')
    
    Returns: MLPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MLPipeline(device=device)
    return _pipeline_instance