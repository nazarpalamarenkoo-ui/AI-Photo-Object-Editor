import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, List
from ultralytics import YOLO
import numpy as np
from PIL import Image
 
from app.ml.experiment_tracker import ExperimentTracker, get_tracker

class YOLODetector:
    """
    YOLO Object Detector
    
    Wrapper around Ultralytics YOLO for object detection.
    Provides async interface and MLflow tracking integration.
    
    Features:
    1. Object detection with confidence thresholding
    2. Class filtering
    3. Async processing
    4. MLflow metrics tracking
    5. Automatic device management (CUDA/CPU)
    
    Handles:
    1. Model loading and inference
    2. Result parsing (bbox, class, confidence)
    3. Metrics calculation
    4. MLflow tracking
    """
    
    def __init__(
        self, 
        model_path: str = 'yolov10n.pt', 
        device: str = 'cuda', 
        conf_threshold: float = 0.5,
        tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize YOLO Detector.
        
        Args:
            model_path: Path to YOLO model weights (default: 'yolov10n.pt')
            device: Device to use ('cuda' or 'cpu', default: 'cuda')
            conf_threshold: Default confidence threshold (default: 0.5)
            tracker: ExperimentTracker for MLflow (default: auto-created)
        """
        
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        
        self.tracker = tracker or get_tracker()
        #load model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        print(f"YOLO loaded. Classes: {len(self.model.names)}")
        
    async def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[str]] = None,
        track_metrics: bool = True
    ) -> Dict:
        """
        Detect objects in image using YOLO.
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold (0.0-1.0, default: uses instance default)
            classes: Optional list of class names to filter (e.g., ['car', 'person'])
            track_metrics: Track metrics to MLflow (default: True)
        
        Returns:
            Dict:
                - detections: List[Dict] - detected objects
                    Each detection: {
                        'x1', 'y1', 'x2', 'y2': int - bbox coordinates
                        'detected_class': str - class name
                        'confidence': float - confidence score
                        'class_id': int - COCO class ID
                    }
        """
        
        conf = conf_threshold or self.conf_threshold
        start_time = time.time()
        
        detections = await asyncio.to_thread(
            self._detect_sync,
            image_path,
            conf,
            classes
        )
        inference_time = (time.time() - start_time) * 1000
        metrics = self._calculate_metrics(detections, inference_time)
        
        if track_metrics:
            await self._track_metrics(metrics, image_path)
        
        return {
            'detections': detections,
            'metrics': metrics
        }
    
    def _detect_sync(
        self,
        image_path: str,
        conf_threshold: float,
        classes: Optional[List[str]] = None
    ) -> List[Dict]:
        # Run YOLO inference
        results = self.model(
            image_path,
            conf = conf_threshold,
            device = self.device,
            verbose = False
        )
        # Parse results        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get class and cofidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.model.names[class_id]

                # Filter by class if specified
                if classes and class_name not in classes:
                    continue
                
                detections.append({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'detected_class': class_name,
                    'confidence': confidence,
                    'class_id': class_id
                })

        return detections
    
    def get_class_names(self) -> List[str]:
        
        return list(self.model.names.values())
   
    def _calculate_metrics(
        self,
        detections: List[Dict],
        inference_time_ms: float            
    ) -> Dict:
        
        if not detections:
            return {
                'num_detections': 0,
                'avg_confidence': 0.0,
                'inference_time_ms': inference_time_ms,
                'classes_detected': []
            }
            
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        classes_detected = list(set(d['detected_class'] for d in detections))
        
        return {
            'num_detections': len(detections),
            'avg_confidence': avg_confidence,
            'inference_time_ms': inference_time_ms,
            'classes_detected': classes_detected,
            'min_confidence': min(d['confidence'] for d in detections),
            'max_confidence': max(d['confidence'] for d in detections)
        }
    
    async def _track_metrics(
        self,
        metrics: Dict,
        image_path: Optional[str] = None
    ) -> None:
        """
        Track detection metrics to MLflow asynchronously.
        
        Args:
            metrics: Metrics dict with detection stats
            image_path: Optional path to image (for tagging)
        """
        def log_sync():
            # Convert inference_time_ms to seconds for tracker
            inference_time_sec = metrics['inference_time_ms'] / 1000.0
            
            self.tracker.log_detection_metrics(
                num_detections=metrics['num_detections'],
                avg_confidence=metrics['avg_confidence'],
                inference_time=inference_time_sec,  # Expects seconds
                model_name=self.model_path
            )
        
        await asyncio.to_thread(log_sync)
    
_detector_instance = None

def get_detector(
    model_path: str = 'yolov10n.pt',
    device: str = 'cuda',
    tracker: Optional[ExperimentTracker] = None
) -> YOLODetector:
    
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector(model_path, device, tracker=tracker)
    return _detector_instance