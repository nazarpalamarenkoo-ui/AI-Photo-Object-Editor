import mlflow
from typing import List, Dict, Optional
from datetime import datetime

import mlflow.tracking


class ExperimentTracker:
    """
    MLflow experiment tracker for object detection system.
    
    Tracks:
    - Detection metrics (num_detections, confidence, inference time)
    - Inpainting metrics (processing time, mask size)
    - Batch performance
    - Model comparisons
    """
    
    def __init__(
        self,
        tracking_uri: str = 'http://localhost:5000',
        experiment_name: str = 'object-detection-system'
    ):
        """
        Initialize experiment tracker.
        
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            
        print(f"Experiment tracker ready: {experiment_name}")
        
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start new MLflow run.
        
        run_name: Name for this run
        tags: Optional tags dict
        
        Returns:
            mlflow.ActiveRun
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
    
    def log_detection_metrics(
        self,
        num_detections: int,
        inference_time: float,
        avg_confidence: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        image_id: Optional[int] = None
    ) -> None:
        """
        Log detection metrics to MLflow.
        Args:
            1. num_detections: Number of objects detected
            2. inference_time: Inference time in SECONDS (will convert to ms)
            3. avg_confidence: Average confidence of detections (optional)
            4. conf_threshold: Confidence threshold used (optional)
            5. model_name: Model name (optional)
            6. image_id: Image ID (optional)
        """
        # Convert seconds to milliseconds
        inference_time_ms = inference_time * 1000
        
        metrics = {
            'num_detections': num_detections,
            'inference_time_ms': inference_time_ms,
            'inference_time_sec': inference_time
        }
        
        if avg_confidence is not None:
            metrics['avg_confidence'] = avg_confidence
        
        if conf_threshold is not None:
            metrics['conf_threshold'] = conf_threshold
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
        if model_name:
            mlflow.set_tag('model_name', model_name)
        if image_id:
            mlflow.set_tag('image_id', image_id)
    
    def log_metrics(self, metrics: Dict[str, any]) -> None:
        """
        Log arbitrary metrics dict to MLflow.
        
        metrics: Dict of metric_name -> value
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)
            else:
                mlflow.set_tag(name, str(value))
            
    def log_inpaint_metrics(
        self,
        processing_time_ms: float,
        mask_size_pixels: int,
        image_size: tuple,
        model_name: Optional[str] = None
    ) -> None:
        """
        Log inpainting metrics to MLflow.
        Args:
            1. processing_time_ms: Processing time in milliseconds
            2. mask_size_pixels: Number of pixels in mask
            3. image_size: (width, height) tuple
            4. model_name: Model name (optional)
        """
        metrics = {
            'inpaint_time_ms': processing_time_ms,
            'mask_size_pixels': mask_size_pixels,
            'image_width': image_size[0],
            'image_height': image_size[1]
        }
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        if model_name:
            mlflow.set_tag('inpaint_model', model_name)
            
    def log_batch_performance(
        self,
        batch_size: int,
        total_time_ms: float,
        total_detections: int,
        avg_confidence: float
    ) -> None:
        """
        Log batch processing metrics.
        
            1. batch_size: Number of images in batch
            2. total_time_ms: Total processing time in milliseconds
            3. total_detections: Total number of detections
            4. avg_confidence: Average confidence across batch
        """
        metrics = {
            'batch_size': batch_size,
            'total_time_ms': total_time_ms,
            'avg_time_per_image_ms': total_time_ms / batch_size,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / batch_size,
            'avg_confidence': avg_confidence
        }
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
    def log_model_comparison(
        self,
        model_a: str,
        model_b: str,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float]
    ) -> None:
        """
        Log A/B model comparison.
        
        Args
            1. model_a: Name of model A
            2. model_b: Name of model B
            3. metrics_a: Metrics dict for model A
            4. metrics_b: Metrics dict for model B
        """
        mlflow.set_tag('comparison', 'A/B test')
        mlflow.set_tag('model_a', model_a)
        mlflow.set_tag('model_b', model_b)
        
        for name, value in metrics_a.items():
            mlflow.log_metric(f'model_a_{name}', value)
        
        for name, value in metrics_b.items():
            mlflow.log_metric(f'model_b_{name}', value)
        
        # Log differences
        for name in metrics_a.keys():
            if name in metrics_b:
                diff = metrics_b[name] - metrics_a[name]
                mlflow.log_metric(f'diff_{name}', diff)
                
    def get_best_run(
        self,
        metric_name: str = 'avg_confidence',
        ascending: bool = False
    ) -> Dict:
        """
        Get best run by metric.
        
        metric_name: Metric to sort by
        ascending: Sort ascending (True) or descending (False)
        
        Returns:
            Dict with run info or None if no runs
        """
        client = mlflow.tracking.MlflowClient()
        
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if not runs:
            return None
        
        run = runs[0]
        
        return {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName'),
            'metrics': run.data.metrics,
            'start_time': run.info.start_time
        }
        
    def get_experiment_summary(self) -> Dict:
        """
        Get experiment summary statistics.
        
        Returns:
            Dict with summary stats
        """
        client = mlflow.tracking.MlflowClient()
        
        runs = client.search_runs(
            experiment_ids=[self.experiment_id]
        )
        
        if not runs:
            return {'total_runs': 0}
        
        total_runs = len(runs)
        
        all_confidences = []
        all_times = []
        
        for run in runs:
            metrics = run.data.metrics
            if 'avg_confidence' in metrics:
                all_confidences.append(metrics['avg_confidence'])
            if 'inference_time_ms' in metrics:
                all_times.append(metrics['inference_time_ms'])
                
        summary = {
            'total_runs': total_runs,
            'experiment_name': self.experiment_name
        }
        
        if all_confidences:
            summary['avg_confidence_mean'] = sum(all_confidences) / len(all_confidences)
            summary['avg_confidence_max'] = max(all_confidences)
        
        if all_times:
            summary['inference_time_mean'] = sum(all_times) / len(all_times)
            summary['inference_time_min'] = min(all_times)
        
        return summary


_tracker_instance = None


def get_tracker(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "object-detection-system"
) -> ExperimentTracker:
    """
    Singleton getter for ExperimentTracker.
    
    tracking_uri: MLflow tracking URI
    experiment_name: Experiment name
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExperimentTracker(tracking_uri, experiment_name)
    return _tracker_instance