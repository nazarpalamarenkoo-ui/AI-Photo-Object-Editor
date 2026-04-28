import mlflow
from typing import Optional, Dict, List
from pathlib import Path
import mlflow.artifacts
import mlflow.tracking
import os

class ModelManager:
    """
    MLflow model manager for object detection system.
    
    Provides:
        1. Model registration and versioning
        2. Model loading from MLflow registry
        3. Stage promotion (Staging -> Production -> Archived)
        4. Model version comparison
    
    Handles:
        1. MLflow experiment setup
        2. Artifact storage
        3. Stage transitions
        4. Metrics comparison
    """
    
    def __init__(
        self,
        tracking_uri: str = 'http://localhost:5000',
        experiment_name: str = 'object-detection-system'
    ):
        """
        Initialize Model Manager.
        
        Args:
            tracking_uri: MLflow tracking server URI (default: 'http://localhost:5000')
            experiment_name: Name of experiment (default: 'object-detection-system')
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
            
        print(f"MLflow initialized: {tracking_uri}")
        print(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
        
    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register model in MLflow registry.
        
        Pipeline:
            1. Start MLflow run
            2. Log metrics and tags
            3. Log model artifact
            4. Register model version
        
        Args:
            1. model_path: Local path to model file
            2. model_name: Name to register model under
            3. metrics: Optional dict of metric_name -> value (default: None)
            4. tags: Optional dict of tag_name -> value (default: None)
            5. description: Optional model description tag (default: None)
        
        Returns: str - Registered model version number
        """
        with mlflow.start_run(experiment_id=self.experiment_id):
            
            # Log metrics if provided
            if metrics:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
            
            # Log tags if provided
            if tags:
                for tag_name, value in tags.items():
                    mlflow.set_tag(tag_name, value)
                    
            if description:
                mlflow.set_tag('description', description)
                
            # Upload model artifact
            mlflow.log_artifact(model_path, artifact_path='model')
            
            model_uri = f'run://{mlflow.active_run().info.run_id}/model'
            result = mlflow.register_model(model_uri, model_name)
            
            version = result.version
            print(f'Registered {model_name} version {version}')
            
            return str(version)
        
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> str:
        """
        Load model artifact from MLflow registry.
        
        Args:
            1. model_name: Registered model name
            2. version: Specific version to load (default: None)
            3. stage: Stage to load from - 'Staging', 'Production' (default: 'Production')
        
        Returns: str - Local path to downloaded model artifact
        
        Raises:
            ValueError: If both version and stage are specified
        """
        if version and stage:
            raise ValueError('Specify either version or stage, not both')
        
        if not version and not stage:
            stage = 'Production'
            
        # Build model URI based on version or stage
        if version:
            model_uri = f'models:/{model_name}/{version}'
        else:
            model_uri = f'models:/{model_name}/{stage}'
        
        model_path = mlflow.artifacts.download_artifacts(model_uri)
        
        print(f"Loaded {model_name} from {model_uri}")
        
        return model_path
    
    def promote_to_production(
        self,
        model_name: str,
        version: str
    ) -> None:
        """
        Promote model version to Production stage.
        
        Pipeline:
            1. Archive current Production version (if exists)
            2. Transition target version to Production
        
        Args:
            1. model_name: Registered model name
            2. version: Version number to promote
        """
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Archive existing production version
            prod_version = client.get_latest_versions(
                model_name,
                stages=['Production']
            )
            for mv in prod_version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage='Archived'
                )
        except Exception as e:
            print(f'No previous production version: {e}')
        
        # Promote target version to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage='Production'
        )
        
        print(f"Promoted {model_name} v{version} to Production")
        
    def get_model_versions(
        self,
        model_name: str
    ) -> List[Dict]:
        """
        Get all versions of a registered model.
        
        Args:
        model_name: Registered model name
        
        Returns:
            List[Dict] - each item:
            {
                - version: str - Version number
                - stage: str - Current stage
                - creation_timestamp: int - Unix timestamp
                - run_id: str - MLflow run ID
            }
        """
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        result = []
        
        # Parse version metadata
        for mv in versions:
            result.append({
                'version': mv.version,
                'stage': mv.current_stage,
                'creation_timestamp': mv.creation_timestamp,
                'run_id': mv.run_id
            })
            
        return result
    
    def compare_models(
        self,
        model_name: str,
        version_a: str,
        version_b: str
    ) -> Dict:
        """
        Compare metrics between two model versions.
        
        Args:
            1. model_name: Registered model name
            2. version_a: First version to compare
            3. version_b: Second version to compare
        
        Returns:
            Dict {
                - version_a: str - First version number
                - version_b: str - Second version number
                - metrics: Dict - per-metric comparison:
                    {
                        metric_name: {
                            'version_a': float,
                            'version_b': float,
                            'diff': float (b - a)
                        }
                    }
            }
        """
        client = mlflow.tracking.MlflowClient()
        
        # Fetch run data for both versions
        mv_a = client.get_model_version(model_name, version_a)
        mv_b = client.get_model_version(model_name, version_b)
        
        run_a = client.get_run(mv_a.run_id)
        run_b = client.get_run(mv_b.run_id)
        
        metrics_a = run_a.data.metrics
        metrics_b = run_b.data.metrics
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics': {}
        }
        
        # Build comparison dict for shared metrics
        for metrics_name in metrics_a.keys():
            if metrics_name in metrics_b:
                comparison['metrics'][metrics_name] = {
                    'version_a': metrics_a[metrics_name],
                    'version_b': metrics_b[metrics_name],
                    'diff': metrics_b[metrics_name] - metrics_a[metrics_name]
                }
                
        return comparison
    

_model_manager_instance = None
 

def get_model_manager(
    tracking_uri: str = None,
    experiment_name: str = "object-detection-system"
) -> ModelManager:
    global _model_manager_instance
    if _model_manager_instance is None:
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        _model_manager_instance = ModelManager(uri, experiment_name)
    return _model_manager_instance
