import mlflow
from typing import Optional, Dict, List
from pathlib import Path
import mlflow.artifacts
import mlflow.tracking
import os
import json
import tempfile
from datetime import datetime

from app.ml.experiment_tracker import ExperimentTracker, get_tracker


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
        tracking_uri: str = 'http://mlflow:5000',
        experiment_name: str = 'object-detection-system'
    ):
        """
        Initialize Model Manager.

        Args:
            tracking_uri: MLflow tracking server URI (default: 'http://mlflow:5000')
            experiment_name: Name of experiment (default: 'object-detection-system')
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.set_experiment(experiment_name)
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
            1. Start MLflow run (explicitly attached to this manager's experiment)
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
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"register_{model_name}") as run:

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

            # NOTE: the scheme is "runs:/", not "run://" (the original had a
            # typo here that would make mlflow.register_model() fail outright).
            model_uri = f'runs:/{run.info.run_id}/model'
            result = mlflow.register_model(model_uri, model_name)

            version = result.version
            print(f'Registered {model_name} version {version}')

            return str(version)

    def register_pipeline(
        self,
        pipeline_name: str = "ml-pipeline",
        operations: Optional[List[str]] = None,
        description: Optional[str] = None,
        tracker: Optional[ExperimentTracker] = None,
        require_all: bool = False,
    ) -> str:
        """
        Register ONE pipeline version instead of registering each model
        (YOLO/LaMa/SAM2) separately with hand-typed metrics.
        Pipeline:
            1. Pull latest real run for each stage in `operations`
            2. Merge into one namespaced metrics/params dict
               (missing stages are recorded in the 'missing_stages' tag,
               not silently zero-filled)
            3. Write a manifest (which run_id backs each stage) as the
               registered artifact — this IS the pipeline's "model":
               a pointer to the exact component runs it's built from
            4. One mlflow run, one register_model() call

        Args:
            pipeline_name: Name to register the pipeline under
                           (default: 'ml-pipeline')
            operations:    Which stage tags to pull, in order
                           (default: ['detect', 'inpaint',
                           'sam2_segment_auto']).
            description:   Optional description tag
            tracker:       ExperimentTracker to read runs from
                           (default: auto-created)
            require_all:   If True, raise instead of registering when any
                           requested stage has no run yet (default: False
                           — register with whatever stages are available)

        Returns: str - Registered pipeline version number

        Raises:
            ValueError: If require_all=True and a stage has no run yet,
                        or if NO requested stage has any run at all.
        """
        operations = operations or ["detect", "inpaint", "sam2_segment_auto"]
        tracker = tracker or get_tracker()

        aggregated_metrics: Dict[str, float] = {}
        aggregated_params: Dict[str, str] = {}
        stage_run_ids: Dict[str, str] = {}
        missing_stages: List[str] = []

        for op in operations:
            latest = tracker.get_latest_run_by_operation(op)

            if latest is None:
                missing_stages.append(op)
                if require_all:
                    raise ValueError(
                        f"No MLflow run found for operation='{op}' yet — "
                        f"run it at least once before registering the "
                        f"pipeline, or pass require_all=False."
                    )
                continue

            stage_run_ids[op] = latest["run_id"]

            for name, value in latest["metrics"].items():
                aggregated_metrics[f"{op}_{name}"] = value

            for name, value in latest["params"].items():
                aggregated_params[f"{op}_{name}"] = value

        if not stage_run_ids:
            raise ValueError(
                f"None of the requested operations {operations} have any "
                f"MLflow runs yet — nothing real to register. Run the "
                f"pipeline at least once first."
            )

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"register_{pipeline_name}"
        ) as run:
            if aggregated_metrics:
                mlflow.log_metrics(aggregated_metrics)
            if aggregated_params:
                mlflow.log_params(aggregated_params)

            mlflow.set_tag("pipeline_stages", ",".join(stage_run_ids.keys()))
            mlflow.set_tag("source_run_ids", json.dumps(stage_run_ids))
            if missing_stages:
                mlflow.set_tag("missing_stages", ",".join(missing_stages))
            if description:
                mlflow.set_tag("description", description)

            manifest = {
                "pipeline_name": pipeline_name,
                "registered_at": datetime.utcnow().isoformat() + "Z",
                "stage_run_ids": stage_run_ids,
                "missing_stages": missing_stages,
            }
            with tempfile.TemporaryDirectory() as tmp_dir:
                manifest_path = Path(tmp_dir) / "pipeline_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2))
                mlflow.log_artifact(str(manifest_path), artifact_path="model")

            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, pipeline_name)

            version = result.version
            print(
                f"Registered pipeline '{pipeline_name}' version {version} "
                f"from stages: {list(stage_run_ids.keys())}"
                + (f" (missing: {missing_stages})" if missing_stages else "")
            )

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
    tracking_uri: Optional[str] = None,
    experiment_name: str = "object-detection-system"
) -> ModelManager:
    global _model_manager_instance
    if _model_manager_instance is None:
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        _model_manager_instance = ModelManager(uri, experiment_name)
    return _model_manager_instance