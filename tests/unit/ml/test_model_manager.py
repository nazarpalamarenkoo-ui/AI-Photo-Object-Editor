import pytest
from unittest.mock import Mock, patch, MagicMock
import mlflow
from app.ml.experiment_tracker import ExperimentTracker
from app.ml.model_manager import ModelManager

@pytest.mark.unit
def test_model_manager_initialization():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="123"):
            manager = ModelManager(
                tracking_uri="http://localhost:5000",
                experiment_name="test-experiment"
            )
            
            assert manager.tracking_uri == "http://localhost:5000"
            assert manager.experiment_name == "test-experiment"
            assert manager.experiment_id == "123"
 
 
@pytest.mark.unit
def test_register_model():
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.create_experiment', return_value="123"), \
         patch('mlflow.start_run') as mock_run, \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.register_model') as mock_register:

        mock_active_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))
        mock_register.return_value = Mock(version="5")
        mock_run.__enter__ = MagicMock(return_value=MagicMock())
        mock_run.__exit__ = MagicMock(return_value=False)

        manager = ModelManager()
        version = manager.register_model(
            model_path="test_model.pt",
            model_name="test-model",
            metrics={'mAP': 0.45},
            tags={'architecture': 'yolo'}
        )

        assert version == "5"
 
 
@pytest.mark.unit
def test_promote_to_production():
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="123")
            
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                mock_client.get_latest_versions.return_value = []
                
                manager = ModelManager()
                
                manager.promote_to_production("test-model", "3")
                
                # Verify transition was called
                mock_client.transition_model_version_stage.assert_called_once_with(
                    name="test-model",
                    version="3",
                    stage="Production"
                )
 
 
@pytest.mark.unit
def test_get_model_versions():
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="123")
            
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                
                # Mock versions
                mock_versions = [
                    Mock(version="1", current_stage="Archived", 
                         creation_timestamp=1000, run_id="run1"),
                    Mock(version="2", current_stage="Production", 
                         creation_timestamp=2000, run_id="run2")
                ]
                mock_client.search_model_versions.return_value = mock_versions
                
                manager = ModelManager()
                versions = manager.get_model_versions("test-model")
                
                assert len(versions) == 2
                assert versions[0]['version'] == "1"
                assert versions[1]['stage'] == "Production"
 
 
 
@pytest.mark.unit
def test_tracker_initialization():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="456"):
            tracker = ExperimentTracker(
                tracking_uri="http://localhost:5000",
                experiment_name="test-tracking"
            )
            
            assert tracker.tracking_uri == "http://localhost:5000"
            assert tracker.experiment_name == "test-tracking"
            assert tracker.experiment_id == "456"
 
 
@pytest.mark.unit
def test_log_detection_metrics():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")
            
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()
                    
                    tracker.log_detection_metrics(
                        num_detections=5,
                        avg_confidence=0.87,
                        inference_time = 0.023,
                        model_name="yolov8n"
                    )
                    
                    # Verify metrics logged: num_detections, inference_time_ms, inference_time_sec, avg_confidence = 4
                    assert mock_log_metric.call_count == 4
                    mock_set_tag.assert_called_with('model_name', 'yolov8n')
 
 
@pytest.mark.unit
def test_log_batch_performance():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")
            
            with patch('mlflow.log_metric') as mock_log_metric:
                tracker = ExperimentTracker()
                
                tracker.log_batch_performance(
                    batch_size=10,
                    total_time_ms=250,
                    total_detections=47,
                    avg_confidence=0.83
                )
                
                # Verify all metrics logged
                assert mock_log_metric.call_count >= 5
 
 
@pytest.mark.unit
def test_get_best_run():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")
            
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                
                # Mock best run
                mock_run = Mock()
                mock_run.info.run_id = "best_run_123"
                mock_run.info.start_time = 1000
                mock_run.data.tags = {'mlflow.runName': 'Best Detection'}
                mock_run.data.metrics = {'avg_confidence': 0.95}
                
                mock_client.search_runs.return_value = [mock_run]
                
                tracker = ExperimentTracker()
                best_run = tracker.get_best_run('avg_confidence')
                
                assert best_run['run_id'] == "best_run_123"
                assert best_run['metrics']['avg_confidence'] == 0.95
 
 
@pytest.mark.unit
def test_get_experiment_summary():
    
    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")
            
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                
                # Mock runs
                mock_runs = [
                    Mock(data=Mock(metrics={'avg_confidence': 0.8, 'inference_time_ms': 20})),
                    Mock(data=Mock(metrics={'avg_confidence': 0.9, 'inference_time_ms': 25}))
                ]
                mock_client.search_runs.return_value = mock_runs
                
                tracker = ExperimentTracker()
                summary = tracker.get_experiment_summary()
                
                assert summary['total_runs'] == 2
                assert summary['avg_confidence_mean'] == pytest.approx(0.85)
                assert summary['inference_time_mean'] == pytest.approx(22.5)