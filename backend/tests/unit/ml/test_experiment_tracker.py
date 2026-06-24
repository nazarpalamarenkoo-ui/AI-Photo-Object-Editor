import pytest
from unittest.mock import Mock, patch, MagicMock
import mlflow
from app.ml.experiment_tracker import ExperimentTracker, get_tracker


@pytest.mark.unit
def test_tracker_initialization_create_new_experiment():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="789") as mock_create:
            tracker = ExperimentTracker(
                tracking_uri="http://localhost:5000",
                experiment_name="new-experiment"
            )

            assert tracker.tracking_uri == "http://localhost:5000"
            assert tracker.experiment_name == "new-experiment"
            assert tracker.experiment_id == "789"
            mock_create.assert_called_once_with("new-experiment")


@pytest.mark.unit
def test_tracker_initialization_existing_experiment():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', side_effect=Exception("Already exists")):
            with patch('mlflow.get_experiment_by_name') as mock_get_exp:
                mock_get_exp.return_value = Mock(experiment_id="321")

                tracker = ExperimentTracker(
                    tracking_uri="http://localhost:5000",
                    experiment_name="existing-experiment"
                )

                assert tracker.experiment_id == "321"
                mock_get_exp.assert_called_once_with("existing-experiment")


@pytest.mark.unit
def test_start_run():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.start_run') as mock_start_run:
                mock_run = Mock()
                mock_start_run.return_value = mock_run

                tracker = ExperimentTracker()
                result = tracker.start_run(
                    run_name="test-run",
                    tags={"env": "test"}
                )

                assert result == mock_run
                mock_start_run.assert_called_once_with(
                    experiment_id="111",
                    run_name="test-run",
                    tags={"env": "test"}
                )


@pytest.mark.unit
def test_log_detection_metrics_all_params():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_detection_metrics(
                        num_detections=5,
                        inference_time=0.023,
                        avg_confidence=0.87,
                        conf_threshold=0.5,
                        model_name="yolov8n",
                        image_id=42
                    )

                    # num_detections, inference_time_ms, inference_time_sec, avg_confidence, conf_threshold = 5 metrics
                    assert mock_log_metric.call_count == 5
                    mock_set_tag.assert_any_call('model_name', 'yolov8n')
                    mock_set_tag.assert_any_call('image_id', 42)


@pytest.mark.unit
def test_log_detection_metrics_converts_seconds_to_ms():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag'):
                    tracker = ExperimentTracker()

                    tracker.log_detection_metrics(
                        num_detections=3,
                        inference_time=0.05
                    )

                    logged_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
                    assert logged_calls['inference_time_ms'] == pytest.approx(50.0)
                    assert logged_calls['inference_time_sec'] == pytest.approx(0.05)


@pytest.mark.unit
def test_log_detection_metrics_optional_fields_not_logged():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_detection_metrics(
                        num_detections=2,
                        inference_time=0.01
                    )

                    # Only num_detections, inference_time_ms, inference_time_sec
                    assert mock_log_metric.call_count == 3
                    mock_set_tag.assert_not_called()


@pytest.mark.unit
def test_log_metrics_numeric_values():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_metrics({
                        'accuracy': 0.95,
                        'loss': 0.12,
                        'epoch': 10
                    })

                    assert mock_log_metric.call_count == 3
                    mock_set_tag.assert_not_called()


@pytest.mark.unit
def test_log_metrics_non_numeric_values_become_tags():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_metrics({
                        'model': 'yolov8n',
                        'loss': 0.12
                    })

                    assert mock_log_metric.call_count == 1
                    mock_set_tag.assert_called_once_with('model', 'yolov8n')


@pytest.mark.unit
def test_log_inpaint_metrics():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_inpaint_metrics(
                        processing_time_ms=120.5,
                        mask_size_pixels=4096,
                        image_size=(1920, 1080),
                        model_name="lama"
                    )

                    # inpaint_time_ms, mask_size_pixels, image_width, image_height = 4 metrics
                    assert mock_log_metric.call_count == 4
                    mock_set_tag.assert_called_once_with('inpaint_model', 'lama')


@pytest.mark.unit
def test_log_inpaint_metrics_no_model_name():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric'):
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_inpaint_metrics(
                        processing_time_ms=80.0,
                        mask_size_pixels=2048,
                        image_size=(640, 480)
                    )

                    mock_set_tag.assert_not_called()


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

                # batch_size, total_time_ms, avg_time_per_image_ms, total_detections,
                # avg_detections_per_image, avg_confidence = 6 metrics
                assert mock_log_metric.call_count == 6


@pytest.mark.unit
def test_log_batch_performance_derived_metrics():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric') as mock_log_metric:
                tracker = ExperimentTracker()

                tracker.log_batch_performance(
                    batch_size=4,
                    total_time_ms=200,
                    total_detections=20,
                    avg_confidence=0.9
                )

                logged_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
                assert logged_calls['avg_time_per_image_ms'] == pytest.approx(50.0)
                assert logged_calls['avg_detections_per_image'] == pytest.approx(5.0)


@pytest.mark.unit
def test_log_model_comparison():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag') as mock_set_tag:
                    tracker = ExperimentTracker()

                    tracker.log_model_comparison(
                        model_a="yolov8n",
                        model_b="yolov8s",
                        metrics_a={'mAP': 0.40, 'inference_time_ms': 10.0},
                        metrics_b={'mAP': 0.50, 'inference_time_ms': 20.0}
                    )

                    mock_set_tag.assert_any_call('comparison', 'A/B test')
                    mock_set_tag.assert_any_call('model_a', 'yolov8n')
                    mock_set_tag.assert_any_call('model_b', 'yolov8s')

                    # 2 model_a + 2 model_b + 2 diff = 6 metrics
                    assert mock_log_metric.call_count == 6


@pytest.mark.unit
def test_log_model_comparison_diff_values():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.set_tag'):
                    tracker = ExperimentTracker()

                    tracker.log_model_comparison(
                        model_a="modelA",
                        model_b="modelB",
                        metrics_a={'mAP': 0.40},
                        metrics_b={'mAP': 0.55}
                    )

                    logged_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
                    assert logged_calls['diff_mAP'] == pytest.approx(0.15)


@pytest.mark.unit
def test_get_best_run():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value

                mock_run = Mock()
                mock_run.info.run_id = "best_run_123"
                mock_run.info.start_time = 1000
                mock_run.data.tags = {'mlflow.runName': 'Best Detection'}
                mock_run.data.metrics = {'avg_confidence': 0.95}

                mock_client.search_runs.return_value = [mock_run]

                tracker = ExperimentTracker()
                best_run = tracker.get_best_run('avg_confidence')

                assert best_run['run_id'] == "best_run_123"
                assert best_run['run_name'] == "Best Detection"
                assert best_run['metrics']['avg_confidence'] == 0.95
                assert best_run['start_time'] == 1000


@pytest.mark.unit
def test_get_best_run_no_runs_returns_none():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                mock_client.search_runs.return_value = []

                tracker = ExperimentTracker()
                result = tracker.get_best_run('avg_confidence')

                assert result is None


@pytest.mark.unit
def test_get_best_run_ascending_order():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                mock_client.search_runs.return_value = [Mock(
                    info=Mock(run_id="run1", start_time=500),
                    data=Mock(tags={}, metrics={'inference_time_ms': 8.0})
                )]

                tracker = ExperimentTracker()
                tracker.get_best_run('inference_time_ms', ascending=True)

                mock_client.search_runs.assert_called_once_with(
                    experiment_ids=["111"],
                    order_by=["metrics.inference_time_ms ASC"]
                )


@pytest.mark.unit
def test_get_experiment_summary():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.get_experiment_by_name') as mock_get_exp:
            mock_get_exp.return_value = Mock(experiment_id="456")

            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value

                mock_runs = [
                    Mock(data=Mock(metrics={'avg_confidence': 0.8, 'inference_time_ms': 20})),
                    Mock(data=Mock(metrics={'avg_confidence': 0.9, 'inference_time_ms': 25}))
                ]
                mock_client.search_runs.return_value = mock_runs

                tracker = ExperimentTracker()
                summary = tracker.get_experiment_summary()

                assert summary['total_runs'] == 2
                assert summary['avg_confidence_mean'] == pytest.approx(0.85)
                assert summary['avg_confidence_max'] == pytest.approx(0.9)
                assert summary['inference_time_mean'] == pytest.approx(22.5)
                assert summary['inference_time_min'] == pytest.approx(20.0)


@pytest.mark.unit
def test_get_experiment_summary_no_runs():

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="111"):
            with patch('mlflow.tracking.MlflowClient') as MockClient:
                mock_client = MockClient.return_value
                mock_client.search_runs.return_value = []

                tracker = ExperimentTracker()
                summary = tracker.get_experiment_summary()

                assert summary == {'total_runs': 0}


@pytest.mark.unit
def test_get_tracker_singleton():

    import app.ml.experiment_tracker as tracker_module
    tracker_module._tracker_instance = None

    with patch('mlflow.set_tracking_uri'):
        with patch('mlflow.create_experiment', return_value="999"):
            t1 = get_tracker("http://localhost:5000", "singleton-test")
            t2 = get_tracker("http://localhost:5000", "singleton-test")

            assert t1 is t2

    tracker_module._tracker_instance = None


@pytest.mark.unit
def test_get_tracker_returns_existing_instance():

    import app.ml.experiment_tracker as tracker_module

    mock_instance = Mock(spec=ExperimentTracker)
    tracker_module._tracker_instance = mock_instance

    result = get_tracker()

    assert result is mock_instance

    tracker_module._tracker_instance = None