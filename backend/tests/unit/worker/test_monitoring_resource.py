import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.workers.worker import _resource_monitor


pytestmark = pytest.mark.unit


async def _run_monitor_iterations(iterations: int, **kwargs):
    """
    Drives _resource_monitor for exactly `iterations` loop passes, then
    stops the infinite loop cleanly by having asyncio.sleep raise
    CancelledError on the (iterations + 1)-th call. Returns the sleep
    mock so callers can assert on how sleep was invoked (e.g. interval).
    """
    sleep_mock = AsyncMock(side_effect=[None] * iterations + [asyncio.CancelledError()])
    with patch("app.workers.worker.asyncio.sleep", sleep_mock):
        with pytest.raises(asyncio.CancelledError):
            await _resource_monitor(**kwargs)
    return sleep_mock


@pytest.mark.asyncio
class TestResourceMonitorLoopMechanics:
    async def test_uses_default_interval_seconds(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger"):
            sleep_mock = await _run_monitor_iterations(1)

        sleep_mock.assert_any_call(60)

    async def test_respects_custom_interval_seconds(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger"):
            sleep_mock = await _run_monitor_iterations(1, interval_seconds=5)

        sleep_mock.assert_any_call(5)

    async def test_increments_iteration_counter_each_loop(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(3)

        iterations_logged = [
            call.kwargs["iteration"] for call in mock_logger.warning.call_args_list
        ]
        assert iterations_logged == [1, 2, 3]

    async def test_logs_warning_event_name_each_iteration(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(2)

        assert mock_logger.warning.call_count == 2
        for call in mock_logger.warning.call_args_list:
            assert call.args == ("worker_resource_snapshot",)


@pytest.mark.asyncio
class TestResourceMonitorThreadGrouping:
    async def test_groups_thread_names_by_base_pattern(self):
        thread_names = [
            "MainThread",
            "ThreadPoolExecutor-0_0",
            "ThreadPoolExecutor-0_1",
            "ThreadPoolExecutor-1_0",
            "asyncio_0",
            "asyncio_1",
        ]
        fake_threads = []
        for name in thread_names:
            t = MagicMock()
            t.name = name
            fake_threads.append(t)

        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=fake_threads
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["num_threads"] == 6
        # Note: the base-name logic strips BOTH a trailing "_N" and a
        # trailing "-N" suffix, so "ThreadPoolExecutor-0_0",
        # "ThreadPoolExecutor-0_1" and "ThreadPoolExecutor-1_0" all
        # collapse into a single "ThreadPoolExecutor" group - pool index
        # is not preserved, only the "_worker index" and "-pool index"
        # suffixes are stripped away.
        assert log_fields["thread_groups"] == {
            "MainThread": 1,
            "ThreadPoolExecutor": 3,
            "asyncio": 2,
        }

    async def test_strips_both_underscore_and_dash_suffixes(self):
        """
        Documents the exact two-stage stripping behaviour:
        rsplit("_", 1)[0] removes a trailing "_<suffix>",
        then rsplit("-", 1)[0] removes a trailing "-<suffix>" from what's left.
        """
        cases = {
            "ThreadPoolExecutor-0_0": "ThreadPoolExecutor",
            "ThreadPoolExecutor-3_12": "ThreadPoolExecutor",
            "asyncio_0": "asyncio",
            "MainThread": "MainThread",
            "worker-5": "worker",
            "plain_name_7": "plain_name",
            "a-b_c": "a",
        }
        fake_threads = []
        for name in cases:
            t = MagicMock()
            t.name = name
            fake_threads.append(t)

        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=fake_threads
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        expected_groups = {}
        for base in cases.values():
            expected_groups[base] = expected_groups.get(base, 0) + 1
        assert log_fields["thread_groups"] == expected_groups

    async def test_handles_no_threads(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["num_threads"] == 0
        assert log_fields["thread_groups"] == {}


@pytest.mark.asyncio
class TestResourceMonitorPsutilBranch:
    def _make_process_mock(self, *, has_num_fds=True, has_net_connections=True):
        process = MagicMock()
        mem = MagicMock()
        mem.rss = 200 * 1024 * 1024
        mem.vms = 500 * 1024 * 1024
        process.memory_info.return_value = mem
        process.open_files.return_value = [MagicMock(), MagicMock()]

        if has_num_fds:
            process.num_fds.return_value = 42
        else:
            del process.num_fds

        if has_net_connections:
            process.net_connections.return_value = [MagicMock(), MagicMock(), MagicMock()]
        else:
            del process.net_connections

        return process

    async def test_omits_process_metrics_when_psutil_unavailable(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        for key in ("rss_mb", "vms_mb", "num_fds", "open_files", "connections"):
            assert key not in log_fields

    async def test_includes_process_metrics_when_psutil_available(self):
        process = self._make_process_mock()

        with patch("app.workers.worker._HAS_PSUTIL", True), patch(
            "app.workers.worker.psutil.Process", return_value=process
        ), patch("app.workers.worker.threading.enumerate", return_value=[]), patch(
            "app.workers.worker.logger"
        ) as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["rss_mb"] == 200.0
        assert log_fields["vms_mb"] == 500.0
        assert log_fields["num_fds"] == 42
        assert log_fields["open_files"] == 2
        assert log_fields["connections"] == 3

    async def test_num_fds_is_none_when_platform_unsupported(self):
        process = self._make_process_mock(has_num_fds=False)

        with patch("app.workers.worker._HAS_PSUTIL", True), patch(
            "app.workers.worker.psutil.Process", return_value=process
        ), patch("app.workers.worker.threading.enumerate", return_value=[]), patch(
            "app.workers.worker.logger"
        ) as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["num_fds"] is None

    async def test_connections_is_none_when_platform_unsupported(self):
        process = self._make_process_mock(has_net_connections=False)

        with patch("app.workers.worker._HAS_PSUTIL", True), patch(
            "app.workers.worker.psutil.Process", return_value=process
        ), patch("app.workers.worker.threading.enumerate", return_value=[]), patch(
            "app.workers.worker.logger"
        ) as mock_logger:
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["connections"] is None

    async def test_process_created_once_at_startup_not_per_iteration(self):
        process = self._make_process_mock()

        with patch("app.workers.worker._HAS_PSUTIL", True), patch(
            "app.workers.worker.psutil.Process", return_value=process
        ) as mock_process_cls, patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger"):
            await _run_monitor_iterations(3)

        # psutil.Process() is only constructed once, before the loop starts
        mock_process_cls.assert_called_once_with()
        assert process.memory_info.call_count == 3


@pytest.mark.asyncio
class TestResourceMonitorCudaBranch:
    async def test_includes_cuda_metrics_when_available(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024
        fake_torch.cuda.memory_reserved.return_value = 300 * 1024 * 1024

        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger, patch.dict(
            sys.modules, {"torch": fake_torch}
        ):
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert log_fields["cuda_allocated_mb"] == 100.0
        assert log_fields["cuda_reserved_mb"] == 300.0

    async def test_omits_cuda_metrics_when_cuda_not_available(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger, patch.dict(
            sys.modules, {"torch": fake_torch}
        ):
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert "cuda_allocated_mb" not in log_fields
        assert "cuda_reserved_mb" not in log_fields

    async def test_swallows_missing_torch_dependency(self):
        # Setting sys.modules["torch"] = None makes `import torch` raise ImportError,
        # simulating an environment where torch is not installed.
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger, patch.dict(
            sys.modules, {"torch": None}
        ):
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert "cuda_allocated_mb" not in log_fields
        assert "cuda_reserved_mb" not in log_fields
        # loop must not have crashed - warning was still logged
        assert mock_logger.warning.call_count == 1

    async def test_swallows_arbitrary_exception_from_torch_calls(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.side_effect = RuntimeError("driver not found")

        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger, patch.dict(
            sys.modules, {"torch": fake_torch}
        ):
            await _run_monitor_iterations(1)

        log_fields = mock_logger.warning.call_args_list[0].kwargs
        assert "cuda_allocated_mb" not in log_fields
        assert mock_logger.warning.call_count == 1


@pytest.mark.asyncio
class TestResourceMonitorCancellation:
    async def test_propagates_cancellation_without_logging_extra_snapshot(self):
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger") as mock_logger:
            await _run_monitor_iterations(2)

        # exactly 2 snapshots logged, none after the CancelledError-raising sleep
        assert mock_logger.warning.call_count == 2

    async def test_can_be_cancelled_via_real_task(self):
        """
        Sanity check against the real asyncio scheduler (no sleep mocking):
        creating the coroutine as a task and cancelling it mid-sleep should
        raise CancelledError and not hang.
        """
        with patch("app.workers.worker._HAS_PSUTIL", False), patch(
            "app.workers.worker.threading.enumerate", return_value=[]
        ), patch("app.workers.worker.logger"):
            task = asyncio.create_task(_resource_monitor(interval_seconds=3600))
            await asyncio.sleep(0)  # let it reach the sleep
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task