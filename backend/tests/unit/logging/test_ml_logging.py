from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.logging.mllogging import MLOperationHandle, log_ml_operation

pytestmark = pytest.mark.unit

class TestMLOperationHandle:
    def test_starts_with_empty_extra(self):
        handle = MLOperationHandle()
        assert handle.extra == {}

    def test_set_output_stores_fields(self):
        handle = MLOperationHandle()
        handle.set_output(num_detections=3, mask_area_px=1024)
        assert handle.extra == {"num_detections": 3, "mask_area_px": 1024}

    def test_set_output_called_multiple_times_merges_fields(self):
        handle = MLOperationHandle()
        handle.set_output(num_detections=3)
        handle.set_output(mask_area_px=1024)
        assert handle.extra == {"num_detections": 3, "mask_area_px": 1024}

    def test_set_output_overwrites_existing_key(self):
        handle = MLOperationHandle()
        handle.set_output(num_detections=3)
        handle.set_output(num_detections=5)
        assert handle.extra == {"num_detections": 5}

    def test_two_handles_do_not_share_state(self):
        # Guards against the classic mutable-default-argument bug, since
        # `extra` uses a dataclass field(default_factory=dict).
        a = MLOperationHandle()
        b = MLOperationHandle()
        a.set_output(x=1)
        assert b.extra == {}


@pytest.fixture
def mock_logger():
    with patch("app.core.logging.mllogging._logger") as mocked:
        yield mocked


class TestLogMlOperationSuccess:
    @pytest.mark.asyncio
    async def test_logs_started_event_on_entry(self, mock_logger):
        async with log_ml_operation("segment", model="sam2", device="cuda", image_size=(640, 480)):
            pass

        mock_logger.info.assert_any_call(
            "segment_started", model="sam2", device="cuda", image_size=(640, 480)
        )

    @pytest.mark.asyncio
    async def test_logs_finished_event_on_clean_exit(self, mock_logger):
        async with log_ml_operation("segment", model="sam2", device="cuda"):
            pass

        finished_calls = [
            c for c in mock_logger.info.call_args_list if c.args[0] == "segment_finished"
        ]
        assert len(finished_calls) == 1
        _, kwargs = finished_calls[0]
        assert kwargs["model"] == "sam2"
        assert kwargs["device"] == "cuda"
        assert "duration_ms" in kwargs

    @pytest.mark.asyncio
    async def test_yields_a_handle_that_attaches_output_fields_to_finished_log(self, mock_logger):
        async with log_ml_operation("detect", model="yolo") as op:
            assert isinstance(op, MLOperationHandle)
            op.set_output(num_detections=7)

        finished_call = next(
            c for c in mock_logger.info.call_args_list if c.args[0] == "detect_finished"
        )
        assert finished_call.kwargs["num_detections"] == 7

    @pytest.mark.asyncio
    async def test_extra_input_fields_are_passed_through_to_started_log(self, mock_logger):
        async with log_ml_operation("detect", model="yolo", confidence_threshold=0.5):
            pass

        mock_logger.info.assert_any_call(
            "detect_started",
            model="yolo",
            device=None,
            image_size=None,
            confidence_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_does_not_log_error_on_clean_exit(self, mock_logger):
        async with log_ml_operation("detect", model="yolo"):
            pass
        mock_logger.error.assert_not_called()


class TestLogMlOperationFailure:
    @pytest.mark.asyncio
    async def test_logs_failed_event_with_exc_info_when_exception_raised(self, mock_logger):
        exc = RuntimeError("cuda oom")
        with pytest.raises(RuntimeError):
            async with log_ml_operation("segment", model="sam2", device="cuda"):
                raise exc

        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert args[0] == "segment_failed"
        assert kwargs["model"] == "sam2"
        assert kwargs["device"] == "cuda"
        assert kwargs["exc_info"] is exc
        assert "duration_ms" in kwargs

    @pytest.mark.asyncio
    async def test_exception_propagates_out_of_the_context_manager(self, mock_logger):
        with pytest.raises(ValueError, match="bad input"):
            async with log_ml_operation("detect", model="yolo"):
                raise ValueError("bad input")

    @pytest.mark.asyncio
    async def test_does_not_log_finished_event_when_exception_raised(self, mock_logger):
        with pytest.raises(RuntimeError):
            async with log_ml_operation("segment", model="sam2"):
                raise RuntimeError("boom")

        finished_calls = [
            c for c in mock_logger.info.call_args_list if c.args[0] == "segment_finished"
        ]
        assert finished_calls == []

    @pytest.mark.asyncio
    async def test_output_set_before_failure_is_not_lost_but_also_not_required(self, mock_logger):
        # set_output data is only wired into the *_finished log; on failure it's
        # simply not emitted. This test documents that behavior.
        with pytest.raises(RuntimeError):
            async with log_ml_operation("segment", model="sam2") as op:
                op.set_output(num_detections=1)
                raise RuntimeError("boom")

        mock_logger.error.assert_called_once()
        assert "num_detections" not in mock_logger.error.call_args.kwargs