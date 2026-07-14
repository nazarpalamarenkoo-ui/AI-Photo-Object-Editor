from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.logging.logger import get_logger

_logger = get_logger("app.ml")


@dataclass
class MLOperationHandle:
    """Returned by `async with log_ml_operation(...) as op:` — call
    `op.set_output(**fields)` any time before the block exits to attach
    output stats (num_detections, num_segments, mask_area_px, ...) to the
    `_finished` log line."""

    extra: dict[str, Any] = field(default_factory=dict)

    def set_output(self, **fields: Any) -> None:
        self.extra.update(fields)


class log_ml_operation:
    """Async context manager wrapping a single AI model invocation."""

    def __init__(
        self,
        operation: str,
        *,
        model: str,
        device: Optional[str] = None,
        image_size: tuple[int, int] | None = None,
        **extra_input_fields: Any,
    ):
        self.operation = operation
        self.model = model
        self.device = device
        self.image_size = image_size
        self.extra_input_fields = extra_input_fields
        self._start = 0.0
        self._handle = MLOperationHandle()

    async def __aenter__(self) -> MLOperationHandle:
        self._start = time.perf_counter()
        _logger.info(
            f"{self.operation}_started",
            model=self.model,
            device=self.device,
            image_size=self.image_size,
            **self.extra_input_fields,
        )
        return self._handle

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type is not None:
            _logger.error(
                f"{self.operation}_failed",
                model=self.model,
                device=self.device,
                duration_ms=duration_ms,
                exc_info=exc,
            )
            return False

        _logger.info(
            f"{self.operation}_finished",
            model=self.model,
            device=self.device,
            duration_ms=duration_ms,
            **self._handle.extra,
        )
        return False