from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable

from app.core.logging.logger import get_logger

_default_logger = get_logger(__name__)


class log_execution:
    def __init__(
        self,
        operation: str,
        *,
        logger: Any = None,
        level: str = "info",
        **extra_fields: Any,
    ):
        self.operation = operation
        self.logger = logger or _default_logger
        self.level = level
        self.extra_fields = extra_fields
        self._start: float = 0.0

    def __enter__(self) -> "log_execution":
        self._start = time.perf_counter()
        getattr(self.logger, self.level)(f"{self.operation}_started", **self.extra_fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type is not None:
            self.logger.error(
                f"{self.operation}_failed",
                duration_ms=duration_ms,
                exc_info=exc,
                **self.extra_fields,
            )
            return False
        getattr(self.logger, self.level)(
            f"{self.operation}_finished", duration_ms=duration_ms, **self.extra_fields
        )
        return False

    async def __aenter__(self) -> "log_execution":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def __call__(self, func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                async with log_execution(
                    self.operation, logger=self.logger, level=self.level, **self.extra_fields
                ):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            with log_execution(
                self.operation, logger=self.logger, level=self.level, **self.extra_fields
            ):
                return func(*args, **kwargs)

        return sync_wrapper