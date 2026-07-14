from __future__ import annotations

import functools
import time
from typing import Any, Callable

from app.core.logging.context import bind_worker_context, clear_context
from app.core.logging.logger import get_logger

_logger = get_logger("app.worker")


def log_job(queue: str = "arq:queue") -> Callable[[Callable], Callable]:
    """Decorator factory for arq task functions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(ctx: dict, *args: Any, **kwargs: Any) -> Any:
            job_id = str(ctx.get("job_id", "unknown"))
            worker_name = str(ctx.get("worker_name") or ctx.get("job_id", "worker"))
            bind_worker_context(job_id=job_id, worker_name=worker_name, queue=queue)

            start = time.perf_counter()
            _logger.info("job_started", task=func.__name__, job_try=ctx.get("job_try"))
            try:
                result = await func(ctx, *args, **kwargs)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                _logger.error(
                    "job_failed", task=func.__name__, duration_ms=duration_ms, exc_info=exc
                )
                raise
            else:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                _logger.info("job_finished", task=func.__name__, duration_ms=duration_ms)
                return result
            finally:
                clear_context()

        return wrapper

    return decorator