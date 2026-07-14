from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import structlog


def new_request_id() -> str:
    return str(uuid.uuid4())


def bind_request_context(
    *,
    request_id: str,
    method: str | None = None,
    endpoint: str | None = None,
    user_id: int | str | None = None,
    **extra: Any,
) -> None:
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=method,
        endpoint=endpoint,
        user_id=user_id,
        **extra,
    )


def bind_worker_context(
    *,
    job_id: str,
    worker_name: str,
    queue: str | None = None,
    **extra: Any,
) -> None:
    structlog.contextvars.bind_contextvars(
        job_id=job_id,
        worker_name=worker_name,
        queue=queue,
        **extra,
    )


def bind_user(user_id: int | str) -> None:
    """
    Call once you know who the request belongs to — e.g. right after
    `get_current_user` resolves — so subsequent logs in the same request
    carry `user_id` too.
    """
    structlog.contextvars.bind_contextvars(user_id=user_id)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


@contextmanager
def request_context(**fields: Any) -> Iterator[None]:
    """Generic scoped-binding helper for use outside HTTP/worker contexts"""
    tokens = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)