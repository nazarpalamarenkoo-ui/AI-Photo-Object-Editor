from __future__ import annotations

from typing import Protocol

import structlog
from structlog.stdlib import BoundLogger


class ILoggerFactory(Protocol):
    def get_logger(self, name: str) -> BoundLogger: ...


class StructlogLoggerFactory:
    """Default production factory — thin wrapper over structlog."""

    def get_logger(self, name: str) -> BoundLogger:
        return structlog.get_logger(name)


_default_factory: ILoggerFactory = StructlogLoggerFactory()


def get_logger(name: str) -> BoundLogger:
    """Convenience function used everywhere in business/service code"""
    return _default_factory.get_logger(name)


def get_logger_factory() -> ILoggerFactory:
    """
    FastAPI dependency for classes that prefer constructor injection over
    the module-level convenience function — e.g."""
    return _default_factory