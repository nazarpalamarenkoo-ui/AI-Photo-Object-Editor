from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Literal

import structlog
from structlog.types import EventDict, WrappedLogger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """
    Logging-only settings, intentionally decoupled from `app.config.settings`.

    Integration options with your existing `Settings` class:
      a) simplest — keep this class as-is; it reads the same env vars
         (LOG_LEVEL, LOG_FORMAT, DEBUG, SERVICE_NAME, ENVIRONMENT) independently.
      b) tighter coupling — make your `Settings` inherit from this class,
         or add these fields to it and pass a `LoggingSettings(**subset)`
         into `configure_logging()` explicitly.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    SERVICE_NAME: str = Field(default="image-editor-api")
    ENVIRONMENT: str = Field(default="local")  # local | staging | production
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: Literal["json", "console"] = Field(default="json")
    DEBUG: bool = Field(default=False)


@lru_cache
def get_logging_settings() -> LoggingSettings:
    return LoggingSettings()


def _rename_event_to_message(
    _: WrappedLogger, __: str, event_dict: EventDict
) -> EventDict:
    """structlog's main log text lives in 'event' — the spec wants 'message'."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def _add_service_metadata(settings: LoggingSettings) -> structlog.types.Processor:
    def processor(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", settings.SERVICE_NAME)
        event_dict.setdefault("environment", settings.ENVIRONMENT)
        return event_dict

    return processor


_FIELD_ORDER = [
    "timestamp", "level", "logger", "service", "environment",
    "request_id", "worker_name", "job_id", "queue", "endpoint", "method",
    "status_code", "duration_ms", "user_id", "device", "model",
    "message", "exception",
]


def _order_keys(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """
    Stable, predictable JSON field order so log shippers can rely on it.
    Anything not in the preferred list (ad-hoc ML/business fields) is kept,
    just appended after the standard fields.
    """
    ordered: EventDict = {k: event_dict.pop(k) for k in _FIELD_ORDER if k in event_dict}
    ordered.update(event_dict)
    return ordered


def _level_to_int(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def _configure_logging_impl(settings: LoggingSettings) -> None:
    """
    Configure stdlib `logging` + `structlog` once, at process startup, using
    structlog's canonical stdlib-integration recipe: structlog builds the
    event dict, then hands off to a `ProcessorFormatter` running on a
    regular `logging.StreamHandler(stdout)`. """
    level = _level_to_int(settings.LOG_LEVEL)

    # Processors that run for EVERY log line, structlog or plain stdlib.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", key="timestamp", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_service_metadata(settings),
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    render_processor: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.LOG_FORMAT == "json"
        # Local-dev-only human-readable renderer. Still stdout, never a file.
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Runs only for records that DIDN'T come from a structlog logger.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _rename_event_to_message,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _order_keys,
            render_processor,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # We log request start/finish ourselves in RequestLoggingMiddleware
    # with request_id/duration/etc — disable uvicorn's plain-text duplicate.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """
    Public entry point. Call sites (exactly these two, nowhere else):
      - FastAPI: first line of `app/main.py`, before `FastAPI()` is created.
      - ARQ worker: first line of `app/worker.py` (module import time,
        since arq imports the module before running anything) — safe to
        call again in `WorkerSettings.on_startup`, it's idempotent.
    """
    _configure_logging_impl(settings or get_logging_settings())