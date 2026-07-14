from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from app.core.logging.logger import (
    ILoggerFactory,
    StructlogLoggerFactory,
    get_logger,
    get_logger_factory,
    _default_factory,
)

pytestmark = pytest.mark.unit
class TestStructlogLoggerFactory:
    def test_get_logger_delegates_to_structlog_with_given_name(self):
        factory = StructlogLoggerFactory()
        sentinel = MagicMock(name="sentinel-logger")
        with patch("app.core.logging.logger.structlog.get_logger", return_value=sentinel) as mock_get:
            result = factory.get_logger("app.my_module")

        mock_get.assert_called_once_with("app.my_module")
        assert result is sentinel

    def test_get_logger_returns_a_usable_logger_object(self):
        factory = StructlogLoggerFactory()
        logger = factory.get_logger("app.smoke")
        assert logger is not None

    def test_satisfies_ilogger_factory_shape(self):
        factory = StructlogLoggerFactory()
        assert hasattr(factory, "get_logger")
        assert callable(factory.get_logger)


class TestModuleLevelHelpers:
    def test_get_logger_convenience_function_delegates_to_default_factory(self):
        sentinel = MagicMock(name="sentinel-logger")
        with patch.object(_default_factory, "get_logger", return_value=sentinel) as mock_get:
            result = get_logger("app.convenience")

        mock_get.assert_called_once_with("app.convenience")
        assert result is sentinel

    def test_get_logger_factory_returns_ilogger_factory_instance(self):
        factory = get_logger_factory()
        assert isinstance(factory, StructlogLoggerFactory)
        assert hasattr(factory, "get_logger")

    def test_get_logger_factory_returns_the_module_singleton(self):
        assert get_logger_factory() is get_logger_factory()
        assert get_logger_factory() is _default_factory

    def test_different_names_produce_independent_structlog_calls(self):
        with patch("app.core.logging.logger.structlog.get_logger") as mock_get:
            get_logger("app.a")
            get_logger("app.b")

        assert mock_get.call_args_list == [(("app.a",), {}), (("app.b",), {})]