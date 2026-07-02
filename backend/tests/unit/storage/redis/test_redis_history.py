import pytest
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

from app.storage.redis.redis_history import RedisHistory


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.delete = AsyncMock()
    redis.lpop = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.lpush = MagicMock()
    pipe.ltrim = MagicMock()
    pipe.expire = MagicMock()
    pipe.delete = MagicMock()
    pipe.llen = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, None, True, None, 1])
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


@pytest.fixture
def history(mock_redis):
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        return RedisHistory()


@pytest.mark.unit
def test_max_history_constant():
    assert RedisHistory.MAX_HISTORY == 10


@pytest.mark.unit
@pytest.mark.asyncio
class TestPushUndoState:
    async def test_returns_length_from_last_pipeline_result(self, history, mock_redis):
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[1, None, True, None, 3])
        result = await history.push_undo_state(1, b"img", "remove bbox_id=5")
        assert result == 3

    async def test_clears_redo_stack(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(1, b"img", "remove bbox_id=5")
        pipe.delete.assert_called_once_with("image:1:redo_stack")

    async def test_uses_correct_undo_key(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(42, b"img", "label")
        pipe.lpush.assert_called_once()
        args = pipe.lpush.call_args[0]
        assert args[0] == "image:42:undo_stack"

    async def test_serializes_bytes_and_label(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(1, b"img_bytes", "my_label")
        args = pipe.lpush.call_args[0]
        entry = pickle.loads(args[1])
        assert entry == {"bytes": b"img_bytes", "label": "my_label"}

    async def test_trims_to_max_history(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(1, b"img", "label")
        pipe.ltrim.assert_called_once_with("image:1:undo_stack", 0, RedisHistory.MAX_HISTORY - 1)

    async def test_sets_expire_with_default_ttl(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(1, b"img", "label")
        pipe.expire.assert_called_once_with("image:1:undo_stack", 7200)

    async def test_sets_expire_with_custom_ttl(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_undo_state(1, b"img", "label", ttl=500)
        pipe.expire.assert_called_once_with("image:1:undo_stack", 500)


@pytest.mark.unit
@pytest.mark.asyncio
class TestPopUndoState:
    async def test_returns_deserialized_dict(self, history, mock_redis):
        entry = pickle.dumps({"bytes": b"img", "label": "remove bbox_id=1"})
        mock_redis.lpop.return_value = entry

        result = await history.pop_undo_state(1)

        assert result == {"bytes": b"img", "label": "remove bbox_id=1"}

    async def test_returns_none_when_empty(self, history, mock_redis):
        mock_redis.lpop.return_value = None
        result = await history.pop_undo_state(1)
        assert result is None

    async def test_uses_correct_key(self, history, mock_redis):
        mock_redis.lpop.return_value = None
        await history.pop_undo_state(99)
        mock_redis.lpop.assert_awaited_once_with("image:99:undo_stack")


@pytest.mark.unit
@pytest.mark.asyncio
class TestPushRedoState:
    async def test_uses_correct_redo_key(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_redo_state(1, b"img", "redo")
        pipe.lpush.assert_called_once()
        args = pipe.lpush.call_args[0]
        assert args[0] == "image:1:redo_stack"

    async def test_serializes_bytes_and_label(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_redo_state(1, b"redo_bytes", "redo_label")
        args = pipe.lpush.call_args[0]
        entry = pickle.loads(args[1])
        assert entry == {"bytes": b"redo_bytes", "label": "redo_label"}

    async def test_trims_to_max_history(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_redo_state(1, b"img", "redo")
        pipe.ltrim.assert_called_once_with("image:1:redo_stack", 0, RedisHistory.MAX_HISTORY - 1)

    async def test_sets_expire_with_default_ttl(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_redo_state(1, b"img", "redo")
        pipe.expire.assert_called_once_with("image:1:redo_stack", 7200)

    async def test_does_not_touch_undo_stack(self, history, mock_redis):
        pipe = mock_redis.pipeline.return_value
        await history.push_redo_state(1, b"img", "redo")
        pipe.delete.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
class TestPopRedoState:
    async def test_returns_deserialized_dict(self, history, mock_redis):
        entry = pickle.dumps({"bytes": b"img2", "label": "redo"})
        mock_redis.lpop.return_value = entry

        result = await history.pop_redo_state(1)

        assert result == {"bytes": b"img2", "label": "redo"}

    async def test_returns_none_when_empty(self, history, mock_redis):
        mock_redis.lpop.return_value = None
        result = await history.pop_redo_state(1)
        assert result is None

    async def test_uses_correct_key(self, history, mock_redis):
        mock_redis.lpop.return_value = None
        await history.pop_redo_state(7)
        mock_redis.lpop.assert_awaited_once_with("image:7:redo_stack")


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetHistoryLabels:
    async def test_returns_labels_in_order(self, history, mock_redis):
        entries = [
            pickle.dumps({"bytes": b"a", "label": "remove bbox_id=1"}),
            pickle.dumps({"bytes": b"b", "label": "replace bbox_id=2"}),
        ]
        mock_redis.lrange.return_value = entries

        result = await history.get_history_labels(1)

        assert result == ["remove bbox_id=1", "replace bbox_id=2"]

    async def test_empty_stack_returns_empty_list(self, history, mock_redis):
        mock_redis.lrange.return_value = []
        result = await history.get_history_labels(1)
        assert result == []

    async def test_skips_corrupt_entries(self, history, mock_redis):
        valid = pickle.dumps({"bytes": b"a", "label": "valid"})
        mock_redis.lrange.return_value = [valid, b"corrupt_data"]

        result = await history.get_history_labels(1)

        assert result == ["valid"]

    async def test_uses_correct_key_and_full_range(self, history, mock_redis):
        mock_redis.lrange.return_value = []
        await history.get_history_labels(5)
        mock_redis.lrange.assert_awaited_once_with("image:5:undo_stack", 0, -1)


@pytest.mark.unit
@pytest.mark.asyncio
class TestClearHistory:
    async def test_deletes_both_stacks(self, history, mock_redis):
        await history.clear_history(1)
        mock_redis.delete.assert_awaited_once_with(
            "image:1:undo_stack",
            "image:1:redo_stack",
        )