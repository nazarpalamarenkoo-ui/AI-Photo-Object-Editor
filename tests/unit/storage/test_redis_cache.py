import pytest
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

from app.storage.redis_storage import RedisImageCache


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.setex = AsyncMock()
    redis.get = AsyncMock()
    redis.delete = AsyncMock()
    redis.exists = AsyncMock(return_value=1)
    redis.close = AsyncMock()
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
def cache(mock_redis):
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        return RedisImageCache()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_calls_redis(cache, mock_redis):
    await cache.set("key", b"data", ttl=10)
    mock_redis.setex.assert_awaited_once_with("key", 10, b"data")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_value(cache, mock_redis):
    mock_redis.get.return_value = b"data"
    result = await cache.get("key")
    assert result == b"data"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_true(cache, mock_redis):
    result = await cache.exists("key")
    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_image_generates_key(cache):
    key = await cache.cache_image(1, b"img")
    assert key == "image:1:processed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_detections_serialization(cache, mock_redis):
    detections = [{"a": 1}]
    await cache.cache_detections(1, detections)
    mock_redis.setex.assert_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_cached_detections(cache, mock_redis):
    data = [{"x": 1}]
    mock_redis.get.return_value = pickle.dumps(data)
    result = await cache.get_cached_detections(1)
    assert result == data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalidate_image_calls_delete(cache, mock_redis):
    await cache.invalidate_image(1)
    mock_redis.delete.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_calls_redis_close(cache, mock_redis):
    mock_redis.aclose = AsyncMock()
    await cache.close()
    mock_redis.aclose.assert_awaited_once()


# --- push_undo_state ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_undo_state_returns_length(cache, mock_redis):
    mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[1, None, True, None, 3])
    result = await cache.push_undo_state(1, b"img", "remove bbox_id=5")
    assert result == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_undo_state_clears_redo(cache, mock_redis):
    pipe = mock_redis.pipeline.return_value
    await cache.push_undo_state(1, b"img", "remove bbox_id=5")
    pipe.delete.assert_called_once_with("image:1:redo_stack")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_undo_state_uses_correct_key(cache, mock_redis):
    pipe = mock_redis.pipeline.return_value
    await cache.push_undo_state(42, b"img", "label")
    pipe.lpush.assert_called_once()
    args = pipe.lpush.call_args[0]
    assert args[0] == "image:42:undo_stack"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_undo_state_trims_to_max_history(cache, mock_redis):
    pipe = mock_redis.pipeline.return_value
    await cache.push_undo_state(1, b"img", "label")
    pipe.ltrim.assert_called_once_with("image:1:undo_stack", 0, RedisImageCache.MAX_HISTORY - 1)


# --- pop_undo_state ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_pop_undo_state_returns_dict(cache, mock_redis):
    entry = pickle.dumps({'bytes': b"img", 'label': 'remove bbox_id=1'})
    mock_redis.lpop.return_value = entry

    result = await cache.pop_undo_state(1)

    assert result == {'bytes': b"img", 'label': 'remove bbox_id=1'}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pop_undo_state_returns_none_when_empty(cache, mock_redis):
    mock_redis.lpop.return_value = None
    result = await cache.pop_undo_state(1)
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pop_undo_state_uses_correct_key(cache, mock_redis):
    mock_redis.lpop.return_value = None
    await cache.pop_undo_state(99)
    mock_redis.lpop.assert_awaited_once_with("image:99:undo_stack")


# --- push_redo_state ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_redo_state_uses_correct_key(cache, mock_redis):
    pipe = mock_redis.pipeline.return_value
    await cache.push_redo_state(1, b"img", "redo")
    pipe.lpush.assert_called_once()
    args = pipe.lpush.call_args[0]
    assert args[0] == "image:1:redo_stack"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_push_redo_state_trims_to_max_history(cache, mock_redis):
    pipe = mock_redis.pipeline.return_value
    await cache.push_redo_state(1, b"img", "redo")
    pipe.ltrim.assert_called_once_with("image:1:redo_stack", 0, RedisImageCache.MAX_HISTORY - 1)


# --- pop_redo_state ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_pop_redo_state_returns_dict(cache, mock_redis):
    entry = pickle.dumps({'bytes': b"img2", 'label': 'redo'})
    mock_redis.lpop.return_value = entry

    result = await cache.pop_redo_state(1)

    assert result == {'bytes': b"img2", 'label': 'redo'}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pop_redo_state_returns_none_when_empty(cache, mock_redis):
    mock_redis.lpop.return_value = None
    result = await cache.pop_redo_state(1)
    assert result is None


# --- get_history_labels ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_labels_returns_labels(cache, mock_redis):
    entries = [
        pickle.dumps({'bytes': b"a", 'label': 'remove bbox_id=1'}),
        pickle.dumps({'bytes': b"b", 'label': 'replace bbox_id=2'}),
    ]
    mock_redis.lrange.return_value = entries

    result = await cache.get_history_labels(1)

    assert result == ['remove bbox_id=1', 'replace bbox_id=2']


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_labels_empty(cache, mock_redis):
    mock_redis.lrange.return_value = []
    result = await cache.get_history_labels(1)
    assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_history_labels_skips_corrupt_entries(cache, mock_redis):
    valid = pickle.dumps({'bytes': b"a", 'label': 'valid'})
    mock_redis.lrange.return_value = [valid, b"corrupt_data"]

    result = await cache.get_history_labels(1)

    assert result == ['valid']


# --- clear_history ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_clear_history_deletes_both_stacks(cache, mock_redis):
    await cache.clear_history(1)
    mock_redis.delete.assert_awaited_once_with(
        "image:1:undo_stack",
        "image:1:redo_stack"
    )