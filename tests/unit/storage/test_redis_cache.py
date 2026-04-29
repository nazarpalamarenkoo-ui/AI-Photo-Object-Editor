import pytest
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
    import pickle

    data = [{"x": 1}]
    mock_redis.get.return_value = pickle.dumps(data)

    result = await cache.get_cached_detections(1)

    assert result == data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalidate_image_calls_delete(cache, mock_redis):
    await cache.invalidate_image(1)

    assert mock_redis.delete.await_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_calls_redis_close(cache, mock_redis):
    await cache.close()
    mock_redis.close.assert_awaited_once()