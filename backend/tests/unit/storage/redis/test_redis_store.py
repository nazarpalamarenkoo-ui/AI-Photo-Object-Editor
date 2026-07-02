import pytest
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

from app.storage.redis.redis_storage import RedisStorage


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.setex = AsyncMock()
    redis.get = AsyncMock()
    redis.delete = AsyncMock()
    redis.exists = AsyncMock(return_value=1)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def storage(mock_redis):
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        return RedisStorage()


@pytest.mark.unit
class TestRedisStorageInit:
    def test_init_sets_default_ttl(self, storage):
        assert storage.default_ttl == 3600

    def test_init_calls_from_url(self, mock_redis):
        with patch("redis.asyncio.from_url", return_value=mock_redis) as mock_from_url:
            RedisStorage()
            mock_from_url.assert_called_once()
            _, kwargs = mock_from_url.call_args
            assert kwargs["encoding"] == "utf8"
            assert kwargs["decode_responses"] is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestSetGetDeleteExists:
    async def test_set_calls_setex_with_given_ttl(self, storage, mock_redis):
        await storage.set("key", b"data", ttl=10)
        mock_redis.setex.assert_awaited_once_with("key", 10, b"data")

    async def test_set_uses_default_ttl_when_none(self, storage, mock_redis):
        await storage.set("key", b"data")
        mock_redis.setex.assert_awaited_once_with("key", storage.default_ttl, b"data")

    async def test_get_returns_value(self, storage, mock_redis):
        mock_redis.get.return_value = b"data"
        result = await storage.get("key")
        assert result == b"data"
        mock_redis.get.assert_awaited_once_with("key")

    async def test_get_returns_none_when_missing(self, storage, mock_redis):
        mock_redis.get.return_value = None
        result = await storage.get("key")
        assert result is None

    async def test_delete_forwards_all_keys(self, storage, mock_redis):
        await storage.delete("k1", "k2", "k3")
        mock_redis.delete.assert_awaited_once_with("k1", "k2", "k3")

    async def test_exists_true(self, storage, mock_redis):
        mock_redis.exists.return_value = 1
        result = await storage.exists("key")
        assert result is True

    async def test_exists_false(self, storage, mock_redis):
        mock_redis.exists.return_value = 0
        result = await storage.exists("key")
        assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestImageCache:
    async def test_cache_image_generates_default_key(self, storage, mock_redis):
        key = await storage.cache_image(1, b"img")
        assert key == "image:1:processed"

    async def test_cache_image_custom_suffix(self, storage, mock_redis):
        key = await storage.cache_image(1, b"thumb", suffix="thumbnail")
        assert key == "image:1:thumbnail"

    async def test_cache_image_stores_with_ttl(self, storage, mock_redis):
        await storage.cache_image(1, b"img", ttl=100)
        mock_redis.setex.assert_awaited_once_with("image:1:processed", 100, b"img")

    async def test_cache_image_default_ttl(self, storage, mock_redis):
        await storage.cache_image(1, b"img")
        mock_redis.setex.assert_awaited_once_with("image:1:processed", 7200, b"img")

    async def test_get_cache_image_default_suffix(self, storage, mock_redis):
        mock_redis.get.return_value = b"img"
        result = await storage.get_cache_image(1)
        assert result == b"img"
        mock_redis.get.assert_awaited_once_with("image:1:processed")

    async def test_get_cache_image_custom_suffix(self, storage, mock_redis):
        mock_redis.get.return_value = b"thumb"
        result = await storage.get_cache_image(1, suffix="thumbnail")
        assert result == b"thumb"
        mock_redis.get.assert_awaited_once_with("image:1:thumbnail")


@pytest.mark.unit
@pytest.mark.asyncio
class TestDetectionsCache:
    async def test_cache_detections_serializes_with_pickle(self, storage, mock_redis):
        detections = [{"a": 1}]
        await storage.cache_detections(1, detections)
        mock_redis.setex.assert_awaited_once()
        args, _ = mock_redis.setex.call_args
        assert args[0] == "detections:1"
        assert pickle.loads(args[2]) == detections

    async def test_cache_detections_default_ttl(self, storage, mock_redis):
        await storage.cache_detections(1, [{"a": 1}])
        args, _ = mock_redis.setex.call_args
        assert args[1] == 3600

    async def test_get_cached_detections_found(self, storage, mock_redis):
        data = [{"x": 1}]
        mock_redis.get.return_value = pickle.dumps(data)
        result = await storage.get_cached_detections(1)
        assert result == data

    async def test_get_cached_detections_missing(self, storage, mock_redis):
        mock_redis.get.return_value = None
        result = await storage.get_cached_detections(1)
        assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentsCache:
    async def test_cache_segments_serializes_with_pickle(self, storage, mock_redis):
        segments = [{"mask_id": 1}]
        await storage.cache_segments(1, segments)
        mock_redis.setex.assert_awaited_once()
        args, _ = mock_redis.setex.call_args
        assert args[0] == "segments:1"
        assert pickle.loads(args[2]) == segments

    async def test_cache_segments_default_ttl(self, storage, mock_redis):
        await storage.cache_segments(1, [{"mask_id": 1}])
        args, _ = mock_redis.setex.call_args
        assert args[1] == 7200

    async def test_get_cached_segments_found(self, storage, mock_redis):
        data = [{"mask_id": 1}]
        mock_redis.get.return_value = pickle.dumps(data)
        result = await storage.get_cached_segments(1)
        assert result == data

    async def test_get_cached_segments_missing(self, storage, mock_redis):
        mock_redis.get.return_value = None
        result = await storage.get_cached_segments(1)
        assert result is None

    async def test_invalidate_segments_calls_delete(self, storage, mock_redis):
        await storage.invalidate_segments(1)
        mock_redis.delete.assert_awaited_once_with("segments:1")


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvalidateImageAndClose:
    async def test_invalidate_image_deletes_all_related_keys(self, storage, mock_redis):
        await storage.invalidate_image(1)
        mock_redis.delete.assert_awaited_once_with(
            "image:1:processed",
            "image:1:thumbnail",
            "detections:1",
            "segments:1",
        )

    async def test_close_calls_redis_aclose(self, storage, mock_redis):
        await storage.close()
        mock_redis.aclose.assert_awaited_once()