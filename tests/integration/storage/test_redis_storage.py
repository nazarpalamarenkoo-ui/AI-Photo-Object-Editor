import pytest
import asyncio
import fakeredis.aioredis

from app.storage.redis_storage import RedisImageCache


@pytest.fixture
def redis_cache(monkeypatch):
    fake_redis = fakeredis.aioredis.FakeRedis()

    monkeypatch.setattr(
        "redis.asyncio.from_url",
        lambda *args, **kwargs: fake_redis
    )

    return RedisImageCache()


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_flow(redis_cache):
    cache = redis_cache

    key = await cache.cache_image(1, b"img")
    assert key == "image:1:processed"

    result = await cache.get_cache_image(1)
    assert result == b"img"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_overwrite(redis_cache):
    cache = redis_cache

    await cache.cache_image(1, b"old")
    await cache.cache_image(1, b"new")

    result = await cache.get_cache_image(1)
    assert result == b"new"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttl_expiration(redis_cache):
    cache = redis_cache

    await cache.set("k", b"data", ttl=1)

    await asyncio.sleep(1.1)

    result = await cache.get("k")
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_partial(redis_cache):
    cache = redis_cache

    await cache.cache_image(1, b"processed", "processed")
    await cache.cache_image(1, b"thumb", "thumbnail")
    await cache.cache_detections(1, [{"a": 1}])

    await cache.invalidate_image(1)

    assert await cache.get_cache_image(1, "processed") is None
    assert await cache.get_cache_image(1, "thumbnail") is None
    assert await cache.get_cached_detections(1) is None