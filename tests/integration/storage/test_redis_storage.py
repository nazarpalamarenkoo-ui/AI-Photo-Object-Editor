import pytest
import asyncio
import pickle
import fakeredis.aioredis

from app.storage.redis_storage import RedisImageCache


@pytest.fixture
def redis_cache(monkeypatch):
    fake_redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: fake_redis)
    return RedisImageCache()


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_flow(redis_cache):
    key = await redis_cache.cache_image(1, b"img")
    assert key == "image:1:processed"
    result = await redis_cache.get_cache_image(1)
    assert result == b"img"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_overwrite(redis_cache):
    await redis_cache.cache_image(1, b"old")
    await redis_cache.cache_image(1, b"new")
    result = await redis_cache.get_cache_image(1)
    assert result == b"new"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttl_expiration(redis_cache):
    await redis_cache.set("k", b"data", ttl=1)
    await asyncio.sleep(1.1)
    result = await redis_cache.get("k")
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_partial(redis_cache):
    await redis_cache.cache_image(1, b"processed", "processed")
    await redis_cache.cache_image(1, b"thumb", "thumbnail")
    await redis_cache.cache_detections(1, [{"a": 1}])
    await redis_cache.invalidate_image(1)
    assert await redis_cache.get_cache_image(1, "processed") is None
    assert await redis_cache.get_cache_image(1, "thumbnail") is None
    assert await redis_cache.get_cached_detections(1) is None


# --- push/pop undo ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_undo_state_and_pop(redis_cache):
    length = await redis_cache.push_undo_state(1, b"img_bytes", "remove bbox_id=5")
    assert length == 1

    result = await redis_cache.pop_undo_state(1)
    assert result == {'bytes': b"img_bytes", 'label': "remove bbox_id=5"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_undo_state_clears_redo(redis_cache):
    await redis_cache.push_redo_state(1, b"redo_bytes", "redo")
    await redis_cache.push_undo_state(1, b"img", "remove bbox_id=1")

    result = await redis_cache.pop_redo_state(1)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_stack_max_history(redis_cache):
    for i in range(RedisImageCache.MAX_HISTORY + 3):
        await redis_cache.push_undo_state(1, b"img", f"op_{i}")

    labels = await redis_cache.get_history_labels(1)
    assert len(labels) == RedisImageCache.MAX_HISTORY


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_stack_lifo_order(redis_cache):
    await redis_cache.push_undo_state(1, b"a", "first")
    await redis_cache.push_undo_state(1, b"b", "second")
    await redis_cache.push_undo_state(1, b"c", "third")

    result = await redis_cache.pop_undo_state(1)
    assert result['label'] == "third"

    result = await redis_cache.pop_undo_state(1)
    assert result['label'] == "second"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pop_undo_empty_returns_none(redis_cache):
    result = await redis_cache.pop_undo_state(999)
    assert result is None


# --- push/pop redo ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_redo_state_and_pop(redis_cache):
    await redis_cache.push_redo_state(1, b"redo_img", "redo")
    result = await redis_cache.pop_redo_state(1)
    assert result == {'bytes': b"redo_img", 'label': "redo"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pop_redo_empty_returns_none(redis_cache):
    result = await redis_cache.pop_redo_state(999)
    assert result is None


# --- get_history_labels ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_labels_order(redis_cache):
    await redis_cache.push_undo_state(1, b"a", "op_1")
    await redis_cache.push_undo_state(1, b"b", "op_2")
    await redis_cache.push_undo_state(1, b"c", "op_3")

    labels = await redis_cache.get_history_labels(1)
    # newest first (LPUSH = stack)
    assert labels == ["op_3", "op_2", "op_1"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_labels_empty(redis_cache):
    labels = await redis_cache.get_history_labels(999)
    assert labels == []


# --- clear_history ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_history(redis_cache):
    await redis_cache.push_undo_state(1, b"a", "op_1")
    await redis_cache.push_redo_state(1, b"b", "redo")

    await redis_cache.clear_history(1)

    assert await redis_cache.pop_undo_state(1) is None
    assert await redis_cache.pop_redo_state(1) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_history_does_not_affect_other_images(redis_cache):
    await redis_cache.push_undo_state(1, b"a", "op_1")
    await redis_cache.push_undo_state(2, b"b", "op_2")

    await redis_cache.clear_history(1)

    labels = await redis_cache.get_history_labels(2)
    assert labels == ["op_2"]


# --- invalidate включає history ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_image_clears_history(redis_cache):
    await redis_cache.push_undo_state(1, b"a", "op_1")
    await redis_cache.push_redo_state(1, b"b", "redo")

    await redis_cache.invalidate_image(1)

    assert await redis_cache.pop_undo_state(1) is None
    assert await redis_cache.pop_redo_state(1) is None