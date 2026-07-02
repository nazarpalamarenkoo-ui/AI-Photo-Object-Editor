import pytest
import asyncio
import fakeredis.aioredis

from app.storage.redis.redis_storage import RedisStorage
from app.storage.redis.redis_history import RedisHistory


@pytest.fixture
def fake_redis_client():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def storage(monkeypatch, fake_redis_client):
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: fake_redis_client)
    return RedisStorage()


@pytest.fixture
def history(monkeypatch, fake_redis_client):
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: fake_redis_client)
    return RedisHistory()

@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_set_and_get_roundtrip(storage):
    await storage.set("k", b"data", ttl=60)
    result = await storage.get("k")
    assert result == b"data"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_missing_key_returns_none(storage):
    result = await storage.get("missing")
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_delete_removes_key(storage):
    await storage.set("k", b"data")
    await storage.delete("k")
    result = await storage.get("k")
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_exists_true_and_false(storage):
    await storage.set("k", b"data")
    assert await storage.exists("k") is True
    assert await storage.exists("missing") is False


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_ttl_expiration(storage):
    await storage.set("k", b"data", ttl=1)
    await asyncio.sleep(1.1)
    result = await storage.get("k")
    assert result is None

@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_flow(storage):
    key = await storage.cache_image(1, b"img")
    assert key == "image:1:processed"
    result = await storage.get_cache_image(1)
    assert result == b"img"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_overwrite(storage):
    await storage.cache_image(1, b"old")
    await storage.cache_image(1, b"new")
    result = await storage.get_cache_image(1)
    assert result == b"new"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_custom_suffix_isolated(storage):
    await storage.cache_image(1, b"processed_bytes", suffix="processed")
    await storage.cache_image(1, b"thumb_bytes", suffix="thumbnail")

    assert await storage.get_cache_image(1, "processed") == b"processed_bytes"
    assert await storage.get_cache_image(1, "thumbnail") == b"thumb_bytes"

@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_detections_roundtrip(storage):
    detections = [{"class": "person", "confidence": 0.9}]
    await storage.cache_detections(1, detections)
    result = await storage.get_cached_detections(1)
    assert result == detections


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_cached_detections_missing_returns_none(storage):
    result = await storage.get_cached_detections(999)
    assert result is None

@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_segments_roundtrip(storage):
    segments = [{"mask_id": 1, "area": 500}]
    await storage.cache_segments(1, segments)
    result = await storage.get_cached_segments(1)
    assert result == segments


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_cached_segments_missing_returns_none(storage):
    result = await storage.get_cached_segments(999)
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_invalidate_segments_removes_only_segments(storage):
    await storage.cache_segments(1, [{"mask_id": 1}])
    await storage.cache_detections(1, [{"class": "person"}])

    await storage.invalidate_segments(1)

    assert await storage.get_cached_segments(1) is None
    assert await storage.get_cached_detections(1) == [{"class": "person"}]

@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_invalidate_image_clears_processed_thumbnail_detections_segments(storage):
    await storage.cache_image(1, b"processed", "processed")
    await storage.cache_image(1, b"thumb", "thumbnail")
    await storage.cache_detections(1, [{"a": 1}])
    await storage.cache_segments(1, [{"mask_id": 1}])

    await storage.invalidate_image(1)

    assert await storage.get_cache_image(1, "processed") is None
    assert await storage.get_cache_image(1, "thumbnail") is None
    assert await storage.get_cached_detections(1) is None
    assert await storage.get_cached_segments(1) is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_invalidate_image_does_not_touch_other_images(storage):
    await storage.cache_image(1, b"a")
    await storage.cache_image(2, b"b")

    await storage.invalidate_image(1)

    assert await storage.get_cache_image(1) is None
    assert await storage.get_cache_image(2) == b"b"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_invalidate_image_does_not_clear_undo_redo_history(storage, history):
    await history.push_undo_state(1, b"a", "op_1")
    await history.push_redo_state(1, b"b", "redo")

    await storage.invalidate_image(1)

    assert await history.pop_undo_state(1) == {"bytes": b"a", "label": "op_1"}
    assert await history.pop_redo_state(1) == {"bytes": b"b", "label": "redo"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_undo_state_and_pop(history):
    length = await history.push_undo_state(1, b"img_bytes", "remove bbox_id=5")
    assert length == 1

    result = await history.pop_undo_state(1)
    assert result == {"bytes": b"img_bytes", "label": "remove bbox_id=5"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_undo_state_clears_redo(history):
    await history.push_redo_state(1, b"redo_bytes", "redo")
    await history.push_undo_state(1, b"img", "remove bbox_id=1")

    result = await history.pop_redo_state(1)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_stack_max_history(history):
    for i in range(RedisHistory.MAX_HISTORY + 3):
        await history.push_undo_state(1, b"img", f"op_{i}")

    labels = await history.get_history_labels(1)
    assert len(labels) == RedisHistory.MAX_HISTORY


@pytest.mark.integration
@pytest.mark.asyncio
async def test_undo_stack_lifo_order(history):
    await history.push_undo_state(1, b"a", "first")
    await history.push_undo_state(1, b"b", "second")
    await history.push_undo_state(1, b"c", "third")

    result = await history.pop_undo_state(1)
    assert result["label"] == "third"

    result = await history.pop_undo_state(1)
    assert result["label"] == "second"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pop_undo_empty_returns_none(history):
    result = await history.pop_undo_state(999)
    assert result is None

@pytest.mark.integration
@pytest.mark.asyncio
async def test_push_redo_state_and_pop(history):
    await history.push_redo_state(1, b"redo_img", "redo")
    result = await history.pop_redo_state(1)
    assert result == {"bytes": b"redo_img", "label": "redo"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pop_redo_empty_returns_none(history):
    result = await history.pop_redo_state(999)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redo_stack_does_not_affect_undo_stack(history):
    await history.push_undo_state(1, b"a", "op_1")
    await history.push_redo_state(1, b"b", "redo")

    labels = await history.get_history_labels(1)
    assert labels == ["op_1"]

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_labels_order(history):
    await history.push_undo_state(1, b"a", "op_1")
    await history.push_undo_state(1, b"b", "op_2")
    await history.push_undo_state(1, b"c", "op_3")

    labels = await history.get_history_labels(1)
    # newest first (LPUSH = stack)
    assert labels == ["op_3", "op_2", "op_1"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_history_labels_empty(history):
    labels = await history.get_history_labels(999)
    assert labels == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_history(history):
    await history.push_undo_state(1, b"a", "op_1")
    await history.push_redo_state(1, b"b", "redo")

    await history.clear_history(1)

    assert await history.pop_undo_state(1) is None
    assert await history.pop_redo_state(1) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_history_does_not_affect_other_images(history):
    await history.push_undo_state(1, b"a", "op_1")
    await history.push_undo_state(2, b"b", "op_2")

    await history.clear_history(1)

    labels = await history.get_history_labels(2)
    assert labels == ["op_2"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_history_does_not_touch_image_cache(storage, history):
    await storage.cache_image(1, b"processed")
    await history.push_undo_state(1, b"a", "op_1")

    await history.clear_history(1)

    assert await storage.get_cache_image(1) == b"processed"