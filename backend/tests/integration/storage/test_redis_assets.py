import asyncio
from io import BytesIO

import fakeredis.aioredis
import pytest
from PIL import Image

from app.storage.redis.redis_assets import RedisAssetsStorage


@pytest.fixture
def fake_redis_client():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def assets_storage(monkeypatch, fake_redis_client):
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: fake_redis_client)
    return RedisAssetsStorage()


@pytest.fixture
def extracted_bytes():
    """RGBA PNG object with a fully-opaque region, as produced by a real cutout"""
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    for x in range(40, 160):
        for y in range(40, 160):
            img.putpixel((x, y), (10, 200, 30, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_save_asset_returns_full_metadata(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1,
        extracted_bytes=extracted_bytes,
        source_image_id=10,
        object_size=(120, 120),
        area_pixels=14400,
        label="cutout 1",
        s3_url="s3://bucket/cutout1.png",
    )

    assert meta["user_id"] == 1
    assert meta["source_image_id"] == 10
    assert meta["object_size"] == (120, 120)
    assert meta["area_pixels"] == 14400
    assert meta["label"] == "cutout 1"
    assert meta["s3_url"] == "s3://bucket/cutout1.png"
    assert len(meta["asset_id"]) == 32


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_save_asset_persists_retrievable_data_and_thumbnail(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1,
        extracted_bytes=extracted_bytes,
        source_image_id=10,
        object_size=(120, 120),
        area_pixels=14400,
    )
    asset_id = meta["asset_id"]

    fetched = await assets_storage.get_asset(1, asset_id, with_bytes=True)
    assert fetched["extracted_bytes"] == extracted_bytes

    thumb_bytes = await assets_storage.get_thumbnail(1, asset_id)
    assert thumb_bytes is not None
    thumb_img = Image.open(BytesIO(thumb_bytes))
    assert thumb_img.width <= RedisAssetsStorage.THUMB_SIZE[0]
    assert thumb_img.height <= RedisAssetsStorage.THUMB_SIZE[1]


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_save_asset_appears_in_index(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1,
        extracted_bytes=extracted_bytes,
        source_image_id=10,
        object_size=(120, 120),
        area_pixels=14400,
    )

    assets = await assets_storage.list_assets(1)
    assert len(assets) == 1
    assert assets[0]["asset_id"] == meta["asset_id"]


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_save_asset_applies_custom_ttl(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1,
        extracted_bytes=extracted_bytes,
        source_image_id=10,
        object_size=(50, 50),
        area_pixels=2500,
        ttl=120,
    )

    keys = assets_storage._keys(1, meta["asset_id"])
    ttl = await assets_storage.redis.ttl(keys["meta"])
    assert 0 < ttl <= 120


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_asset_missing_returns_none(assets_storage):
    result = await assets_storage.get_asset(1, "nonexistent")
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_asset_without_bytes_omits_extracted_bytes(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1,
        extracted_bytes=extracted_bytes,
        source_image_id=10,
        object_size=(50, 50),
        area_pixels=2500,
    )

    fetched = await assets_storage.get_asset(1, meta["asset_id"], with_bytes=False)
    assert "extracted_bytes" not in fetched


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_get_thumbnail_missing_returns_none(assets_storage):
    result = await assets_storage.get_thumbnail(1, "nonexistent")
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_list_assets_orders_newest_first(assets_storage, extracted_bytes):
    ids = []
    for i in range(3):
        meta = await assets_storage.save_asset(
            user_id=1,
            extracted_bytes=extracted_bytes,
            source_image_id=i,
            object_size=(10, 10),
            area_pixels=100,
            label=f"asset_{i}",
        )
        ids.append(meta["asset_id"])
        await asyncio.sleep(0.02)

    assets = await assets_storage.list_assets(1)
    returned_ids = [a["asset_id"] for a in assets]

    assert returned_ids == list(reversed(ids))


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_list_assets_respects_limit_and_offset(assets_storage, extracted_bytes):
    for i in range(5):
        await assets_storage.save_asset(
            user_id=1,
            extracted_bytes=extracted_bytes,
            source_image_id=i,
            object_size=(10, 10),
            area_pixels=100,
            label=f"asset_{i}",
        )

    page1 = await assets_storage.list_assets(1, limit=2, offset=0)
    page2 = await assets_storage.list_assets(1, limit=2, offset=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert {a["asset_id"] for a in page1}.isdisjoint({a["asset_id"] for a in page2})


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_list_assets_empty_for_user_with_no_assets(assets_storage):
    assets = await assets_storage.list_assets(1)
    assert assets == []


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_list_assets_isolated_per_user(assets_storage, extracted_bytes):
    await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=1,
        object_size=(10, 10), area_pixels=100, label="user1_asset",
    )
    await assets_storage.save_asset(
        user_id=2, extracted_bytes=extracted_bytes, source_image_id=2,
        object_size=(10, 10), area_pixels=100, label="user2_asset",
    )

    user1_assets = await assets_storage.list_assets(1)
    user2_assets = await assets_storage.list_assets(2)

    assert len(user1_assets) == 1
    assert user1_assets[0]["label"] == "user1_asset"
    assert len(user2_assets) == 1
    assert user2_assets[0]["label"] == "user2_asset"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_rename_asset_updates_label(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=1,
        object_size=(10, 10), area_pixels=100, label="old label",
    )

    updated = await assets_storage.rename_asset(1, meta["asset_id"], "new label")
    assert updated["label"] == "new label"

    fetched = await assets_storage.get_asset(1, meta["asset_id"], with_bytes=False)
    assert fetched["label"] == "new label"


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_rename_asset_missing_returns_none(assets_storage):
    result = await assets_storage.rename_asset(1, "nonexistent", "new label")
    assert result is None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_rename_asset_preserves_data_and_thumbnail(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=1,
        object_size=(10, 10), area_pixels=100, label="old label",
    )

    await assets_storage.rename_asset(1, meta["asset_id"], "new label")

    fetched = await assets_storage.get_asset(1, meta["asset_id"], with_bytes=True)
    assert fetched["extracted_bytes"] == extracted_bytes
    assert await assets_storage.get_thumbnail(1, meta["asset_id"]) is not None


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_delete_asset_removes_meta_data_thumbnail_and_index_entry(assets_storage, extracted_bytes):
    meta = await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=1,
        object_size=(10, 10), area_pixels=100,
    )
    asset_id = meta["asset_id"]

    deleted = await assets_storage.delete_asset(1, asset_id)
    assert deleted is True

    assert await assets_storage.get_asset(1, asset_id) is None
    assert await assets_storage.get_thumbnail(1, asset_id) is None
    assert await assets_storage.list_assets(1) == []


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_delete_asset_missing_returns_false(assets_storage):
    result = await assets_storage.delete_asset(1, "nonexistent")
    assert result is False


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_delete_asset_does_not_affect_other_assets(assets_storage, extracted_bytes):
    meta1 = await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=1,
        object_size=(10, 10), area_pixels=100, label="keep me",
    )
    meta2 = await assets_storage.save_asset(
        user_id=1, extracted_bytes=extracted_bytes, source_image_id=2,
        object_size=(10, 10), area_pixels=100, label="delete me",
    )

    await assets_storage.delete_asset(1, meta2["asset_id"])

    remaining = await assets_storage.list_assets(1)
    assert len(remaining) == 1
    assert remaining[0]["asset_id"] == meta1["asset_id"]


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_enforce_limit_deletes_oldest_assets_when_exceeding_max(
    monkeypatch, assets_storage, extracted_bytes
):
    monkeypatch.setattr(RedisAssetsStorage, "MAX_ASSETS_PER_USER", 3)

    ids = []
    for i in range(5):
        meta = await assets_storage.save_asset(
            user_id=1, extracted_bytes=extracted_bytes, source_image_id=i,
            object_size=(10, 10), area_pixels=100, label=f"asset_{i}",
        )
        ids.append(meta["asset_id"])
        # guarantee strictly increasing created_at scores across saves,
        # see test_list_assets_orders_newest_first for rationale
        await asyncio.sleep(0.02)

    remaining = await assets_storage.list_assets(1)
    remaining_ids = {a["asset_id"] for a in remaining}

    assert len(remaining) == 3
    # the 3 most recently created assets should survive
    assert remaining_ids == set(ids[-3:])


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_enforce_limit_does_not_trigger_under_max(monkeypatch, assets_storage, extracted_bytes):
    monkeypatch.setattr(RedisAssetsStorage, "MAX_ASSETS_PER_USER", 10)

    for i in range(3):
        await assets_storage.save_asset(
            user_id=1, extracted_bytes=extracted_bytes, source_image_id=i,
            object_size=(10, 10), area_pixels=100,
        )

    remaining = await assets_storage.list_assets(1)
    assert len(remaining) == 3