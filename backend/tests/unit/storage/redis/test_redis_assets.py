import pickle
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.storage.redis.redis_assets import RedisAssetsStorage


@pytest.fixture
def storage():
    """
    RedisAssetsStorage instance with its Redis client fully mocked out.
    Bypasses RedisStorage.__init__ (which would try to connect to a real
    Redis server) via __new__, then attaches a MagicMock as `.redis`.
    """
    instance = RedisAssetsStorage.__new__(RedisAssetsStorage)
    instance.redis = MagicMock()
    return instance


@pytest.fixture
def mock_pipeline(storage):
    """Attach a mock pipeline (as returned by redis.pipeline()) to storage.redis"""
    pipe = MagicMock()
    pipe.setex = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.delete = MagicMock(return_value=pipe)
    pipe.zrem = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[True, True, True, 1, True])
    storage.redis.pipeline = MagicMock(return_value=pipe)
    return pipe


@pytest.fixture
def rgba_image_bytes():
    """Simple 400x300 RGBA PNG with a fully-opaque region and a transparent one"""
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    for x in range(50, 350):
        for y in range(50, 250):
            img.putpixel((x, y), (255, 0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.unit
def test_keys_builds_expected_key_names(storage):
    keys = storage._keys(42, "abc123")
    assert keys == {
        "meta": "asset:42:abc123:meta",
        "data": "asset:42:abc123:data",
        "thumb": "asset:42:abc123:thumb",
    }


@pytest.mark.unit
def test_index_key_builds_expected_key_name(storage):
    assert storage._index_key(42) == "assets_index:42"


@pytest.mark.unit
def test_make_thumbnail_returns_valid_png_within_bounds(storage, rgba_image_bytes):
    thumb_bytes = storage._make_thumbnail(rgba_image_bytes)

    assert isinstance(thumb_bytes, bytes)
    thumb_img = Image.open(BytesIO(thumb_bytes))
    assert thumb_img.format == "PNG"
    assert thumb_img.width <= storage.THUMB_SIZE[0]
    assert thumb_img.height <= storage.THUMB_SIZE[1]


@pytest.mark.unit
def test_make_thumbnail_preserves_alpha_channel(storage, rgba_image_bytes):
    thumb_bytes = storage._make_thumbnail(rgba_image_bytes)
    thumb_img = Image.open(BytesIO(thumb_bytes))
    assert thumb_img.mode == "RGBA"


@pytest.mark.unit
def test_make_thumbnail_preserves_aspect_ratio(storage, rgba_image_bytes):
    # source is 400x300 (4:3)
    thumb_bytes = storage._make_thumbnail(rgba_image_bytes)
    thumb_img = Image.open(BytesIO(thumb_bytes))
    original_ratio = 400 / 300
    thumb_ratio = thumb_img.width / thumb_img.height
    assert abs(original_ratio - thumb_ratio) < 0.05


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_asset_returns_metadata_with_expected_fields(storage, mock_pipeline, rgba_image_bytes):
    storage._enforce_limit = AsyncMock()

    meta = await storage.save_asset(
        user_id=1,
        extracted_bytes=rgba_image_bytes,
        source_image_id=99,
        object_size=(300, 200),
        area_pixels=12345,
        label="my cutout",
        s3_url="s3://bucket/obj.png",
    )

    assert meta["user_id"] == 1
    assert meta["source_image_id"] == 99
    assert meta["object_size"] == (300, 200)
    assert meta["area_pixels"] == 12345
    assert meta["label"] == "my cutout"
    assert meta["s3_url"] == "s3://bucket/obj.png"
    assert "asset_id" in meta and len(meta["asset_id"]) == 32  # uuid4().hex
    assert "created_at" in meta


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_asset_defaults_label_and_s3_url_to_none(storage, mock_pipeline, rgba_image_bytes):
    storage._enforce_limit = AsyncMock()

    meta = await storage.save_asset(
        user_id=1,
        extracted_bytes=rgba_image_bytes,
        source_image_id=99,
        object_size=(300, 200),
        area_pixels=12345,
    )

    assert meta["label"] is None
    assert meta["s3_url"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_asset_writes_meta_data_thumb_with_ttl(storage, mock_pipeline, rgba_image_bytes):
    storage._enforce_limit = AsyncMock()

    meta = await storage.save_asset(
        user_id=7,
        extracted_bytes=rgba_image_bytes,
        source_image_id=1,
        object_size=(100, 100),
        area_pixels=500,
        ttl=3600,
    )

    keys = storage._keys(7, meta["asset_id"])

    setex_calls = mock_pipeline.setex.call_args_list
    assert len(setex_calls) == 3

    meta_call = setex_calls[0]
    assert meta_call.args[0] == keys["meta"]
    assert meta_call.args[1] == 3600
    assert pickle.loads(meta_call.args[2])["asset_id"] == meta["asset_id"]

    data_call = setex_calls[1]
    assert data_call.args[0] == keys["data"]
    assert data_call.args[1] == 3600
    assert data_call.args[2] == rgba_image_bytes

    thumb_call = setex_calls[2]
    assert thumb_call.args[0] == keys["thumb"]
    assert thumb_call.args[1] == 3600


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_asset_indexes_asset_with_created_at_score(storage, mock_pipeline, rgba_image_bytes):
    storage._enforce_limit = AsyncMock()

    meta = await storage.save_asset(
        user_id=7,
        extracted_bytes=rgba_image_bytes,
        source_image_id=1,
        object_size=(100, 100),
        area_pixels=500,
    )

    zadd_args = mock_pipeline.zadd.call_args
    assert zadd_args.args[0] == "assets_index:7"
    assert meta["asset_id"] in zadd_args.args[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_asset_calls_enforce_limit(storage, mock_pipeline, rgba_image_bytes):
    storage._enforce_limit = AsyncMock()

    await storage.save_asset(
        user_id=1,
        extracted_bytes=rgba_image_bytes,
        source_image_id=1,
        object_size=(10, 10),
        area_pixels=1,
    )

    storage._enforce_limit.assert_awaited_once_with(1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enforce_limit_does_nothing_when_under_limit(storage):
    storage.redis.zcard = AsyncMock(return_value=storage.MAX_ASSETS_PER_USER - 1)
    storage.redis.zrange = AsyncMock()
    storage.delete_asset = AsyncMock()

    await storage._enforce_limit(1)

    storage.redis.zrange.assert_not_called()
    storage.delete_asset.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enforce_limit_does_nothing_when_at_limit(storage):
    storage.redis.zcard = AsyncMock(return_value=storage.MAX_ASSETS_PER_USER)
    storage.redis.zrange = AsyncMock()
    storage.delete_asset = AsyncMock()

    await storage._enforce_limit(1)

    storage.redis.zrange.assert_not_called()
    storage.delete_asset.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enforce_limit_deletes_oldest_overflowing_assets(storage):
    storage.redis.zcard = AsyncMock(return_value=storage.MAX_ASSETS_PER_USER + 2)
    storage.redis.zrange = AsyncMock(return_value=[b"old_id_1", b"old_id_2"])
    storage.delete_asset = AsyncMock()

    await storage._enforce_limit(5)

    storage.redis.zrange.assert_awaited_once_with("assets_index:5", 0, 1)
    assert storage.delete_asset.await_count == 2
    storage.delete_asset.assert_any_await(5, "old_id_1")
    storage.delete_asset.assert_any_await(5, "old_id_2")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enforce_limit_handles_string_ids(storage):
    storage.redis.zcard = AsyncMock(return_value=storage.MAX_ASSETS_PER_USER + 1)
    storage.redis.zrange = AsyncMock(return_value=["already_str_id"])
    storage.delete_asset = AsyncMock()

    await storage._enforce_limit(5)

    storage.delete_asset.assert_awaited_once_with(5, "already_str_id")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_asset_returns_none_when_missing(storage):
    storage.redis.get = AsyncMock(return_value=None)

    result = await storage.get_asset(1, "missing")

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_asset_with_bytes_attaches_extracted_bytes(storage):
    meta = {"asset_id": "abc", "label": None}

    async def fake_get(key):
        if key.endswith(":meta"):
            return pickle.dumps(meta)
        if key.endswith(":data"):
            return b"raw-bytes"
        return None

    storage.redis.get = AsyncMock(side_effect=fake_get)

    result = await storage.get_asset(1, "abc", with_bytes=True)

    assert result["asset_id"] == "abc"
    assert result["extracted_bytes"] == b"raw-bytes"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_asset_without_bytes_skips_data_fetch(storage):
    meta = {"asset_id": "abc", "label": None}
    storage.redis.get = AsyncMock(return_value=pickle.dumps(meta))

    result = await storage.get_asset(1, "abc", with_bytes=False)

    assert "extracted_bytes" not in result
    storage.redis.get.assert_awaited_once_with("asset:1:abc:meta")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_thumbnail_returns_bytes(storage):
    storage.redis.get = AsyncMock(return_value=b"thumb-bytes")

    result = await storage.get_thumbnail(1, "abc")

    assert result == b"thumb-bytes"
    storage.redis.get.assert_awaited_once_with("asset:1:abc:thumb")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_thumbnail_returns_none_when_missing(storage):
    storage.redis.get = AsyncMock(return_value=None)

    result = await storage.get_thumbnail(1, "missing")

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_assets_returns_empty_list_when_no_ids(storage):
    storage.redis.zrevrange = AsyncMock(return_value=[])
    storage.redis.mget = AsyncMock()

    result = await storage.list_assets(1)

    assert result == []
    storage.redis.mget.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_assets_uses_limit_and_offset(storage):
    storage.redis.zrevrange = AsyncMock(return_value=[])

    await storage.list_assets(1, limit=10, offset=20)

    storage.redis.zrevrange.assert_awaited_once_with("assets_index:1", 20, 29)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_assets_decodes_meta_and_skips_missing(storage):
    meta_a = pickle.dumps({"asset_id": "a"})
    meta_c = pickle.dumps({"asset_id": "c"})

    storage.redis.zrevrange = AsyncMock(return_value=[b"a", b"b", b"c"])
    storage.redis.mget = AsyncMock(return_value=[meta_a, None, meta_c])

    result = await storage.list_assets(1)

    assert result == [{"asset_id": "a"}, {"asset_id": "c"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rename_asset_returns_none_when_asset_missing(storage):
    storage.get_asset = AsyncMock(return_value=None)
    storage.redis.setex = AsyncMock()

    result = await storage.rename_asset(1, "missing", "new label")

    assert result is None
    storage.redis.setex.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rename_asset_updates_label_and_preserves_ttl(storage):
    storage.get_asset = AsyncMock(return_value={"asset_id": "abc", "label": "old"})
    storage.redis.ttl = AsyncMock(return_value=1234)
    storage.redis.setex = AsyncMock()

    result = await storage.rename_asset(1, "abc", "new label")

    assert result["label"] == "new label"
    storage.redis.setex.assert_awaited_once()
    call = storage.redis.setex.call_args
    assert call.args[0] == "asset:1:abc:meta"
    assert call.args[1] == 1234
    assert pickle.loads(call.args[2])["label"] == "new label"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rename_asset_falls_back_to_default_ttl_when_ttl_negative(storage):
    storage.get_asset = AsyncMock(return_value={"asset_id": "abc", "label": "old"})
    storage.redis.ttl = AsyncMock(return_value=-1)  # key exists with no TTL, or missing
    storage.redis.setex = AsyncMock()

    await storage.rename_asset(1, "abc", "new label")

    call = storage.redis.setex.call_args
    assert call.args[1] == storage.ASSET_TTL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_asset_returns_true_when_meta_key_deleted(storage, mock_pipeline):
    mock_pipeline.execute = AsyncMock(return_value=[1, 1])

    result = await storage.delete_asset(1, "abc")

    assert result is True
    mock_pipeline.delete.assert_called_once_with(
        "asset:1:abc:meta", "asset:1:abc:data", "asset:1:abc:thumb"
    )
    mock_pipeline.zrem.assert_called_once_with("assets_index:1", "abc")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_asset_returns_false_when_meta_key_missing(storage, mock_pipeline):
    mock_pipeline.execute = AsyncMock(return_value=[0, 0])

    result = await storage.delete_asset(1, "missing")

    assert result is False