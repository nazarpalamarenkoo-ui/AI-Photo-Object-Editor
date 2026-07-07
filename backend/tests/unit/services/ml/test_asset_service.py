import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.assets_service import AssetService

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_s3():
    return AsyncMock()


@pytest.fixture
def mock_redis_storage():
    return AsyncMock()


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    ra = AsyncMock()
    ra.list_assets = AsyncMock(return_value=[])
    ra.get_thumbnail = AsyncMock(return_value=None)
    ra.get_asset = AsyncMock(return_value=None)
    ra.rename_asset = AsyncMock(return_value=None)
    ra.delete_asset = AsyncMock(return_value=False)
    return ra


@pytest.fixture
def service(
    mock_db, mock_s3, mock_redis_storage, mock_redis_history,
    mock_image_repo, mock_detection_repo, mock_pipeline, mock_redis_assets,
):
    return AssetService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
        redis_assets=mock_redis_assets,
    )


def make_asset_meta(asset_id="asset-1", label="my object"):
    return {
        "asset_id": asset_id,
        "label": label,
        "object_size": (100, 100),
        "area_pixels": 5000,
    }


class TestListAssets:
    async def test_returns_redis_result(self, service, mock_redis_assets):
        expected = [make_asset_meta("a1"), make_asset_meta("a2")]
        mock_redis_assets.list_assets = AsyncMock(return_value=expected)

        result = await service.list_assets(user_id=42)

        assert result == expected

    async def test_passes_limit_and_offset(self, service, mock_redis_assets):
        await service.list_assets(user_id=42, limit=10, offset=20)

        mock_redis_assets.list_assets.assert_awaited_once_with(42, limit=10, offset=20)

    async def test_uses_default_limit_and_offset(self, service, mock_redis_assets):
        await service.list_assets(user_id=42)

        mock_redis_assets.list_assets.assert_awaited_once_with(42, limit=50, offset=0)

    async def test_returns_empty_list_when_no_assets(self, service, mock_redis_assets):
        mock_redis_assets.list_assets = AsyncMock(return_value=[])

        result = await service.list_assets(user_id=42)

        assert result == []


class TestGetAssetThumbnail:
    async def test_returns_thumbnail_bytes(self, service, mock_redis_assets):
        mock_redis_assets.get_thumbnail = AsyncMock(return_value=b"thumb-bytes")

        result = await service.get_asset_thumbnail(user_id=42, asset_id="a1")

        assert result == b"thumb-bytes"

    async def test_returns_none_when_missing(self, service, mock_redis_assets):
        mock_redis_assets.get_thumbnail = AsyncMock(return_value=None)

        result = await service.get_asset_thumbnail(user_id=42, asset_id="missing")

        assert result is None

    async def test_passes_correct_args(self, service, mock_redis_assets):
        await service.get_asset_thumbnail(user_id=42, asset_id="a1")

        mock_redis_assets.get_thumbnail.assert_awaited_once_with(42, "a1")


class TestGetAssetImage:
    async def test_returns_extracted_bytes_when_asset_found(self, service, mock_redis_assets):
        mock_redis_assets.get_asset = AsyncMock(return_value={"extracted_bytes": b"full-res-png"})

        result = await service.get_asset_image(user_id=42, asset_id="a1")

        assert result == b"full-res-png"

    async def test_returns_none_when_asset_missing(self, service, mock_redis_assets):
        mock_redis_assets.get_asset = AsyncMock(return_value=None)

        result = await service.get_asset_image(user_id=42, asset_id="missing")

        assert result is None

    async def test_calls_get_asset_with_bytes_true(self, service, mock_redis_assets):
        mock_redis_assets.get_asset = AsyncMock(return_value={"extracted_bytes": b"x"})

        await service.get_asset_image(user_id=42, asset_id="a1")

        mock_redis_assets.get_asset.assert_awaited_once_with(42, "a1", with_bytes=True)


class TestRenameAsset:
    async def test_success_returns_updated_meta(self, service, mock_redis_assets):
        updated = make_asset_meta("a1", label="new label")
        mock_redis_assets.rename_asset = AsyncMock(return_value=updated)

        result = await service.rename_asset(user_id=42, asset_id="a1", label="new label")

        assert result == updated

    async def test_raises_when_not_found(self, service, mock_redis_assets):
        mock_redis_assets.rename_asset = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Asset not found"):
            await service.rename_asset(user_id=42, asset_id="missing", label="x")

    async def test_forwards_correct_args(self, service, mock_redis_assets):
        mock_redis_assets.rename_asset = AsyncMock(return_value=make_asset_meta())

        await service.rename_asset(user_id=42, asset_id="a1", label="new label")

        mock_redis_assets.rename_asset.assert_awaited_once_with(42, "a1", "new label")


class TestDeleteAsset:
    async def test_success_does_not_raise(self, service, mock_redis_assets):
        mock_redis_assets.delete_asset = AsyncMock(return_value=True)

        await service.delete_asset(user_id=42, asset_id="a1")  # should not raise

    async def test_raises_when_not_found(self, service, mock_redis_assets):
        mock_redis_assets.delete_asset = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="Asset not found"):
            await service.delete_asset(user_id=42, asset_id="missing")

    async def test_forwards_correct_args(self, service, mock_redis_assets):
        mock_redis_assets.delete_asset = AsyncMock(return_value=True)

        await service.delete_asset(user_id=42, asset_id="a1")

        mock_redis_assets.delete_asset.assert_awaited_once_with(42, "a1")