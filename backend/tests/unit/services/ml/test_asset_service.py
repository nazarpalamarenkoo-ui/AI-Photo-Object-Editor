import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image as PILImage

from app.services.ml.assets_service import AssetService

pytestmark = pytest.mark.unit


def _png_bytes(size=(40, 40), mode="RGBA"):
    buf = io.BytesIO()
    PILImage.new(mode, size, (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/asset.png")
    s3.download = AsyncMock(return_value=b"downloaded-bytes")
    s3.delete = AsyncMock(return_value=True)
    return s3


@pytest.fixture
def mock_redis_storage():
    redis = AsyncMock()
    redis.get_cache_image = AsyncMock(return_value=None)
    return redis


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_content_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_segmentation_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_edit_history_repo():
    return AsyncMock()


@pytest.fixture
def mock_assets_repo():
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_overflow = AsyncMock(return_value=[])
    repo.delete_many = AsyncMock()
    repo.list_by_user = AsyncMock(return_value=[])
    repo.get_by_public_id = AsyncMock(return_value=None)
    repo.rename = AsyncMock()
    repo.delete = AsyncMock()
    return repo


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.content_id = 100
    return version


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline, sample_image, sample_version,
):
    mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
    mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)

    return AssetService(
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
        image_content_repo=mock_image_content_repo,
        detection_repo=mock_detection_repo,
        segmentation_repo=mock_segmentation_repo,
        edit_history_repo=mock_edit_history_repo,
        assets_repo=mock_assets_repo,
        pipeline=mock_pipeline,
    )


def make_mask(mask_id=1):
    mask = MagicMock()
    mask.id = mask_id + 500
    mask.mask_id = mask_id
    mask.mask_storage_path = "s3://bucket/masks/1.png"
    mask.x1, mask.y1, mask.x2, mask.y2 = 0, 0, 20, 20
    mask.area = 400.0
    mask.score = 0.9
    return mask


def make_asset(public_id="a1", label="my object"):
    asset = MagicMock()
    asset.public_id = public_id
    asset.label = label
    asset.storage_path = f"assets/{public_id}.png"
    asset.thumbnail_path = f"assets/{public_id}_thumb.png"
    return asset


class TestExtractObject:

    async def test_success(
        self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline,
        mock_s3, mock_assets_repo,
    ):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_mask(5)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"mask-bytes")

        extracted = _png_bytes()
        mock_pipeline.sam_extract_object = AsyncMock(return_value={
            "extracted_bytes": extracted,
            "object_size": (40, 40),
            "area_pixels": 1600,
            "cropped_bbox": {"x1": 0, "y1": 0, "x2": 40, "y2": 40},
            "timestamp": "t",
        })
        mock_assets_repo.create = AsyncMock(return_value=make_asset("new-asset"))

        result = await service.extract_object(image_id=1, mask_id=5, user_id=42)

        mock_pipeline.sam_extract_object.assert_awaited_once()
        assert mock_s3.upload_bytes.await_count == 2  # full-res + thumbnail
        mock_assets_repo.create.assert_awaited_once()
        assert result["asset_id"] == "new-asset"
        assert result["object_size"] == (40, 40)

    async def test_evicts_overflow_assets(
        self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline, mock_assets_repo, mock_s3,
    ):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_mask(5)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"mask-bytes")
        mock_pipeline.sam_extract_object = AsyncMock(return_value={
            "extracted_bytes": _png_bytes(), "object_size": (40, 40), "area_pixels": 1600,
            "cropped_bbox": {}, "timestamp": "t",
        })
        mock_assets_repo.create = AsyncMock(return_value=make_asset())
        overflow = [make_asset("old-1"), make_asset("old-2")]
        mock_assets_repo.get_overflow = AsyncMock(return_value=overflow)

        await service.extract_object(image_id=1, mask_id=5, user_id=42)

        assert mock_s3.delete.await_count == 4  # 2 old assets x (main + thumbnail)
        mock_assets_repo.delete_many.assert_awaited_once_with(overflow)

    async def test_segment_not_found(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="not found"):
            await service.extract_object(image_id=1, mask_id=5, user_id=42)

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.extract_object(image_id=1, mask_id=5, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.extract_object(image_id=1, mask_id=5, user_id=42)


class TestListAssets:

    async def test_returns_repo_result(self, service, mock_assets_repo):
        expected = [make_asset("a1"), make_asset("a2")]
        mock_assets_repo.list_by_user = AsyncMock(return_value=expected)

        result = await service.list_assets(user_id=42)

        assert result == expected

    async def test_passes_limit_and_offset(self, service, mock_assets_repo):
        await service.list_assets(user_id=42, limit=10, offset=20)

        mock_assets_repo.list_by_user.assert_awaited_once_with(42, limit=10, offset=20)


class TestGetAssetThumbnail:

    async def test_returns_thumbnail_bytes(self, service, mock_assets_repo, mock_s3):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=make_asset("a1"))
        mock_s3.download = AsyncMock(return_value=b"thumb-bytes")

        result = await service.get_asset_thumbnail(user_id=42, asset_id="a1")

        assert result == b"thumb-bytes"

    async def test_returns_none_when_asset_missing(self, service, mock_assets_repo):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=None)

        result = await service.get_asset_thumbnail(user_id=42, asset_id="missing")

        assert result is None


class TestGetAssetImage:

    async def test_returns_full_res_bytes(self, service, mock_assets_repo, mock_s3):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=make_asset("a1"))
        mock_s3.download = AsyncMock(return_value=b"full-res-png")

        result = await service.get_asset_image(user_id=42, asset_id="a1")

        assert result == b"full-res-png"

    async def test_returns_none_when_missing(self, service, mock_assets_repo):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=None)

        result = await service.get_asset_image(user_id=42, asset_id="missing")

        assert result is None


class TestRenameAsset:

    async def test_success(self, service, mock_assets_repo):
        asset = make_asset("a1")
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=asset)
        renamed = make_asset("a1", label="new label")
        mock_assets_repo.rename = AsyncMock(return_value=renamed)

        result = await service.rename_asset(user_id=42, asset_id="a1", label="new label")

        mock_assets_repo.rename.assert_awaited_once_with(asset, "new label")
        assert result == renamed

    async def test_raises_when_not_found(self, service, mock_assets_repo):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Asset not found"):
            await service.rename_asset(user_id=42, asset_id="missing", label="x")


class TestDeleteAsset:

    async def test_success(self, service, mock_assets_repo, mock_s3):
        asset = make_asset("a1")
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=asset)

        await service.delete_asset(user_id=42, asset_id="a1")

        assert mock_s3.delete.await_count == 2  # main + thumbnail
        mock_assets_repo.delete.assert_awaited_once_with(asset)

    async def test_raises_when_not_found(self, service, mock_assets_repo):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Asset not found"):
            await service.delete_asset(user_id=42, asset_id="missing")

    async def test_s3_failure_does_not_block_db_delete(self, service, mock_assets_repo, mock_s3):
        """S3 cleanup is best-effort — a dangling S3 object should never
        prevent the DB row the user asked to delete from actually going."""
        asset = make_asset("a1")
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=asset)
        mock_s3.delete = AsyncMock(side_effect=IOError("s3 down"))

        await service.delete_asset(user_id=42, asset_id="a1")

        mock_assets_repo.delete.assert_awaited_once_with(asset)


class TestPasteExtractedObject:

    async def test_requires_asset_id_or_extracted_url(self, service):
        with pytest.raises(ValueError, match="Provide either asset_id or extracted_url"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            )

    async def test_success_with_asset_id(
        self, service, mock_assets_repo, mock_s3, mock_pipeline, mock_redis_history,
    ):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=make_asset("a1"))
        mock_s3.download = AsyncMock(return_value=b"extracted-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value={
            "result_bytes": b"pasted", "paste_bbox": {}, "object_size": (10, 10), "timestamp": "t",
        })

        result = await service.paste_extracted_object(
            image_id=1, user_id=42, target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10}, asset_id="a1",
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_paste_extracted_object.assert_awaited_once()
        assert result["result_url"] is not None

    async def test_asset_not_found(self, service, mock_assets_repo):
        mock_assets_repo.get_by_public_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Asset not found"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10}, asset_id="missing",
            )

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.paste_extracted_object(
                image_id=1, user_id=42, target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10}, asset_id="a1",
            )