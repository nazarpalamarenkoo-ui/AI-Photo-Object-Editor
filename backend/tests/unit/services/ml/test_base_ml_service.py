import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.base_ml_service import BaseMLService


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.download = AsyncMock(return_value=b"image-bytes")
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.url/result.jpg")
    return s3


@pytest.fixture
def mock_redis_storage():
    redis = AsyncMock()
    redis.get_cache_image = AsyncMock(return_value=None)
    redis.cache_image = AsyncMock()
    return redis


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    ra = AsyncMock()
    ra.list_assets = AsyncMock(return_value=[])
    return ra


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
    return AsyncMock()


@pytest.fixture
def mock_edit_history_repo():
    return AsyncMock()


@pytest.fixture
def mock_assets_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline,
):
    return BaseMLService(
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


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.storage_path = "images/original.jpg"
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.image_id = 1
    version.content_id = 100
    return version


def make_mask(mask_id=1, storage_path="s3://bucket/masks/1.png"):
    mask = MagicMock()
    mask.id = mask_id + 500
    mask.mask_id = mask_id
    mask.mask_storage_path = storage_path
    mask.x1, mask.y1, mask.x2, mask.y2 = 0, 0, 10, 10
    mask.area = 100.0
    mask.score = 0.9
    return mask


class TestGetImageAuthorized:

    async def test_success(self, service, mock_image_repo, sample_image):
        mock_image_repo.get_by_id.return_value = sample_image

        result = await service._get_image_authorized(1, 42)

        assert result is sample_image
        mock_image_repo.get_by_id.assert_awaited_once_with(1)

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Image 1 not found"):
            await service._get_image_authorized(1, 42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id.return_value = sample_image

        with pytest.raises(ValueError, match="Unauthorized"):
            await service._get_image_authorized(1, 42)


class TestGetCurrentVersionAuthorized:

    async def test_success(self, service, mock_image_repo, mock_image_version_repo, sample_image, sample_version):
        mock_image_repo.get_by_id.return_value = sample_image
        mock_image_version_repo.get_current.return_value = sample_version

        image, version = await service._get_current_version_authorized(1, 42)

        assert image is sample_image
        assert version is sample_version
        mock_image_version_repo.get_current.assert_awaited_once_with(sample_image)

    async def test_no_current_version(self, service, mock_image_repo, mock_image_version_repo, sample_image):
        mock_image_repo.get_by_id.return_value = sample_image
        mock_image_version_repo.get_current.return_value = None

        with pytest.raises(ValueError, match="has no current version"):
            await service._get_current_version_authorized(1, 42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id.return_value = sample_image

        with pytest.raises(ValueError, match="Unauthorized"):
            await service._get_current_version_authorized(1, 42)


class TestGetCurrentImageBytes:

    async def test_returns_cached_image(self, service, mock_redis_storage, mock_s3):
        mock_redis_storage.get_cache_image.return_value = b"cached"

        result = await service._get_current_image_bytes(1, "image.jpg")

        assert result == b"cached"
        mock_s3.download.assert_not_awaited()

    async def test_fallback_to_s3(self, service, mock_redis_storage, mock_s3):
        mock_redis_storage.get_cache_image.return_value = None
        mock_s3.download.return_value = b"s3"

        result = await service._get_current_image_bytes(1, "image.jpg")

        assert result == b"s3"
        mock_s3.download.assert_awaited_once_with("image.jpg")

    async def test_s3_failure(self, service, mock_redis_storage, mock_s3):
        mock_redis_storage.get_cache_image.return_value = None
        mock_s3.download.side_effect = IOError("download failed")

        with pytest.raises(IOError, match="download failed"):
            await service._get_current_image_bytes(1, "image.jpg")


class TestSaveCurrentState:

    async def test_success(self, service, mock_redis_storage):
        await service._save_current_state(1, b"bytes")

        mock_redis_storage.cache_image.assert_awaited_once_with(
            image_id=1, image_data=b"bytes", suffix="current_state", ttl=7200,
        )

    async def test_redis_failure(self, service, mock_redis_storage):
        mock_redis_storage.cache_image.side_effect = RuntimeError("redis")

        with pytest.raises(RuntimeError, match="redis"):
            await service._save_current_state(1, b"bytes")


class TestUploadResult:

    async def test_success(self, service, mock_s3):
        url, presigned = await service._upload_result(b"bytes", "results/test.jpg")

        mock_s3.upload_bytes.assert_awaited_once_with(
            data=b"bytes", path="results/test.jpg", content_type="image/jpeg",
        )
        mock_s3.get_presigned_url.assert_awaited_once_with(path="results/test.jpg", expiration=3600)

        assert url == "s3://bucket/result.jpg"
        assert presigned == "https://presigned.url/result.jpg"

    async def test_custom_content_type(self, service, mock_s3):
        await service._upload_result(b"bytes", "mask.png", content_type="image/png")

        _, kwargs = mock_s3.upload_bytes.call_args
        assert kwargs["content_type"] == "image/png"

    async def test_upload_failure(self, service, mock_s3):
        mock_s3.upload_bytes.side_effect = IOError("upload failed")

        with pytest.raises(IOError, match="upload failed"):
            await service._upload_result(b"bytes", "file.jpg")

    async def test_presigned_failure(self, service, mock_s3):
        mock_s3.get_presigned_url.side_effect = IOError("presigned")

        with pytest.raises(IOError, match="presigned"):
            await service._upload_result(b"bytes", "file.jpg")


class TestGetTempUrl:

    async def test_success(self, service, mock_s3):
        url = await service._get_temp_url_from_bytes(image_id=1, user_id=42, image_bytes=b"abc", op="remove")

        mock_s3.upload_bytes.assert_awaited_once()
        args = mock_s3.upload_bytes.call_args.kwargs
        assert args["data"] == b"abc"
        assert args["content_type"] == "image/jpeg"
        assert args["path"].startswith("temp/42/1/remove_")
        assert url == "https://presigned.url/result.jpg"

    async def test_upload_failure(self, service, mock_s3):
        mock_s3.upload_bytes.side_effect = IOError("upload")

        with pytest.raises(IOError, match="upload"):
            await service._get_temp_url_from_bytes(1, 42, b"bytes", "remove")


class TestGetSegmentOrRaise:
    """
    _get_segment_or_raise is now content-scoped: it reads SegmentationMask
    rows via segmentation_repo.get_by_content(content_id), not a Redis
    segments cache — Postgres is the source of truth for mask metadata,
    Redis only caches the raw mask bytes under f"mask:{content_id}:{mask_id}".
    """

    async def test_success_reads_mask_bytes_from_redis(self, service, mock_segmentation_repo, mock_redis_storage):
        mock_segmentation_repo.get_by_content.return_value = [make_mask(1), make_mask(5)]
        mock_redis_storage.get_cache_image.return_value = b"cached-mask-bytes"

        result = await service._get_segment_or_raise(content_id=100, mask_id=5, image_id=1)

        assert result["mask_id"] == 5
        assert result["mask_bytes"] == b"cached-mask-bytes"
        mock_redis_storage.get_cache_image.assert_awaited_once_with(1, suffix="mask:100:5")
        service.s3.download.assert_not_awaited()

    async def test_falls_back_to_s3_and_rewarms_cache(self, service, mock_segmentation_repo, mock_redis_storage, mock_s3):
        mock_segmentation_repo.get_by_content.return_value = [make_mask(5, storage_path="s3://bucket/masks/5.png")]
        mock_redis_storage.get_cache_image.return_value = None
        mock_s3.download.return_value = b"from-s3"

        result = await service._get_segment_or_raise(content_id=100, mask_id=5, image_id=1)

        mock_s3.download.assert_awaited_once_with("s3://bucket/masks/5.png")
        mock_redis_storage.cache_image.assert_awaited_once()
        assert result["mask_bytes"] == b"from-s3"

    async def test_mask_not_found(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content.return_value = [make_mask(1), make_mask(2)]

        with pytest.raises(ValueError, match="Segment with mask_id=5 not found"):
            await service._get_segment_or_raise(content_id=100, mask_id=5, image_id=1)

    async def test_no_masks_for_content(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content.return_value = []

        with pytest.raises(ValueError, match="Segment with mask_id=5 not found"):
            await service._get_segment_or_raise(content_id=100, mask_id=5, image_id=1)

    async def test_repo_failure_propagates(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content.side_effect = RuntimeError("db down")

        with pytest.raises(RuntimeError, match="db down"):
            await service._get_segment_or_raise(content_id=100, mask_id=5, image_id=1)


class TestExtractModelMeta:

    def test_reads_top_level_fields(self, service):
        meta = service._extract_model_meta(
            {"model_name": "yolov10m", "model_version": "1.2", "metrics": {"inference_time_ms": 42}},
            default_name="fallback",
        )
        assert meta.model_name == "yolov10m"
        assert meta.model_version == "1.2"
        assert meta.inference_time_ms == 42.0

    def test_falls_back_to_defaults(self, service):
        meta = service._extract_model_meta(None, default_name="mobile_sam", default_version="unknown")
        assert meta.model_name == "mobile_sam"
        assert meta.model_version == "unknown"
        assert meta.inference_time_ms == 0.0

    def test_converts_seconds_to_ms(self, service):
        meta = service._extract_model_meta({"inference_time_s": 0.25}, default_name="x")
        assert meta.inference_time_ms == 250.0


class TestExtractProcessingTimeMs:

    def test_reads_processing_time_ms(self, service):
        assert service._extract_processing_time_ms({"processing_time_ms": 120}) == 120

    def test_returns_none_for_missing_metrics(self, service):
        assert service._extract_processing_time_ms(None) is None

    def test_converts_seconds(self, service):
        assert service._extract_processing_time_ms({"inference_time_s": 1.5}) == 1500