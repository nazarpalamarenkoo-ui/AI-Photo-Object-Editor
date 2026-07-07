import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.base_ml_service import BaseMLService


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return AsyncMock()


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
    redis.get_cached_segments = AsyncMock(return_value=None)
    return redis


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
    mock_db,
    mock_s3,
    mock_redis_storage,
    mock_redis_history,
    mock_redis_assets,
    mock_image_repo,
    mock_detection_repo,
    mock_pipeline,
):
    return BaseMLService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.storage_path = "images/original.jpg"
    return image


def make_segment(mask_id=1):
    return {
        "mask_id": mask_id,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "mask_bytes": b"mask",
    }


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


class TestGetCurrentImageBytes:

    async def test_returns_cached_image(
        self,
        service,
        mock_redis_storage,
        mock_s3,
    ):
        mock_redis_storage.get_cache_image.return_value = b"cached"

        result = await service._get_current_image_bytes(
            1,
            "image.jpg",
        )

        assert result == b"cached"

        mock_s3.download.assert_not_awaited()

    async def test_fallback_to_s3(
        self,
        service,
        mock_redis_storage,
        mock_s3,
    ):
        mock_redis_storage.get_cache_image.return_value = None
        mock_s3.download.return_value = b"s3"

        result = await service._get_current_image_bytes(
            1,
            "image.jpg",
        )

        assert result == b"s3"

        mock_s3.download.assert_awaited_once_with("image.jpg")

    async def test_s3_failure(
        self,
        service,
        mock_redis_storage,
        mock_s3,
    ):
        mock_redis_storage.get_cache_image.return_value = None
        mock_s3.download.side_effect = IOError("download failed")

        with pytest.raises(IOError, match="download failed"):
            await service._get_current_image_bytes(
                1,
                "image.jpg",
            )


class TestSaveCurrentState:

    async def test_success(
        self,
        service,
        mock_redis_storage,
    ):
        await service._save_current_state(
            1,
            b"bytes",
        )

        mock_redis_storage.cache_image.assert_awaited_once_with(
            image_id=1,
            image_data=b"bytes",
            suffix="current_state",
            ttl=7200,
        )

    async def test_redis_failure(
        self,
        service,
        mock_redis_storage,
    ):
        mock_redis_storage.cache_image.side_effect = RuntimeError("redis")

        with pytest.raises(RuntimeError, match="redis"):
            await service._save_current_state(
                1,
                b"bytes",
            )


class TestUploadResult:

    async def test_success(
        self,
        service,
        mock_s3,
    ):
        url, presigned = await service._upload_result(
            b"bytes",
            "results/test.jpg",
        )

        mock_s3.upload_bytes.assert_awaited_once_with(
            data=b"bytes",
            path="results/test.jpg",
            content_type="image/jpeg",
        )

        mock_s3.get_presigned_url.assert_awaited_once_with(
            path="results/test.jpg",
            expiration=3600,
        )

        assert url == "s3://bucket/result.jpg"
        assert presigned == "https://presigned.url/result.jpg"

    async def test_custom_content_type(
        self,
        service,
        mock_s3,
    ):
        await service._upload_result(
            b"bytes",
            "mask.png",
            content_type="image/png",
        )

        _, kwargs = mock_s3.upload_bytes.call_args

        assert kwargs["content_type"] == "image/png"

    async def test_upload_failure(
        self,
        service,
        mock_s3,
    ):
        mock_s3.upload_bytes.side_effect = IOError("upload failed")

        with pytest.raises(IOError, match="upload failed"):
            await service._upload_result(
                b"bytes",
                "file.jpg",
            )

    async def test_presigned_failure(
        self,
        service,
        mock_s3,
    ):
        mock_s3.get_presigned_url.side_effect = IOError("presigned")

        with pytest.raises(IOError, match="presigned"):
            await service._upload_result(
                b"bytes",
                "file.jpg",
            )


class TestGetTempUrl:

    async def test_success(
        self,
        service,
        mock_s3,
    ):
        url = await service._get_temp_url_from_bytes(
            image_id=1,
            user_id=42,
            image_bytes=b"abc",
            op="remove",
        )

        mock_s3.upload_bytes.assert_awaited_once()

        args = mock_s3.upload_bytes.call_args.kwargs

        assert args["data"] == b"abc"
        assert args["content_type"] == "image/jpeg"
        assert args["path"].startswith("temp/42/1/remove_")

        assert url == "https://presigned.url/result.jpg"

    async def test_upload_failure(
        self,
        service,
        mock_s3,
    ):
        mock_s3.upload_bytes.side_effect = IOError("upload")

        with pytest.raises(IOError, match="upload"):
            await service._get_temp_url_from_bytes(
                1,
                42,
                b"bytes",
                "remove",
            )


class TestGetSegment:

    async def test_success(
        self,
        service,
        mock_redis_storage,
    ):
        segment = make_segment(5)

        mock_redis_storage.get_cached_segments.return_value = [
            make_segment(1),
            segment,
            make_segment(9),
        ]

        result = await service._get_segment_or_raise(
            1,
            5,
        )

        assert result == segment

    async def test_no_segments(
        self,
        service,
        mock_redis_storage,
    ):
        mock_redis_storage.get_cached_segments.return_value = None

        with pytest.raises(
            ValueError,
            match="No segments found",
        ):
            await service._get_segment_or_raise(
                1,
                5,
            )

    async def test_empty_segments(
        self,
        service,
        mock_redis_storage,
    ):
        mock_redis_storage.get_cached_segments.return_value = []

        with pytest.raises(
            ValueError,
            match="No segments found",
        ):
            await service._get_segment_or_raise(
                1,
                5,
            )

    async def test_mask_not_found(
        self,
        service,
        mock_redis_storage,
    ):
        mock_redis_storage.get_cached_segments.return_value = [
            make_segment(1),
            make_segment(2),
        ]

        with pytest.raises(
            ValueError,
            match="Segment with mask_id=5 not found",
        ):
            await service._get_segment_or_raise(
                1,
                5,
            )

    async def test_redis_failure(
        self,
        service,
        mock_redis_storage,
    ):
        mock_redis_storage.get_cached_segments.side_effect = RuntimeError(
            "redis failed"
        )

        with pytest.raises(
            RuntimeError,
            match="redis failed",
        ):
            await service._get_segment_or_raise(
                1,
                5,
            )