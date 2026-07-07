import pytest
from unittest.mock import AsyncMock

from app.services.ml.base_ml_service import BaseMLService

pytestmark = pytest.mark.integration

@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def ml_service(db_session, mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets, image_repo, detection_repo):
    return BaseMLService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        detection_repo=detection_repo,
        pipeline=AsyncMock(),
    )


class TestGetImageAuthorized:
    @pytest.mark.asyncio
    async def test_returns_image_for_correct_owner(self, ml_service, sample_image, sample_user):
        result = await ml_service._get_image_authorized(sample_image.id, sample_user.id)

        assert result.id == sample_image.id
        assert result.user_id == sample_user.id

    @pytest.mark.asyncio
    async def test_raises_when_image_not_found(self, ml_service, sample_user):
        with pytest.raises(ValueError, match="not found"):
            await ml_service._get_image_authorized(999999, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_owner_mismatch(self, ml_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await ml_service._get_image_authorized(sample_image.id, sample_image.user_id + 1)


class TestGetCurrentImageBytes:
    @pytest.mark.asyncio
    async def test_returns_redis_cached_bytes_without_calling_s3(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        cached_bytes = b"cached-image-bytes"
        mock_redis_cache.get_cache_image = AsyncMock(return_value=cached_bytes)
        mock_s3_storage.download = AsyncMock(return_value=b"s3-bytes")

        result = await ml_service._get_current_image_bytes(sample_image.id, sample_image.storage_path)

        assert result == cached_bytes
        mock_s3_storage.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_s3_on_cache_miss(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        s3_bytes = b"s3-image-bytes"
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_s3_storage.download = AsyncMock(return_value=s3_bytes)

        result = await ml_service._get_current_image_bytes(sample_image.id, sample_image.storage_path)

        assert result == s3_bytes
        mock_s3_storage.download.assert_awaited_once_with(sample_image.storage_path)

    @pytest.mark.asyncio
    async def test_propagates_exception_on_s3_failure(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_s3_storage.download = AsyncMock(side_effect=RuntimeError("s3 down"))

        with pytest.raises(RuntimeError, match="s3 down"):
            await ml_service._get_current_image_bytes(sample_image.id, sample_image.storage_path)


class TestSaveCurrentState:
    @pytest.mark.asyncio
    async def test_persists_state_with_correct_args(self, ml_service, sample_image, mock_redis_cache):
        mock_redis_cache.cache_image = AsyncMock()

        await ml_service._save_current_state(sample_image.id, b"new-state-bytes")

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=b"new-state-bytes",
            suffix="current_state",
            ttl=7200,
        )


class TestUploadResult:
    @pytest.mark.asyncio
    async def test_uploads_and_returns_urls(self, ml_service, mock_s3_storage):
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://presigned-url")

        result_url, presigned_url = await ml_service._upload_result(b"result-bytes", "results/1/2/result.jpg")

        assert result_url == "s3://bucket/result.jpg"
        assert presigned_url == "https://presigned-url"
        mock_s3_storage.upload_bytes.assert_awaited_once_with(
            data=b"result-bytes", path="results/1/2/result.jpg", content_type="image/jpeg"
        )
        mock_s3_storage.get_presigned_url.assert_awaited_once_with(path="results/1/2/result.jpg", expiration=3600)

    @pytest.mark.asyncio
    async def test_propagates_exception_on_upload_failure(self, ml_service, mock_s3_storage):
        mock_s3_storage.upload_bytes = AsyncMock(side_effect=RuntimeError("upload failed"))

        with pytest.raises(RuntimeError, match="upload failed"):
            await ml_service._upload_result(b"result-bytes", "results/1/2/result.jpg")

    @pytest.mark.asyncio
    async def test_propagates_exception_on_presigned_url_failure(self, ml_service, mock_s3_storage):
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(side_effect=RuntimeError("presign failed"))

        with pytest.raises(RuntimeError, match="presign failed"):
            await ml_service._upload_result(b"result-bytes", "results/1/2/result.jpg")


class TestGetTempUrlFromBytes:
    @pytest.mark.asyncio
    async def test_uploads_to_temp_path_and_returns_presigned_url(
        self, ml_service, sample_image, sample_user, mock_s3_storage
    ):
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/temp.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://temp-presigned-url")

        result = await ml_service._get_temp_url_from_bytes(
            sample_image.id, sample_user.id, b"temp-bytes", "undo"
        )

        assert result == "https://temp-presigned-url"
        called_path = mock_s3_storage.upload_bytes.call_args.kwargs["path"]
        assert called_path.startswith(f"temp/{sample_user.id}/{sample_image.id}/undo_")
        mock_s3_storage.get_presigned_url.assert_awaited_once()


class TestGetSegmentOrRaise:
    @pytest.mark.asyncio
    async def test_returns_matching_segment(self, ml_service, sample_image, mock_redis_cache):
        segments = [
            {"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"a"},
            {"mask_id": 2, "bbox": {"x1": 5, "y1": 5, "x2": 15, "y2": 15}, "mask_bytes": b"b"},
        ]
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=segments)

        result = await ml_service._get_segment_or_raise(sample_image.id, 2)

        assert result["mask_id"] == 2
        assert result["mask_bytes"] == b"b"

    @pytest.mark.asyncio
    async def test_raises_when_redis_returns_none(self, ml_service, sample_image, mock_redis_cache):
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await ml_service._get_segment_or_raise(sample_image.id, 1)

    @pytest.mark.asyncio
    async def test_raises_when_segments_list_empty(self, ml_service, sample_image, mock_redis_cache):
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="No segments found"):
            await ml_service._get_segment_or_raise(sample_image.id, 1)

    @pytest.mark.asyncio
    async def test_raises_when_mask_id_not_present(self, ml_service, sample_image, mock_redis_cache):
        mock_redis_cache.get_cached_segments = AsyncMock(
            return_value=[{"mask_id": 1, "bbox": {}, "mask_bytes": b"a"}]
        )

        with pytest.raises(ValueError, match="mask_id=99 not found"):
            await ml_service._get_segment_or_raise(sample_image.id, 99)