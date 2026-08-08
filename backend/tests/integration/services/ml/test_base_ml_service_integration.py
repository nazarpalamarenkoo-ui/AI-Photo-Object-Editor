import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ml.base_ml_service import BaseMLService
from app.db.schemas.model_meta import ModelMeta

pytestmark = pytest.mark.integration



@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def ml_service(
    mock_s3_storage,
    mock_redis_cache,
    mock_redis_history,
    mock_redis_assets,
    image_repo,
    image_version_repo,
    image_content_repo,
    detection_repo,
    segmentation_repo,
    edit_history_repo,
    assets_repo,
):
    return BaseMLService(
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        image_version_repo=image_version_repo,
        image_content_repo=image_content_repo,
        detection_repo=detection_repo,
        segmentation_repo=segmentation_repo,
        edit_history_repo=edit_history_repo,
        assets_repo=assets_repo,
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


class TestGetCurrentVersionAuthorized:
    @pytest.mark.asyncio
    async def test_returns_image_and_version_for_correct_owner(
        self, ml_service, sample_image, sample_image_version, sample_user
    ):
        image, version = await ml_service._get_current_version_authorized(
            sample_image.id, sample_user.id
        )

        assert image.id == sample_image.id
        assert version.id == sample_image_version.id
        assert version.content_id is not None

    @pytest.mark.asyncio
    async def test_raises_when_image_not_found(self, ml_service, sample_user):
        with pytest.raises(ValueError, match="not found"):
            await ml_service._get_current_version_authorized(999999, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, ml_service, sample_image, sample_image_version):
        with pytest.raises(ValueError, match="Unauthorized"):
            await ml_service._get_current_version_authorized(
                sample_image.id, sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_raises_when_no_current_version(self, ml_service, sample_image, sample_user):
        # image exists but has no current version set
        with pytest.raises(ValueError, match="no current version"):
            await ml_service._get_current_version_authorized(sample_image.id, sample_user.id)



class TestGetCurrentImageBytes:
    @pytest.mark.asyncio
    async def test_returns_redis_cached_bytes_without_calling_s3(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        cached_bytes = b"cached-image-bytes"
        mock_redis_cache.get_cache_image = AsyncMock(return_value=cached_bytes)
        mock_s3_storage.download = AsyncMock(return_value=b"s3-bytes")

        result = await ml_service._get_current_image_bytes(
            sample_image.id, sample_image.storage_path
        )

        assert result == cached_bytes
        mock_s3_storage.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_s3_on_cache_miss(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        s3_bytes = b"s3-image-bytes"
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_s3_storage.download = AsyncMock(return_value=s3_bytes)

        result = await ml_service._get_current_image_bytes(
            sample_image.id, sample_image.storage_path
        )

        assert result == s3_bytes
        mock_s3_storage.download.assert_awaited_once_with(sample_image.storage_path)

    @pytest.mark.asyncio
    async def test_propagates_exception_on_s3_failure(
        self, ml_service, sample_image, mock_redis_cache, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_s3_storage.download = AsyncMock(side_effect=RuntimeError("s3 down"))

        with pytest.raises(RuntimeError, match="s3 down"):
            await ml_service._get_current_image_bytes(
                sample_image.id, sample_image.storage_path
            )


class TestSaveCurrentState:
    @pytest.mark.asyncio
    async def test_persists_state_with_correct_args(
        self, ml_service, sample_image, mock_redis_cache
    ):
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

        result_url, presigned_url = await ml_service._upload_result(
            b"result-bytes", "results/1/2/result.jpg"
        )

        assert result_url == "s3://bucket/result.jpg"
        assert presigned_url == "https://presigned-url"
        mock_s3_storage.upload_bytes.assert_awaited_once_with(
            data=b"result-bytes",
            path="results/1/2/result.jpg",
            content_type="image/jpeg",
        )
        mock_s3_storage.get_presigned_url.assert_awaited_once_with(
            path="results/1/2/result.jpg", expiration=3600
        )

    @pytest.mark.asyncio
    async def test_propagates_exception_on_upload_failure(self, ml_service, mock_s3_storage):
        mock_s3_storage.upload_bytes = AsyncMock(side_effect=RuntimeError("upload failed"))

        with pytest.raises(RuntimeError, match="upload failed"):
            await ml_service._upload_result(b"result-bytes", "results/1/2/result.jpg")

    @pytest.mark.asyncio
    async def test_propagates_exception_on_presigned_url_failure(
        self, ml_service, mock_s3_storage
    ):
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/result.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(
            side_effect=RuntimeError("presign failed")
        )

        with pytest.raises(RuntimeError, match="presign failed"):
            await ml_service._upload_result(b"result-bytes", "results/1/2/result.jpg")


class TestGetTempUrlFromBytes:
    @pytest.mark.asyncio
    async def test_uploads_to_temp_path_and_returns_presigned_url(
        self, ml_service, sample_image, sample_user, mock_s3_storage
    ):
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/temp.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(
            return_value="https://temp-presigned-url"
        )

        result = await ml_service._get_temp_url_from_bytes(
            sample_image.id, sample_user.id, b"temp-bytes", "undo"
        )

        assert result == "https://temp-presigned-url"
        called_path = mock_s3_storage.upload_bytes.call_args.kwargs["path"]
        assert called_path.startswith(f"temp/{sample_user.id}/{sample_image.id}/undo_")
        mock_s3_storage.get_presigned_url.assert_awaited_once()


class TestForkVersion:
    @pytest.mark.asyncio
    async def test_creates_new_version_with_content(
        self, ml_service, sample_image, sample_image_version
    ):
        # use a minimal valid JPEG-like bytes so PIL can parse dimensions
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format="JPEG")
        result_bytes = buf.getvalue()
        storage_path = f"results/1/{sample_image.id}/test.jpg"

        new_version = await ml_service._fork_version(sample_image, result_bytes, storage_path)

        assert new_version.id != sample_image_version.id
        assert new_version.image_id == sample_image.id
        assert new_version.content_id is not None

    @pytest.mark.asyncio
    async def test_reuses_content_for_identical_bytes(
        self, ml_service, sample_image, sample_image_version
    ):
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), color=(0, 255, 0)).save(buf, format="JPEG")
        result_bytes = buf.getvalue()
        storage_path = f"results/1/{sample_image.id}/dupe.jpg"

        v1 = await ml_service._fork_version(sample_image, result_bytes, storage_path)
        v2 = await ml_service._fork_version(sample_image, result_bytes, storage_path)

        # Both versions point at the same content_id (content-dedup)
        assert v1.content_id == v2.content_id
        assert v1.id != v2.id


class TestGetSegmentOrRaise:
    @pytest.mark.asyncio
    async def test_returns_matching_segment_from_db_and_redis(
        self, ml_service, sample_image, sample_image_version, mock_redis_cache
    ):
        """
        _get_segment_or_raise now reads SegmentationMask rows from DB
        (segmentation_repo.get_by_content) and mask bytes from Redis/S3.
        Seed the repo with a real row and warm the Redis mock.
        """
        from app.db.models.segmentation import SegmentationMask
        from app.db.enums.segmentation_mode import SegmentationMode
        mask = SegmentationMask(
            content_id=sample_image_version.content_id,
            mask_id=7,
            x1=0, y1=0, x2=10, y2=10,
            area=100,
            score=0.9,
            mask_storage_path="masks/test.png",
            preview_storage_path="masks/test_preview.png",
            segmentation_mode=SegmentationMode.SAM,
            model_name="test-model",
            model_version="1.0",
            inference_time_ms=10.0,
            is_active=True,
        )
        await ml_service.segmentation_repo.create_many([mask])

        cache_suffix = f"mask:{sample_image_version.content_id}:7"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else None
        )

        result = await ml_service._get_segment_or_raise(
            content_id=sample_image_version.content_id,
            mask_id=7,
            image_id=sample_image.id,
        )

        assert result["mask_id"] == 7
        assert result["mask_bytes"] == b"mask-bytes"
        assert result["bbox"] == {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

    @pytest.mark.asyncio
    async def test_falls_back_to_s3_on_redis_miss_and_re_caches(
        self, ml_service, sample_image, sample_image_version, mock_redis_cache, mock_s3_storage
    ):
        from app.db.models.segmentation import SegmentationMask
        from app.db.enums.segmentation_mode import SegmentationMode
        mask = SegmentationMask(
            content_id=sample_image_version.content_id,
            mask_id=3,
            x1=5, y1=5, x2=15, y2=15,
            area=50,
            score=0.8,
            mask_storage_path="masks/3.png",
            preview_storage_path="masks/3_preview.png",
            segmentation_mode=SegmentationMode.SAM,
            model_name="test-model",
            model_version="1.0",
            inference_time_ms=10.0,
            is_active=True,
        )
        await ml_service.segmentation_repo.create_many([mask])

        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"s3-mask-bytes")

        result = await ml_service._get_segment_or_raise(
            content_id=sample_image_version.content_id,
            mask_id=3,
            image_id=sample_image.id,
        )

        assert result["mask_bytes"] == b"s3-mask-bytes"
        mock_s3_storage.download.assert_awaited_once_with("masks/3.png")
        mock_redis_cache.cache_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_mask_id_not_in_db(
        self, ml_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="mask_id=99"):
            await ml_service._get_segment_or_raise(
                content_id=sample_image_version.content_id,
                mask_id=99,
                image_id=sample_image.id,
            )


class TestExtractModelMeta:
    def test_reads_from_top_level_result_keys(self):
        result = {
            "model_name": "yolov10m",
            "model_version": "1.0",
            "metrics": {"inference_time_ms": 42.0},
        }

        meta = BaseMLService._extract_model_meta(result, "default_model")

        assert meta.model_name == "yolov10m"
        assert meta.model_version == "1.0"
        assert meta.inference_time_ms == 42.0

    def test_falls_back_to_default_name_and_version(self):
        meta = BaseMLService._extract_model_meta({}, "fallback_name", "v0")

        assert meta.model_name == "fallback_name"
        assert meta.model_version == "v0"

    def test_reads_inference_time_from_nested_metrics(self):
        result = {"metrics": {"inference_time_ms": 123.5}}

        meta = BaseMLService._extract_model_meta(result, "m")

        assert meta.inference_time_ms == 123.5

    def test_converts_inference_time_s_to_ms(self):
        result = {"metrics": {"inference_time_s": 2.0}}

        meta = BaseMLService._extract_model_meta(result, "m")

        assert meta.inference_time_ms == 2000.0

    def test_converts_processing_time_s_to_ms_as_fallback(self):
        result = {"metrics": {"processing_time_s": 0.5}}

        meta = BaseMLService._extract_model_meta(result, "m")

        assert meta.inference_time_ms == 500.0

    def test_accepts_bare_metrics_dict_without_wrapper(self):
        metrics = {"model_name": "mobile_sam", "inference_time_ms": 77.0}

        meta = BaseMLService._extract_model_meta(metrics, "default")

        assert meta.model_name == "mobile_sam"
        assert meta.inference_time_ms == 77.0

    def test_none_input_uses_all_defaults(self):
        meta = BaseMLService._extract_model_meta(None, "my_model", "2.1")

        assert meta.model_name == "my_model"
        assert meta.model_version == "2.1"
        assert meta.inference_time_ms == 0.0

    def test_returns_model_meta_instance(self):
        meta = BaseMLService._extract_model_meta({}, "x")

        assert isinstance(meta, ModelMeta)

    def test_prefers_ms_over_s_when_both_present(self):
        result = {"metrics": {"inference_time_ms": 10.0, "inference_time_s": 99.0}}

        meta = BaseMLService._extract_model_meta(result, "m")

        assert meta.inference_time_ms == 10.0


class TestExtractProcessingTimeMs:
    def test_reads_processing_time_ms(self):
        assert BaseMLService._extract_processing_time_ms(
            {"processing_time_ms": 200}
        ) == 200

    def test_reads_total_time_ms_as_fallback(self):
        assert BaseMLService._extract_processing_time_ms(
            {"total_time_ms": 150}
        ) == 150

    def test_reads_inference_time_ms_as_last_ms_fallback(self):
        assert BaseMLService._extract_processing_time_ms(
            {"inference_time_ms": 99}
        ) == 99

    def test_converts_inference_time_s_to_ms(self):
        assert BaseMLService._extract_processing_time_ms(
            {"inference_time_s": 1.5}
        ) == 1500

    def test_returns_none_for_empty_dict(self):
        assert BaseMLService._extract_processing_time_ms({}) is None

    def test_returns_none_for_none_input(self):
        assert BaseMLService._extract_processing_time_ms(None) is None

    def test_prefers_processing_time_ms_over_inference_time_s(self):
        assert BaseMLService._extract_processing_time_ms(
            {"processing_time_ms": 42, "inference_time_s": 99}
        ) == 42