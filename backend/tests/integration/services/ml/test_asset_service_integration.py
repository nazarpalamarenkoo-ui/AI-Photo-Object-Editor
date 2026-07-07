import pytest
from unittest.mock import AsyncMock

from app.services.ml.assets_service import AssetService

pytestmark = pytest.mark.integration

@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def asset_service(db_session, mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets, image_repo, detection_repo, mock_pipeline):
    return AssetService(
        db=db_session,
        s3_storage=mock_s3_storage,
        redis_storage=mock_redis_cache,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=image_repo,
        detection_repo=detection_repo,
        pipeline=mock_pipeline,
    )


def _cached_segment():
    return {"mask_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "mask_bytes": b"mask-bytes"}


def _extract_result():
    return {
        "extracted_bytes": b"png-bytes",
        "object_size": (10, 10),
        "area_pixels": 100,
        "cropped_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "timestamp": "ts",
    }


class TestExtractObject:
    @pytest.mark.asyncio
    async def test_success_uploads_png(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=[_cached_segment()])
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/extracted.png")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://extract-url")

        # persist_to_s3 за замовчуванням False у сервісі -> явно вмикаємо,
        # інакше extracted_url лишиться None
        result = await asset_service.extract_object(
            sample_image.id, 1, sample_user.id, persist_to_s3=True
        )

        assert result["extracted_url"] == "s3://bucket/extracted.png"
        mock_s3_storage.upload_bytes.assert_awaited_once()
        assert mock_s3_storage.upload_bytes.call_args.kwargs["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(self, asset_service, sample_image):
        with pytest.raises(ValueError, match="Unauthorized"):
            await asset_service.extract_object(sample_image.id, 1, sample_image.user_id + 1)

    @pytest.mark.asyncio
    async def test_raises_when_segment_missing(self, asset_service, sample_image, sample_user, mock_redis_cache):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="No segments found"):
            await asset_service.extract_object(sample_image.id, 1, sample_user.id)

    @pytest.mark.asyncio
    async def test_propagates_pipeline_exception(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.get_cached_segments = AsyncMock(return_value=[_cached_segment()])
        mock_pipeline.sam_extract_object = AsyncMock(side_effect=RuntimeError("extract failed"))

        with pytest.raises(RuntimeError, match="extract failed"):
            await asset_service.extract_object(sample_image.id, 1, sample_user.id)


class TestPasteExtractedObject:
    def _result(self):
        return {
            "result_bytes": b"pasted-bytes",
            "paste_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "object_size": (10, 10),
            "timestamp": "ts",
        }

    @pytest.mark.asyncio
    async def test_success_downloads_extracted_and_uploads_result(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=self._result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/paste.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://paste-url")

        # реальна сигнатура: (image_id, user_id, target_bbox, asset_id=None, extracted_url=None, ...)
        # extracted_url обов'язково як keyword, інакше піде в asset_id і зайде у гілку asset-бібліотеки
        result = await asset_service.paste_extracted_object(
            sample_image.id,
            sample_user.id,
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        assert result["result_url"] == "s3://bucket/paste.jpg"
        mock_s3_storage.download.assert_awaited_once_with("s3://bucket/extracted.png")

    @pytest.mark.asyncio
    async def test_pushes_undo_history(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_redis_history, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=self._result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/paste.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://paste-url")

        await asset_service.paste_extracted_object(
            sample_image.id,
            sample_user.id,
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_saves_current_state(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=self._result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/paste.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://paste-url")

        await asset_service.paste_extracted_object(
            sample_image.id,
            sample_user.id,
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id, image_data=b"pasted-bytes", suffix="current_state", ttl=7200
        )

    @pytest.mark.asyncio
    async def test_raises_when_extracted_download_fails(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_s3_storage.download = AsyncMock(side_effect=RuntimeError("download failed"))

        with pytest.raises(ValueError, match="Failed to download extracted object"):
            await asset_service.paste_extracted_object(
                sample_image.id,
                sample_user.id,
                {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://bucket/missing.png",
            )

    @pytest.mark.asyncio
    async def test_propagates_exception_on_upload_failure(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=self._result())
        mock_s3_storage.upload_bytes = AsyncMock(side_effect=RuntimeError("upload failed"))

        with pytest.raises(RuntimeError, match="upload failed"):
            await asset_service.paste_extracted_object(
                sample_image.id,
                sample_user.id,
                {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://bucket/extracted.png",
            )

    @pytest.mark.asyncio
    async def test_propagates_exception_on_pipeline_failure(
        self, asset_service, sample_image, sample_user, mock_redis_cache, mock_pipeline, mock_s3_storage
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png-bytes")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(side_effect=RuntimeError("paste failed"))

        with pytest.raises(RuntimeError, match="paste failed"):
            await asset_service.paste_extracted_object(
                sample_image.id,
                sample_user.id,
                {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://bucket/extracted.png",
            )