import io
import pytest
from unittest.mock import AsyncMock
from PIL import Image as PILImage

pytestmark = pytest.mark.integration


def _jpeg_bytes(color=(0, 100, 200)):
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(color=(0, 200, 100)):
    buf = io.BytesIO()
    PILImage.new("RGBA", (10, 10), color=(*color, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_redis_history():
    return AsyncMock()


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


def _make_service(cls, mock_s3_storage, mock_redis_cache, mock_redis_history,
                  mock_redis_assets, image_repo, image_version_repo,
                  image_content_repo, detection_repo, segmentation_repo,
                  edit_history_repo, assets_repo, mock_pipeline):
    return cls(
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
        pipeline=mock_pipeline,
    )

from app.services.ml.assets_service import AssetService


@pytest.fixture
def asset_service(
    mock_s3_storage, mock_redis_cache, mock_redis_history, mock_redis_assets,
    image_repo, image_version_repo, image_content_repo,
    detection_repo, segmentation_repo, edit_history_repo, assets_repo, mock_pipeline,
):
    return _make_service(
        AssetService, mock_s3_storage, mock_redis_cache, mock_redis_history,
        mock_redis_assets, image_repo, image_version_repo, image_content_repo,
        detection_repo, segmentation_repo, edit_history_repo, assets_repo, mock_pipeline,
    )


@pytest.fixture
async def sample_mask(segmentation_repo, sample_image_version):
    from app.db.models.segmentation import SegmentationMask
    from app.db.enums.segmentation_mode import SegmentationMode
    mask = SegmentationMask(
        content_id=sample_image_version.content_id,
        mask_id=1,
        x1=0, y1=0, x2=10, y2=10,
        area=100.0, score=0.9,
        mask_storage_path="masks/1.png",
        preview_storage_path="masks/1.png",
        segmentation_mode=SegmentationMode.SAM,
        model_name="mobile_sam", model_version="unknown", inference_time_ms=0.0,
    )
    await segmentation_repo.create_many([mask])
    return mask


def _extract_pipeline_result():
    return {
        "extracted_bytes": _png_bytes(),
        "object_size": (10, 10),
        "area_pixels": 100,
        "cropped_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "timestamp": "ts",
    }


class TestExtractObject:
    @pytest.mark.asyncio
    async def test_success_uploads_png_and_thumbnail_and_persists_asset(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/asset.png")

        result = await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )

        assert "asset_id" in result
        assert "storage_path" in result
        assert "thumbnail_path" in result
        # two upload_bytes calls: extracted + thumbnail
        assert mock_s3_storage.upload_bytes.await_count == 2
        calls = [c.kwargs["content_type"] for c in mock_s3_storage.upload_bytes.call_args_list]
        assert all(ct == "image/png" for ct in calls)

    @pytest.mark.asyncio
    async def test_asset_row_persisted_to_db(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/asset.png")

        result = await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id, label="my object"
        )

        assets = await asset_service.assets_repo.list_by_user(sample_user.id)
        assert len(assets) == 1
        assert assets[0].public_id == result["asset_id"]

    @pytest.mark.asyncio
    async def test_raises_when_mask_not_found(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="mask_id=99"):
            await asset_service.extract_object(
                sample_image.id, mask_id=99, user_id=sample_user.id
            )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, asset_service, sample_image, sample_image_version, sample_mask
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await asset_service.extract_object(
                sample_image.id, mask_id=1, user_id=sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(
            side_effect=RuntimeError("extract failed")
        )

        with pytest.raises(RuntimeError, match="extract failed"):
            await asset_service.extract_object(
                sample_image.id, mask_id=1, user_id=sample_user.id
            )

    @pytest.mark.asyncio
    async def test_passes_params_to_pipeline(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/a.png")

        await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id, padding_pixels=16
        )

        kw = mock_pipeline.sam_extract_object.call_args.kwargs
        assert kw["padding_pixels"] == 16
        assert kw["mask_bytes"] == b"mask-bytes"


class TestPasteExtractedObject:
    def _paste_result(self):
        return {
            "result_bytes": _jpeg_bytes(),
            "paste_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "object_size": (10, 10),
            "timestamp": "ts",
        }

    def _setup(self, mock_redis_cache, mock_s3_storage, mock_pipeline):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=self._paste_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/paste.jpg")
        mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://paste-url")

    @pytest.mark.asyncio
    async def test_success_with_extracted_url(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        self._setup(mock_redis_cache, mock_s3_storage, mock_pipeline)

        result = await asset_service.paste_extracted_object(
            sample_image.id, sample_user.id,
            target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        assert result["result_url"] == "s3://bucket/paste.jpg"
        assert result["presigned_url"] == "https://paste-url"
        mock_s3_storage.download.assert_awaited_once_with("s3://bucket/extracted.png")

    @pytest.mark.asyncio
    async def test_pushes_undo_state(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        self._setup(mock_redis_cache, mock_s3_storage, mock_pipeline)

        await asset_service.paste_extracted_object(
            sample_image.id, sample_user.id,
            target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        label = mock_redis_history.push_undo_state.call_args.kwargs["label"]
        assert "paste extracted" in label

    @pytest.mark.asyncio
    async def test_saves_current_state_in_redis(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        self._setup(mock_redis_cache, mock_s3_storage, mock_pipeline)
        res = self._paste_result()
        mock_pipeline.sam_paste_extracted_object = AsyncMock(return_value=res)

        await asset_service.paste_extracted_object(
            sample_image.id, sample_user.id,
            target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
        )

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=res["result_bytes"],
            suffix="current_state",
            ttl=7200,
        )

    @pytest.mark.asyncio
    async def test_raises_when_no_source_provided(
        self, asset_service, sample_image, sample_user
    ):
        with pytest.raises(ValueError, match="Provide either"):
            await asset_service.paste_extracted_object(
                sample_image.id, sample_user.id,
                target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            )

    @pytest.mark.asyncio
    async def test_raises_when_download_fails(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_s3_storage.download = AsyncMock(side_effect=RuntimeError("S3 error"))

        with pytest.raises(ValueError, match="Failed to download extracted object"):
            await asset_service.paste_extracted_object(
                sample_image.id, sample_user.id,
                target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://bucket/missing.png",
            )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, asset_service, sample_image
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await asset_service.paste_extracted_object(
                sample_image.id, sample_image.user_id + 1,
                target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://x",
            )

    @pytest.mark.asyncio
    async def test_forwards_optional_params_to_pipeline(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        self._setup(mock_redis_cache, mock_s3_storage, mock_pipeline)

        await asset_service.paste_extracted_object(
            sample_image.id, sample_user.id,
            target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            extracted_url="s3://bucket/extracted.png",
            scale=1.5,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method="histogram",
        )

        kw = mock_pipeline.sam_paste_extracted_object.call_args.kwargs
        assert kw["scale"] == 1.5
        assert kw["use_color_matching"] is True
        assert kw["use_edge_blending"] is True
        assert kw["color_match_method"] == "histogram"

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        asset_service,
        sample_image,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_s3_storage.download = AsyncMock(return_value=b"extracted-png")
        mock_pipeline.sam_paste_extracted_object = AsyncMock(
            side_effect=RuntimeError("paste failed")
        )

        with pytest.raises(RuntimeError, match="paste failed"):
            await asset_service.paste_extracted_object(
                sample_image.id, sample_user.id,
                target_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                extracted_url="s3://bucket/extracted.png",
            )


class TestListAssets:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_assets(
        self, asset_service, sample_user
    ):
        result = await asset_service.list_assets(sample_user.id)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_persisted_assets(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/a.png")

        await asset_service.extract_object(sample_image.id, mask_id=1, user_id=sample_user.id)
        assets = await asset_service.list_assets(sample_user.id)

        assert len(assets) == 1


class TestRenameAsset:
    @pytest.mark.asyncio
    async def test_success_renames_asset(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/a.png")

        extract_result = await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )
        renamed = await asset_service.rename_asset(
            sample_user.id, extract_result["asset_id"], "new label"
        )

        assert renamed.label == "new label"

    @pytest.mark.asyncio
    async def test_raises_when_asset_not_found(self, asset_service, sample_user):
        with pytest.raises(ValueError, match="Asset not found"):
            await asset_service.rename_asset(sample_user.id, "nonexistent-id", "label")


class TestDeleteAsset:
    @pytest.mark.asyncio
    async def test_success_deletes_asset_from_db_and_s3(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/a.png")
        mock_s3_storage.delete = AsyncMock()

        extract_result = await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )
        await asset_service.delete_asset(sample_user.id, extract_result["asset_id"])

        assets = await asset_service.list_assets(sample_user.id)
        assert assets == []
        assert mock_s3_storage.delete.await_count >= 1

    @pytest.mark.asyncio
    async def test_raises_when_asset_not_found(self, asset_service, sample_user):
        with pytest.raises(ValueError, match="Asset not found"):
            await asset_service.delete_asset(sample_user.id, "nonexistent-id")

    @pytest.mark.asyncio
    async def test_s3_failure_does_not_block_db_deletion(
        self,
        asset_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """_delete_asset_files swallows S3 errors — DB row should still be gone."""
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_extract_object = AsyncMock(return_value=_extract_pipeline_result())
        mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/a.png")
        mock_s3_storage.delete = AsyncMock(side_effect=RuntimeError("s3 down"))

        extract_result = await asset_service.extract_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )
        # should NOT raise even though S3 delete fails
        await asset_service.delete_asset(sample_user.id, extract_result["asset_id"])

        assets = await asset_service.list_assets(sample_user.id)
        assert assets == []