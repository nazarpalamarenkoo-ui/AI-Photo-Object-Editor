import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image as PILImage

from app.services.ml.segmentation_service import (
    SegmentationService,
    _mask_to_data_url,
    _segments_for_response,
)

pytestmark = pytest.mark.integration


def _jpeg_bytes(color=(0, 128, 255)):
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=color).save(buf, format="JPEG")
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


@pytest.fixture
def segmentation_service(
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
    mock_pipeline,
):
    return SegmentationService(
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


def _segment(mask_id=0, x1=0, y1=0, x2=10, y2=10, source=None):
    seg = {
        "mask_id": mask_id,
        "bbox_id": mask_id,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "area": 100.0,
        "stability_score": 0.9,
        "mask_bytes": b"mask-bytes",
    }
    if source:
        seg["source"] = source
    return seg


def _pipeline_seg_result(n=1, image_size=(100, 100)):
    return {
        "segments": [_segment(mask_id=i) for i in range(n)],
        "metrics": {"inference_time_ms": 10.0},
        "image_size": image_size,
    }


def _setup_s3_upload(mock_s3_storage):
    mock_s3_storage.upload_bytes = AsyncMock(return_value="s3://bucket/mask.png")
    mock_s3_storage.get_presigned_url = AsyncMock(return_value="https://presigned")


class TestSegmentObjects:
    @pytest.mark.asyncio
    async def test_success_persists_masks_and_returns_segments(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value=_pipeline_seg_result(n=2))
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_objects(sample_image.id, sample_user.id)

        assert len(result["segments"]) == 2
        assert all("mask_bytes" not in s for s in result["segments"])
        assert "image_size" in result
        persisted = await segmentation_service.segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        assert len(persisted) == 2

    @pytest.mark.asyncio
    async def test_second_call_served_from_db_cache_skips_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value=_pipeline_seg_result(n=1))
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_objects(sample_image.id, sample_user.id)
        mock_pipeline.sam_segment_objects.reset_mock()

        result = await segmentation_service.segment_objects(sample_image.id, sample_user.id)

        mock_pipeline.sam_segment_objects.assert_not_called()
        assert result["metrics"] == {"cache_hit": True}
        assert len(result["segments"]) == 1

    @pytest.mark.asyncio
    async def test_mask_ids_start_at_zero_on_fresh_content(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value=_pipeline_seg_result(n=3))
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_objects(sample_image.id, sample_user.id)

        persisted = await segmentation_service.segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        mask_ids = sorted(m.mask_id for m in persisted)
        assert mask_ids == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_objects(
                sample_image.id, sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_raises_when_no_current_version(
        self, segmentation_service, sample_image, sample_user
    ):
        with pytest.raises(ValueError, match="no current version"):
            await segmentation_service.segment_objects(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_passes_params_to_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value=_pipeline_seg_result())
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_objects(
            sample_image.id, sample_user.id, min_area=1000, max_segments=10
        )

        kw = mock_pipeline.sam_segment_objects.call_args.kwargs
        assert kw["min_area"] == 1000
        assert kw["max_segments"] == 10

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_segment_objects = AsyncMock(
            side_effect=RuntimeError("sam crashed")
        )

        with pytest.raises(RuntimeError, match="sam crashed"):
            await segmentation_service.segment_objects(sample_image.id, sample_user.id)


class TestSegmentWithPrompt:
    @pytest.mark.asyncio
    async def test_success_persists_and_returns_segments(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(
            return_value=_pipeline_seg_result(n=1)
        )
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_with_prompt(
            sample_image.id, sample_user.id,
            point_coords=[(5, 5)], point_labels=[1],
        )

        assert len(result["segments"]) == 1
        assert "mask_bytes" not in result["segments"][0]
        persisted = await segmentation_service.segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        assert len(persisted) == 1

    @pytest.mark.asyncio
    async def test_passes_prompt_params_to_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(
            return_value=_pipeline_seg_result()
        )
        _setup_s3_upload(mock_s3_storage)
        bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}

        await segmentation_service.segment_with_prompt(
            sample_image.id, sample_user.id,
            point_coords=[(5, 5)], point_labels=[1],
            bbox=bbox, multimask_output=True,
        )

        kw = mock_pipeline.sam_segment_with_prompt.call_args.kwargs
        assert kw["point_coords"] == [(5, 5)]
        assert kw["point_labels"] == [1]
        assert kw["bbox"] == bbox
        assert kw["multimask_output"] is True

    @pytest.mark.asyncio
    async def test_mask_ids_increment_after_existing_masks(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
        segmentation_repo,
    ):
        """If content already has masks 0..2, new masks should start at 3."""
        from app.db.models.segmentation import SegmentationMask
        from app.db.enums.segmentation_mode import SegmentationMode
        existing = [
            SegmentationMask(
                content_id=sample_image_version.content_id,
                mask_id=i, x1=0, y1=0, x2=5, y2=5,
                area=25.0, score=0.8,
                mask_storage_path="masks/x.png",
                preview_storage_path="masks/x.png",
                segmentation_mode=SegmentationMode.SAM,
                model_name="m", model_version="v", inference_time_ms=0.0,
            )
            for i in range(3)
        ]
        await segmentation_repo.create_many(existing)

        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_with_prompt = AsyncMock(
            return_value=_pipeline_seg_result(n=2)
        )
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_with_prompt(
            sample_image.id, sample_user.id, point_coords=[(5, 5)], point_labels=[1]
        )

        all_masks = await segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        new_ids = sorted(m.mask_id for m in all_masks if m.mask_id >= 3)
        assert new_ids == [3, 4]

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_with_prompt(
                sample_image.id, sample_image.user_id + 1, point_coords=[(5, 5)]
            )

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_segment_with_prompt = AsyncMock(
            side_effect=RuntimeError("prompt failed")
        )

        with pytest.raises(RuntimeError, match="prompt failed"):
            await segmentation_service.segment_with_prompt(
                sample_image.id, sample_user.id, point_coords=[(5, 5)]
            )


class TestSegmentByPolygon:
    @pytest.mark.asyncio
    async def test_success_persists_and_returns_segment(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(
            return_value=_pipeline_seg_result(n=1)
        )
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)]
        )

        assert len(result["segments"]) == 1
        assert "mask_bytes" not in result["segments"][0]
        persisted = await segmentation_service.segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        assert len(persisted) == 1

    @pytest.mark.asyncio
    async def test_passes_polygon_params_to_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(
            return_value=_pipeline_seg_result()
        )
        _setup_s3_upload(mock_s3_storage)
        points = [(0, 0), (10, 0), (5, 10)]

        await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=points,
            smooth=False, smoothing_factor=0.4, feather_px=3,
        )

        kw = mock_pipeline.sam_segment_by_polygon.call_args.kwargs
        assert kw["points"] == points
        assert kw["smooth"] is False
        assert kw["smoothing_factor"] == 0.4
        assert kw["feather_px"] == 3

    @pytest.mark.asyncio
    async def test_all_polygon_segments_share_same_mask_id_offset(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """segment_by_polygon assigns the same offset to ALL segments (single lasso)."""
        raw = _pipeline_seg_result(n=1)
        raw["segments"][0]["mask_id"] = 999  # will be overwritten by offset logic
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value=raw)
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_by_polygon(
            sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)]
        )

        # offset = max_mask_id (0 initially, no existing masks) → assigned as 0
        assert result["segments"][0]["mask_id"] == 0

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_by_polygon(
                sample_image.id, sample_image.user_id + 1, points=[(0, 0)]
            )

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_segment_by_polygon = AsyncMock(
            side_effect=RuntimeError("polygon failed")
        )

        with pytest.raises(RuntimeError, match="polygon failed"):
            await segmentation_service.segment_by_polygon(
                sample_image.id, sample_user.id, points=[(0, 0), (10, 0), (5, 10)]
            )


class TestSegmentHybrid:
    def _yolo_result(self, detections):
        return {"detections": detections}

    def _det(self, x1=0, y1=0, x2=10, y2=10):
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.9}

    @pytest.mark.asyncio
    async def test_success_combines_yolo_and_fallback_sources(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(
            return_value=self._yolo_result([self._det(0, 0, 10, 10)])
        )
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [_segment(x1=0, y1=0, x2=10, y2=10)],
            "metrics": {},
        })
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(x1=50, y1=50, x2=60, y2=60)],
            "metrics": {},
            "image_size": (300, 300),
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id
        )

        sources = {s["source"] for s in result["segments"]}
        assert sources == {"yolo", "sam_auto"}
        assert len(result["segments"]) == 2
        assert all("mask_bytes" not in s for s in result["segments"])
        assert result["image_size"] == (300, 300)

    @pytest.mark.asyncio
    async def test_overlapping_fallback_segment_dropped(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(
            return_value=self._yolo_result([self._det(0, 0, 10, 10)])
        )
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [_segment(x1=0, y1=0, x2=10, y2=10)],
            "metrics": {},
        })
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(x1=0, y1=0, x2=10, y2=10)],  # identical bbox
            "metrics": {},
            "image_size": (100, 100),
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id, overlap_iou_thresh=0.5
        )

        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "yolo"

    @pytest.mark.asyncio
    async def test_no_yolo_detections_skips_batch_call(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(
            return_value=self._yolo_result([])
        )
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock()
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment()],
            "metrics": {},
            "image_size": (50, 50),
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id
        )

        mock_pipeline.sam_segment_with_prompts_batch.assert_not_called()
        assert result["segments"][0]["source"] == "sam_auto"

    @pytest.mark.asyncio
    async def test_passes_yolo_and_fallback_params_to_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(return_value=self._yolo_result([]))
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_hybrid(
            sample_image.id, sample_user.id,
            yolo_conf_threshold=0.7, yolo_classes=["dog"],
            fallback_min_area=900, fallback_max_segments=20,
        )

        det_kw = mock_pipeline.detect_objects.call_args.kwargs
        assert det_kw["conf_threshold"] == 0.7
        assert det_kw["classes"] == ["dog"]
        fb_kw = mock_pipeline.sam_segment_objects.call_args.kwargs
        assert fb_kw["min_area"] == 900
        assert fb_kw["max_segments"] == 20

    @pytest.mark.asyncio
    async def test_persists_all_segments_to_db(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.detect_objects = AsyncMock(
            return_value=self._yolo_result([self._det(0, 0, 10, 10)])
        )
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [_segment(x1=0, y1=0, x2=10, y2=10)],
            "metrics": {},
        })
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [_segment(x1=50, y1=50, x2=60, y2=60)],
            "metrics": {},
            "image_size": (100, 100),
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

        persisted = await segmentation_service.segmentation_repo.get_by_content(
            sample_image_version.content_id, active_only=True
        )
        assert len(persisted) == 2

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.segment_hybrid(
                sample_image.id, sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_propagates_yolo_exception(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(
            side_effect=RuntimeError("yolo crashed")
        )

        with pytest.raises(RuntimeError, match="yolo crashed"):
            await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)

    @pytest.mark.asyncio
    async def test_propagates_fallback_exception(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.detect_objects = AsyncMock(return_value=self._yolo_result([]))
        mock_pipeline.sam_segment_objects = AsyncMock(
            side_effect=RuntimeError("fallback crashed")
        )

        with pytest.raises(RuntimeError, match="fallback crashed"):
            await segmentation_service.segment_hybrid(sample_image.id, sample_user.id)


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


class TestSamRemoveObject:
    @pytest.mark.asyncio
    async def test_success_uploads_result_and_returns_new_version(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.sam_remove_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )

        assert result["result_url"] == "s3://bucket/mask.png"
        assert "image_version_id" in result
        assert result["image_version_id"] != sample_image_version.id

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.sam_remove_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )

        versions = await segmentation_service.image_version_repo.list_by_image(
            sample_image.id
        )
        assert len(versions) == 2

    @pytest.mark.asyncio
    async def test_pushes_undo_state_before_pipeline(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.sam_remove_object(
            sample_image.id, mask_id=1, user_id=sample_user.id
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        assert mock_redis_history.push_undo_state.call_args.kwargs["label"] == "sam_remove mask_id=1"

    @pytest.mark.asyncio
    async def test_raises_when_mask_not_found(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="mask_id=999"):
            await segmentation_service.sam_remove_object(
                sample_image.id, mask_id=999, user_id=sample_user.id
            )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version, sample_mask
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.sam_remove_object(
                sample_image.id, mask_id=1, user_id=sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_passes_lama_params_to_pipeline(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.sam_remove_object(
            sample_image.id, mask_id=1, user_id=sample_user.id,
            expand_mask_pixels=20, ldm_steps=50, ldm_sampler="ddim",
        )

        kw = mock_pipeline.sam_remove_object.call_args.kwargs
        assert kw["expand_mask_pixels"] == 20
        assert kw["ldm_steps"] == 50
        assert kw["ldm_sampler"] == "ddim"
        assert kw["mask_bytes"] == b"mask-bytes"

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates_after_undo_push(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_pipeline.sam_remove_object = AsyncMock(
            side_effect=RuntimeError("lama failed")
        )

        with pytest.raises(RuntimeError, match="lama failed"):
            await segmentation_service.sam_remove_object(
                sample_image.id, mask_id=1, user_id=sample_user.id
            )

        mock_redis_history.push_undo_state.assert_awaited_once()


class TestSamReplaceObject:
    @pytest.mark.asyncio
    async def test_success_passes_replacement_bytes_to_pipeline(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.sam_replace_object(
            sample_image.id, mask_id=1,
            replacement_image_bytes=b"replacement-bytes",
            user_id=sample_user.id,
        )

        assert "result_url" in result
        assert "image_version_id" in result
        kw = mock_pipeline.sam_replace_object.call_args.kwargs
        assert kw["replacement_image_bytes"] == b"replacement-bytes"
        assert kw["mask_bytes"] == b"mask-bytes"

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        result = await segmentation_service.sam_replace_object(
            sample_image.id, mask_id=1, replacement_image_bytes=b"rep", user_id=sample_user.id
        )

        assert result["image_version_id"] != sample_image_version.id

    @pytest.mark.asyncio
    async def test_pushes_undo_state(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_mask,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        cache_suffix = f"mask:{sample_image_version.content_id}:1"
        mock_redis_cache.get_cache_image = AsyncMock(
            side_effect=lambda image_id, suffix: b"mask-bytes" if suffix == cache_suffix else b"image-bytes"
        )
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.sam_replace_object(
            sample_image.id, mask_id=1, replacement_image_bytes=b"rep", user_id=sample_user.id
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        assert "sam_replace" in mock_redis_history.push_undo_state.call_args.kwargs["label"]

    @pytest.mark.asyncio
    async def test_raises_when_mask_not_found(
        self,
        segmentation_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="mask_id=999"):
            await segmentation_service.sam_replace_object(
                sample_image.id, mask_id=999, replacement_image_bytes=b"rep",
                user_id=sample_user.id,
            )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, segmentation_service, sample_image, sample_image_version, sample_mask
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await segmentation_service.sam_replace_object(
                sample_image.id, mask_id=1, replacement_image_bytes=b"rep",
                user_id=sample_image.user_id + 1,
            )

    @pytest.mark.asyncio
    async def test_passes_optional_params_to_pipeline(
        self,
        segmentation_service,
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
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": _jpeg_bytes(), "metrics": {}, "timestamp": "ts",
        })
        _setup_s3_upload(mock_s3_storage)

        await segmentation_service.sam_replace_object(
            sample_image.id, mask_id=1, replacement_image_bytes=b"rep",
            user_id=sample_user.id,
            use_color_matching=True, color_match_method="histogram",
            replacement_is_cutout=True, expand_mask_pixels=15,
        )

        kw = mock_pipeline.sam_replace_object.call_args.kwargs
        assert kw["use_color_matching"] is True
        assert kw["color_match_method"] == "histogram"
        assert kw["replacement_is_cutout"] is True
        assert kw["expand_mask_pixels"] == 15


class TestMaskToDataUrl:
    def test_returns_correct_prefix(self):
        assert _mask_to_data_url(b"bytes").startswith("data:image/png;base64,")

    def test_base64_payload_round_trips(self):
        import base64
        raw = b"\x89PNG\r\nfake"
        b64 = _mask_to_data_url(raw).split(",", 1)[1]
        assert base64.b64decode(b64) == raw

    def test_empty_bytes_returns_empty_payload(self):
        assert _mask_to_data_url(b"") == "data:image/png;base64,"

    def test_different_inputs_produce_different_urls(self):
        assert _mask_to_data_url(b"a") != _mask_to_data_url(b"b")


class TestSegmentsForResponse:
    def test_strips_mask_bytes(self):
        result = _segments_for_response([{"mask_id": 0, "mask_bytes": b"x"}])
        assert "mask_bytes" not in result[0]

    def test_adds_mask_url_when_bytes_present(self):
        result = _segments_for_response([{"mask_id": 0, "mask_bytes": b"x"}])
        assert result[0]["mask_url"] == _mask_to_data_url(b"x")

    def test_no_mask_url_when_bytes_missing(self):
        result = _segments_for_response([{"mask_id": 0}])
        assert "mask_url" not in result[0]

    def test_no_mask_url_when_bytes_none_or_empty(self):
        result = _segments_for_response([
            {"mask_id": 0, "mask_bytes": None},
            {"mask_id": 1, "mask_bytes": b""},
        ])
        assert "mask_url" not in result[0]
        assert "mask_url" not in result[1]

    def test_preserves_other_fields(self):
        seg = {"mask_id": 5, "area": 100, "source": "yolo", "mask_bytes": b"x"}
        result = _segments_for_response([seg])
        assert result[0]["mask_id"] == 5
        assert result[0]["source"] == "yolo"

    def test_does_not_mutate_original(self):
        original = {"mask_id": 0, "mask_bytes": b"x"}
        _segments_for_response([original])
        assert original["mask_bytes"] == b"x"

    def test_preserves_order(self):
        segs = [{"mask_id": i, "mask_bytes": b"x"} for i in range(5)]
        result = _segments_for_response(segs)
        assert [s["mask_id"] for s in result] == [0, 1, 2, 3, 4]

    def test_empty_list_returns_empty(self):
        assert _segments_for_response([]) == []