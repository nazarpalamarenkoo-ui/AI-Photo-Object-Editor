import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException, UploadFile

import app.api.v1.ml as ml_module
from app.api.v1.ml import (
    remove_object_async,
    remove_multiple_objects_async,
    replace_object_async,
    segment_objects_async,
    segment_with_prompt_async,
    segment_by_polygon,
    segment_by_polygon_async,
    segment_hybrid,
    segment_hybrid_async,
    sam_remove_object_async,
    sam_replace_object_async,
    extract_object_async,
    get_job_status,
)
from app.db.schemas.ml import (
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SegmentRequest,
    SegmentWithPromptRequest,
    SegmentByPolygonRequest,
    SegmentHybridRequest,
    SamRemoveRequest,
    SamReplaceRequest,
    ExtractRequest,
)

try:
    from arq.jobs import JobStatus
except ImportError:
    JobStatus = None


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_file():
    file = MagicMock(spec=UploadFile)
    file.read = AsyncMock(return_value=b"image-bytes")
    return file


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    job = MagicMock()
    job.job_id = "job-123"
    pool.enqueue_job = AsyncMock(return_value=job)
    return pool


@pytest.fixture
def mock_segmentation_service():
    service = MagicMock()
    service.segment_by_polygon = AsyncMock()
    service.segment_hybrid = AsyncMock()
    return service


@pytest.fixture
def mock_asset_service():
    service = MagicMock()
    service.get_asset_image = AsyncMock()
    return service

@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentByPolygon:
    async def test_success(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_by_polygon.return_value = {"segments": []}
        body = SegmentByPolygonRequest(
            points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            smooth=True,
            smoothing_factor=0.5,
            feather_px=3,
        )

        result = await segment_by_polygon(
            image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service
        )

        mock_segmentation_service.segment_by_polygon.assert_called_once_with(
            image_id=1,
            user_id=1,
            points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            smooth=True,
            smoothing_factor=0.5,
            feather_px=3,
        )
        assert result == {"segments": []}

    async def test_not_found(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_by_polygon.side_effect = ValueError("image not found")
        body = SegmentByPolygonRequest(points=[(0, 0), (10, 0), (10, 10)])

        with pytest.raises(HTTPException) as exc:
            await segment_by_polygon(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 404

    async def test_generic_error(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_by_polygon.side_effect = ValueError("polygon self-intersects")
        body = SegmentByPolygonRequest(points=[(0, 0), (10, 0), (10, 10)])

        with pytest.raises(HTTPException) as exc:
            await segment_by_polygon(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentHybrid:
    async def test_success_custom_body(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_hybrid.return_value = {"segments": []}
        body = SegmentHybridRequest(
            yolo_conf_threshold=0.4,
            yolo_classes=["person", "car"],
            fallback_min_area=200,
            fallback_max_segments=20,
            overlap_iou_thresh=0.3,
        )

        result = await segment_hybrid(
            image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service
        )

        mock_segmentation_service.segment_hybrid.assert_called_once_with(
            image_id=1,
            user_id=1,
            yolo_conf_threshold=0.4,
            yolo_classes=["person", "car"],
            fallback_min_area=200,
            fallback_max_segments=20,
            overlap_iou_thresh=0.3,
        )
        assert result == {"segments": []}

    async def test_not_found(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_hybrid.side_effect = ValueError("image not found")
        body = SegmentHybridRequest()

        with pytest.raises(HTTPException) as exc:
            await segment_hybrid(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 404

    async def test_generic_error(self, mock_user, mock_segmentation_service):
        mock_segmentation_service.segment_hybrid.side_effect = ValueError("YOLO model failed")
        body = SegmentHybridRequest()

        with pytest.raises(HTTPException) as exc:
            await segment_hybrid(image_id=1, body=body, current_user=mock_user, service=mock_segmentation_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveObjectAsync:
    async def test_enqueues_job_default_body(self, mock_user, mock_pool):
        result = await remove_object_async(
            image_id=1, bbox_id=2, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "remove_object_task",
            image_id=1,
            bbox_id=2,
            user_id=1,
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result == {"job_id": "job-123"}

    async def test_enqueues_job_custom_body(self, mock_user, mock_pool):
        from app.db.schemas.ml import LdmConfig

        body = RemoveRequest(
            expand_mask_pixels=20,
            use_edge_blending=True,
            ldm_steps=40,
            ldm_sampler="ddim",
            hd_strategy="RESIZE",
        )

        await remove_object_async(image_id=1, bbox_id=2, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "remove_object_task",
            image_id=1,
            bbox_id=2,
            user_id=1,
            expand_mask_pixels=20,
            use_edge_blending=True,
            ldm_steps=40,
            ldm_sampler="ddim",
            hd_strategy="RESIZE",
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveMultipleObjectsAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = RemoveMultipleRequest(bbox_ids=[1, 2, 3])

        result = await remove_multiple_objects_async(
            image_id=1, body=body, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "remove_multiple_objects_task",
            image_id=1,
            bbox_ids=[1, 2, 3],
            user_id=1,
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestReplaceObjectAsync:
    async def test_reads_file_and_enqueues_job(self, mock_user, mock_pool, mock_file):
        body = ReplaceRequest(
            expand_mask_pixels=15,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method="histogram",
        )

        result = await replace_object_async(
            image_id=1,
            bbox_id=2,
            replacement_file=mock_file,
            body=body,
            current_user=mock_user,
            pool=mock_pool,
        )

        mock_file.read.assert_awaited_once()
        mock_pool.enqueue_job.assert_awaited_once_with(
            "replace_object_task",
            image_id=1,
            bbox_id=2,
            replace_image_bytes=b"image-bytes",
            user_id=1,
            expand_mask_pixels=15,
            use_color_matching=True,
            use_edge_blending=True,
            color_match_method="histogram",
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentObjectsAsync:
    async def test_enqueues_job_default_body(self, mock_user, mock_pool):
        result = await segment_objects_async(image_id=1, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_objects_task",
            image_id=1,
            user_id=1,
            min_area=500,
            max_segments=50,
        )
        assert result == {"job_id": "job-123"}

    async def test_enqueues_job_custom_body(self, mock_user, mock_pool):
        body = SegmentRequest(min_area=100, max_segments=10)

        await segment_objects_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_objects_task",
            image_id=1,
            user_id=1,
            min_area=100,
            max_segments=10,
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentWithPromptAsync:
    async def test_enqueues_job_with_points(self, mock_user, mock_pool):
        body = SegmentWithPromptRequest(point_coords=[(10, 20)], point_labels=[1])

        result = await segment_with_prompt_async(
            image_id=1, body=body, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_with_prompt_task",
            image_id=1,
            user_id=1,
            point_coords=[(10, 20)],
            point_labels=[1],
            bbox=None,
            multimask_output=None,
        )
        assert result == {"job_id": "job-123"}

    async def test_enqueues_job_with_bbox(self, mock_user, mock_pool):
        from app.db.schemas.ml import BboxSchema

        bbox = BboxSchema(x1=0, y1=0, x2=50, y2=50)
        body = SegmentWithPromptRequest(bbox=bbox)

        await segment_with_prompt_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        _, kwargs = mock_pool.enqueue_job.call_args
        assert kwargs["bbox"] == bbox.model_dump()


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentByPolygonAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = SegmentByPolygonRequest(
            points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            smooth=False,
            smoothing_factor=0.2,
            feather_px=0,
        )

        result = await segment_by_polygon_async(
            image_id=1, body=body, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_by_polygon_task",
            image_id=1,
            user_id=1,
            points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            smooth=False,
            smoothing_factor=0.2,
            feather_px=0,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentHybridAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = SegmentHybridRequest(
            yolo_conf_threshold=0.4,
            yolo_classes=["person"],
            fallback_min_area=200,
            fallback_max_segments=20,
            overlap_iou_thresh=0.3,
        )

        result = await segment_hybrid_async(
            image_id=1, body=body, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_hybrid_task",
            image_id=1,
            user_id=1,
            yolo_conf_threshold=0.4,
            yolo_classes=["person"],
            fallback_min_area=200,
            fallback_max_segments=20,
            overlap_iou_thresh=0.3,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamRemoveObjectAsync:
    async def test_enqueues_job_default_body(self, mock_user, mock_pool):
        result = await sam_remove_object_async(
            image_id=1, mask_id=3, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_remove_object_task",
            image_id=1,
            mask_id=3,
            user_id=1,
            expand_mask_pixels=12,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObjectAsync:
    async def test_enqueues_job_with_file(self, mock_user, mock_pool, mock_file, mock_asset_service):
        body = SamReplaceRequest()

        result = await sam_replace_object_async(
            image_id=1,
            mask_id=3,
            replacement_file=mock_file,
            asset_id=None,
            body=body,
            current_user=mock_user,
            asset_service=mock_asset_service,
            pool=mock_pool,
        )

        mock_file.read.assert_awaited_once()
        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_replace_object_task",
            image_id=1,
            mask_id=3,
            replacement_image_bytes=b"image-bytes",
            user_id=1,
            expand_mask_pixels=8,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method='color_transfer',
            ldm_steps=25,
            ldm_sampler='plms',
            hd_strategy='CROP',
            replacement_is_cutout=False,
        )
        assert result == {"job_id": "job-123"}

    async def test_enqueues_job_with_asset_id(self, mock_user, mock_pool, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = b"asset-bytes"
        body = SamReplaceRequest()

        await sam_replace_object_async(
            image_id=1,
            mask_id=3,
            replacement_file=None,
            asset_id="asset-1",
            body=body,
            current_user=mock_user,
            asset_service=mock_asset_service,
            pool=mock_pool,
        )

        mock_asset_service.get_asset_image.assert_awaited_once_with(1, "asset-1")
        _, kwargs = mock_pool.enqueue_job.call_args
        assert kwargs["replacement_image_bytes"] == b"asset-bytes"
        assert kwargs["replacement_is_cutout"] is True

    async def test_asset_not_found_returns_404(self, mock_user, mock_pool, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = None
        body = SamReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_async(
                image_id=1,
                mask_id=3,
                replacement_file=None,
                asset_id="missing-asset",
                body=body,
                current_user=mock_user,
                asset_service=mock_asset_service,
                pool=mock_pool,
            )

        assert exc.value.status_code == 404
        mock_pool.enqueue_job.assert_not_awaited()

    async def test_missing_file_and_asset_id_returns_400(self, mock_user, mock_pool, mock_asset_service):
        body = SamReplaceRequest()

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_async(
                image_id=1,
                mask_id=3,
                replacement_file=None,
                asset_id=None,
                body=body,
                current_user=mock_user,
                asset_service=mock_asset_service,
                pool=mock_pool,
            )

        assert exc.value.status_code == 400
        assert exc.value.detail == "Provide replacement_file or asset_id"
        mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
class TestExtractObjectAsync:
    async def test_enqueues_job_default_body(self, mock_user, mock_pool):
        result = await extract_object_async(
            image_id=1, mask_id=3, current_user=mock_user, pool=mock_pool
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_extract_object_task",
            image_id=1,
            mask_id=3,
            user_id=1,
            padding_pixels=8,
            label=None,
            persist_to_s3=False,
        )
        assert result == {"job_id": "job-123"}

    async def test_enqueues_job_custom_body(self, mock_user, mock_pool):
        body = ExtractRequest(padding_pixels=20, label="my-object", persist_to_s3=True)

        await extract_object_async(image_id=1, mask_id=3, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_extract_object_task",
            image_id=1,
            mask_id=3,
            user_id=1,
            padding_pixels=20,
            label="my-object",
            persist_to_s3=True,
        )



@pytest.mark.unit
@pytest.mark.asyncio
class TestGetJobStatus:
    def _patched_job(self, status, result_info=None):
        """Патчить ml_module.Job так, щоб конструктор повертав мок з потрібним статусом."""
        job_instance = MagicMock()
        job_instance.status = AsyncMock(return_value=status)
        job_instance.result_info = AsyncMock(return_value=result_info)
        return patch.object(ml_module, "Job", return_value=job_instance), job_instance

    async def test_not_found_raises_404(self, mock_user):
        pool = MagicMock()
        patcher, _ = self._patched_job(JobStatus.not_found)

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await get_job_status(job_id="missing-job", current_user=mock_user, pool=pool)

        assert exc.value.status_code == 404

    @pytest.mark.parametrize("status", ["deferred", "queued", "in_progress"])
    async def test_pending_statuses_return_status_only(self, mock_user, status):
        pool = MagicMock()
        job_status = getattr(JobStatus, status)
        patcher, job_instance = self._patched_job(job_status)

        with patcher:
            result = await get_job_status(job_id="job-1", current_user=mock_user, pool=pool)

        assert result == {"job_id": "job-1", "status": job_status.value}
        job_instance.result_info.assert_not_awaited()

    async def test_complete_success_includes_result(self, mock_user):
        pool = MagicMock()
        result_info = MagicMock(success=True, result={"result_url": "s3://out.jpg"})
        patcher, _ = self._patched_job(JobStatus.complete, result_info=result_info)

        with patcher:
            result = await get_job_status(job_id="job-2", current_user=mock_user, pool=pool)

        assert result["status"] == JobStatus.complete.value
        assert result["result"] == {"result_url": "s3://out.jpg"}
        assert "error" not in result

    async def test_complete_failure_includes_error(self, mock_user):
        pool = MagicMock()
        result_info = MagicMock(success=False, result=ValueError("inpainting failed"))
        patcher, _ = self._patched_job(JobStatus.complete, result_info=result_info)

        with patcher:
            result = await get_job_status(job_id="job-3", current_user=mock_user, pool=pool)

        assert result["status"] == JobStatus.complete.value
        assert result["error"] == "inpainting failed"
        assert "result" not in result

    async def test_complete_without_result_info(self, mock_user):
        pool = MagicMock()
        patcher, _ = self._patched_job(JobStatus.complete, result_info=None)

        with patcher:
            result = await get_job_status(job_id="job-4", current_user=mock_user, pool=pool)

        assert result == {"job_id": "job-4", "status": JobStatus.complete.value}
        assert "result" not in result
        assert "error" not in result