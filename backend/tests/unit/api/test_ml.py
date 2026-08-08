import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from fastapi import HTTPException, UploadFile

import app.api.v1.ml.jobs as ml_jobs_module

from app.api.v1.ml.detect import detect_objects, get_supported_classes
from app.api.v1.ml.editing import (
    remove_object,
    remove_object_async,
    remove_multiple_objects,
    remove_multiple_objects_async,
    sam_replace_object_diffusion,
    sam_replace_object_diffusion_async,
    replace_object,
    replace_object_async,
)
from app.api.v1.ml.session import (
    get_current_state,
    reset_current_state,
    save_result,
    undo,
    redo,
    get_history,
)
from app.api.v1.ml.segmentation import (
    segment_objects,
    segment_objects_async,
    segment_with_prompt,
    segment_with_prompt_async,
    segment_by_polygon,
    segment_by_polygon_async,
    segment_hybrid,
    segment_hybrid_async,
)
from app.api.v1.ml.sam_ops import (
    sam_remove_object,
    sam_remove_object_async,
    sam_replace_object,
    sam_replace_object_async,
    extract_object,
    extract_object_async,
)
from app.api.v1.ml.assets import (
    list_assets,
    get_asset_thumbnail,
    get_asset_image,
    rename_asset,
    delete_asset,
    paste_extracted_object,
)
from app.api.v1.ml.jobs import get_job_status
from app.api.v1.ml.deps import _http_status

from app.db.enums.ml_task_status import MLTaskType
from app.services.ml.detector_service import DetectorService
from app.services.ml.editing_service import EditingService
from app.services.ml.segmentation_service import SegmentationService
from app.services.ml.assets_service import AssetService

try:
    from arq.jobs import JobStatus
except ImportError:  # pragma: no cover
    JobStatus = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_body(**attrs):
    """MagicMock request body with only the attributes the endpoint touches."""
    body = MagicMock()
    for key, value in attrs.items():
        setattr(body, key, value)
    return body


def make_ldm(ldm_steps=25, ldm_sampler="plms", hd_strategy="CROP"):
    ldm = MagicMock()
    ldm.ldm_steps = ldm_steps
    ldm.ldm_sampler = ldm_sampler
    ldm.hd_strategy = hd_strategy
    return ldm


def make_diffusion_body(**overrides):
    defaults = dict(
        bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        prompt="a red sports car",
        use_color_matching=True,
        color_match_method="color_transfer",
        negative_prompt=None,
        num_inference_steps=30,
        guidance_scale=7.5,
        ip_adapter_scale=0.6,
        strength=0.8,
        seed=42,
    )
    defaults.update(overrides)
    return make_body(**defaults)


def patched_run_tracked(module: str, return_value=None, side_effect=None):
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    return patch(f"app.api.v1.ml.{module}.run_tracked", new=mock), mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_deps():
    """Sentinel deps dict passed straight through to run_tracked untouched."""
    return {"marker": "deps"}


@pytest.fixture
def mock_mljob_service():
    return MagicMock()


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
def mock_detector_service():
    service = MagicMock()
    service.get_supported_classes = MagicMock(return_value=["person", "car", "dog"])
    return service


@pytest.fixture
def mock_asset_service():
    """Only what editing.py / sam_ops.py touch on the injected asset_service."""
    service = MagicMock()
    service.get_asset_image = AsyncMock(return_value=b"asset-bytes")
    return service


@pytest.fixture
def mock_asset_full_service():
    """Full AssetService surface used directly by assets.py routes."""
    service = MagicMock()
    service.list_assets = AsyncMock(return_value=[])
    service.get_asset_thumbnail = AsyncMock(return_value=b"thumb-bytes")
    service.get_asset_image = AsyncMock(return_value=b"asset-bytes")
    service.rename_asset = AsyncMock(return_value={"asset_id": "asset-1", "label": "new-name"})
    service.delete_asset = AsyncMock()
    service.paste_extracted_object = AsyncMock(return_value={"result_url": "s3://bucket/pasted.jpg"})
    return service


@pytest.fixture
def mock_version_history_service():
    service = MagicMock()
    service.get_current_state = AsyncMock(return_value={"presigned_url": "https://presigned.url/current.jpg"})
    service.reset_current_state = AsyncMock()
    service.save_result = AsyncMock(return_value={"id": 42, "filename": "edited.jpg"})
    service.undo = AsyncMock(return_value={"detail": "Undone"})
    service.redo = AsyncMock(return_value={"detail": "Redone"})
    service.get_history = AsyncMock(return_value={"history": []})
    return service


# ---------------------------------------------------------------------------
# detect.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestDetectObjects:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(conf_threshold=0.7, classes=["person"])
        patcher, mock_run_tracked = patched_run_tracked(
            "detect", return_value={"detections": [{"class": "person"}], "count": 1}
        )

        with patcher:
            result = await detect_objects(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            DetectorService, mock_deps, mock_mljob_service, "detect_objects",
            1, 1, MLTaskType.DETECTION,
            conf_threshold=0.7,
            classes=["person"],
        )
        assert result["count"] == 1

    async def test_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(conf_threshold=0.5, classes=None)
        patcher, _ = patched_run_tracked("detect", side_effect=ValueError("Image not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await detect_objects(
                    image_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404

    async def test_unauthorized(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(conf_threshold=0.5, classes=None)
        patcher, _ = patched_run_tracked("detect", side_effect=ValueError("unauthorized"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await detect_objects(
                    image_id=1, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetSupportedClasses:
    async def test_success(self, mock_user, mock_detector_service):
        result = await get_supported_classes(current_user=mock_user, service=mock_detector_service)

        assert result == ["person", "car", "dog"]
        mock_detector_service.get_supported_classes.assert_called_once()


# ---------------------------------------------------------------------------
# editing.py - remove
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveObject:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(expand_mask_pixels=10, use_edge_blending=False, ldm=make_ldm())
        patcher, mock_run_tracked = patched_run_tracked(
            "editing", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            result = await remove_object(
                image_id=1, bbox_id=2, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            EditingService, mock_deps, mock_mljob_service, "remove_object",
            1, 1, MLTaskType.REMOVE_OBJECT,
            bbox_id=2,
            expand_mask_pixels=10,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
        )
        assert result["result_url"] == "s3://bucket/result.jpg"

    async def test_bbox_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(expand_mask_pixels=5, use_edge_blending=False, ldm=make_ldm())
        patcher, _ = patched_run_tracked("editing", side_effect=ValueError("bbox not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await remove_object(
                    image_id=1, bbox_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveObjectAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(
            expand_mask_pixels=20, use_edge_blending=True,
            ldm=make_ldm(ldm_steps=40, ldm_sampler="ddim", hd_strategy="RESIZE"),
        )

        result = await remove_object_async(
            image_id=1, bbox_id=2, body=body, current_user=mock_user, pool=mock_pool,
        )

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
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveMultipleObjects:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(
            bbox_ids=[1, 2, 3], expand_mask_pixels=5, use_edge_blending=False, ldm=make_ldm(),
        )
        patcher, mock_run_tracked = patched_run_tracked(
            "editing", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            result = await remove_multiple_objects(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            EditingService, mock_deps, mock_mljob_service, "remove_multiple_objects",
            1, 1, MLTaskType.REMOVE_MULTIPLE_OBJECTS,
            bbox_ids=[1, 2, 3],
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
        )
        assert result["result_url"] == "s3://bucket/result.jpg"

    async def test_unauthorized(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(bbox_ids=[1], expand_mask_pixels=5, use_edge_blending=False, ldm=make_ldm())
        patcher, _ = patched_run_tracked("editing", side_effect=ValueError("unauthorized"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await remove_multiple_objects(
                    image_id=1, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestRemoveMultipleObjectsAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(
            bbox_ids=[1, 2, 3], expand_mask_pixels=5, use_edge_blending=False, ldm=make_ldm(),
        )

        result = await remove_multiple_objects_async(
            image_id=1, body=body, current_user=mock_user, pool=mock_pool,
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "remove_multiple_objects_task",
            image_id=1,
            bbox_ids=[1, 2, 3],
            user_id=1,
            expand_mask_pixels=5,
            use_edge_blending=False,
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


# ---------------------------------------------------------------------------
# editing.py - SAM diffusion replace
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObjectDiffusion:
    async def test_success_with_reference_file(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")
        reference_file = MagicMock(spec=UploadFile)
        reference_file.read = AsyncMock(return_value=b"reference-bytes")
        body = make_diffusion_body()
        patcher, mock_run_tracked = patched_run_tracked(
            "editing", return_value={"result_url": "s3://bucket/diffusion.jpg"}
        )

        with patcher:
            result = await sam_replace_object_diffusion(
                image_id=1, mask_file=mask_file, reference_file=reference_file, asset_id=None,
                body=body, current_user=mock_user, deps=mock_deps,
                mljob_service=mock_mljob_service, asset_service=mock_asset_service,
            )

        mock_asset_service.get_asset_image.assert_not_awaited()
        mock_run_tracked.assert_awaited_once_with(
            EditingService, mock_deps, mock_mljob_service, "sam_replace_object_diffusion",
            1, 1, MLTaskType.DIFFUSION,
            mask_bytes=b"mask-bytes",
            bbox=body.bbox,
            reference_image_bytes=b"reference-bytes",
            prompt=body.prompt,
            use_color_matching=body.use_color_matching,
            color_match_method=body.color_match_method,
            negative_prompt=body.negative_prompt,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            ip_adapter_scale=body.ip_adapter_scale,
            strength=body.strength,
            seed=body.seed,
        )
        assert result["result_url"] == "s3://bucket/diffusion.jpg"

    async def test_success_with_asset_id(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")
        body = make_diffusion_body()
        patcher, mock_run_tracked = patched_run_tracked(
            "editing", return_value={"result_url": "s3://bucket/diffusion.jpg"}
        )

        with patcher:
            result = await sam_replace_object_diffusion(
                image_id=1, mask_file=mask_file, reference_file=None, asset_id="asset-1",
                body=body, current_user=mock_user, deps=mock_deps,
                mljob_service=mock_mljob_service, asset_service=mock_asset_service,
            )

        mock_asset_service.get_asset_image.assert_awaited_once_with(1, "asset-1")
        assert mock_run_tracked.await_args.kwargs["reference_image_bytes"] == b"asset-bytes"
        assert result["result_url"] == "s3://bucket/diffusion.jpg"

    async def test_missing_reference_and_asset_id_returns_400(
        self, mock_user, mock_deps, mock_mljob_service, mock_asset_service
    ):
        mask_file = MagicMock(spec=UploadFile)

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_diffusion(
                image_id=1, mask_file=mask_file, reference_file=None, asset_id=None,
                body=make_diffusion_body(), current_user=mock_user, deps=mock_deps,
                mljob_service=mock_mljob_service, asset_service=mock_asset_service,
            )

        assert exc.value.status_code == 400

    async def test_asset_not_found_returns_404(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = None
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_diffusion(
                image_id=1, mask_file=mask_file, reference_file=None, asset_id="missing",
                body=make_diffusion_body(), current_user=mock_user, deps=mock_deps,
                mljob_service=mock_mljob_service, asset_service=mock_asset_service,
            )

        assert exc.value.status_code == 404

    async def test_service_value_error(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")
        reference_file = MagicMock(spec=UploadFile)
        reference_file.read = AsyncMock(return_value=b"reference-bytes")
        patcher, _ = patched_run_tracked("editing", side_effect=ValueError("image not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await sam_replace_object_diffusion(
                    image_id=999, mask_file=mask_file, reference_file=reference_file, asset_id=None,
                    body=make_diffusion_body(), current_user=mock_user, deps=mock_deps,
                    mljob_service=mock_mljob_service, asset_service=mock_asset_service,
                )

        assert exc.value.status_code == 404

    async def test_diffusion_runtime_error_returns_502(
        self, mock_user, mock_deps, mock_mljob_service, mock_asset_service
    ):
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")
        reference_file = MagicMock(spec=UploadFile)
        reference_file.read = AsyncMock(return_value=b"reference-bytes")
        patcher, _ = patched_run_tracked("editing", side_effect=RuntimeError("NaN in output"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await sam_replace_object_diffusion(
                    image_id=1, mask_file=mask_file, reference_file=reference_file, asset_id=None,
                    body=make_diffusion_body(), current_user=mock_user, deps=mock_deps,
                    mljob_service=mock_mljob_service, asset_service=mock_asset_service,
                )

        assert exc.value.status_code == 502


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObjectDiffusionAsync:
    async def test_enqueues_job_with_reference_file(self, mock_user, mock_pool, mock_asset_service):
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")
        reference_file = MagicMock(spec=UploadFile)
        reference_file.read = AsyncMock(return_value=b"reference-bytes")

        result = await sam_replace_object_diffusion_async(
            image_id=1, mask_file=mask_file, reference_file=reference_file, asset_id=None,
            body=make_diffusion_body(), current_user=mock_user,
            asset_service=mock_asset_service, pool=mock_pool,
        )

        mock_pool.enqueue_job.assert_awaited_once()
        args, kwargs = mock_pool.enqueue_job.call_args
        assert args[0] == "sam_replace_object_diffusion_task"
        assert kwargs["mask_bytes"] == b"mask-bytes"
        assert kwargs["reference_image_bytes"] == b"reference-bytes"
        assert result == {"job_id": "job-123"}

    async def test_missing_reference_and_asset_id_returns_400(self, mock_user, mock_pool, mock_asset_service):
        mask_file = MagicMock(spec=UploadFile)

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_diffusion_async(
                image_id=1, mask_file=mask_file, reference_file=None, asset_id=None,
                body=make_diffusion_body(), current_user=mock_user,
                asset_service=mock_asset_service, pool=mock_pool,
            )

        assert exc.value.status_code == 400
        mock_pool.enqueue_job.assert_not_awaited()

    async def test_asset_not_found_returns_404(self, mock_user, mock_pool, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = None
        mask_file = MagicMock(spec=UploadFile)
        mask_file.read = AsyncMock(return_value=b"mask-bytes")

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_diffusion_async(
                image_id=1, mask_file=mask_file, reference_file=None, asset_id="missing",
                body=make_diffusion_body(), current_user=mock_user,
                asset_service=mock_asset_service, pool=mock_pool,
            )

        assert exc.value.status_code == 404
        mock_pool.enqueue_job.assert_not_awaited()


# ---------------------------------------------------------------------------
# editing.py - replace (YOLO)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestReplaceObject:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service, mock_file):
        body = make_body(
            expand_mask_pixels=12, use_color_matching=True, use_edge_blending=False,
            color_match_method="histogram", ldm=make_ldm(),
        )
        patcher, mock_run_tracked = patched_run_tracked(
            "editing", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            result = await replace_object(
                image_id=1, bbox_id=1, replacement_file=mock_file, body=body,
                current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_file.read.assert_awaited_once()
        mock_run_tracked.assert_awaited_once_with(
            EditingService, mock_deps, mock_mljob_service, "replace_object",
            1, 1, MLTaskType.REPLACE_OBJECT,
            bbox_id=1,
            replace_image_bytes=b"image-bytes",
            expand_mask_pixels=12,
            use_color_matching=True,
            use_edge_blending=False,
            color_match_method="histogram",
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
        )
        assert result["result_url"] == "s3://bucket/result.jpg"

    async def test_generic_error_returns_400(self, mock_user, mock_deps, mock_mljob_service, mock_file):
        body = make_body(
            expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
            color_match_method="color_transfer", ldm=make_ldm(),
        )
        patcher, _ = patched_run_tracked("editing", side_effect=ValueError("invalid replacement image"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await replace_object(
                    image_id=1, bbox_id=1, replacement_file=mock_file, body=body,
                    current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestReplaceObjectAsync:
    async def test_reads_file_and_enqueues_job(self, mock_user, mock_pool, mock_file):
        body = make_body(
            expand_mask_pixels=15, use_color_matching=True, use_edge_blending=True,
            color_match_method="histogram", ldm=make_ldm(),
        )

        result = await replace_object_async(
            image_id=1, bbox_id=2, replacement_file=mock_file, body=body,
            current_user=mock_user, pool=mock_pool,
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
            ldm_sampler="plms",
            hd_strategy="CROP",
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


# ---------------------------------------------------------------------------
# session.py - working-state management
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestGetCurrentState:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await get_current_state(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.get_current_state.assert_awaited_once_with(image_id=1, user_id=1)
        assert result["presigned_url"] == "https://presigned.url/current.jpg"

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.get_current_state.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await get_current_state(image_id=999, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestResetCurrentState:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await reset_current_state(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.reset_current_state.assert_awaited_once_with(1, 1)
        assert result == {"detail": "State reset to original image"}

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.reset_current_state.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await reset_current_state(image_id=999, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveResult:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await save_result(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.save_result.assert_awaited_once_with(image_id=1, user_id=1)
        assert result["id"] == 42

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.save_result.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await save_result(image_id=999, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestUndo:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await undo(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.undo.assert_awaited_once_with(1, 1)
        assert result == {"detail": "Undone"}

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.undo.side_effect = ValueError("nothing to undo")

        with pytest.raises(HTTPException) as exc:
            await undo(image_id=1, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestRedo:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await redo(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.redo.assert_awaited_once_with(1, 1)
        assert result == {"detail": "Redone"}

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.redo.side_effect = ValueError("nothing to redo")

        with pytest.raises(HTTPException) as exc:
            await redo(image_id=1, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetHistory:
    async def test_success(self, mock_user, mock_version_history_service):
        result = await get_history(image_id=1, current_user=mock_user, service=mock_version_history_service)

        mock_version_history_service.get_history.assert_awaited_once_with(1, 1)
        assert result == {"history": []}

    async def test_not_found(self, mock_user, mock_version_history_service):
        mock_version_history_service.get_history.side_effect = ValueError("not found")

        with pytest.raises(HTTPException) as exc:
            await get_history(image_id=999, current_user=mock_user, service=mock_version_history_service)

        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# segmentation.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentObjects:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(min_area=100, max_segments=50)
        patcher, mock_run_tracked = patched_run_tracked("segmentation", return_value={"segments": []})

        with patcher:
            result = await segment_objects(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "segment_objects",
            1, 1, MLTaskType.SEGMENTATION,
            min_area=100,
            max_segments=50,
        )
        assert result == {"segments": []}

    async def test_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(min_area=100, max_segments=50)
        patcher, _ = patched_run_tracked("segmentation", side_effect=ValueError("not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await segment_objects(
                    image_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentObjectsAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(min_area=100, max_segments=50)

        result = await segment_objects_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_objects_task",
            image_id=1,
            user_id=1,
            min_area=100,
            max_segments=50,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentWithPrompt:
    async def test_success_with_bbox(self, mock_user, mock_deps, mock_mljob_service):
        bbox = MagicMock()
        bbox.model_dump = MagicMock(return_value={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        body = make_body(point_coords=[[1, 2]], point_labels=[1], bbox=bbox, multimask_output=True)
        patcher, mock_run_tracked = patched_run_tracked("segmentation", return_value={"segments": []})

        with patcher:
            result = await segment_with_prompt(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "segment_with_prompt",
            1, 1, MLTaskType.SEGMENTATION_PROMPT,
            point_coords=[[1, 2]],
            point_labels=[1],
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            multimask_output=True,
        )
        assert result == {"segments": []}

    async def test_success_without_bbox(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(point_coords=[[1, 2]], point_labels=[1], bbox=None, multimask_output=False)
        patcher, mock_run_tracked = patched_run_tracked("segmentation", return_value={"segments": []})

        with patcher:
            await segment_with_prompt(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        assert mock_run_tracked.await_args.kwargs["bbox"] is None

    async def test_unauthorized(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(point_coords=[[1, 2]], point_labels=[1], bbox=None, multimask_output=False)
        patcher, _ = patched_run_tracked("segmentation", side_effect=ValueError("unauthorized"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await segment_with_prompt(
                    image_id=1, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentWithPromptAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        bbox = MagicMock()
        bbox.model_dump = MagicMock(return_value={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        body = make_body(point_coords=[[1, 2]], point_labels=[1], bbox=bbox, multimask_output=True)

        result = await segment_with_prompt_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_with_prompt_task",
            image_id=1,
            user_id=1,
            point_coords=[[1, 2]],
            point_labels=[1],
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            multimask_output=True,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentByPolygon:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(points=[[0, 0], [10, 0], [10, 10]], smooth=True, smoothing_factor=0.5, feather_px=3)
        patcher, mock_run_tracked = patched_run_tracked("segmentation", return_value={"segments": []})

        with patcher:
            result = await segment_by_polygon(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "segment_by_polygon",
            1, 1, MLTaskType.SEGMENTATION_POLYGON,
            points=[[0, 0], [10, 0], [10, 10]],
            smooth=True,
            smoothing_factor=0.5,
            feather_px=3,
        )
        assert result == {"segments": []}

    async def test_generic_error_returns_400(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(points=[[0, 0]], smooth=False, smoothing_factor=0.0, feather_px=0)
        patcher, _ = patched_run_tracked("segmentation", side_effect=ValueError("too few points"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await segment_by_polygon(
                    image_id=1, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentByPolygonAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(points=[[0, 0], [10, 0], [10, 10]], smooth=True, smoothing_factor=0.5, feather_px=3)

        result = await segment_by_polygon_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_by_polygon_task",
            image_id=1,
            user_id=1,
            points=[[0, 0], [10, 0], [10, 10]],
            smooth=True,
            smoothing_factor=0.5,
            feather_px=3,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentHybrid:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(
            yolo_conf_threshold=0.6, yolo_classes=["person"],
            fallback_min_area=200, fallback_max_segments=30, overlap_iou_thresh=0.4,
        )
        patcher, mock_run_tracked = patched_run_tracked("segmentation", return_value={"segments": []})

        with patcher:
            result = await segment_hybrid(
                image_id=1, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "segment_hybrid",
            1, 1, MLTaskType.SEGMENTATION_HYBRID,
            yolo_conf_threshold=0.6,
            yolo_classes=["person"],
            fallback_min_area=200,
            fallback_max_segments=30,
            overlap_iou_thresh=0.4,
        )
        assert result == {"segments": []}

    async def test_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(
            yolo_conf_threshold=0.5, yolo_classes=None,
            fallback_min_area=100, fallback_max_segments=50, overlap_iou_thresh=0.3,
        )
        patcher, _ = patched_run_tracked("segmentation", side_effect=ValueError("not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await segment_hybrid(
                    image_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSegmentHybridAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(
            yolo_conf_threshold=0.6, yolo_classes=["person"],
            fallback_min_area=200, fallback_max_segments=30, overlap_iou_thresh=0.4,
        )

        result = await segment_hybrid_async(image_id=1, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "segment_hybrid_task",
            image_id=1,
            user_id=1,
            yolo_conf_threshold=0.6,
            yolo_classes=["person"],
            fallback_min_area=200,
            fallback_max_segments=30,
            overlap_iou_thresh=0.4,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


# ---------------------------------------------------------------------------
# sam_ops.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestSamRemoveObject:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(expand_mask_pixels=8, use_edge_blending=True, ldm=make_ldm())
        patcher, mock_run_tracked = patched_run_tracked(
            "sam_ops", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            result = await sam_remove_object(
                image_id=1, mask_id=3, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "sam_remove_object",
            1, 1, MLTaskType.SAM_REMOVE_OBJECT,
            mask_id=3,
            expand_mask_pixels=8,
            use_edge_blending=True,
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
        )
        assert result["result_url"] == "s3://bucket/result.jpg"

    async def test_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(expand_mask_pixels=5, use_edge_blending=False, ldm=make_ldm())
        patcher, _ = patched_run_tracked("sam_ops", side_effect=ValueError("mask not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await sam_remove_object(
                    image_id=1, mask_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamRemoveObjectAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(expand_mask_pixels=8, use_edge_blending=True, ldm=make_ldm())

        result = await sam_remove_object_async(
            image_id=1, mask_id=3, body=body, current_user=mock_user, pool=mock_pool,
        )

        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_remove_object_task",
            image_id=1,
            mask_id=3,
            user_id=1,
            expand_mask_pixels=8,
            use_edge_blending=True,
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObject:
    async def test_success_with_replacement_file(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service, mock_file):
        body = make_body(
            expand_mask_pixels=5, use_color_matching=True, use_edge_blending=False,
            color_match_method="color_transfer", ldm=make_ldm(),
        )
        patcher, mock_run_tracked = patched_run_tracked(
            "sam_ops", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            result = await sam_replace_object(
                image_id=1, mask_id=3, replacement_file=mock_file, asset_id=None, body=body,
                current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
                asset_service=mock_asset_service,
            )

        mock_file.read.assert_awaited_once()
        mock_run_tracked.assert_awaited_once_with(
            SegmentationService, mock_deps, mock_mljob_service, "sam_replace_object",
            1, 1, MLTaskType.SAM_REPLACE_OBJECT,
            mask_id=3,
            replacement_image_bytes=b"image-bytes",
            expand_mask_pixels=5,
            use_color_matching=True,
            use_edge_blending=False,
            color_match_method="color_transfer",
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
            replacement_is_cutout=False,
        )
        assert result["result_url"] == "s3://bucket/result.jpg"

    async def test_success_with_asset_id(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        body = make_body(
            expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
            color_match_method="color_transfer", ldm=make_ldm(),
        )
        patcher, mock_run_tracked = patched_run_tracked(
            "sam_ops", return_value={"result_url": "s3://bucket/result.jpg"}
        )

        with patcher:
            await sam_replace_object(
                image_id=1, mask_id=3, replacement_file=None, asset_id="asset-1", body=body,
                current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
                asset_service=mock_asset_service,
            )

        mock_asset_service.get_asset_image.assert_awaited_once_with(1, "asset-1")
        assert mock_run_tracked.await_args.kwargs["replacement_image_bytes"] == b"asset-bytes"
        assert mock_run_tracked.await_args.kwargs["replacement_is_cutout"] is True

    async def test_missing_replacement_and_asset_id_returns_400(
        self, mock_user, mock_deps, mock_mljob_service, mock_asset_service
    ):
        with pytest.raises(HTTPException) as exc:
            await sam_replace_object(
                image_id=1, mask_id=3, replacement_file=None, asset_id=None,
                body=make_body(expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
                                color_match_method="color_transfer", ldm=make_ldm()),
                current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
                asset_service=mock_asset_service,
            )

        assert exc.value.status_code == 400

    async def test_asset_not_found_returns_404(self, mock_user, mock_deps, mock_mljob_service, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = None

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object(
                image_id=1, mask_id=3, replacement_file=None, asset_id="missing",
                body=make_body(expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
                                color_match_method="color_transfer", ldm=make_ldm()),
                current_user=mock_user, deps=mock_deps, mljob_service=mock_mljob_service,
                asset_service=mock_asset_service,
            )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestSamReplaceObjectAsync:
    async def test_enqueues_job_with_replacement_file(self, mock_user, mock_pool, mock_asset_service, mock_file):
        body = make_body(
            expand_mask_pixels=5, use_color_matching=True, use_edge_blending=False,
            color_match_method="color_transfer", ldm=make_ldm(),
        )

        result = await sam_replace_object_async(
            image_id=1, mask_id=3, replacement_file=mock_file, asset_id=None, body=body,
            current_user=mock_user, asset_service=mock_asset_service, pool=mock_pool,
        )

        mock_file.read.assert_awaited_once()
        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_replace_object_task",
            image_id=1,
            mask_id=3,
            replacement_image_bytes=b"image-bytes",
            user_id=1,
            expand_mask_pixels=5,
            use_color_matching=True,
            use_edge_blending=False,
            color_match_method="color_transfer",
            ldm_steps=25,
            ldm_sampler="plms",
            hd_strategy="CROP",
            replacement_is_cutout=False,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}

    async def test_missing_replacement_and_asset_id_returns_400(self, mock_user, mock_pool, mock_asset_service):
        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_async(
                image_id=1, mask_id=3, replacement_file=None, asset_id=None,
                body=make_body(expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
                                color_match_method="color_transfer", ldm=make_ldm()),
                current_user=mock_user, asset_service=mock_asset_service, pool=mock_pool,
            )

        assert exc.value.status_code == 400
        mock_pool.enqueue_job.assert_not_awaited()

    async def test_asset_not_found_returns_404(self, mock_user, mock_pool, mock_asset_service):
        mock_asset_service.get_asset_image.return_value = None

        with pytest.raises(HTTPException) as exc:
            await sam_replace_object_async(
                image_id=1, mask_id=3, replacement_file=None, asset_id="missing",
                body=make_body(expand_mask_pixels=5, use_color_matching=False, use_edge_blending=False,
                                color_match_method="color_transfer", ldm=make_ldm()),
                current_user=mock_user, asset_service=mock_asset_service, pool=mock_pool,
            )

        assert exc.value.status_code == 404
        mock_pool.enqueue_job.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
class TestExtractObject:
    async def test_success(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(padding_pixels=4, label="cutout", persist_to_s3=True)
        patcher, mock_run_tracked = patched_run_tracked(
            "sam_ops", return_value={"asset_id": "asset-1", "extracted_url": "s3://bucket/obj.png"}
        )

        with patcher:
            result = await extract_object(
                image_id=1, mask_id=3, body=body, current_user=mock_user,
                deps=mock_deps, mljob_service=mock_mljob_service,
            )

        mock_run_tracked.assert_awaited_once_with(
            AssetService, mock_deps, mock_mljob_service, "extract_object",
            1, 1, MLTaskType.EXTRACT_OBJECT,
            mask_id=3,
            padding_pixels=4,
            label="cutout",
            persist_to_s3=True,
        )
        assert result["asset_id"] == "asset-1"

    async def test_not_found(self, mock_user, mock_deps, mock_mljob_service):
        body = make_body(padding_pixels=4, label=None, persist_to_s3=False)
        patcher, _ = patched_run_tracked("sam_ops", side_effect=ValueError("mask not found"))

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await extract_object(
                    image_id=1, mask_id=999, body=body, current_user=mock_user,
                    deps=mock_deps, mljob_service=mock_mljob_service,
                )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestExtractObjectAsync:
    async def test_enqueues_job(self, mock_user, mock_pool):
        body = make_body(padding_pixels=4, label="cutout", persist_to_s3=True)

        result = await extract_object_async(image_id=1, mask_id=3, body=body, current_user=mock_user, pool=mock_pool)

        mock_pool.enqueue_job.assert_awaited_once_with(
            "sam_extract_object_task",
            image_id=1,
            mask_id=3,
            user_id=1,
            padding_pixels=4,
            label="cutout",
            persist_to_s3=True,
            _trace_carrier=ANY,
        )
        assert result == {"job_id": "job-123"}


# ---------------------------------------------------------------------------
# assets.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestListAssets:
    async def test_success(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.list_assets.return_value = [{"asset_id": "a1"}, {"asset_id": "a2"}]

        result = await list_assets(limit=10, offset=5, current_user=mock_user, service=mock_asset_full_service)

        mock_asset_full_service.list_assets.assert_awaited_once_with(1, limit=10, offset=5)
        assert len(result) == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetAssetThumbnail:
    async def test_success(self, mock_user, mock_asset_full_service):
        result = await get_asset_thumbnail(asset_id="asset-1", current_user=mock_user, service=mock_asset_full_service)

        assert result.body == b"thumb-bytes"
        assert result.media_type == "image/png"

    async def test_not_found(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.get_asset_thumbnail.return_value = None

        with pytest.raises(HTTPException) as exc:
            await get_asset_thumbnail(asset_id="missing", current_user=mock_user, service=mock_asset_full_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetAssetImage:
    async def test_success(self, mock_user, mock_asset_full_service):
        result = await get_asset_image(asset_id="asset-1", current_user=mock_user, service=mock_asset_full_service)

        assert result.body == b"asset-bytes"
        assert result.media_type == "image/png"

    async def test_not_found(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.get_asset_image.return_value = None

        with pytest.raises(HTTPException) as exc:
            await get_asset_image(asset_id="missing", current_user=mock_user, service=mock_asset_full_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestRenameAsset:
    async def test_success(self, mock_user, mock_asset_full_service):
        body = make_body(label="new-name")

        result = await rename_asset(
            asset_id="asset-1", body=body, current_user=mock_user, service=mock_asset_full_service,
        )

        mock_asset_full_service.rename_asset.assert_awaited_once_with(1, "asset-1", "new-name")
        assert result["label"] == "new-name"

    async def test_not_found(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.rename_asset.side_effect = ValueError("asset not found")

        with pytest.raises(HTTPException) as exc:
            await rename_asset(
                asset_id="missing", body=make_body(label="x"),
                current_user=mock_user, service=mock_asset_full_service,
            )

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestDeleteAsset:
    async def test_success(self, mock_user, mock_asset_full_service):
        result = await delete_asset(asset_id="asset-1", current_user=mock_user, service=mock_asset_full_service)

        mock_asset_full_service.delete_asset.assert_awaited_once_with(1, "asset-1")
        assert result == {"detail": "Asset deleted"}

    async def test_not_found(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.delete_asset.side_effect = ValueError("asset not found")

        with pytest.raises(HTTPException) as exc:
            await delete_asset(asset_id="missing", current_user=mock_user, service=mock_asset_full_service)

        assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
class TestPasteExtractedObject:
    async def test_success(self, mock_user, mock_asset_full_service):
        target_bbox = MagicMock()
        target_bbox.model_dump = MagicMock(return_value={"x1": 0, "y1": 0, "x2": 50, "y2": 50})
        body = make_body(
            asset_id=None,
            extracted_url="s3://bucket/obj.png",
            target_bbox=target_bbox,
            scale=1.5,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method="color_transfer",
        )

        result = await paste_extracted_object(
            image_id=1, body=body, current_user=mock_user, service=mock_asset_full_service,
        )

        mock_asset_full_service.paste_extracted_object.assert_awaited_once_with(
            image_id=1,
            user_id=1,
            asset_id=None,
            extracted_url="s3://bucket/obj.png",
            target_bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
            scale=1.5,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method="color_transfer",
        )
        assert result["result_url"] == "s3://bucket/pasted.jpg"

    async def test_not_found(self, mock_user, mock_asset_full_service):
        mock_asset_full_service.paste_extracted_object.side_effect = ValueError("extracted object not found")
        target_bbox = MagicMock()
        target_bbox.model_dump = MagicMock(return_value={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        body = make_body(
            asset_id=None,
            extracted_url="s3://bucket/missing.png",
            target_bbox=target_bbox,
            scale=1.0,
            use_color_matching=False,
            use_edge_blending=False,
            color_match_method="color_transfer",
        )

        with pytest.raises(HTTPException) as exc:
            await paste_extracted_object(
                image_id=1, body=body, current_user=mock_user, service=mock_asset_full_service,
            )

        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# jobs.py
# ---------------------------------------------------------------------------

def _patched_job(status, result_info=None):
    job_instance = MagicMock()
    job_instance.status = AsyncMock(return_value=status)
    job_instance.result_info = AsyncMock(return_value=result_info)
    return patch.object(ml_jobs_module, "Job", return_value=job_instance), job_instance


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetJobStatus:
    async def test_not_found_raises_404(self, mock_user):
        pool = MagicMock()
        patcher, _ = _patched_job(JobStatus.not_found)

        with patcher:
            with pytest.raises(HTTPException) as exc:
                await get_job_status(job_id="missing-job", current_user=mock_user, pool=pool)

        assert exc.value.status_code == 404

    @pytest.mark.parametrize("status", ["deferred", "queued", "in_progress"])
    async def test_pending_statuses_return_status_only(self, mock_user, status):
        pool = MagicMock()
        job_status = getattr(JobStatus, status)
        patcher, job_instance = _patched_job(job_status)

        with patcher:
            result = await get_job_status(job_id="job-1", current_user=mock_user, pool=pool)

        assert result == {"job_id": "job-1", "status": job_status.value}
        job_instance.result_info.assert_not_awaited()

    async def test_complete_success_includes_result(self, mock_user):
        pool = MagicMock()
        result_info = MagicMock(success=True, result={"result_url": "s3://out.jpg"})
        patcher, _ = _patched_job(JobStatus.complete, result_info=result_info)

        with patcher:
            result = await get_job_status(job_id="job-2", current_user=mock_user, pool=pool)

        assert result["status"] == JobStatus.complete.value
        assert result["result"] == {"result_url": "s3://out.jpg"}
        assert "error" not in result

    async def test_complete_failure_includes_error(self, mock_user):
        pool = MagicMock()
        result_info = MagicMock(success=False, result=ValueError("inpainting failed"))
        patcher, _ = _patched_job(JobStatus.complete, result_info=result_info)

        with patcher:
            result = await get_job_status(job_id="job-3", current_user=mock_user, pool=pool)

        assert result["status"] == JobStatus.complete.value
        assert result["error"] == "inpainting failed"
        assert "result" not in result

    async def test_complete_without_result_info(self, mock_user):
        pool = MagicMock()
        patcher, _ = _patched_job(JobStatus.complete, result_info=None)

        with patcher:
            result = await get_job_status(job_id="job-4", current_user=mock_user, pool=pool)

        assert result == {"job_id": "job-4", "status": JobStatus.complete.value}
        assert "result" not in result
        assert "error" not in result


# ---------------------------------------------------------------------------
# deps.py - _http_status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHttpStatus:
    def test_not_found(self):
        assert _http_status(ValueError("Image not found")) == 404

    def test_no_valid_detections(self):
        assert _http_status(ValueError("no valid detections")) == 404

    def test_unauthorized(self):
        assert _http_status(ValueError("unauthorized access")) == 403

    def test_generic_returns_400(self):
        assert _http_status(ValueError("something else went wrong")) == 400

    def test_case_insensitive(self):
        assert _http_status(ValueError("NOT FOUND")) == 404