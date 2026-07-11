import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import (
    segment_objects_task,
    segment_with_prompt_task,
    segment_by_polygon_task,
    segment_hybrid_task,
    sam_remove_object_task,
    sam_replace_object_task,
)


class _FakeDepsCM:
    """Fake async context manager mimicking _build_ml_deps()."""

    def __init__(self, deps):
        self._deps = deps

    async def __aenter__(self):
        return self._deps

    async def __aexit__(self, *exc):
        return False


class _FakeDbCM:
    """Fake async context manager mimicking get_db_session()."""

    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_deps():
    return {
        "db": MagicMock(name="db"),
        "s3_storage": MagicMock(name="s3_storage"),
        "redis_storage": MagicMock(name="redis_storage"),
        "redis_history": MagicMock(name="redis_history"),
        "redis_assets": MagicMock(name="redis_assets"),
        "image_repo": MagicMock(name="image_repo"),
        "detection_repo": MagicMock(name="detection_repo"),
        "pipeline": MagicMock(name="pipeline"),
    }


@pytest.fixture(autouse=True)
def patch_db_and_deps(fake_deps):
    with patch(
        "app.workers.worker.get_db_session", return_value=_FakeDbCM(fake_deps["db"])
    ), patch("app.workers.worker._build_ml_deps", return_value=_FakeDepsCM(fake_deps)):
        yield fake_deps


@pytest.mark.asyncio
class TestSegmentObjectsTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_objects = AsyncMock(return_value={"ok": True})

            result = await segment_objects_task(
                ctx={}, image_id=1, user_id=2, min_area=100, max_segments=10,
            )

            MockService.assert_called_once_with(**fake_deps)
            instance.segment_objects.assert_awaited_once_with(
                image_id=1, user_id=2, min_area=100, max_segments=10,
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_objects = AsyncMock(return_value={})

            await segment_objects_task(ctx={}, image_id=1, user_id=2)

            instance.segment_objects.assert_awaited_once_with(
                image_id=1, user_id=2, min_area=500, max_segments=50,
            )


@pytest.mark.asyncio
class TestSegmentWithPromptTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_with_prompt = AsyncMock(return_value={"ok": True})

            result = await segment_with_prompt_task(
                ctx={},
                image_id=1,
                user_id=2,
                point_coords=[(1, 2)],
                point_labels=[1],
                bbox={"x": 0, "y": 0, "w": 1, "h": 1},
                multimask_output=True,
            )

            instance.segment_with_prompt.assert_awaited_once_with(
                image_id=1,
                user_id=2,
                point_coords=[(1, 2)],
                point_labels=[1],
                bbox={"x": 0, "y": 0, "w": 1, "h": 1},
                multimask_output=True,
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_with_prompt = AsyncMock(return_value={})

            await segment_with_prompt_task(ctx={}, image_id=1, user_id=2)

            instance.segment_with_prompt.assert_awaited_once_with(
                image_id=1,
                user_id=2,
                point_coords=None,
                point_labels=None,
                bbox=None,
                multimask_output=None,
            )


@pytest.mark.asyncio
class TestSegmentByPolygonTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_by_polygon = AsyncMock(return_value={"ok": True})

            result = await segment_by_polygon_task(
                ctx={},
                image_id=1,
                user_id=2,
                points=[(0, 0), (1, 1)],
                smooth=False,
                smoothing_factor=0.5,
                feather_px=3,
            )

            instance.segment_by_polygon.assert_awaited_once_with(
                image_id=1,
                user_id=2,
                points=[(0, 0), (1, 1)],
                smooth=False,
                smoothing_factor=0.5,
                feather_px=3,
            )
            assert result == {"ok": True}


@pytest.mark.asyncio
class TestSegmentHybridTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.segment_hybrid = AsyncMock(return_value={"ok": True})

            result = await segment_hybrid_task(
                ctx={},
                image_id=1,
                user_id=2,
                yolo_conf_threshold=0.7,
                yolo_classes=["cat"],
                fallback_min_area=900,
                fallback_max_segments=20,
                overlap_iou_thresh=0.4,
            )

            instance.segment_hybrid.assert_awaited_once_with(
                image_id=1,
                user_id=2,
                yolo_conf_threshold=0.7,
                yolo_classes=["cat"],
                fallback_min_area=900,
                fallback_max_segments=20,
                overlap_iou_thresh=0.4,
            )
            assert result == {"ok": True}


@pytest.mark.asyncio
class TestSamRemoveObjectTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.sam_remove_object = AsyncMock(return_value={"ok": True})

            result = await sam_remove_object_task(
                ctx={},
                image_id=1,
                mask_id=5,
                user_id=2,
                expand_mask_pixels=20,
                use_edge_blending=True,
                ldm_steps=30,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )

            instance.sam_remove_object.assert_awaited_once_with(
                image_id=1,
                mask_id=5,
                user_id=2,
                expand_mask_pixels=20,
                use_edge_blending=True,
                ldm_steps=30,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )
            assert result == {"ok": True}


@pytest.mark.asyncio
class TestSamReplaceObjectTask:
    async def test_calls_segmentation_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.SegmentationService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object = AsyncMock(return_value={"ok": True})

            result = await sam_replace_object_task(
                ctx={},
                image_id=1,
                mask_id=5,
                replacement_image_bytes=b"binary-data",
                user_id=2,
                expand_mask_pixels=15,
                use_color_matching=True,
                use_edge_blending=True,
                color_match_method="reinhard",
                ldm_steps=40,
                ldm_sampler="plms",
                hd_strategy="CROP",
                replacement_is_cutout=True,
            )

            instance.sam_replace_object.assert_awaited_once_with(
                image_id=1,
                mask_id=5,
                replacement_image_bytes=b"binary-data",
                user_id=2,
                expand_mask_pixels=15,
                use_color_matching=True,
                use_edge_blending=True,
                color_match_method="reinhard",
                ldm_steps=40,
                ldm_sampler="plms",
                hd_strategy="CROP",
                replacement_is_cutout=True,
            )
            assert result == {"ok": True}