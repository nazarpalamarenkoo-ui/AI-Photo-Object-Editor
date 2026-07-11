import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import (
    remove_object_task,
    remove_multiple_objects_task,
    replace_object_task,
)


class _FakeDepsCM:
    def __init__(self, deps):
        self._deps = deps

    async def __aenter__(self):
        return self._deps

    async def __aexit__(self, *exc):
        return False


class _FakeDbCM:
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
class TestRemoveObjectTask:
    async def test_calls_editing_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.remove_object = AsyncMock(return_value={"ok": True})

            result = await remove_object_task(
                ctx={},
                image_id=1,
                bbox_id=7,
                user_id=2,
                expand_mask_pixels=12,
                use_edge_blending=False,
                ldm_steps=50,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )

            MockService.assert_called_once_with(**fake_deps)
            instance.remove_object.assert_awaited_once_with(
                image_id=1,
                bbox_id=7,
                user_id=2,
                expand_mask_pixels=12,
                use_edge_blending=False,
                ldm_steps=50,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.remove_object = AsyncMock(return_value={})

            await remove_object_task(ctx={}, image_id=1, bbox_id=7, user_id=2)

            instance.remove_object.assert_awaited_once_with(
                image_id=1,
                bbox_id=7,
                user_id=2,
                expand_mask_pixels=5,
                use_edge_blending=True,
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="CROP",
            )


@pytest.mark.asyncio
class TestRemoveMultipleObjectsTask:
    async def test_calls_editing_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.remove_multiple_objects = AsyncMock(return_value={"ok": True})

            result = await remove_multiple_objects_task(
                ctx={},
                image_id=1,
                bbox_ids=[1, 2, 3],
                user_id=2,
                expand_mask_pixels=9,
                use_edge_blending=False,
                ldm_steps=10,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )

            instance.remove_multiple_objects.assert_awaited_once_with(
                image_id=1,
                bbox_ids=[1, 2, 3],
                user_id=2,
                expand_mask_pixels=9,
                use_edge_blending=False,
                ldm_steps=10,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.remove_multiple_objects = AsyncMock(return_value={})

            await remove_multiple_objects_task(
                ctx={}, image_id=1, bbox_ids=[1, 2], user_id=2,
            )

            instance.remove_multiple_objects.assert_awaited_once_with(
                image_id=1,
                bbox_ids=[1, 2],
                user_id=2,
                expand_mask_pixels=5,
                use_edge_blending=True,
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="CROP",
            )


@pytest.mark.asyncio
class TestReplaceObjectTask:
    async def test_calls_editing_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.replace_object = AsyncMock(return_value={"ok": True})

            result = await replace_object_task(
                ctx={},
                image_id=1,
                bbox_id=7,
                replace_image_bytes=b"binary-data",
                user_id=2,
                expand_mask_pixels=30,
                use_color_matching=True,
                use_edge_blending=True,
                color_match_method="color_transfer",
                ldm_steps=15,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )

            instance.replace_object.assert_awaited_once_with(
                image_id=1,
                bbox_id=7,
                replace_image_bytes=b"binary-data",
                user_id=2,
                expand_mask_pixels=30,
                use_color_matching=True,
                use_edge_blending=True,
                color_match_method="color_transfer",
                ldm_steps=15,
                ldm_sampler="ddim",
                hd_strategy="RESIZE",
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.replace_object = AsyncMock(return_value={})

            await replace_object_task(
                ctx={},
                image_id=1,
                bbox_id=7,
                replace_image_bytes=b"data",
                user_id=2,
            )

            instance.replace_object.assert_awaited_once_with(
                image_id=1,
                bbox_id=7,
                replace_image_bytes=b"data",
                user_id=2,
                expand_mask_pixels=25,
                use_color_matching=False,
                use_edge_blending=False,
                color_match_method="mean_std",
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="CROP",
            )