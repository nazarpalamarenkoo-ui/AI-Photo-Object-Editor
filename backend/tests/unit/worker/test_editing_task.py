import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.worker import (
    remove_object_task,
    remove_multiple_objects_task,
    replace_object_task,
    sam_replace_object_diffusion_task,
)


class _FakeDepsCM:
    def __init__(self, deps, mljob_service):
        self._deps = deps
        self._mljob_service = mljob_service

    async def __aenter__(self):
        return self._deps, self._mljob_service

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_deps():
    return {
        "s3_storage": MagicMock(name="s3_storage"),
        "redis_storage": MagicMock(name="redis_storage"),
        "redis_history": MagicMock(name="redis_history"),
        "redis_assets": MagicMock(name="redis_assets"),
        "image_repo": MagicMock(name="image_repo"),
        "image_version_repo": MagicMock(name="image_version_repo"),
        "image_content_repo": MagicMock(name="image_content_repo"),
        "detection_repo": MagicMock(name="detection_repo"),
        "segmentation_repo": MagicMock(name="segmentation_repo"),
        "edit_history_repo": MagicMock(name="edit_history_repo"),
        "assets_repo": MagicMock(name="assets_repo"),
        "pipeline": MagicMock(name="pipeline"),
    }


@pytest.fixture
def fake_mljob_service():
    return MagicMock(name="mljob_service")


@pytest.fixture(autouse=True)
def patch_build_ml_deps(fake_deps, fake_mljob_service):
    with patch(
        "app.workers.worker._build_ml_deps",
        return_value=_FakeDepsCM(fake_deps, fake_mljob_service),
    ):
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


@pytest.mark.asyncio
class TestSamReplaceObjectDiffusionTask:
    async def test_calls_editing_service_with_correct_args(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(return_value={"ok": True})

            result = await sam_replace_object_diffusion_task(
                ctx={},
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
                prompt="a red sports car",
                use_color_matching=True,
                color_match_method="reinhard",
                negative_prompt="blurry, low quality",
                num_inference_steps=40,
                guidance_scale=7.5,
                ip_adapter_scale=0.6,
                strength=0.85,
                seed=42,
            )

            MockService.assert_called_once_with(**fake_deps)
            instance.sam_replace_object_diffusion.assert_awaited_once_with(
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
                prompt="a red sports car",
                use_color_matching=True,
                color_match_method="reinhard",
                negative_prompt="blurry, low quality",
                num_inference_steps=40,
                guidance_scale=7.5,
                ip_adapter_scale=0.6,
                strength=0.85,
                seed=42,
            )
            assert result == {"ok": True}

    async def test_uses_default_params(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(return_value={})

            await sam_replace_object_diffusion_task(
                ctx={},
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
            )

            instance.sam_replace_object_diffusion.assert_awaited_once_with(
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
                prompt="",
                use_color_matching=False,
                color_match_method="color_transfer",
                negative_prompt=None,
                num_inference_steps=None,
                guidance_scale=None,
                ip_adapter_scale=None,
                strength=None,
                seed=0,
            )

    async def test_propagates_optional_overrides_individually(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(return_value={})

            await sam_replace_object_diffusion_task(
                ctx={},
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
                guidance_scale=9.0,
            )

            call_kwargs = instance.sam_replace_object_diffusion.await_args.kwargs
            assert call_kwargs["guidance_scale"] == 9.0
            assert call_kwargs["negative_prompt"] is None
            assert call_kwargs["num_inference_steps"] is None
            assert call_kwargs["ip_adapter_scale"] is None
            assert call_kwargs["strength"] is None

    async def test_passes_through_raw_binary_payloads_unmodified(self, fake_deps):
        mask_bytes = bytes(range(256))
        reference_bytes = b"\x00\x01\xfe\xff" * 100

        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(return_value={})

            await sam_replace_object_diffusion_task(
                ctx={},
                image_id=1,
                mask_bytes=mask_bytes,
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=reference_bytes,
                user_id=2,
            )

            call_kwargs = instance.sam_replace_object_diffusion.await_args.kwargs
            assert call_kwargs["mask_bytes"] is mask_bytes
            assert call_kwargs["reference_image_bytes"] is reference_bytes

    async def test_propagates_exception_from_service(self, fake_deps):
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(
                side_effect=ValueError("Image not found")
            )

            with pytest.raises(ValueError, match="Image not found"):
                await sam_replace_object_diffusion_task(
                    ctx={},
                    image_id=999,
                    mask_bytes=b"mask-bytes",
                    bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                    reference_image_bytes=b"ref-bytes",
                    user_id=2,
                )

    async def test_returns_result_dict_shape_from_service(self, fake_deps):
        expected = {
            "result_url": "s3://bucket/results/2/1/sam_replace_diffusion_123.jpg",
            "presigned_url": "https://signed.example/...",
            "metrics": {"latency_ms": 812},
            "timestamp": "2026-07-28T12:00:00Z",
        }
        with patch("app.workers.worker.EditingService") as MockService:
            instance = MockService.return_value
            instance.sam_replace_object_diffusion = AsyncMock(return_value=expected)

            result = await sam_replace_object_diffusion_task(
                ctx={},
                image_id=1,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=2,
            )

            assert result == expected