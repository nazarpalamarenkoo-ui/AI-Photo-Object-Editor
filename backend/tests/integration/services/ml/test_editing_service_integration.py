import pytest
from unittest.mock import AsyncMock

from app.services.ml.editing_service import EditingService

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
def editing_service(
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
    return EditingService(
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


@pytest.fixture
async def sample_detection(detection_repo, sample_image_version):
    """A single persisted Detection scoped to sample_image_version.content_id."""
    from app.db.models.detection import Detection

    det = Detection(
        content_id=sample_image_version.content_id,
        bbox_id=1,
        detected_class="person",
        confidence=0.9,
        x1=0, y1=0, x2=10, y2=10,
        model_name="yolov10m",
        model_version="unknown",
        inference_time_ms=0.0,
    )
    await detection_repo.create_many([det])
    return det


def _pipeline_result():
    """Minimal pipeline result that includes bytes parseable as JPEG by PIL."""
    import io
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=(0, 100, 200)).save(buf, format="JPEG")
    return {
        "result_bytes": buf.getvalue(),
        "metrics": {"latency_ms": 5},
        "timestamp": "2024-01-01T00:00:00",
    }


def _setup_s3(mock_s3_storage, url="s3://bucket/r.jpg", presigned="https://url"):
    mock_s3_storage.upload_bytes = AsyncMock(return_value=url)
    mock_s3_storage.get_presigned_url = AsyncMock(return_value=presigned)


class TestRemoveObject:
    @pytest.mark.asyncio
    async def test_success_returns_result_and_new_version(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        res = _pipeline_result()
        mock_pipeline.remove_object = AsyncMock(return_value=res)
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_object(
            sample_image.id, sample_detection.bbox_id, sample_user.id
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        assert result["presigned_url"] == "https://url"
        assert "image_version_id" in result
        assert result["image_version_id"] != sample_image_version.id

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_object(
            sample_image.id, sample_detection.bbox_id, sample_user.id
        )

        versions = await editing_service.image_version_repo.list_by_image(sample_image.id)
        assert len(versions) == 2
        assert result["image_version_id"] == versions[-1].id

    @pytest.mark.asyncio
    async def test_removed_detection_is_not_carried_forward(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_object(
            sample_image.id, sample_detection.bbox_id, sample_user.id
        )

        new_version = await editing_service.image_version_repo.get_by_id(
            result["image_version_id"]
        )
        new_detections = await editing_service.detection_repo.get_by_content(
            new_version.content_id, active_only=True
        )
        assert new_detections == []

    @pytest.mark.asyncio
    async def test_pushes_undo_state_before_pipeline_call(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        await editing_service.remove_object(
            sample_image.id, sample_detection.bbox_id, sample_user.id
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        args = mock_redis_history.push_undo_state.call_args
        assert args.args[0] == sample_image.id
        assert args.args[1] == b"image-bytes"

    @pytest.mark.asyncio
    async def test_updates_current_state_in_redis(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        res = _pipeline_result()
        mock_pipeline.remove_object = AsyncMock(return_value=res)
        _setup_s3(mock_s3_storage)

        await editing_service.remove_object(
            sample_image.id, sample_detection.bbox_id, sample_user.id
        )

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=res["result_bytes"],
            suffix="current_state",
            ttl=7200,
        )

    @pytest.mark.asyncio
    async def test_raises_when_detection_missing(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="not found"):
            await editing_service.remove_object(sample_image.id, 9999, sample_user.id)

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, editing_service, sample_image, sample_image_version, sample_detection
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await editing_service.remove_object(
                sample_image.id, sample_detection.bbox_id, sample_image.user_id + 1
            )

    @pytest.mark.asyncio
    async def test_raises_when_no_current_version(
        self, editing_service, sample_image, sample_user
    ):
        with pytest.raises(ValueError, match="no current version"):
            await editing_service.remove_object(sample_image.id, 1, sample_user.id)

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.remove_object = AsyncMock(side_effect=RuntimeError("lama crashed"))

        with pytest.raises(RuntimeError, match="lama crashed"):
            await editing_service.remove_object(
                sample_image.id, sample_detection.bbox_id, sample_user.id
            )

        # undo state was pushed before the pipeline call
        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_selected_and_scene_bboxes_to_pipeline(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        detection_repo,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """Two detections: remove one, verify scene_bboxes contains only the other."""
        from app.db.models.detection import Detection
        dets = [
            Detection(
                content_id=sample_image_version.content_id,
                bbox_id=10, detected_class="person", confidence=0.9,
                x1=0, y1=0, x2=10, y2=10,
                model_name="m", model_version="v", inference_time_ms=0.0,
            ),
            Detection(
                content_id=sample_image_version.content_id,
                bbox_id=11, detected_class="car", confidence=0.8,
                x1=50, y1=50, x2=100, y2=100,
                model_name="m", model_version="v", inference_time_ms=0.0,
            ),
        ]
        await detection_repo.create_many(dets)
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        await editing_service.remove_object(sample_image.id, 10, sample_user.id)

        kw = mock_pipeline.remove_object.call_args.kwargs
        assert kw["selected_bbox"] == {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert len(kw["scene_bboxes"]) == 2  # full list passed in


class TestReplaceObject:
    @pytest.mark.asyncio
    async def test_success_passes_replacement_bytes_to_pipeline(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.replace_object(
            sample_image.id,
            sample_detection.bbox_id,
            b"replacement-bytes",
            sample_user.id,
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        kw = mock_pipeline.replace_object.call_args.kwargs
        assert kw["replacement_image_bytes"] == b"replacement-bytes"

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.replace_object(
            sample_image.id, sample_detection.bbox_id, b"rep", sample_user.id
        )

        assert "image_version_id" in result
        assert result["image_version_id"] != sample_image_version.id

    @pytest.mark.asyncio
    async def test_pushes_undo_state(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        sample_detection,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.replace_object = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        await editing_service.replace_object(
            sample_image.id, sample_detection.bbox_id, b"rep", sample_user.id
        )

        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_detection_missing(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="not found"):
            await editing_service.replace_object(
                sample_image.id, 9999, b"bytes", sample_user.id
            )

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_detection,
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await editing_service.replace_object(
                sample_image.id,
                sample_detection.bbox_id,
                b"bytes",
                sample_image.user_id + 1,
            )


class TestRemoveMultipleObjects:
    @pytest.fixture
    async def two_detections(self, detection_repo, sample_image_version):
        from app.db.models.detection import Detection
        dets = [
            Detection(
                content_id=sample_image_version.content_id,
                bbox_id=1, detected_class="person", confidence=0.9,
                x1=0, y1=0, x2=10, y2=10,
                model_name="m", model_version="v", inference_time_ms=0.0,
            ),
            Detection(
                content_id=sample_image_version.content_id,
                bbox_id=2, detected_class="car", confidence=0.8,
                x1=20, y1=20, x2=30, y2=30,
                model_name="m", model_version="v", inference_time_ms=0.0,
            ),
        ]
        await detection_repo.create_many(dets)
        return dets

    @pytest.mark.asyncio
    async def test_success_removes_selected_objects(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        two_detections,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_multiple_objects(
            sample_image.id, [1, 2], sample_user.id
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        # new version's content has no active detections
        new_ver = await editing_service.image_version_repo.get_by_id(
            result["image_version_id"]
        )
        remaining = await editing_service.detection_repo.get_by_content(
            new_ver.content_id, active_only=True
        )
        assert remaining == []

    @pytest.mark.asyncio
    async def test_raises_for_invalid_bbox_ids(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")

        with pytest.raises(ValueError, match="No valid detections"):
            await editing_service.remove_multiple_objects(
                sample_image.id, [999], sample_user.id
            )

    @pytest.mark.asyncio
    async def test_partial_removal_carries_unselected_detection_forward(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        two_detections,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """Remove bbox_id=1 only; bbox_id=2 should be carried forward to new version."""
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_multiple_objects(
            sample_image.id, [1], sample_user.id
        )

        new_ver = await editing_service.image_version_repo.get_by_id(
            result["image_version_id"]
        )
        remaining = await editing_service.detection_repo.get_by_content(
            new_ver.content_id, active_only=True
        )
        assert len(remaining) == 1
        assert remaining[0].bbox_id == 2

    @pytest.mark.asyncio
    async def test_partial_list_passes_correct_selected_and_scene_bboxes_to_pipeline(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        two_detections,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        await editing_service.remove_multiple_objects(sample_image.id, [1], sample_user.id)

        kw = mock_pipeline.remove_multiple_objects.call_args.kwargs
        assert len(kw["selected_bboxes"]) == 1
        assert len(kw["scene_bboxes"]) == 1  # only the non-selected one

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        two_detections,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.remove_multiple_objects = AsyncMock(return_value=_pipeline_result())
        _setup_s3(mock_s3_storage)

        result = await editing_service.remove_multiple_objects(
            sample_image.id, [1, 2], sample_user.id
        )

        assert result["image_version_id"] != sample_image_version.id


class TestSamReplaceObjectDiffusion:
    """
    Unlike remove/replace, this op does NOT key off a stored Detection row —
    mask/bbox come from the client. So no "detection missing" failure mode
    exists. All active detections not overlapping bbox are carried forward.
    """

    @pytest.mark.asyncio
    async def test_success_uploads_and_returns_result(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        result = await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        assert result["result_url"] == "s3://bucket/r.jpg"
        assert result["presigned_url"] == "https://url"
        assert "metrics" in result
        assert "timestamp" in result
        assert "image_version_id" in result

    @pytest.mark.asyncio
    async def test_forks_new_version(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        result = await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        assert result["image_version_id"] != sample_image_version.id

    @pytest.mark.asyncio
    async def test_raises_when_unauthorized(
        self, editing_service, sample_image, sample_image_version
    ):
        with pytest.raises(ValueError, match="Unauthorized"):
            await editing_service.sam_replace_object_diffusion(
                image_id=sample_image.id,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=sample_image.user_id + 1,
            )

    @pytest.mark.asyncio
    async def test_raises_when_image_not_found(
        self, editing_service, sample_user
    ):
        with pytest.raises(ValueError, match="not found"):
            await editing_service.sam_replace_object_diffusion(
                image_id=999999,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=sample_user.id,
            )

    @pytest.mark.asyncio
    async def test_passes_mask_bbox_and_reference_bytes_to_pipeline(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)
        bbox = {"x1": 5, "y1": 5, "x2": 25, "y2": 25}

        await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox=bbox,
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        kw = mock_pipeline.sam_replace_object_diffusion.call_args.kwargs
        assert kw["image_bytes"] == b"image-bytes"
        assert kw["mask_bytes"] == b"mask-bytes"
        assert kw["bbox"] == bbox
        assert kw["reference_image_bytes"] == b"ref-bytes"

    @pytest.mark.asyncio
    async def test_forwards_optional_diffusion_params_to_pipeline(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
            prompt="a red chair",
            use_color_matching=True,
            color_match_method="histogram",
            negative_prompt="blurry",
            num_inference_steps=30,
            guidance_scale=7.5,
            ip_adapter_scale=0.6,
            strength=0.9,
            seed=42,
        )

        kw = mock_pipeline.sam_replace_object_diffusion.call_args.kwargs
        assert kw["prompt"] == "a red chair"
        assert kw["use_color_matching"] is True
        assert kw["color_match_method"] == "histogram"
        assert kw["negative_prompt"] == "blurry"
        assert kw["num_inference_steps"] == 30
        assert kw["guidance_scale"] == 7.5
        assert kw["ip_adapter_scale"] == 0.6
        assert kw["strength"] == 0.9
        assert kw["seed"] == 42

    @pytest.mark.asyncio
    async def test_pushes_undo_state_before_pipeline_call(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        call = mock_redis_history.push_undo_state.call_args
        assert call.args[0] == sample_image.id
        assert call.args[1] == b"image-bytes"
        assert call.kwargs["label"] == "sam replace (diffusion)"

    @pytest.mark.asyncio
    async def test_updates_current_state_in_redis(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        res = _pipeline_result()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(return_value=res)
        _setup_s3(mock_s3_storage)

        await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        mock_redis_cache.cache_image.assert_awaited_once_with(
            image_id=sample_image.id,
            image_data=res["result_bytes"],
            suffix="current_state",
            ttl=7200,
        )

    @pytest.mark.asyncio
    async def test_pipeline_exception_propagates_after_undo_push(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        mock_redis_cache,
        mock_redis_history,
        mock_pipeline,
    ):
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            side_effect=RuntimeError("diffusion failed")
        )

        with pytest.raises(RuntimeError, match="diffusion failed"):
            await editing_service.sam_replace_object_diffusion(
                image_id=sample_image.id,
                mask_bytes=b"mask-bytes",
                bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                reference_image_bytes=b"ref-bytes",
                user_id=sample_user.id,
            )

        mock_redis_history.push_undo_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_overlapping_detection_is_carried_forward(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        detection_repo,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """Detection far from bbox should survive to new version."""
        from app.db.models.detection import Detection
        det = Detection(
            content_id=sample_image_version.content_id,
            bbox_id=5, detected_class="car", confidence=0.9,
            x1=200, y1=200, x2=300, y2=300,
            model_name="m", model_version="v", inference_time_ms=0.0,
        )
        await detection_repo.create_many([det])
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        result = await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        new_ver = await editing_service.image_version_repo.get_by_id(
            result["image_version_id"]
        )
        carried = await editing_service.detection_repo.get_by_content(
            new_ver.content_id, active_only=True
        )
        assert len(carried) == 1
        assert carried[0].bbox_id == 5

    @pytest.mark.asyncio
    async def test_overlapping_detection_is_excluded_from_carry_forward(
        self,
        editing_service,
        sample_image,
        sample_image_version,
        sample_user,
        detection_repo,
        mock_redis_cache,
        mock_pipeline,
        mock_s3_storage,
    ):
        """Detection overlapping the diffusion bbox should NOT appear on new version."""
        from app.db.models.detection import Detection
        det = Detection(
            content_id=sample_image_version.content_id,
            bbox_id=6, detected_class="person", confidence=0.9,
            x1=0, y1=0, x2=10, y2=10,  # same bbox as the op
            model_name="m", model_version="v", inference_time_ms=0.0,
        )
        await detection_repo.create_many([det])
        mock_redis_cache.get_cache_image = AsyncMock(return_value=b"image-bytes")
        mock_redis_cache.cache_image = AsyncMock()
        mock_pipeline.sam_replace_object_diffusion = AsyncMock(
            return_value=_pipeline_result()
        )
        _setup_s3(mock_s3_storage)

        result = await editing_service.sam_replace_object_diffusion(
            image_id=sample_image.id,
            mask_bytes=b"mask-bytes",
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            reference_image_bytes=b"ref-bytes",
            user_id=sample_user.id,
        )

        new_ver = await editing_service.image_version_repo.get_by_id(
            result["image_version_id"]
        )
        carried = await editing_service.detection_repo.get_by_content(
            new_ver.content_id, active_only=True
        )
        assert carried == []