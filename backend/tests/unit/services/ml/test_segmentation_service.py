import base64
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.segmentation_service import (
    SegmentationService,
    _mask_to_data_url,
    _segments_for_response,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_s3():
    s3 = AsyncMock()
    s3.download = AsyncMock(return_value=b"original-bytes")
    s3.upload_bytes = AsyncMock(return_value="s3://bucket/path.jpg")
    s3.get_presigned_url = AsyncMock(return_value="https://presigned.example/path.jpg")
    return s3


@pytest.fixture
def mock_redis_storage():
    redis_storage = AsyncMock()
    redis_storage.get_cache_image = AsyncMock(return_value=None)
    redis_storage.cache_image = AsyncMock(return_value=None)
    redis_storage.delete = AsyncMock(return_value=None)
    return redis_storage


@pytest.fixture
def mock_redis_history():
    history = AsyncMock()
    history.push_undo_state = AsyncMock(return_value=None)
    return history


@pytest.fixture
def mock_redis_assets():
    return AsyncMock()


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_image_version_repo():
    repo = AsyncMock()
    repo.create_next = AsyncMock()
    return repo


@pytest.fixture
def mock_image_content_repo():
    repo = AsyncMock()
    content = MagicMock()
    content.id = 200
    repo.get_or_create = AsyncMock(return_value=(content, True))
    return repo


@pytest.fixture
def mock_detection_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_segmentation_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.max_mask_id = AsyncMock(return_value=-1)
    repo.create_many = AsyncMock(side_effect=lambda masks: masks)
    repo.soft_delete = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_edit_history_repo():
    return AsyncMock()


@pytest.fixture
def mock_assets_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock()
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    image.width = 200
    image.height = 200
    return image


@pytest.fixture
def sample_version():
    version = MagicMock()
    version.id = 10
    version.content_id = 100
    version.storage_path = "raw/42/1/original.jpg"
    return version


@pytest.fixture
def new_version():
    version = MagicMock()
    version.id = 11
    version.content_id = 200
    return version


@pytest.fixture
def service(
    mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_image_version_repo, mock_image_content_repo,
    mock_detection_repo, mock_segmentation_repo, mock_edit_history_repo,
    mock_assets_repo, mock_pipeline, sample_image, sample_version, new_version,
):
    mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
    mock_image_version_repo.get_current = AsyncMock(return_value=sample_version)
    mock_image_version_repo.create_next = AsyncMock(return_value=new_version)

    svc = SegmentationService(
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        image_version_repo=mock_image_version_repo,
        image_content_repo=mock_image_content_repo,
        detection_repo=mock_detection_repo,
        segmentation_repo=mock_segmentation_repo,
        edit_history_repo=mock_edit_history_repo,
        assets_repo=mock_assets_repo,
        pipeline=mock_pipeline,
    )

    svc._read_dimensions = MagicMock(return_value=(100, 100))
    return svc


def make_raw_segment(mask_id=1, bbox=None, area=1000):
    return {
        "mask_id": mask_id, "bbox_id": mask_id,
        "bbox": bbox or {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "area": area, "stability_score": 0.95, "mask_bytes": b"mask-bytes",
    }


def make_db_mask(mask_id=1, storage_path="s3://bucket/masks/1.png"):
    mask = MagicMock()
    mask.mask_id = mask_id
    mask.mask_storage_path = storage_path
    mask.x1, mask.y1, mask.x2, mask.y2 = 0, 0, 10, 10
    mask.area = 100.0
    mask.score = 0.9
    return mask


def make_detection(x1=0, y1=0, x2=10, y2=10, conf=0.9, label="cat"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "detected_class": label}


class TestSegmentObjects:

    async def test_runs_pipeline_and_persists_when_no_cache(
        self, service, mock_segmentation_repo, mock_pipeline,
    ):
        segment = make_raw_segment(mask_id=999)
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [segment], "metrics": {"latency_ms": 5}, "image_size": (200, 200),
        })

        result = await service.segment_objects(image_id=1, user_id=42)

        mock_pipeline.sam_segment_objects.assert_awaited_once()
        mock_segmentation_repo.create_many.assert_awaited_once()
        assert result["segments"][0]["mask_id"] == 0  # offset from max_mask_id() + 1 == 0
        assert "mask_bytes" not in result["segments"][0]
        assert result["image_size"] == (200, 200)

    async def test_served_from_content_cache_skips_pipeline(
        self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline,
    ):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_db_mask(3)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"cached-mask-bytes")

        result = await service.segment_objects(image_id=1, user_id=42)

        mock_pipeline.sam_segment_objects.assert_not_called()
        assert result["metrics"] == {"cache_hit": True}
        assert len(result["segments"]) == 1
        assert result["segments"][0]["mask_id"] == 3

    async def test_offset_continues_from_max_existing_mask_id(
        self, service, mock_segmentation_repo, mock_pipeline,
    ):
        mock_segmentation_repo.max_mask_id = AsyncMock(return_value=4)
        segments = [make_raw_segment(mask_id=999), make_raw_segment(mask_id=998)]
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": segments, "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_objects(image_id=1, user_id=42)

        ids = sorted(seg["mask_id"] for seg in result["segments"])
        assert ids == [5, 6]

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_objects(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_objects(image_id=1, user_id=42)

    async def test_pipeline_exception_propagates(self, service, mock_pipeline):
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("sam crashed"))

        with pytest.raises(RuntimeError, match="sam crashed"):
            await service.segment_objects(image_id=1, user_id=42)


class TestSegmentByPolygon:

    async def test_success(self, service, mock_pipeline, mock_segmentation_repo):
        segment = make_raw_segment(mask_id=999)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [segment], "metrics": {"latency_ms": 5}, "image_size": (200, 200),
        })

        result = await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        assert result["segments"][0]["mask_id"] == 0
        assert "mask_bytes" not in result["segments"][0]
        mock_segmentation_repo.create_many.assert_awaited_once()

    async def test_passes_polygon_params_to_pipeline(self, service, mock_pipeline):
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })
        points = [(0, 0), (10, 0), (5, 10)]

        await service.segment_by_polygon(
            image_id=1, user_id=42, points=points, smooth=False, smoothing_factor=0.7, feather_px=4,
        )

        _, kwargs = mock_pipeline.sam_segment_by_polygon.call_args
        assert kwargs["points"] == points
        assert kwargs["smooth"] is False
        assert kwargs["smoothing_factor"] == 0.7
        assert kwargs["feather_px"] == 4

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_by_polygon(image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)])

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_by_polygon(image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)])

    async def test_pipeline_exception_propagates(self, service, mock_pipeline):
        mock_pipeline.sam_segment_by_polygon = AsyncMock(side_effect=RuntimeError("polygon failed"))

        with pytest.raises(RuntimeError, match="polygon failed"):
            await service.segment_by_polygon(image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)])

    async def test_empty_segments_result(self, service, mock_pipeline):
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_by_polygon(image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)])

        assert result["segments"] == []


class TestSegmentHybrid:

    async def test_yolo_and_nonoverlapping_fallback_both_included(
        self, service, mock_pipeline, mock_segmentation_repo,
    ):
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_detection(0, 0, 10, 10)],
        })
        yolo_seg = make_raw_segment(mask_id=100, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={"segments": [yolo_seg]})
        fallback_seg = make_raw_segment(mask_id=200, bbox={"x1": 50, "y1": 50, "x2": 60, "y2": 60})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [fallback_seg], "image_size": (300, 300),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        sources = {seg["source"] for seg in result["segments"]}
        assert sources == {"yolo", "sam_auto"}
        assert len(result["segments"]) == 2
        assert result["image_size"] == (300, 300)

    async def test_overlapping_fallback_segment_is_dropped(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_detection(0, 0, 10, 10)],
        })
        yolo_seg = make_raw_segment(mask_id=1, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={"segments": [yolo_seg]})
        duplicate_seg = make_raw_segment(mask_id=2, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [duplicate_seg], "image_size": (100, 100),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42, overlap_iou_thresh=0.5)

        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "yolo"

    async def test_no_yolo_detections_skips_batch_sam_call(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock()
        fallback_seg = make_raw_segment(mask_id=1, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [fallback_seg], "image_size": (50, 50),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        mock_pipeline.sam_segment_with_prompts_batch.assert_not_called()
        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "sam_auto"

    async def test_strips_mask_bytes_from_response(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [make_raw_segment(mask_id=1)], "image_size": (10, 10),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        assert all("mask_bytes" not in seg for seg in result["segments"])

    async def test_passes_yolo_params_to_detect_objects(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={"segments": [], "image_size": (1, 1)})

        await service.segment_hybrid(
            image_id=1, user_id=42, yolo_conf_threshold=0.6, yolo_classes=["dog"],
            fallback_min_area=1200, fallback_max_segments=10,
        )

        _, detect_kwargs = mock_pipeline.detect_objects.call_args
        assert detect_kwargs["conf_threshold"] == 0.6
        assert detect_kwargs["classes"] == ["dog"]
        _, fallback_kwargs = mock_pipeline.sam_segment_objects.call_args
        assert fallback_kwargs["min_area"] == 1200
        assert fallback_kwargs["max_segments"] == 10

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_hybrid(image_id=1, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_hybrid(image_id=1, user_id=42)

    async def test_detect_objects_exception_propagates(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("yolo crashed"))

        with pytest.raises(RuntimeError, match="yolo crashed"):
            await service.segment_hybrid(image_id=1, user_id=42)

    async def test_fallback_exception_propagates(self, service, mock_pipeline):
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("fallback crashed"))

        with pytest.raises(RuntimeError, match="fallback crashed"):
            await service.segment_hybrid(image_id=1, user_id=42)


class TestSamRemoveObject:

    async def test_success(
        self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline,
        mock_redis_history, mock_s3, mock_edit_history_repo,
    ):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_db_mask(5)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"mask-bytes")
        mock_pipeline.sam_remove_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

        mock_redis_history.push_undo_state.assert_awaited_once()
        mock_pipeline.sam_remove_object.assert_awaited_once()
        mock_s3.upload_bytes.assert_awaited_once()
        mock_edit_history_repo.create.assert_awaited_once()
        assert result["image_version_id"] == 11

    async def test_mask_not_found(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="not found"):
            await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)

    async def test_pipeline_exception(self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_db_mask(5)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"mask-bytes")
        mock_pipeline.sam_remove_object = AsyncMock(side_effect=RuntimeError("lama failure"))

        with pytest.raises(RuntimeError, match="lama failure"):
            await service.sam_remove_object(image_id=1, mask_id=5, user_id=42)


class TestSamReplaceObject:

    async def test_success(
        self, service, mock_segmentation_repo, mock_redis_storage, mock_pipeline, mock_redis_history,
    ):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[make_db_mask(5)])
        mock_redis_storage.get_cache_image = AsyncMock(return_value=b"mask-bytes")
        mock_pipeline.sam_replace_object = AsyncMock(return_value={
            "result_bytes": b"result", "metrics": {}, "timestamp": "t",
        })

        result = await service.sam_replace_object(
            image_id=1, mask_id=5, replacement_image_bytes=b"new-obj", user_id=42,
        )

        mock_redis_history.push_undo_state.assert_awaited_once()
        _, kwargs = mock_pipeline.sam_replace_object.call_args
        assert kwargs["replacement_image_bytes"] == b"new-obj"
        assert "image_version_id" in result

    async def test_mask_not_found(self, service, mock_segmentation_repo):
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="not found"):
            await service.sam_replace_object(
                image_id=1, mask_id=5, replacement_image_bytes=b"x", user_id=42,
            )

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.sam_replace_object(
                image_id=1, mask_id=5, replacement_image_bytes=b"x", user_id=42,
            )


class TestNextMaskOffset:
    """_next_mask_offset is now content-scoped and DB-backed
    (segmentation_repo.max_mask_id), not a Redis cached-segments list."""

    async def test_zero_when_no_masks_persisted(self, service, mock_segmentation_repo):
        mock_segmentation_repo.max_mask_id = AsyncMock(return_value=-1)

        offset = await service._next_mask_offset(content_id=100)

        assert offset == 0

    async def test_continues_from_max_plus_one(self, service, mock_segmentation_repo):
        mock_segmentation_repo.max_mask_id = AsyncMock(return_value=5)

        offset = await service._next_mask_offset(content_id=100)

        assert offset == 6


class TestGetSupportedClasses:

    def test_passthrough_to_pipeline(self, service, mock_pipeline):
        mock_pipeline.get_supported_classes = MagicMock(return_value=["person", "car"])

        assert service.get_supported_classes() == ["person", "car"]


class TestIouAndOverlapsAny:

    def test_iou_identical_boxes_is_one(self):
        box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert SegmentationService._iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap_is_zero(self):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 100, "y1": 100, "x2": 110, "y2": 110}
        assert SegmentationService._iou(a, b) == 0.0

    def test_iou_partial_overlap(self):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}
        expected = 25 / 175
        assert SegmentationService._iou(a, b) == pytest.approx(expected)

    def test_iou_degenerate_box_zero_union_returns_zero(self):
        a = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
        b = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
        assert SegmentationService._iou(a, b) == 0.0

    def test_overlaps_any_true_when_above_threshold(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        existing = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        assert SegmentationService._overlaps_any(bbox, existing, 0.5) is True

    def test_overlaps_any_false_when_below_threshold(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        existing = [{"x1": 100, "y1": 100, "x2": 110, "y2": 110}]
        assert SegmentationService._overlaps_any(bbox, existing, 0.5) is False

    def test_overlaps_any_empty_existing_list_is_false(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert SegmentationService._overlaps_any(bbox, [], 0.5) is False


class TestMaskToDataUrl:

    def test_returns_correct_data_url_prefix(self):
        result = _mask_to_data_url(b"fake-png-bytes")
        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_matches_input_bytes(self):
        raw = b"\x89PNG\r\n\x1a\nfake-mask-data"
        result = _mask_to_data_url(raw)
        b64_payload = result.split(",", 1)[1]
        assert base64.b64decode(b64_payload) == raw

    def test_empty_bytes_returns_empty_payload_url(self):
        assert _mask_to_data_url(b"") == "data:image/png;base64,"

    def test_different_inputs_produce_different_urls(self):
        assert _mask_to_data_url(b"mask-one") != _mask_to_data_url(b"mask-two")


class TestSegmentsForResponse:

    def test_strips_mask_bytes_key(self):
        segments = [{"mask_id": 0, "bbox_id": 0, "mask_bytes": b"raw-bytes"}]
        result = _segments_for_response(segments)
        assert "mask_bytes" not in result[0]

    def test_adds_mask_url_when_mask_bytes_present(self):
        segments = [{"mask_id": 0, "mask_bytes": b"raw-bytes"}]
        result = _segments_for_response(segments)
        assert result[0]["mask_url"] == _mask_to_data_url(b"raw-bytes")

    def test_no_mask_url_when_mask_bytes_key_missing(self):
        segments = [{"mask_id": 0, "bbox_id": 0}]
        result = _segments_for_response(segments)
        assert "mask_url" not in result[0]

    def test_no_mask_url_when_mask_bytes_is_empty_or_none(self):
        segments = [{"mask_id": 0, "mask_bytes": b""}, {"mask_id": 1, "mask_bytes": None}]
        result = _segments_for_response(segments)
        assert "mask_url" not in result[0]
        assert "mask_url" not in result[1]

    def test_preserves_other_fields(self):
        segments = [{
            "mask_id": 5, "bbox_id": 5, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "area": 100, "stability_score": 0.87, "source": "yolo", "mask_bytes": b"raw-bytes",
        }]
        result = _segments_for_response(segments)
        assert result[0]["mask_id"] == 5
        assert result[0]["source"] == "yolo"

    def test_empty_list_returns_empty_list(self):
        assert _segments_for_response([]) == []

    def test_does_not_mutate_original_segments(self):
        original = {"mask_id": 0, "mask_bytes": b"raw-bytes"}
        _segments_for_response([original])
        assert original["mask_bytes"] == b"raw-bytes"

    def test_preserves_segment_order(self):
        segments = [{"mask_id": i, "mask_bytes": b"x"} for i in range(5)]
        result = _segments_for_response(segments)
        assert [seg["mask_id"] for seg in result] == [0, 1, 2, 3, 4]