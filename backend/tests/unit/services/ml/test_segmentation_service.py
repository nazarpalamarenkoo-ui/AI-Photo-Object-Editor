import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.ml.segmentation_service import (
    SegmentationService,
    _mask_to_data_url,
    _segments_for_response,
)
from app.db.models.image import Image


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return AsyncMock()


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
    redis_storage.cache_segments = AsyncMock(return_value=None)
    redis_storage.get_cached_segments = AsyncMock(return_value=None)
    return redis_storage


@pytest.fixture
def mock_redis_history():
    history = AsyncMock()
    history.push_undo_state = AsyncMock(return_value=None)
    history.pop_undo_state = AsyncMock(return_value=None)
    history.push_redo_state = AsyncMock(return_value=None)
    history.pop_redo_state = AsyncMock(return_value=None)
    history.get_history_labels = AsyncMock(return_value=[])
    history.clear_history = AsyncMock(return_value=None)
    return history


@pytest.fixture
def mock_redis_assets():
    ra = AsyncMock()
    ra.list_assets = AsyncMock(return_value=[])
    ra.get_thumbnail = AsyncMock(return_value=None)
    ra.get_asset = AsyncMock(return_value=None)
    ra.rename_asset = AsyncMock(return_value=None)
    ra.delete_asset = AsyncMock(return_value=False)
    return ra


@pytest.fixture
def mock_image_repo():
    return AsyncMock()


@pytest.fixture
def mock_detection_repo():
    return AsyncMock()


@pytest.fixture
def mock_pipeline():
    return AsyncMock()


@pytest.fixture
def sample_image():
    image = MagicMock(spec=Image)
    image.id = 1
    image.user_id = 42
    image.storage_path = "raw/42/1/original.jpg"
    image.filename = "original.jpg"
    return image


@pytest.fixture
def service(
    mock_db, mock_s3, mock_redis_storage, mock_redis_history, mock_redis_assets,
    mock_image_repo, mock_detection_repo, mock_pipeline,
):
    return SegmentationService(
        db=mock_db,
        s3_storage=mock_s3,
        redis_storage=mock_redis_storage,
        redis_history=mock_redis_history,
        redis_assets=mock_redis_assets,
        image_repo=mock_image_repo,
        detection_repo=mock_detection_repo,
        pipeline=mock_pipeline,
    )


def make_segment(mask_id=1, bbox=None, area=1000, source=None):
    seg = {
        "mask_id": mask_id,
        "bbox_id": mask_id,
        "bbox": bbox or {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "area": area,
        "stability_score": 0.95,
        "mask_bytes": b"mask-bytes",
    }
    if source is not None:
        seg["source"] = source
    return seg


def make_detection(x1=0, y1=0, x2=10, y2=10, conf=0.9, label="cat"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "label": label}


class TestSegmentByPolygon:
    async def test_success_single_segment_gets_offset_as_id(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)
        segment = make_segment(mask_id=999)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [segment], "metrics": {"latency_ms": 5}, "image_size": (200, 200),
        })

        result = await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        assert result["segments"][0]["mask_id"] == 0
        assert result["segments"][0]["bbox_id"] == 0
        assert "mask_bytes" not in result["segments"][0]
        assert result["image_size"] == (200, 200)
        assert "timestamp" in result and result["timestamp"]

    async def test_all_returned_segments_share_same_offset_id(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[make_segment(mask_id=3)])
        segments = [make_segment(mask_id=10), make_segment(mask_id=11)]
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": segments, "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        ids = {seg["mask_id"] for seg in result["segments"]}
        assert ids == {4}

    async def test_appends_to_existing_cached_segments(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        existing = [make_segment(mask_id=0)]
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=existing)
        new_segment = make_segment(mask_id=999)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [new_segment], "metrics": {}, "image_size": (1, 1),
        })

        await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        _, kwargs = mock_redis_storage.cache_segments.call_args
        assert kwargs["image_id"] == 1
        assert kwargs["ttl"] == 7200
        assert len(kwargs["segments"]) == 2
        assert kwargs["segments"][0] is existing[0]

    async def test_offset_zero_when_nothing_cached(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [make_segment(mask_id=999)], "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        assert result["segments"][0]["mask_id"] == 0

    async def test_passes_polygon_params_to_pipeline(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })
        points = [(0, 0), (10, 0), (5, 10)]

        await service.segment_by_polygon(
            image_id=1, user_id=42, points=points,
            smooth=False, smoothing_factor=0.7, feather_px=4,
        )

        _, kwargs = mock_pipeline.sam_segment_by_polygon.call_args
        assert kwargs["points"] == points
        assert kwargs["smooth"] is False
        assert kwargs["smoothing_factor"] == 0.7
        assert kwargs["feather_px"] == 4

    async def test_image_not_found(self, service, mock_image_repo):
        mock_image_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            await service.segment_by_polygon(
                image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
            )

    async def test_unauthorized(self, service, mock_image_repo, sample_image):
        sample_image.user_id = 999
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)

        with pytest.raises(ValueError, match="Unauthorized"):
            await service.segment_by_polygon(
                image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
            )

    async def test_pipeline_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(side_effect=RuntimeError("polygon failed"))

        with pytest.raises(RuntimeError, match="polygon failed"):
            await service.segment_by_polygon(
                image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
            )

    async def test_empty_segments_result(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)
        mock_pipeline.sam_segment_by_polygon = AsyncMock(return_value={
            "segments": [], "metrics": {}, "image_size": (1, 1),
        })

        result = await service.segment_by_polygon(
            image_id=1, user_id=42, points=[(0, 0), (10, 0), (5, 10)],
        )

        assert result["segments"] == []


class TestSegmentHybrid:
    async def test_yolo_and_nonoverlapping_fallback_both_included(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_detection(0, 0, 10, 10)],
        })
        yolo_seg = make_segment(mask_id=100, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [yolo_seg],
        })
        fallback_seg = make_segment(mask_id=200, bbox={"x1": 50, "y1": 50, "x2": 60, "y2": 60})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [fallback_seg], "image_size": (300, 300),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        sources = {seg["source"] for seg in result["segments"]}
        assert sources == {"yolo", "sam_auto"}
        assert len(result["segments"]) == 2
        ids = sorted(seg["mask_id"] for seg in result["segments"])
        assert ids == [0, 1]
        assert result["image_size"] == (300, 300)
        assert "metrics" not in result

    async def test_overlapping_fallback_segment_is_dropped(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={
            "detections": [make_detection(0, 0, 10, 10)],
        })
        yolo_seg = make_segment(mask_id=1, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock(return_value={
            "segments": [yolo_seg],
        })
        duplicate_seg = make_segment(mask_id=2, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [duplicate_seg], "image_size": (100, 100),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42, overlap_iou_thresh=0.5)

        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "yolo"

    async def test_no_yolo_detections_skips_batch_sam_call(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_with_prompts_batch = AsyncMock()
        fallback_seg = make_segment(mask_id=1, bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [fallback_seg], "image_size": (50, 50),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        mock_pipeline.sam_segment_with_prompts_batch.assert_not_called()
        assert len(result["segments"]) == 1
        assert result["segments"][0]["source"] == "sam_auto"

    async def test_caches_final_segments(
        self, service, mock_image_repo, sample_image, mock_pipeline, mock_redis_storage,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [make_segment(mask_id=1)], "image_size": (10, 10),
        })

        await service.segment_hybrid(image_id=1, user_id=42)

        mock_redis_storage.cache_segments.assert_awaited_once()
        _, kwargs = mock_redis_storage.cache_segments.call_args
        assert kwargs["image_id"] == 1
        assert kwargs["ttl"] == 7200

    async def test_strips_mask_bytes_from_response(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [make_segment(mask_id=1)], "image_size": (10, 10),
        })

        result = await service.segment_hybrid(image_id=1, user_id=42)

        assert all("mask_bytes" not in seg for seg in result["segments"])

    async def test_passes_yolo_params_to_detect_objects(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(return_value={
            "segments": [], "image_size": (1, 1),
        })

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

    async def test_detect_objects_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(side_effect=RuntimeError("yolo crashed"))

        with pytest.raises(RuntimeError, match="yolo crashed"):
            await service.segment_hybrid(image_id=1, user_id=42)

    async def test_fallback_exception_propagates(
        self, service, mock_image_repo, sample_image, mock_pipeline,
    ):
        mock_image_repo.get_by_id = AsyncMock(return_value=sample_image)
        mock_pipeline.detect_objects = AsyncMock(return_value={"detections": []})
        mock_pipeline.sam_segment_objects = AsyncMock(side_effect=RuntimeError("fallback crashed"))

        with pytest.raises(RuntimeError, match="fallback crashed"):
            await service.segment_hybrid(image_id=1, user_id=42)


class TestNextMaskOffset:
    async def test_returns_zero_when_no_segments_cached(self, service, mock_redis_storage):
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=None)

        offset = await service._next_mask_offset(image_id=1)

        assert offset == 0

    async def test_returns_zero_when_empty_list_cached(self, service, mock_redis_storage):
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[])

        offset = await service._next_mask_offset(image_id=1)

        assert offset == 0

    async def test_returns_max_plus_one(self, service, mock_redis_storage):
        mock_redis_storage.get_cached_segments = AsyncMock(return_value=[
            make_segment(mask_id=0), make_segment(mask_id=5), make_segment(mask_id=2),
        ])

        offset = await service._next_mask_offset(image_id=1)

        assert offset == 6


class TestIouAndOverlapsAny:
    def test_iou_identical_boxes_is_one(self, service):
        box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

        assert SegmentationService._iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap_is_zero(self, service):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 100, "y1": 100, "x2": 110, "y2": 110}

        assert SegmentationService._iou(a, b) == 0.0

    def test_iou_partial_overlap(self, service):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}
        expected = 25 / 175

        assert SegmentationService._iou(a, b) == pytest.approx(expected)

    def test_iou_degenerate_box_zero_union_returns_zero(self, service):
        a = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
        b = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}

        assert SegmentationService._iou(a, b) == 0.0

    def test_overlaps_any_true_when_above_threshold(self, service):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        existing = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

        assert SegmentationService._overlaps_any(bbox, existing, 0.5) is True

    def test_overlaps_any_false_when_below_threshold(self, service):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        existing = [{"x1": 100, "y1": 100, "x2": 110, "y2": 110}]

        assert SegmentationService._overlaps_any(bbox, existing, 0.5) is False

    def test_overlaps_any_boundary_is_exclusive(self, service):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        other = {"x1": 0, "y1": 0, "x2": 20, "y2": 5}
        iou = SegmentationService._iou(bbox, other)

        assert SegmentationService._overlaps_any(bbox, [other], iou) is False

    def test_overlaps_any_empty_existing_list_is_false(self, service):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}

        assert SegmentationService._overlaps_any(bbox, [], 0.5) is False

    def test_overlaps_any_checks_all_boxes(self, service):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        existing = [
            {"x1": 100, "y1": 100, "x2": 110, "y2": 110},
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        ]

        assert SegmentationService._overlaps_any(bbox, existing, 0.5) is True


class TestMaskToDataUrl:
    def test_returns_correct_data_url_prefix(self):
        result = _mask_to_data_url(b"fake-png-bytes")

        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_matches_input_bytes(self):
        import base64
        raw = b"\x89PNG\r\n\x1a\nfake-mask-data"

        result = _mask_to_data_url(raw)

        b64_payload = result.split(",", 1)[1]
        assert base64.b64decode(b64_payload) == raw

    def test_empty_bytes_returns_empty_payload_url(self):
        result = _mask_to_data_url(b"")

        assert result == "data:image/png;base64,"

    def test_different_inputs_produce_different_urls(self):
        url1 = _mask_to_data_url(b"mask-one")
        url2 = _mask_to_data_url(b"mask-two")

        assert url1 != url2


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
        segments = [
            {"mask_id": 0, "mask_bytes": b""},
            {"mask_id": 1, "mask_bytes": None},
        ]

        result = _segments_for_response(segments)

        assert "mask_url" not in result[0]
        assert "mask_url" not in result[1]

    def test_preserves_other_fields(self):
        segments = [{
            "mask_id": 5,
            "bbox_id": 5,
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "area": 100,
            "stability_score": 0.87,
            "source": "yolo",
            "mask_bytes": b"raw-bytes",
        }]

        result = _segments_for_response(segments)

        assert result[0]["mask_id"] == 5
        assert result[0]["bbox"] == {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert result[0]["source"] == "yolo"

    def test_empty_list_returns_empty_list(self):
        assert _segments_for_response([]) == []

    def test_does_not_mutate_original_segments(self):
        original = {"mask_id": 0, "mask_bytes": b"raw-bytes"}
        segments = [original]

        _segments_for_response(segments)

        assert original["mask_bytes"] == b"raw-bytes"

    def test_preserves_segment_order(self):
        segments = [{"mask_id": i, "mask_bytes": b"x"} for i in range(5)]

        result = _segments_for_response(segments)

        assert [seg["mask_id"] for seg in result] == [0, 1, 2, 3, 4]