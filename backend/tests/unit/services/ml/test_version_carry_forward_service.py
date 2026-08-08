import pytest
from unittest.mock import AsyncMock

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base

from app.services.ml.version_carry_forward import (
    VersionCarryForwardMixin,
    _clone_for_content,
    _iou,
    _overlaps_any,
)

pytestmark = pytest.mark.unit

Base = declarative_base()


class _FakeRow(Base):
    """Minimal stand-in mapped class shaped like Detection/SegmentationMask
    — enough columns for sa_inspect(obj).mapper.columns to work."""

    __tablename__ = "fake_carry_forward_rows"

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer)
    created_at = Column(String)
    bbox_id = Column(Integer)
    detected_class = Column(String)
    confidence = Column(Float)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def make_row(id=1, content_id=100, bbox_id=0, x1=0, y1=0, x2=10, y2=10,
             detected_class="car", confidence=0.9):
    return _FakeRow(
        id=id, content_id=content_id, created_at="2024-01-01", bbox_id=bbox_id,
        detected_class=detected_class, confidence=confidence,
        x1=x1, y1=y1, x2=x2, y2=y2,
    )


class _Host(VersionCarryForwardMixin):
    """Stand-in for EditingService/SegmentationService, which mix this in
    and provide detection_repo/segmentation_repo via their own __init__."""

    def __init__(self, detection_repo, segmentation_repo):
        self.detection_repo = detection_repo
        self.segmentation_repo = segmentation_repo


@pytest.fixture
def mock_detection_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.soft_delete = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_segmentation_repo():
    repo = AsyncMock()
    repo.get_by_content = AsyncMock(return_value=[])
    repo.soft_delete = AsyncMock(return_value=None)
    repo.create_many = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def host(mock_detection_repo, mock_segmentation_repo):
    return _Host(detection_repo=mock_detection_repo, segmentation_repo=mock_segmentation_repo)


class TestCloneForContent:

    def test_copies_columns_onto_new_content_id(self):
        row = make_row(id=1, content_id=100, bbox_id=3, detected_class="dog", confidence=0.7)

        clone = _clone_for_content(row, new_content_id=200)

        assert clone.content_id == 200
        assert clone.bbox_id == 3
        assert clone.detected_class == "dog"
        assert clone.confidence == 0.7
        assert clone.x1 == row.x1
        assert clone.y1 == row.y1
        assert clone.x2 == row.x2
        assert clone.y2 == row.y2

    def test_excludes_id_and_created_at(self):
        row = make_row(id=42)

        clone = _clone_for_content(row, new_content_id=200)

        assert clone.id is None
        assert clone.created_at is None

    def test_returns_new_instance_of_same_class(self):
        row = make_row()

        clone = _clone_for_content(row, new_content_id=200)

        assert isinstance(clone, _FakeRow)
        assert clone is not row

    def test_does_not_mutate_original_row(self):
        row = make_row(id=1, content_id=100)

        _clone_for_content(row, new_content_id=200)

        assert row.content_id == 100
        assert row.id == 1


class TestIou:

    def test_identical_boxes_is_one(self):
        box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 100, "y1": 100, "x2": 110, "y2": 110}
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        b = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}
        expected = 25 / 175
        assert _iou(a, b) == pytest.approx(expected)

    def test_degenerate_box_zero_union_returns_zero(self):
        a = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
        b = {"x1": 5, "y1": 5, "x2": 5, "y2": 5}
        assert _iou(a, b) == 0.0


class TestOverlapsAny:

    def test_true_when_above_threshold(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        others = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        assert _overlaps_any(bbox, others, 0.5) is True

    def test_false_when_below_threshold(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        others = [{"x1": 100, "y1": 100, "x2": 110, "y2": 110}]
        assert _overlaps_any(bbox, others, 0.5) is False

    def test_false_when_others_empty(self):
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        assert _overlaps_any(bbox, [], 0.5) is False

    def test_boundary_not_strictly_greater_is_excluded(self):
        # _overlaps_any uses a strict > comparison, so an IoU exactly at
        # threshold does not count as an overlap.
        bbox = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        others = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        assert _overlaps_any(bbox, others, 1.0) is False


class TestCarryForwardDetectionsByOverlap:

    async def test_clones_non_overlapping_and_soft_deletes_overlapping(
        self, host, mock_detection_repo,
    ):
        hit = make_row(id=1, content_id=100, bbox_id=1, x1=0, y1=0, x2=10, y2=10)
        safe = make_row(id=2, content_id=100, bbox_id=2, x1=100, y1=100, x2=110, y2=110)
        mock_detection_repo.get_by_content = AsyncMock(return_value=[hit, safe])
        affected_boxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

        await host._carry_forward_detections_by_overlap(100, 200, affected_boxes)

        mock_detection_repo.get_by_content.assert_awaited_once_with(100, active_only=True)
        mock_detection_repo.soft_delete.assert_awaited_once_with(1)
        carried = mock_detection_repo.create_many.call_args.args[0]
        assert len(carried) == 1
        assert carried[0].bbox_id == 2
        assert carried[0].content_id == 200

    async def test_no_affected_boxes_carries_everything(self, host, mock_detection_repo):
        rows = [make_row(id=1, bbox_id=1), make_row(id=2, bbox_id=2)]
        mock_detection_repo.get_by_content = AsyncMock(return_value=rows)

        await host._carry_forward_detections_by_overlap(100, 200, [])

        mock_detection_repo.soft_delete.assert_not_awaited()
        carried = mock_detection_repo.create_many.call_args.args[0]
        assert len(carried) == 2

    async def test_no_rows_skips_create_many(self, host, mock_detection_repo):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[])

        await host._carry_forward_detections_by_overlap(
            100, 200, [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        )

        mock_detection_repo.create_many.assert_not_awaited()

    async def test_uses_detection_repo_not_segmentation_repo(
        self, host, mock_detection_repo, mock_segmentation_repo,
    ):
        await host._carry_forward_detections_by_overlap(100, 200, [])

        mock_detection_repo.get_by_content.assert_awaited_once()
        mock_segmentation_repo.get_by_content.assert_not_called()


class TestCarryForwardMasks:

    async def test_clones_non_overlapping_and_soft_deletes_overlapping(
        self, host, mock_segmentation_repo,
    ):
        hit = make_row(id=1, content_id=100, bbox_id=1, x1=0, y1=0, x2=10, y2=10)
        safe = make_row(id=2, content_id=100, bbox_id=2, x1=100, y1=100, x2=110, y2=110)
        mock_segmentation_repo.get_by_content = AsyncMock(return_value=[hit, safe])
        affected_boxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

        await host._carry_forward_masks(100, 200, affected_boxes)

        mock_segmentation_repo.get_by_content.assert_awaited_once_with(100, active_only=True)
        mock_segmentation_repo.soft_delete.assert_awaited_once_with(1)
        carried = mock_segmentation_repo.create_many.call_args.args[0]
        assert len(carried) == 1
        assert carried[0].bbox_id == 2

    async def test_uses_segmentation_repo_not_detection_repo(
        self, host, mock_detection_repo, mock_segmentation_repo,
    ):
        await host._carry_forward_masks(100, 200, [])

        mock_segmentation_repo.get_by_content.assert_awaited_once()
        mock_detection_repo.get_by_content.assert_not_called()


class TestCarryForwardDetections:
    """Excludes by known bbox_id set rather than by geometric overlap, and
    additionally reports back the boxes of what got excluded."""

    async def test_excludes_by_bbox_id_and_returns_removed_boxes(
        self, host, mock_detection_repo,
    ):
        det1 = make_row(id=1, bbox_id=1, x1=0, y1=0, x2=10, y2=10)
        det2 = make_row(id=2, bbox_id=2, x1=20, y1=20, x2=30, y2=30)
        mock_detection_repo.get_by_content = AsyncMock(return_value=[det1, det2])

        removed = await host._carry_forward_detections(100, 200, frozenset({1}))

        mock_detection_repo.soft_delete.assert_awaited_once_with(1)
        carried = mock_detection_repo.create_many.call_args.args[0]
        assert len(carried) == 1
        assert carried[0].bbox_id == 2
        assert carried[0].content_id == 200
        assert removed == [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]

    async def test_excludes_multiple_bbox_ids(self, host, mock_detection_repo):
        dets = [make_row(id=1, bbox_id=1), make_row(id=2, bbox_id=2), make_row(id=3, bbox_id=3)]
        mock_detection_repo.get_by_content = AsyncMock(return_value=dets)

        removed = await host._carry_forward_detections(100, 200, frozenset({1, 2}))

        assert mock_detection_repo.soft_delete.await_count == 2
        carried = mock_detection_repo.create_many.call_args.args[0]
        assert len(carried) == 1
        assert carried[0].bbox_id == 3
        assert len(removed) == 2

    async def test_no_matching_bbox_ids_carries_all_and_returns_empty(
        self, host, mock_detection_repo,
    ):
        dets = [make_row(id=1, bbox_id=1), make_row(id=2, bbox_id=2)]
        mock_detection_repo.get_by_content = AsyncMock(return_value=dets)

        removed = await host._carry_forward_detections(100, 200, frozenset({99}))

        assert removed == []
        mock_detection_repo.soft_delete.assert_not_awaited()
        carried = mock_detection_repo.create_many.call_args.args[0]
        assert len(carried) == 2

    async def test_empty_detections_returns_empty_list_and_skips_create_many(
        self, host, mock_detection_repo,
    ):
        mock_detection_repo.get_by_content = AsyncMock(return_value=[])

        removed = await host._carry_forward_detections(100, 200, frozenset({1}))

        assert removed == []
        mock_detection_repo.create_many.assert_not_awaited()