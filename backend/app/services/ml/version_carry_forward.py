from typing import Dict, List
from app.repository.detection_repo import DetectionRepository
from app.repository.segmentation_repo import SegmentationRepository
from sqlalchemy import inspect as sa_inspect


def _clone_for_content(obj, new_content_id: int):
    """Copy a Detection/SegmentationMask row onto a new content_id."""
    mapper = sa_inspect(obj).mapper
    data = {
        col.key: getattr(obj, col.key)
        for col in mapper.columns
        if col.key not in ("id", "created_at", "content_id")
    }
    data["content_id"] = new_content_id
    return obj.__class__(**data)


def _iou(a: Dict, b: Dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps_any(bbox: Dict, others: List[Dict], iou_thresh: float) -> bool:
    return any(_iou(bbox, o) > iou_thresh for o in others)


class VersionCarryForwardMixin:
    """
    Shared "clone active rows onto the new ImageVersion's content, except
    ones that no longer describe reality" logic.
    """
    detection_repo: DetectionRepository
    segmentation_repo: SegmentationRepository

    CARRY_FORWARD_IOU_THRESH = 0.3

    async def _carry_forward_by_overlap(
        self,
        repo,
        old_content_id: int,
        new_content_id: int,
        affected_boxes: List[Dict],
    ) -> None:
        """
        Clone active rows from old_content to new_content, except ones
        overlapping affected_boxes. Works for both DetectionRepository and
        SegmentationRepository — both expose get_by_content(active_only)/
        soft_delete(id)/create_many(rows) with the same shape, and both
        row types carry x1/y1/x2/y2.
        """
        rows = await repo.get_by_content(old_content_id, active_only=True)
        carried = []
        for row in rows:
            row_bbox = {"x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2}
            if affected_boxes and _overlaps_any(row_bbox, affected_boxes, self.CARRY_FORWARD_IOU_THRESH):
                await repo.soft_delete(row.id)
            else:
                carried.append(_clone_for_content(row, new_content_id))
        if carried:
            await repo.create_many(carried)

    async def _carry_forward_detections_by_overlap(
        self,
        old_content_id: int,
        new_content_id: int,
        affected_boxes: List[Dict],
    ) -> None:
        """
        Detection variant of _carry_forward_by_overlap — used where there's
        no bbox_id to exclude by (e.g. sam_replace_object_diffusion,
        sam_remove_object: the acted-on region comes from a mask/bbox,
        not a stored Detection row).
        """
        await self._carry_forward_by_overlap(
            self.detection_repo, old_content_id, new_content_id, affected_boxes
        )

    async def _carry_forward_masks(
        self,
        old_content_id: int,
        new_content_id: int,
        affected_boxes: List[Dict],
    ) -> None:
        """SegmentationMask variant of _carry_forward_by_overlap."""
        await self._carry_forward_by_overlap(
            self.segmentation_repo, old_content_id, new_content_id, affected_boxes
        )

    async def _carry_forward_detections(
        self,
        old_content_id: int,
        new_content_id: int,
        excluded_bbox_ids: frozenset,
    ) -> List[Dict]:
        """
        Detection-only variant that excludes by known bbox_id instead of
        overlap — used where the caller already knows exactly which
        bbox_id(s) were acted on.
        Returns the excluded detections' boxes,
        so the caller can carry-forward masks using the same affected
        region.
        """
        detections = await self.detection_repo.get_by_content(old_content_id, active_only=True)
        carried = []
        removed_boxes = []
        for det in detections:
            if det.bbox_id in excluded_bbox_ids:
                await self.detection_repo.soft_delete(det.id)
                removed_boxes.append({"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2})
            else:
                carried.append(_clone_for_content(det, new_content_id))
        if carried:
            await self.detection_repo.create_many(carried)
        return removed_boxes