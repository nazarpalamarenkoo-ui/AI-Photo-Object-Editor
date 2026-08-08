import pytest
from app.db.models.detection import Detection
from app.repository.detection_repo import DetectionRepository
from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.services.detection_service import DetectionService

pytestmark = pytest.mark.integration


def _make_service(db_session) -> DetectionService:
    """DetectionService has no db/redis in its constructor — it's a pure
    reader/soft-deleter over already-persisted, content-scoped rows."""
    return DetectionService(
        detection_repo=DetectionRepository(db_session),
        image_repo=ImageRepository(db_session),
        image_version_repo=ImageVersionRepository(db_session),
    )


async def _add_detection(db_session, content_id: int, bbox_id: int, cls: str = "person", conf: float = 0.9):
    repo = DetectionRepository(db_session)
    created = await repo.create_many([Detection(
        content_id=content_id,
        bbox_id=bbox_id,
        detected_class=cls,
        confidence=conf,
        x1=10, y1=10, x2=100, y2=200,
        model_name="m", model_version="v", inference_time_ms=0.0,
    )])
    return created[0]


class TestGetDetections:
    @pytest.mark.asyncio
    async def test_returns_active_detections_for_current_version(
        self, db_session, sample_image_version, sample_image, sample_user,
    ):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=0)
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=1)
        service = _make_service(db_session)

        detections = await service.get_detections(sample_image.id, sample_user.id)

        assert len(detections) == 2

    @pytest.mark.asyncio
    async def test_scoped_to_explicit_version_id(
        self, db_session, sample_image_version, another_image_version, sample_image, sample_user,
    ):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=0)
        service = _make_service(db_session)

        # another_image_version belongs to a different image; asking for
        # sample_image's own version explicitly should still work
        detections = await service.get_detections(
            sample_image.id, sample_user.id, version_id=sample_image_version.id
        )

        assert len(detections) == 1

    @pytest.mark.asyncio
    async def test_version_mismatch_raises(
        self, db_session, sample_image_version, another_image_version, sample_image, sample_user,
    ):
        service = _make_service(db_session)

        with pytest.raises(ValueError, match="not found"):
            await service.get_detections(
                sample_image.id, sample_user.id, version_id=another_image_version.id
            )

    @pytest.mark.asyncio
    async def test_image_not_found(self, db_session, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.get_detections(99999, sample_user.id)

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, sample_image_version, sample_image, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.get_detections(sample_image.id, sample_user.id + 999)

    @pytest.mark.asyncio
    async def test_no_current_version_raises(self, db_session, sample_image, sample_user):
        # sample_image without sample_image_version has no current version
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="no current version"):
            await service.get_detections(sample_image.id, sample_user.id)


class TestGetDetectionByBboxId:
    @pytest.mark.asyncio
    async def test_success(self, db_session, sample_image_version, sample_image, sample_user):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=3)
        service = _make_service(db_session)

        det = await service.get_detection_by_bbox_id(sample_image.id, 3, sample_user.id)

        assert det.bbox_id == 3

    @pytest.mark.asyncio
    async def test_not_found(self, db_session, sample_image_version, sample_image, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.get_detection_by_bbox_id(sample_image.id, 99, sample_user.id)


class TestSoftDeleteDetection:
    @pytest.mark.asyncio
    async def test_removes_single_detection(self, db_session, sample_image_version, sample_image, sample_user):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=0)
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=1)
        service = _make_service(db_session)

        await service.soft_delete_detection(sample_image.id, 0, sample_user.id)

        remaining = await service.get_detections(sample_image.id, sample_user.id)
        assert {d.bbox_id for d in remaining} == {1}

    @pytest.mark.asyncio
    async def test_not_found(self, db_session, sample_image_version, sample_image, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.soft_delete_detection(sample_image.id, 99, sample_user.id)


class TestDeleteVersionDetections:
    @pytest.mark.asyncio
    async def test_hard_deletes_all_for_content(self, db_session, sample_image_version, sample_image, sample_user):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=0)
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=1)
        service = _make_service(db_session)

        count = await service.delete_version_detections(sample_image.id, sample_user.id)

        assert count == 2
        remaining = await service.get_detections(sample_image.id, sample_user.id, active_only=False)
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_unauthorized(self, db_session, sample_image_version, sample_image, sample_user):
        service = _make_service(db_session)
        with pytest.raises(ValueError, match="Unauthorized"):
            await service.delete_version_detections(sample_image.id, sample_user.id + 999)


class TestGetDetectionStats:
    @pytest.mark.asyncio
    async def test_computes_stats(self, db_session, sample_image_version, sample_image, sample_user):
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=0, cls="person", conf=0.9)
        await _add_detection(db_session, sample_image_version.content_id, bbox_id=1, cls="car", conf=0.7)
        service = _make_service(db_session)

        stats = await service.get_detection_stats(sample_image.id, sample_user.id)

        assert stats["total_detections"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.8)
        assert stats["min_confidence"] == pytest.approx(0.7)
        assert stats["max_confidence"] == pytest.approx(0.9)
        assert set(stats["classes"]) == {"person", "car"}

    @pytest.mark.asyncio
    async def test_empty_stats(self, db_session, sample_image_version, sample_image, sample_user):
        service = _make_service(db_session)
        stats = await service.get_detection_stats(sample_image.id, sample_user.id)
        assert stats["total_detections"] == 0
        assert stats["avg_confidence"] == 0.0