import pytest
from app.repository.segmentation_repo import SegmentationRepository
from app.db.models.segmentation import SegmentationMask
from app.db.enums.segmentation_mode import SegmentationMode


def _make_mask(content_id, mask_id, score=0.9):
    return SegmentationMask(
        content_id=content_id,
        mask_id=mask_id,
        mask_storage_path=f"s3://bucket/masks/mask_{mask_id}.png",
        preview_storage_path=f"s3://bucket/masks/preview_{mask_id}.png",
        x1=10, y1=10, x2=100, y2=100,
        area=8100.0,
        score=score,
        segmentation_mode=SegmentationMode.SAM,
        model_name="mobile_sam",
        model_version="v1",
        inference_time_ms=120.0,
    )


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_many_masks(db_session, sample_image_content):
    """Test bulk-inserting multiple SegmentationMask records"""
    repo = SegmentationRepository(db_session)

    masks = [
        _make_mask(sample_image_content.id, 0, score=0.92),
        _make_mask(sample_image_content.id, 1, score=0.85),
    ]

    created = await repo.create_many(masks)

    assert len(created) == 2
    assert created[0].mask_id == 0
    assert created[1].mask_id == 1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_content_active_only(db_session, sample_image_content):
    """get_by_content with active_only=True must exclude soft-deleted masks"""
    repo = SegmentationRepository(db_session)

    masks = [
        _make_mask(sample_image_content.id, 0, score=0.9),
        _make_mask(sample_image_content.id, 1, score=0.75),
    ]
    created = await repo.create_many(masks)

    # Soft-delete the first one
    await repo.soft_delete(created[0].id)

    active = await repo.get_by_content(sample_image_content.id, active_only=True)

    assert all(m.is_active for m in active)
    assert not any(m.id == created[0].id for m in active)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_content_all(db_session, sample_image_content):
    """get_by_content with active_only=False must include soft-deleted masks"""
    repo = SegmentationRepository(db_session)

    masks = [
        _make_mask(sample_image_content.id, 0, score=0.9),
    ]
    created = await repo.create_many(masks)
    await repo.soft_delete(created[0].id)

    all_masks = await repo.get_by_content(sample_image_content.id, active_only=False)

    assert any(m.id == created[0].id for m in all_masks)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id(db_session, sample_segmentation_mask):
    """Test fetching a SegmentationMask by primary key"""
    repo = SegmentationRepository(db_session)

    mask = await repo.get_by_id(sample_segmentation_mask.id)

    assert mask is not None
    assert mask.id == sample_segmentation_mask.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id_not_found(db_session):
    """Test fetching non-existent mask returns None"""
    repo = SegmentationRepository(db_session)

    mask = await repo.get_by_id(999999)

    assert mask is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_max_mask_id(db_session, sample_image_content):
    """max_mask_id must return the highest mask_id across all masks (including inactive)"""
    repo = SegmentationRepository(db_session)

    masks = [
        _make_mask(sample_image_content.id, 0, score=0.5),
        _make_mask(sample_image_content.id, 3, score=0.6),
    ]
    created = await repo.create_many(masks)
    # Soft-delete the one with mask_id=3 — it must still count toward max
    await repo.soft_delete(created[1].id)

    max_id = await repo.max_mask_id(sample_image_content.id)

    assert max_id == 3


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_max_mask_id_empty_returns_minus_one(db_session, sample_image_content):
    """max_mask_id must return -1 when there are no masks"""
    repo = SegmentationRepository(db_session)

    max_id = await repo.max_mask_id(sample_image_content.id)

    assert max_id == -1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_soft_delete(db_session, sample_image_content):
    """soft_delete must set is_active=False without removing the row"""
    repo = SegmentationRepository(db_session)

    created = await repo.create_many([
        _make_mask(sample_image_content.id, 0, score=0.9),
    ])
    mask_id = created[0].id

    deleted = await repo.soft_delete(mask_id)

    assert deleted is not None
    assert deleted.is_active is False

    # Row still exists
    still_there = await repo.get_by_id(mask_id)
    assert still_there is not None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_by_content(db_session, sample_image_content):
    """delete_by_content must hard-delete all masks and return the count"""
    repo = SegmentationRepository(db_session)

    await repo.create_many([
        _make_mask(sample_image_content.id, 0, score=0.9),
        _make_mask(sample_image_content.id, 1, score=0.8),
    ])

    count = await repo.delete_by_content(sample_image_content.id)

    assert count == 2
    remaining = await repo.get_by_content(sample_image_content.id, active_only=False)
    assert len(remaining) == 0