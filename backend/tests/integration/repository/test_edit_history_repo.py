import pytest
from app.repository.edit_history_repo import ImageEditHistoryRepository
from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_edit_history_entry(db_session, sample_image_version):
    """Test creating a basic edit history entry (DETECT via YOLO)"""
    repo = ImageEditHistoryRepository(db_session)

    entry = await repo.create(
        image_version_id=sample_image_version.id,
        operation=EditOperation.DETECT,
        engine=EngineType.YOLO,
        parameters={"confidence_threshold": 0.5},
        processing_time_ms=42,
    )

    assert entry.id is not None
    assert entry.image_version_id == sample_image_version.id
    assert entry.operation == EditOperation.DETECT
    assert entry.engine == EngineType.YOLO
    assert entry.parameters == {"confidence_threshold": 0.5}
    assert entry.processing_time_ms == 42


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_edit_history_without_optional_fields(db_session, sample_image_version):
    """Test creating an entry with no parameters and no processing time"""
    repo = ImageEditHistoryRepository(db_session)

    entry = await repo.create(
        image_version_id=sample_image_version.id,
        operation=EditOperation.SEGMENT,
        engine=EngineType.SAM,
    )

    assert entry.id is not None
    assert entry.operation == EditOperation.SEGMENT
    assert entry.engine == EngineType.SAM
    assert entry.parameters is None
    assert entry.processing_time_ms is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_edit_history_by_id(db_session, sample_edit_history_entry):
    """Test fetching a history entry by its primary key"""
    repo = ImageEditHistoryRepository(db_session)

    entry = await repo.get_by_id(sample_edit_history_entry.id)

    assert entry is not None
    assert entry.id == sample_edit_history_entry.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_edit_history_by_id_not_found(db_session):
    """Test fetching a non-existent entry returns None"""
    repo = ImageEditHistoryRepository(db_session)

    entry = await repo.get_by_id(999999)

    assert entry is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_edit_history_by_version(db_session, sample_image_version):
    """Test fetching all entries for a version, ordered ascending by created_at"""
    repo = ImageEditHistoryRepository(db_session)

    e1 = await repo.create(
        image_version_id=sample_image_version.id,
        operation=EditOperation.REMOVE,
        engine=EngineType.LAMA,
        parameters={"mask_id": 3},
        processing_time_ms=10,
    )
    e2 = await repo.create(
        image_version_id=sample_image_version.id,
        operation=EditOperation.REPLACE,
        engine=EngineType.DIFFUSION,
        parameters={"prompt": "sunset background"},
        processing_time_ms=20,
    )

    entries = await repo.get_by_version(sample_image_version.id)

    assert len(entries) >= 2
    ids = [e.id for e in entries]
    assert e1.id in ids
    assert e2.id in ids
    # Must be in ascending order (created_at asc)
    assert entries[0].created_at <= entries[-1].created_at
    # And specifically e1 (created first) must come before e2
    assert ids.index(e1.id) < ids.index(e2.id)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_edit_history_by_version_empty(db_session, sample_image_version):
    """Test fetching entries for a version that has none returns empty list"""
    repo = ImageEditHistoryRepository(db_session)

    entries = await repo.get_by_version(sample_image_version.id)

    assert entries == []


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_edit_history_isolated_by_version(db_session, sample_image_version, another_image_version):
    """Entries for one version must not appear when querying another"""
    repo = ImageEditHistoryRepository(db_session)

    await repo.create(
        image_version_id=sample_image_version.id,
        operation=EditOperation.DETECT,
        engine=EngineType.YOLO,
    )

    entries_other = await repo.get_by_version(another_image_version.id)

    assert all(e.image_version_id == another_image_version.id for e in entries_other)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,engine",
    [
        (EditOperation.DETECT, EngineType.YOLO),
        (EditOperation.SEGMENT, EngineType.SAM),
        (EditOperation.REMOVE, EngineType.LAMA),
        (EditOperation.REPLACE, EngineType.DIFFUSION),
    ],
)
async def test_create_edit_history_all_operation_engine_pairs(
    db_session, sample_image_version, operation, engine
):
    """Every real (operation, engine) pairing must round-trip through create/get_by_id"""
    repo = ImageEditHistoryRepository(db_session)

    entry = await repo.create(
        image_version_id=sample_image_version.id,
        operation=operation,
        engine=engine,
    )

    fetched = await repo.get_by_id(entry.id)

    assert fetched is not None
    assert fetched.operation == operation
    assert fetched.engine == engine