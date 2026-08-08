import pytest
from app.repository.image_content_repo import ImageContentRepository
from app.db.models.image_content import ImageContent


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_image_content(db_session):
    """Test creating a new ImageContent record"""
    repo = ImageContentRepository(db_session)

    content = await repo.create(
        content_hash="abc123hash",
        storage_path="s3://bucket/images/abc123.jpg",
        width=1920,
        height=1080,
        file_size=204800,
    )

    assert content.id is not None
    assert content.content_hash == "abc123hash"
    assert content.storage_path == "s3://bucket/images/abc123.jpg"
    assert content.width == 1920
    assert content.height == 1080
    assert content.file_size == 204800


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_content_by_id(db_session, sample_image_content):
    """Test fetching ImageContent by primary key"""
    repo = ImageContentRepository(db_session)

    content = await repo.get_by_id(sample_image_content.id)

    assert content is not None
    assert content.id == sample_image_content.id
    assert content.content_hash == sample_image_content.content_hash


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_content_by_id_not_found(db_session):
    """Test fetching non-existent ImageContent returns None"""
    repo = ImageContentRepository(db_session)

    content = await repo.get_by_id(999999)

    assert content is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_content_by_hash(db_session, sample_image_content):
    """Test fetching ImageContent by content hash"""
    repo = ImageContentRepository(db_session)

    content = await repo.get_by_hash(sample_image_content.content_hash)

    assert content is not None
    assert content.content_hash == sample_image_content.content_hash
    assert content.id == sample_image_content.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_content_by_hash_not_found(db_session):
    """Test fetching ImageContent by non-existent hash returns None"""
    repo = ImageContentRepository(db_session)

    content = await repo.get_by_hash("nonexistent_hash_xyz")

    assert content is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_or_create_creates_new(db_session):
    """Test get_or_create inserts when hash is absent"""
    repo = ImageContentRepository(db_session)

    content, created = await repo.get_or_create(
        content_hash="unique_hash_001",
        storage_path="s3://bucket/unique_001.jpg",
        width=800,
        height=600,
        file_size=102400,
    )

    assert created is True
    assert content.id is not None
    assert content.content_hash == "unique_hash_001"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_or_create_returns_existing(db_session, sample_image_content):
    """Test get_or_create returns existing record without inserting a duplicate"""
    repo = ImageContentRepository(db_session)

    content, created = await repo.get_or_create(
        content_hash=sample_image_content.content_hash,
        storage_path="s3://bucket/doesnt_matter.jpg",
        width=999,
        height=999,
        file_size=999,
    )

    assert created is False
    assert content.id == sample_image_content.id
    # Original values must be preserved
    assert content.width == sample_image_content.width
    assert content.height == sample_image_content.height


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_or_create_idempotent_second_call(db_session):
    """Two consecutive get_or_create calls with the same hash must both succeed"""
    repo = ImageContentRepository(db_session)

    first, created_first = await repo.get_or_create(
        content_hash="idempotent_hash_002",
        storage_path="s3://bucket/idempotent_002.jpg",
        width=640,
        height=480,
        file_size=51200,
    )
    second, created_second = await repo.get_or_create(
        content_hash="idempotent_hash_002",
        storage_path="s3://bucket/idempotent_002.jpg",
        width=640,
        height=480,
        file_size=51200,
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id