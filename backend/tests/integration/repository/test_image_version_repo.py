import pytest
from app.repository.image_version_repo import ImageVersionRepository


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_original_version(db_session, sample_image, sample_image_content):
    """Test creating version_number=0 original version for a fresh image"""
    repo = ImageVersionRepository(db_session)

    version = await repo.create_original(sample_image, sample_image_content.id)

    assert version.id is not None
    assert version.image_id == sample_image.id
    assert version.version_number == 0
    assert version.parent_version_id is None
    assert version.content_id == sample_image_content.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_original_sets_current_version(db_session, sample_image, sample_image_content):
    """create_original must also update image.current_version_id"""
    repo = ImageVersionRepository(db_session)

    version = await repo.create_original(sample_image, sample_image_content.id)

    assert sample_image.current_version_id == version.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_next_version(db_session, sample_image, sample_image_content, another_image_content):
    """Test forking a next version from the current one"""
    repo = ImageVersionRepository(db_session)

    original = await repo.create_original(sample_image, sample_image_content.id)
    next_ver = await repo.create_next(
        sample_image,
        storage_path="s3://bucket/edited_v1.jpg",
        content_id=another_image_content.id,
    )

    assert next_ver.id is not None
    assert next_ver.version_number == 1
    assert next_ver.parent_version_id == original.id
    assert next_ver.content_id == another_image_content.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_next_increments_version_number(
    db_session, sample_image, sample_image_content, another_image_content, third_image_content
):
    """Each successive create_next must increment the global max, not just parent+1"""
    repo = ImageVersionRepository(db_session)

    await repo.create_original(sample_image, sample_image_content.id)
    v1 = await repo.create_next(sample_image, "s3://bucket/v1.jpg", another_image_content.id)
    v2 = await repo.create_next(sample_image, "s3://bucket/v2.jpg", third_image_content.id)

    assert v1.version_number == 1
    assert v2.version_number == 2


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_current_version(db_session, sample_image, sample_image_content, another_image_content):
    """get_current must return the version pointed to by image.current_version_id"""
    repo = ImageVersionRepository(db_session)

    await repo.create_original(sample_image, sample_image_content.id)
    next_ver = await repo.create_next(sample_image, "s3://bucket/v1.jpg", another_image_content.id)

    current = await repo.get_current(sample_image)

    assert current is not None
    assert current.id == next_ver.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id(db_session, sample_image, sample_image_content):
    """Test fetching a specific version by its primary key"""
    repo = ImageVersionRepository(db_session)

    original = await repo.create_original(sample_image, sample_image_content.id)
    fetched = await repo.get_by_id(original.id)

    assert fetched is not None
    assert fetched.id == original.id
    assert fetched.version_number == 0


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_by_id_not_found(db_session):
    """Test fetching non-existent version returns None"""
    repo = ImageVersionRepository(db_session)

    version = await repo.get_by_id(999999)

    assert version is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_list_by_image(db_session, sample_image, sample_image_content, another_image_content):
    """list_by_image must return all versions in ascending order"""
    repo = ImageVersionRepository(db_session)

    await repo.create_original(sample_image, sample_image_content.id)
    await repo.create_next(sample_image, "s3://bucket/v1.jpg", another_image_content.id)

    versions = await repo.list_by_image(sample_image.id)

    assert len(versions) == 2
    assert versions[0].version_number < versions[1].version_number


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_set_current_moves_pointer(db_session, sample_image, sample_image_content, another_image_content):
    """set_current must move current_version_id back (undo simulation)"""
    repo = ImageVersionRepository(db_session)

    original = await repo.create_original(sample_image, sample_image_content.id)
    await repo.create_next(sample_image, "s3://bucket/v1.jpg", another_image_content.id)

    # Undo — go back to the original
    reverted = await repo.set_current(sample_image, original.id)

    assert reverted.id == original.id
    assert sample_image.current_version_id == original.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_set_current_wrong_image_raises(db_session, sample_image, another_image, sample_image_content):
    """set_current must raise ValueError when version belongs to another image"""
    repo = ImageVersionRepository(db_session)

    # Create a version that belongs to another_image
    other_version = await repo.create_original(another_image, sample_image_content.id)

    with pytest.raises(ValueError):
        await repo.set_current(sample_image, other_version.id)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_next_without_original_raises(db_session, sample_image, sample_image_content):
    """create_next must raise ValueError when image has no current version yet"""
    repo = ImageVersionRepository(db_session)

    with pytest.raises(ValueError):
        await repo.create_next(sample_image, "s3://bucket/v1.jpg", sample_image_content.id)