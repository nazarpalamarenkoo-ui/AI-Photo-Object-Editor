import pytest
from app.repository.image_repo import ImageRepository


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_image(db_session, sample_user):
    repo = ImageRepository(db_session)

    image = await repo.create(
        filename="test.jpg",
        storage_path="s3://test.jpg",
        user_id=sample_user.id,
        mime_type="image/jpeg",
        width=1920,
        height=1080,
        file_size=204800,
    )

    assert image.id is not None
    assert image.filename == "test.jpg"
    assert image.storage_path == "s3://test.jpg"
    assert image.user_id == sample_user.id
    assert image.mime_type == "image/jpeg"
    assert image.width == 1920
    assert image.height == 1080
    assert image.file_size == 204800
    assert image.cache_key is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_image_with_cache_key(db_session, sample_user):
    repo = ImageRepository(db_session)

    image = await repo.create(
        filename="cached.jpg",
        storage_path="s3://cached.jpg",
        user_id=sample_user.id,
        mime_type="image/jpeg",
        width=800,
        height=600,
        file_size=102400,
        cache_key="cache_abc123",
    )

    assert image.cache_key == "cache_abc123"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_by_id(db_session, sample_image):
    repo = ImageRepository(db_session)
    image = await repo.get_by_id(sample_image.id)
    assert image is not None
    assert image.id == sample_image.id


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_image_by_id_not_found(db_session):
    repo = ImageRepository(db_session)
    image = await repo.get_by_id(999999)
    assert image is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_user_images(db_session, multiple_images, sample_user):
    repo = ImageRepository(db_session)
    images = await repo.get_user_images(sample_user.id)
    assert len(images) == 3


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_user_images_ordered_desc(db_session, sample_user):
    """get_user_images must return results ordered by uploaded_at descending"""
    repo = ImageRepository(db_session)

    first = await repo.create(
        filename="first.jpg", storage_path="s3://first.jpg", user_id=sample_user.id,
        mime_type="image/jpeg", width=100, height=100, file_size=1000,
    )
    second = await repo.create(
        filename="second.jpg", storage_path="s3://second.jpg", user_id=sample_user.id,
        mime_type="image/jpeg", width=100, height=100, file_size=1000,
    )

    images = await repo.get_user_images(sample_user.id)

    ids = [img.id for img in images]
    assert ids.index(second.id) < ids.index(first.id)


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_get_user_images_empty_for_unknown_user(db_session):
    repo = ImageRepository(db_session)
    images = await repo.get_user_images(999999)
    assert images == []


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_update_image(db_session, sample_user):
    """`update` must pick up changes made on a detached instance via merge()"""
    repo = ImageRepository(db_session)

    image = await repo.create(
        filename="original.jpg",
        storage_path="s3://original.jpg",
        user_id=sample_user.id,
        mime_type="image/jpeg",
        width=800,
        height=600,
        file_size=51200,
    )

    image.filename = "renamed.jpg"
    updated = await repo.update(image)

    assert updated.id == image.id
    assert updated.filename == "renamed.jpg"

    # Persisted, not just returned in-memory
    refetched = await repo.get_by_id(image.id)
    assert refetched.filename == "renamed.jpg"


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_image(db_session, sample_image):
    repo = ImageRepository(db_session)
    image_id = sample_image.id
    result = await repo.delete(image_id)
    assert result is True
    image = await repo.get_by_id(image_id)
    assert image is None


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_delete_image_not_found(db_session):
    repo = ImageRepository(db_session)
    result = await repo.delete(999999)
    assert result is False