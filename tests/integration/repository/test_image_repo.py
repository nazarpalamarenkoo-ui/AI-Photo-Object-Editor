import pytest
from app.repository.image_repo import ImageRepository


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_create_image(db_session, sample_user):
    repo = ImageRepository(db_session)
    image = await repo.create("test.jpg", "s3://test.jpg", sample_user.id)
    assert image.id is not None
    assert image.filename == "test.jpg"


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
async def test_get_user_images(db_session, multiple_images, sample_user):
    repo = ImageRepository(db_session)
    images = await repo.get_user_images(sample_user.id)
    assert len(images) == 3


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