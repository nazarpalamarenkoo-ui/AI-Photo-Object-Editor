import pytest
from unittest.mock import AsyncMock, MagicMock
from app.repository.image_repo import ImageRepository
from app.db.models.image import Image


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_image():
    db = AsyncMock()
    repo = ImageRepository(db)

    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    image = await repo.create("file.jpg", "path", 1)

    assert image.filename == "file.jpg"
    db.add.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_id():
    db = AsyncMock()
    repo = ImageRepository(db)

    mock_image = Image(id=1, filename="f", storage_path="p", user_id=1)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = mock_image
    db.execute = AsyncMock(return_value=result_mock)

    image = await repo.get_by_id(1)

    assert image == mock_image


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_found():
    db = AsyncMock()
    repo = ImageRepository(db)

    image = Image(id=1, filename="f", storage_path="p", user_id=1)

    repo.get_by_id = AsyncMock(return_value=image)
    db.delete = AsyncMock()
    db.commit = AsyncMock()

    result = await repo.delete(1)

    assert result is True
    db.delete.assert_called_once_with(image)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_image_not_found():
    db = AsyncMock()
    repo = ImageRepository(db)

    repo.get_by_id = AsyncMock(return_value=None)

    result = await repo.delete(1)

    assert result is False