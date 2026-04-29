import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.storage.s3_storage import S3Storage


@pytest.fixture
def mock_s3_client():
    client = MagicMock()

    client.upload_fileobj = AsyncMock()

    mock_body = MagicMock()
    mock_body.read = AsyncMock(return_value=b"data")

    client.get_object = AsyncMock(return_value={"Body": mock_body})
    client.delete_object = AsyncMock()
    client.head_object = AsyncMock()

    return client


@pytest.fixture
def storage(mock_s3_client):
    mock_session = MagicMock()

    class MockClientContext:
        async def __aenter__(self):
            return mock_s3_client

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_session.client = MagicMock(return_value=MockClientContext())

    with patch("aioboto3.Session", return_value=mock_session):
        return S3Storage()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_bytes(storage, mock_s3_client):
    result = await storage.upload_bytes(b"data", "test.jpg")

    assert result.startswith("s3://")
    mock_s3_client.upload_fileobj.assert_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download(storage):
    result = await storage.download("test.jpg")

    assert result == b"data"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete(storage, mock_s3_client):
    result = await storage.delete("test.jpg")

    assert result is True
    mock_s3_client.delete_object.assert_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_true(storage):
    result = await storage.exists("test.jpg")

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_false(storage, mock_s3_client):
    mock_s3_client.head_object.side_effect = Exception()

    result = await storage.exists("test.jpg")

    assert result is False


@pytest.mark.unit
def test_parse_key():
    s = S3Storage()

    assert s._parse_key("s3://bucket/file.jpg") == "file.jpg"
    assert s._parse_key("file.jpg") == "file.jpg"