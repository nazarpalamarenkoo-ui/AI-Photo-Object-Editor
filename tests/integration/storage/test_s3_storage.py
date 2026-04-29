import pytest
import boto3
from moto import mock_aws

from app.storage.s3_storage import S3Storage
from app.config.settings import settings


@pytest.fixture
def aws_mock():
    with mock_aws():
        yield


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


@pytest.fixture
def s3_client(aws_mock, aws_credentials):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    return s3


@pytest.fixture
def mock_s3_settings(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(settings, "R2_ENDPOINT", None)
    monkeypatch.setattr(settings, "ACCESS_KEY", "testing")
    monkeypatch.setattr(settings, "SECRET_KEY", "testing")
    monkeypatch.setattr(settings, "R2_PUBLIC_URL", "http://localhost")


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_s3_upload_delete_exists_flow(s3_client, mock_s3_settings, monkeypatch):
    storage = S3Storage()
    path = "test.jpg"

    # upload
    s3_client.put_object(
        Bucket="test-bucket",
        Key=path,
        Body=b"hello"
    )

    class MockClient:
        async def head_object(self, Bucket, Key):
            return s3_client.head_object(Bucket=Bucket, Key=Key)

        async def delete_object(self, Bucket, Key):
            return s3_client.delete_object(Bucket=Bucket, Key=Key)

    class MockCtx:
        async def __aenter__(self):
            return MockClient()

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        "aioboto3.Session.client",
        lambda *args, **kwargs: MockCtx()
    )

    # exists
    assert await storage.exists(path) is True

    # delete
    await storage.delete(path)

    # exists after delete
    assert await storage.exists(path) is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s3_download_mock(monkeypatch, mock_s3_settings):
    storage = S3Storage()

    class MockBody:
        async def read(self):
            return b"hello"

    class MockClient:
        async def get_object(self, *args, **kwargs):
            return {"Body": MockBody()}

    class MockCtx:
        async def __aenter__(self):
            return MockClient()

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        "aioboto3.Session.client",
        lambda *args, **kwargs: MockCtx()
    )

    data = await storage.download("test.jpg")
    assert data == b"hello"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_s3_exists_false(s3_client, mock_s3_settings, monkeypatch):
    storage = S3Storage()

    class MockClient:
        async def head_object(self, Bucket, Key):
            raise Exception()  # emulate not found

    class MockCtx:
        async def __aenter__(self):
            return MockClient()

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        "aioboto3.Session.client",
        lambda *args, **kwargs: MockCtx()
    )

    result = await storage.exists("not_exists.jpg")
    assert result is False