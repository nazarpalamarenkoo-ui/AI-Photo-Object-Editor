import pytest
import boto3
from moto import mock_aws
from botocore.exceptions import ClientError
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
    monkeypatch.setattr(settings, "S3_ENDPOINT", None)


@pytest.mark.integration
@pytest.mark.storage
def test_mock_s3_basic_operations(s3_client):

    s3_client.put_object(
        Bucket="test-bucket",
        Key="test.txt",
        Body=b"test data"
    )

    response = s3_client.get_object(
        Bucket="test-bucket",
        Key="test.txt"
    )

    data = response["Body"].read()

    assert data == b"test data"

    s3_client.delete_object(
        Bucket="test-bucket",
        Key="test.txt"
    )

    with pytest.raises(ClientError):
        s3_client.head_object(
            Bucket="test-bucket",
            Key="test.txt"
        )