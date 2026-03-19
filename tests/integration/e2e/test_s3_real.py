import pytest
from app.storage.s3_storage import S3Storage


@pytest.mark.e2e
@pytest.mark.storage
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_s3_upload(mock_upload_file):
    storage = S3Storage()
    
    path = await storage.upload(mock_upload_file, "e2e-test/upload.jpg")
    
    assert path.startswith("s3://")
    
    # Cleanup
    await storage.delete("e2e-test/upload.jpg")


@pytest.mark.e2e
@pytest.mark.storage
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_s3_download():
    storage = S3Storage()
    
    # Upload first
    test_data = b"real e2e test data"
    await storage.upload_bytes(test_data, "e2e-test/download.jpg")
    
    # Download
    downloaded = await storage.download("e2e-test/download.jpg")
    
    assert downloaded == test_data
    
    # Cleanup
    await storage.delete("e2e-test/download.jpg")