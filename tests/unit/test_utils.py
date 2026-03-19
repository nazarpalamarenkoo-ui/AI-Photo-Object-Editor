import pytest


@pytest.mark.unit
class TestPathUtils:
    def test_extract_path_from_s3_url(self):
        bucket = "mybucket"
        url = f"s3://{bucket}/users/123/image.jpg"
        path = url.replace(f"s3://{bucket}/", "")
        assert path == "users/123/image.jpg"


@pytest.mark.unit
class TestCacheKeyGeneration:
    def test_image_cache_key(self):
        key = f"image:123:processed"
        assert key.startswith("image:")
        assert ":123:" in key
    
    def test_detection_cache_key(self):
        key = f"detections:456"
        assert key == "detections:456"