import pytest
from pydantic import ValidationError
from app.db.schemas.user import UserCreate
from app.db.schemas.image import ImageCreate
from app.db.schemas.detection import DetectionCreate


@pytest.mark.unit
class TestUserSchemas:
    def test_user_create_valid(self):
        user = UserCreate(username="john", email="john@test.com", password="pass123")
        assert user.username == "john"
    
    def test_user_create_invalid_email(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="invalid", password="pass123")
    
    def test_user_create_short_username(self):
        with pytest.raises(ValidationError):
            UserCreate(username="ab", email="john@test.com", password="pass123")


@pytest.mark.unit
class TestImageSchemas:
    def test_image_create_valid(self):
        img = ImageCreate(filename="test.jpg", file_path="s3://test.jpg", user_id=1)
        assert img.filename == "test.jpg"


@pytest.mark.unit
class TestDetectionSchemas:
    def test_detection_create_valid(self):
        det = DetectionCreate(image_id=1, x1=10, y1=10, x2=100, y2=100, detected_class="person", confidence=0.9)
        assert det.confidence == 0.9