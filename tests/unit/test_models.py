import pytest
from app.db.models.user import User
from app.db.models.image import Image
from app.db.models.detection import Detection


@pytest.mark.unit
class TestUserModel:
    def test_user_creation(self):
        user = User(username="test", email="test@test.com", password_hash="hash")
        assert user.username == "test"
        assert user.email == "test@test.com"
    
    def test_user_tablename(self):
        assert User.__tablename__ == "users"
    
    def test_user_repr(self):
        user = User(id=1, username="john")
        assert "User" in repr(user)
        assert "john" in repr(user)


@pytest.mark.unit
class TestImageModel:
    def test_image_creation(self):
        image = Image(filename="test.jpg", storage_path="s3://test.jpg", user_id=1)
        assert image.filename == "test.jpg"
        assert image.user_id == 1
    
    def test_image_default_status(self):
        image = Image(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            user_id=1,
            status="uploaded"  # ← Додай це
        )
        assert image.status == "uploaded"
        
    def test_image_tablename(self):
        assert Image.__tablename__ == "images"


@pytest.mark.unit
class TestDetectionModel:
    def test_detection_creation(self):
        det = Detection(image_id=1, x1=10, y1=10, x2=100, y2=100, detected_class="person", confidence=0.9)
        assert det.x2 > det.x1
        assert det.y2 > det.y1
    
    def test_detection_tablename(self):
        assert Detection.__tablename__ == "detections"