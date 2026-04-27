import pytest
from pydantic import ValidationError
from app.db.schemas.user import UserCreate
from app.db.schemas.image import ImageCreate
from app.db.schemas.detection import DetectionCreate
from app.db.schemas.ml import (
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    MLResultResponse
)

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
        img = ImageCreate(filename="test.jpg", storage_path="s3://test.jpg", user_id=1)
        assert img.filename == "test.jpg"
        


@pytest.mark.unit
class TestDetectionSchemas:
    def test_detection_create_valid(self):
        det = DetectionCreate(image_id=1, x1=10, y1=10, x2=100, y2=100, detected_class="person", confidence=0.9)
        assert det.confidence == 0.9
        
        
@pytest.mark.unit
class TestMLSchemas:

    def test_detect_request_valid(self):
        req = DetectRequest(conf_threshold=0.7, classes=["person"])
        assert req.conf_threshold == 0.7
        assert req.classes == ["person"]

    def test_detect_request_invalid_threshold(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=1.5)

    def test_remove_request_defaults(self):
        req = RemoveRequest(expand_mask_pixels = 5)
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is True

    def test_remove_multiple_valid(self):
        req = RemoveMultipleRequest(expand_mask_pixels = 0, bbox_ids=[1, 2, 3])
        assert len(req.bbox_ids) == 3

    def test_remove_multiple_invalid_empty(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest(expand_mask_pixels = 0,bbox_ids=[])

    def test_replace_request_valid(self):
        req = ReplaceRequest(expand_mask_pixels = 0, color_match_method="histogram")
        assert req.color_match_method == "histogram"

    def test_replace_request_invalid_method(self):
        with pytest.raises(ValidationError):
            ReplaceRequest(expand_mask_pixels = 0,color_match_method="invalid") # type: ignore

    def test_ml_result_response_valid(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"time": 0.5},
            timestamp="2025-01-01T00:00:00"
        )
        assert res.result_url is not None