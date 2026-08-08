import pytest
from datetime import datetime
from pydantic import ValidationError

from app.db.schemas.common import BboxSchema

from app.db.schemas.assets import (
    AssetBase,
    AssetCreate,
    AssetUpdate,
    AssetResponse,
)

from app.db.schemas.detection import (
    DetectionBase,
    DetectionCreate,
    DetectionUpdate,
    DetectionResponse,
)

from app.db.schemas.image import (
    ImageBase,
    ImageCreate,
    ImageResponse,
)

from app.db.schemas.image_content import (
    ImageContentBase,
    ImageContentCreate,
    ImageContentResponse,
)

from app.db.schemas.image_edit_history import (
    ImageEditHistoryBase,
    ImageEditHistoryCreate,
    ImageEditHistoryResponse,
)

from app.db.schemas.image_version import (
    ImageVersionBase,
    ImageVersionCreate,
    ImageVersionResponse,
)

from app.db.schemas.model_meta import ModelMeta

from app.db.schemas.segmentation import (
    SegmentRequest,
    SegmentWithPromptRequest,
    SegmentByPolygonRequest,
    SegmentHybridRequest,
    SegmentInfo,
    SegmentResponse,
    SegmentationMaskBase,
    SegmentationMaskCreate,
    SegmentationMaskUpdate,
    SegmentationMaskResponse,
)

from app.db.schemas.user import (
    UserBase,
    UserCreate,
    UserResponse,
    UserUpdate,
    ChangePassword,
)

from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType
from app.db.enums.segmentation_mode import SegmentationMode

@pytest.mark.unit
class TestBboxSchema:
    def test_valid(self):
        bbox = BboxSchema(x1=10, y1=20, x2=100, y2=200)
        assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10, 20, 100, 200)

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1=10, y1=20, x2=100)  # type: ignore

    def test_wrong_type_invalid(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1="a", y1=20, x2=100, y2=200)  # type: ignore

@pytest.mark.unit
class TestAssetBase:
    def test_valid(self):
        asset = AssetBase(width=100, height=200, area_pixels=20000)
        assert asset.label is None

    def test_with_label(self):
        asset = AssetBase(width=100, height=200, area_pixels=20000, label="cat")
        assert asset.label == "cat"

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            AssetBase(width=100, height=200)  # type: ignore


@pytest.mark.unit
class TestAssetCreate:
    def test_defaults(self):
        asset = AssetCreate(
            width=100, height=200, area_pixels=20000,
            user_id=1, storage_path="s3://a.png",
        )
        assert asset.content_type == "image/png"
        assert asset.thumbnail_path is None
        assert asset.file_size is None
        assert asset.source_image_version_id is None
        assert asset.source_segmentation_mask_id is None

    def test_missing_user_id_invalid(self):
        with pytest.raises(ValidationError):
            AssetCreate(width=100, height=200, area_pixels=20000, storage_path="s3://a.png")  # type: ignore

    def test_custom_content_type(self):
        asset = AssetCreate(
            width=100, height=200, area_pixels=20000,
            user_id=1, storage_path="s3://a.png", content_type="image/jpeg",
        )
        assert asset.content_type == "image/jpeg"


@pytest.mark.unit
class TestAssetUpdate:
    def test_valid(self):
        assert AssetUpdate(label="new-label").label == "new-label"

    def test_missing_label_invalid(self):
        with pytest.raises(ValidationError):
            AssetUpdate()  # type: ignore


@pytest.mark.unit
class TestAssetResponse:
    def test_valid(self):
        resp = AssetResponse(
            width=100, height=200, area_pixels=20000,
            public_id="pub-1", storage_path="s3://a.png",
            content_type="image/png", created_at=datetime(2025, 1, 1),
        )
        assert resp.public_id == "pub-1"
        assert resp.thumbnail_path is None

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            AssetResponse(
                width=100, height=200, area_pixels=20000,
                public_id="pub-1", storage_path="s3://a.png",
                content_type="image/png",
            )  # type: ignore

    def test_from_attributes_config_enabled(self):
        assert AssetResponse.model_config.get("from_attributes") is True

@pytest.mark.unit
class TestDetectionBase:
    def test_defaults(self):
        det = DetectionBase(x1=1, y1=2, x2=3, y2=4, confidence=0.5)
        assert det.detected_class == "unknown"

    def test_missing_confidence_invalid(self):
        with pytest.raises(ValidationError):
            DetectionBase(x1=1, y1=2, x2=3, y2=4)  # type: ignore

    def test_custom_class(self):
        det = DetectionBase(x1=1, y1=2, x2=3, y2=4, confidence=0.9, detected_class="dog")
        assert det.detected_class == "dog"


@pytest.mark.unit
class TestDetectionCreate:
    def test_valid(self):
        det = DetectionCreate(
            x1=1, y1=2, x2=3, y2=4, confidence=0.9,
            content_id=1, bbox_id=1,
            model_name="yolo", model_version="v8", inference_time_ms=12.5,
        )
        assert det.content_id == 1
        assert det.model_name == "yolo"

    def test_missing_model_name_invalid(self):
        with pytest.raises(ValidationError):
            DetectionCreate(
                x1=1, y1=2, x2=3, y2=4, confidence=0.9,
                content_id=1, bbox_id=1,
                model_version="v8", inference_time_ms=12.5,
            )  # type: ignore


@pytest.mark.unit
class TestDetectionUpdate:
    def test_default_none(self):
        assert DetectionUpdate().is_active is None

    def test_set_false(self):
        assert DetectionUpdate(is_active=False).is_active is False


@pytest.mark.unit
class TestDetectionResponse:
    def test_valid(self):
        resp = DetectionResponse(
            x1=1, y1=2, x2=3, y2=4, confidence=0.9,
            id=1, content_id=1, bbox_id=1, is_active=True,
            model_name="yolo", model_version="v8", inference_time_ms=12.5,
            created_at=datetime(2025, 1, 1),
        )
        assert resp.id == 1
        assert resp.is_active is True

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            DetectionResponse(
                x1=1, y1=2, x2=3, y2=4, confidence=0.9,
                id=1, content_id=1, bbox_id=1, is_active=True,
                model_name="yolo", model_version="v8",
                created_at=datetime(2025, 1, 1),
            )  # type: ignore

@pytest.mark.unit
class TestImageBase:
    def test_valid(self):
        img = ImageBase(filename="a.jpg", storage_path="s3://a.jpg")
        assert img.filename == "a.jpg"

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            ImageBase(filename="a.jpg")  # type: ignore


@pytest.mark.unit
class TestImageCreate:
    def test_valid(self):
        img = ImageCreate(filename="a.jpg", storage_path="s3://a.jpg", user_id=1)
        assert img.user_id == 1


@pytest.mark.unit
class TestImageResponse:
    def test_valid(self):
        resp = ImageResponse(
            filename="a.jpg", storage_path="s3://a.jpg",
            id=1, uploaded_at=datetime(2025, 1, 1, 12, 30),
        )
        assert resp.cache_key is None

    def test_uploaded_at_serialized_to_isoformat(self):
        resp = ImageResponse(
            filename="a.jpg", storage_path="s3://a.jpg",
            id=1, uploaded_at=datetime(2025, 1, 1, 12, 30),
        )
        dumped = resp.model_dump(mode="json")
        assert dumped["uploaded_at"] == datetime(2025, 1, 1, 12, 30).isoformat()

    def test_with_cache_key(self):
        resp = ImageResponse(
            filename="a.jpg", storage_path="s3://a.jpg",
            id=1, uploaded_at=datetime(2025, 1, 1), cache_key="abc123",
        )
        assert resp.cache_key == "abc123"

@pytest.mark.unit
class TestImageContentBase:
    def test_valid(self):
        content = ImageContentBase(width=100, height=100, file_size=1024)
        assert content.file_size == 1024

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            ImageContentBase(width=100, height=100)  # type: ignore


@pytest.mark.unit
class TestImageContentCreate:
    def test_valid(self):
        content = ImageContentCreate(
            width=100, height=100, file_size=1024,
            content_hash="a" * 64, storage_path="s3://c.png",
        )
        assert content.content_hash == "a" * 64


@pytest.mark.unit
class TestImageContentResponse:
    def test_valid(self):
        resp = ImageContentResponse(
            width=100, height=100, file_size=1024,
            id=1, content_hash="a" * 64, storage_path="s3://c.png",
            created_at=datetime(2025, 1, 1),
        )
        assert resp.id == 1

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            ImageContentResponse(
                width=100, height=100, file_size=1024,
                id=1, content_hash="a" * 64,
                created_at=datetime(2025, 1, 1),
            )  # type: ignore

@pytest.mark.unit
class TestImageEditHistoryBase:
    def test_valid(self):
        entry = ImageEditHistoryBase(
            image_version_id=1,
            operation=EditOperation.REMOVE,
            engine=EngineType.LAMA,
        )
        assert entry.parameters is None
        assert entry.processing_time_ms is None

    def test_invalid_operation_value(self):
        with pytest.raises(ValidationError):
            ImageEditHistoryBase(
                image_version_id=1, operation="not_a_real_op", engine=EngineType.LAMA
            )  # type: ignore

    def test_accepts_plain_string_matching_enum_value(self):
        entry = ImageEditHistoryBase(
            image_version_id=1, operation="detect", engine="yolo"
        )
        assert entry.operation == EditOperation.DETECT
        assert entry.engine == EngineType.YOLO

    def test_with_parameters(self):
        entry = ImageEditHistoryBase(
            image_version_id=1, operation=EditOperation.SEGMENT, engine=EngineType.SAM,
            parameters={"threshold": 0.5}, processing_time_ms=120,
        )
        assert entry.parameters == {"threshold": 0.5}
        assert entry.processing_time_ms == 120


@pytest.mark.unit
class TestImageEditHistoryCreate:
    def test_valid(self):
        entry = ImageEditHistoryCreate(
            image_version_id=1, operation=EditOperation.REPLACE, engine=EngineType.DIFFUSION,
        )
        assert entry.operation == EditOperation.REPLACE


@pytest.mark.unit
class TestImageEditHistoryResponse:
    def test_valid(self):
        resp = ImageEditHistoryResponse(
            image_version_id=1, operation=EditOperation.DETECT, engine=EngineType.YOLO,
            id=1, created_at=datetime(2025, 1, 1),
        )
        assert resp.id == 1

    def test_missing_created_at_invalid(self):
        with pytest.raises(ValidationError):
            ImageEditHistoryResponse(
                image_version_id=1, operation=EditOperation.DETECT, engine=EngineType.YOLO,
                id=1,
            )  # type: ignore

@pytest.mark.unit
class TestImageVersionBase:
    def test_valid(self):
        version = ImageVersionBase(image_id=1, version_number=1, storage_path="s3://v1.png")
        assert version.parent_version_id is None

    def test_with_parent(self):
        version = ImageVersionBase(
            image_id=1, version_number=2, storage_path="s3://v2.png", parent_version_id=1,
        )
        assert version.parent_version_id == 1

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            ImageVersionBase(image_id=1, storage_path="s3://v1.png")  # type: ignore


@pytest.mark.unit
class TestImageVersionCreate:
    def test_valid(self):
        version = ImageVersionCreate(image_id=1, version_number=1, storage_path="s3://v1.png")
        assert version.version_number == 1


@pytest.mark.unit
class TestImageVersionResponse:
    def test_valid(self):
        resp = ImageVersionResponse(
            image_id=1, version_number=1, storage_path="s3://v1.png",
            id=1, created_at=datetime(2025, 1, 1),
        )
        assert resp.id == 1

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            ImageVersionResponse(
                image_id=1, version_number=1, storage_path="s3://v1.png", id=1,
            )  # type: ignore
@pytest.mark.unit
class TestModelMeta:
    def test_valid(self):
        meta = ModelMeta(model_name="yolo", model_version="v8", inference_time_ms=12.5)
        assert meta.model_name == "yolo"
        assert meta.inference_time_ms == 12.5

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            ModelMeta(model_name="yolo", model_version="v8")  # type: ignore

    def test_wrong_type_invalid(self):
        with pytest.raises(ValidationError):
            ModelMeta(model_name="yolo", model_version="v8", inference_time_ms="fast")  # type: ignore

@pytest.mark.unit
class TestSegmentRequest:
    def test_defaults(self):
        req = SegmentRequest()
        assert req.min_area == 500
        assert req.max_segments == 50

    def test_min_area_negative_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(min_area=-1)

    def test_max_segments_bounds(self):
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=0)
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=201)


@pytest.mark.unit
class TestSegmentWithPromptRequest:
    def test_all_optional(self):
        req = SegmentWithPromptRequest()
        assert req.point_coords is None
        assert req.bbox is None

    def test_with_bbox(self):
        req = SegmentWithPromptRequest(bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10))
        assert req.bbox.x2 == 10


@pytest.mark.unit
class TestSegmentByPolygonRequest:
    def test_valid(self):
        req = SegmentByPolygonRequest(points=[(0, 0), (10, 0), (5, 10)])
        assert req.smooth is True
        assert req.smoothing_factor == 0.0
        assert req.feather_px == 0

    def test_fewer_than_three_points_invalid(self):
        with pytest.raises(ValidationError):
            SegmentByPolygonRequest(points=[(0, 0), (10, 0)])


@pytest.mark.unit
class TestSegmentHybridRequest:
    def test_defaults(self):
        req = SegmentHybridRequest()
        assert req.yolo_conf_threshold == 0.35
        assert req.yolo_classes is None
        assert req.fallback_min_area == 800
        assert req.fallback_max_segments == 50
        assert req.overlap_iou_thresh == 0.5

    def test_custom_values(self):
        req = SegmentHybridRequest(
            yolo_conf_threshold=0.6, yolo_classes=["person"],
            fallback_min_area=1000, fallback_max_segments=10, overlap_iou_thresh=0.7,
        )
        assert req.yolo_conf_threshold == 0.6
        assert req.yolo_classes == ["person"]


@pytest.mark.unit
class TestSegmentInfoAndResponse:
    def test_segment_info_valid(self):
        info = SegmentInfo(
            mask_id=1, bbox_id=2, bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10), area=100,
        )
        assert info.stability_score is None
        assert info.mask_url is None

    def test_segment_response_valid(self):
        resp = SegmentResponse(
            segments=[], metrics={}, image_size=(100, 100), timestamp=datetime(2025, 1, 1),
        )
        assert resp.segments == []


@pytest.mark.unit
class TestSegmentationMaskBase:
    def test_valid(self):
        mask = SegmentationMaskBase(x1=0, y1=0, x2=50, y2=50, area=2500.0, score=0.95)
        assert mask.score == 0.95

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            SegmentationMaskBase(x1=0, y1=0, x2=50, y2=50, area=2500.0)  # type: ignore


@pytest.mark.unit
class TestSegmentationMaskCreate:
    def test_valid(self):
        mask = SegmentationMaskCreate(
            x1=0, y1=0, x2=50, y2=50, area=2500.0, score=0.95,
            content_id=1, mask_id=1,
            mask_storage_path="s3://m.png", preview_storage_path="s3://p.png",
            segmentation_mode=SegmentationMode.SAM,
            model_name="sam", model_version="v1", inference_time_ms=15.0,
        )
        assert mask.segmentation_mode == SegmentationMode.SAM

    def test_invalid_segmentation_mode(self):
        with pytest.raises(ValidationError):
            SegmentationMaskCreate(
                x1=0, y1=0, x2=50, y2=50, area=2500.0, score=0.95,
                content_id=1, mask_id=1,
                mask_storage_path="s3://m.png", preview_storage_path="s3://p.png",
                segmentation_mode="not_a_mode",
                model_name="sam", model_version="v1", inference_time_ms=15.0,
            )  # type: ignore


@pytest.mark.unit
class TestSegmentationMaskUpdate:
    def test_default_none(self):
        assert SegmentationMaskUpdate().is_active is None

    def test_set_value(self):
        assert SegmentationMaskUpdate(is_active=False).is_active is False


@pytest.mark.unit
class TestSegmentationMaskResponse:
    def test_valid(self):
        resp = SegmentationMaskResponse(
            x1=0, y1=0, x2=50, y2=50, area=2500.0, score=0.95,
            id=1, content_id=1, mask_id=1, is_active=True,
            segmentation_mode=SegmentationMode.HYBRID,
            model_name="sam", model_version="v1", inference_time_ms=15.0,
            created_at=datetime(2025, 1, 1),
        )
        assert resp.is_active is True

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            SegmentationMaskResponse(
                x1=0, y1=0, x2=50, y2=50, area=2500.0, score=0.95,
                id=1, content_id=1, mask_id=1, is_active=True,
                segmentation_mode=SegmentationMode.HYBRID,
                model_name="sam", model_version="v1",
                created_at=datetime(2025, 1, 1),
            )  # type: ignore

@pytest.mark.unit
class TestUserBase:
    def test_valid(self):
        user = UserBase(username="john", email="john@test.com")
        assert user.username == "john"

    def test_username_too_short_invalid(self):
        with pytest.raises(ValidationError):
            UserBase(username="jo", email="john@test.com")

    def test_username_too_long_invalid(self):
        with pytest.raises(ValidationError):
            UserBase(username="j" * 51, email="john@test.com")

    def test_invalid_email(self):
        with pytest.raises(ValidationError):
            UserBase(username="john", email="not-an-email")


@pytest.mark.unit
class TestUserCreate:
    def test_valid(self):
        user = UserCreate(username="john", email="john@test.com", password="secret1")
        assert user.password == "secret1"

    def test_password_too_short_invalid(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="john@test.com", password="123")

    def test_password_too_long_invalid(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="john@test.com", password="x" * 101)


@pytest.mark.unit
class TestUserResponse:
    def test_valid(self):
        resp = UserResponse(
            username="john", email="john@test.com", id=1, created_at=datetime(2025, 1, 1),
        )
        assert resp.id == 1

    def test_from_attributes_config_enabled(self):
        assert UserResponse.model_config.get("from_attributes") is True


@pytest.mark.unit
class TestUserUpdate:
    def test_all_optional_defaults_none(self):
        update = UserUpdate()
        assert update.username is None
        assert update.email is None

    def test_partial_update(self):
        update = UserUpdate(username="newname")
        assert update.username == "newname"
        assert update.email is None

    def test_invalid_email_still_validated(self):
        with pytest.raises(ValidationError):
            UserUpdate(email="not-an-email")


@pytest.mark.unit
class TestChangePassword:
    def test_valid(self):
        cp = ChangePassword(old_password="secret1", new_password="secret2")
        assert cp.new_password == "secret2"

    def test_old_password_too_short_invalid(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="123", new_password="secret2")

    def test_new_password_too_short_invalid(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="secret1", new_password="123")

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="secret1")  # type: ignore