import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.db.schemas.user import (
    UserCreate,
    UserResponse,
    UserUpdate,
    ChangePassword,
)
from app.db.schemas.image import ImageCreate, ImageResponse
from app.db.schemas.detection import (
    DetectionCreate,
    DetectionUpdate,
    DetectionResponse,
)
from app.db.schemas.ml import (
    BboxSchema,
    DetectRequest,
    ExtractRequest,
    ExtractResponse,
    LdmConfig,
    MLResultResponse,
    PasteRequest,
    PasteResponse,
    RemoveMultipleRequest,
    RemoveRequest,
    ReplaceRequest,
    SamRemoveRequest,
    SamReplaceRequest,
    SegmentInfo,
    SegmentRequest,
    SegmentResponse,
    SegmentWithPromptRequest,
)

@pytest.mark.unit
class TestUserSchemas:
    def test_user_create_valid(self):
        user = UserCreate(username="john", email="john@test.com", password="pass123")
        assert user.username == "john"
        assert user.email == "john@test.com"
        assert user.password == "pass123"

    def test_user_create_invalid_email(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="invalid", password="pass123")

    def test_user_create_short_username(self):
        with pytest.raises(ValidationError):
            UserCreate(username="ab", email="john@test.com", password="pass123")

    def test_user_create_username_too_long(self):
        with pytest.raises(ValidationError):
            UserCreate(username="a" * 51, email="john@test.com", password="pass123")

    def test_user_create_username_boundary_min(self):
        user = UserCreate(username="abc", email="john@test.com", password="pass123")
        assert user.username == "abc"

    def test_user_create_username_boundary_max(self):
        user = UserCreate(username="a" * 50, email="john@test.com", password="pass123")
        assert len(user.username) == 50

    def test_user_create_password_too_short(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="john@test.com", password="12345")

    def test_user_create_password_too_long(self):
        with pytest.raises(ValidationError):
            UserCreate(username="john", email="john@test.com", password="a" * 101)

    def test_user_create_missing_fields(self):
        with pytest.raises(ValidationError):
            UserCreate()  # type: ignore

    def test_user_response_valid(self):
        resp = UserResponse(
            username="john",
            email="john@test.com",
            id=1,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert resp.id == 1
        assert resp.username == "john"

    def test_user_response_from_attributes_config(self):
        assert UserResponse.model_config.get("from_attributes") is True

    def test_user_update_all_optional(self):
        upd = UserUpdate()
        assert upd.username is None
        assert upd.email is None

    def test_user_update_partial(self):
        upd = UserUpdate(username="newname")
        assert upd.username == "newname"
        assert upd.email is None

    def test_user_update_invalid_email(self):
        with pytest.raises(ValidationError):
            UserUpdate(email="not-an-email")

    def test_change_password_valid(self):
        cp = ChangePassword(old_password="oldpass", new_password="newpass")
        assert cp.old_password == "oldpass"
        assert cp.new_password == "newpass"

    def test_change_password_old_too_short(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="123", new_password="newpass")

    def test_change_password_new_too_short(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="oldpass", new_password="123")

    def test_change_password_missing_field(self):
        with pytest.raises(ValidationError):
            ChangePassword(old_password="oldpass")  # type: ignore


@pytest.mark.unit
class TestImageSchemas:
    def test_image_create_valid(self):
        img = ImageCreate(filename="test.jpg", storage_path="s3://test.jpg", user_id=1)
        assert img.filename == "test.jpg"
        assert img.storage_path == "s3://test.jpg"
        assert img.user_id == 1

    def test_image_create_missing_field(self):
        with pytest.raises(ValidationError):
            ImageCreate(filename="test.jpg", storage_path="s3://test.jpg")  # type: ignore

    def test_image_response_valid(self):
        resp = ImageResponse(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            id=1,
            uploaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert resp.cache_key is None

    def test_image_response_with_cache_key(self):
        resp = ImageResponse(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            id=1,
            uploaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            cache_key="abc123",
        )
        assert resp.cache_key == "abc123"

    def test_image_response_datetime_serialization(self):
        dt = datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc)
        resp = ImageResponse(
            filename="test.jpg",
            storage_path="s3://test.jpg",
            id=1,
            uploaded_at=dt,
        )
        dumped = resp.model_dump(mode="json")
        assert dumped["uploaded_at"] == dt.isoformat()

    def test_image_response_from_attributes_config(self):
        assert ImageResponse.Config.from_attributes is True
@pytest.mark.unit
class TestDetectionSchemas:
    def test_detection_create_valid(self):
        det = DetectionCreate(
            image_id=1, x1=10, y1=10, x2=100, y2=100,
            detected_class="person", confidence=0.9,
        )
        assert det.confidence == 0.9
        assert det.detected_class == "person"

    def test_detection_create_default_class(self):
        det = DetectionCreate(image_id=1, x1=0, y1=0, x2=10, y2=10, confidence=0.5)
        assert det.detected_class == "unknown"

    def test_detection_create_missing_required(self):
        with pytest.raises(ValidationError):
            DetectionCreate(image_id=1, x1=0, y1=0, x2=10)  # type: ignore

    def test_detection_update_valid(self):
        upd = DetectionUpdate(
            image_id=1, x1=1, y1=2, x2=3, y2=4, confidence=0.1,
        )
        assert upd.x1 == 1

    def test_detection_response_valid(self):
        resp = DetectionResponse(
            image_id=1, x1=0, y1=0, x2=10, y2=10,
            confidence=0.75, id=5, bbox_id=7,
        )
        assert resp.id == 5
        assert resp.bbox_id == 7

    def test_detection_response_missing_bbox_id(self):
        with pytest.raises(ValidationError):
            DetectionResponse(
                image_id=1, x1=0, y1=0, x2=10, y2=10,
                confidence=0.75, id=5,
            )  # type: ignore

    def test_detection_response_from_attributes_config(self):
        assert DetectionResponse.Config.from_attributes is True
@pytest.mark.unit
class TestBboxSchema:
    def test_bbox_valid(self):
        bbox = BboxSchema(x1=10, y1=20, x2=100, y2=200)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200

    def test_bbox_zero_coords(self):
        bbox = BboxSchema(x1=0, y1=0, x2=0, y2=0)
        assert bbox.x1 == 0

    def test_bbox_missing_field(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1=10, y1=20, x2=100)  # type: ignore

    def test_bbox_wrong_type(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1="a", y1=20, x2=100, y2=200)  # type: ignore

    def test_bbox_negative_coords(self):
        bbox = BboxSchema(x1=-10, y1=-20, x2=100, y2=200)
        assert bbox.x1 == -10


@pytest.mark.unit
class TestLdmConfig:
    def test_ldm_config_defaults(self):
        cfg = LdmConfig()
        assert cfg.ldm_steps == 25
        assert cfg.ldm_sampler == 'plms'
        assert cfg.hd_strategy == 'CROP'

    def test_ldm_config_valid_custom(self):
        cfg = LdmConfig(ldm_steps=10, ldm_sampler='ddim', hd_strategy='RESIZE')
        assert cfg.ldm_steps == 10
        assert cfg.ldm_sampler == 'ddim'
        assert cfg.hd_strategy == 'RESIZE'

    def test_ldm_config_hd_strategy_original(self):
        cfg = LdmConfig(hd_strategy='ORIGINAL')
        assert cfg.hd_strategy == 'ORIGINAL'

    def test_ldm_config_steps_min(self):
        assert LdmConfig(ldm_steps=5).ldm_steps == 5

    def test_ldm_config_steps_max(self):
        assert LdmConfig(ldm_steps=50).ldm_steps == 50

    def test_ldm_config_steps_below_min(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_steps=4)

    def test_ldm_config_steps_above_max(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_steps=51)

    def test_ldm_config_invalid_sampler(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_sampler='euler')  # type: ignore

    def test_ldm_config_invalid_hd_strategy(self):
        with pytest.raises(ValidationError):
            LdmConfig(hd_strategy='INVALID')  # type: ignore

    def test_ldm_config_plms_sampler(self):
        assert LdmConfig(ldm_sampler='plms').ldm_sampler == 'plms'

    def test_ldm_config_ddim_sampler(self):
        assert LdmConfig(ldm_sampler='ddim').ldm_sampler == 'ddim'


@pytest.mark.unit
class TestDetectRequest:
    def test_detect_request_valid(self):
        req = DetectRequest(conf_threshold=0.7, classes=["person"])
        assert req.conf_threshold == 0.7
        assert req.classes == ["person"]

    def test_detect_request_defaults(self):
        req = DetectRequest()
        assert req.conf_threshold == 0.5
        assert req.classes is None

    def test_detect_request_invalid_threshold_above(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=1.5)

    def test_detect_request_threshold_below_zero(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=-0.1)

    def test_detect_request_threshold_boundary_zero(self):
        assert DetectRequest(conf_threshold=0.0).conf_threshold == 0.0

    def test_detect_request_threshold_boundary_one(self):
        assert DetectRequest(conf_threshold=1.0).conf_threshold == 1.0

    def test_detect_request_empty_classes(self):
        assert DetectRequest(classes=[]).classes == []


@pytest.mark.unit
class TestRemoveRequest:
    def test_defaults(self):
        req = RemoveRequest()
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is True
        assert req.ldm.ldm_steps == 25

    def test_with_ldm(self):
        req = RemoveRequest(ldm=LdmConfig(ldm_steps=10, ldm_sampler='ddim', hd_strategy='RESIZE'))
        assert req.ldm.ldm_steps == 10

    def test_ldm_default_factory_independent(self):
        req1, req2 = RemoveRequest(), RemoveRequest()
        assert req1.ldm is not req2.ldm

    def test_expand_mask_min(self):
        assert RemoveRequest(expand_mask_pixels=0).expand_mask_pixels == 0

    def test_expand_mask_max(self):
        assert RemoveRequest(expand_mask_pixels=50).expand_mask_pixels == 50

    def test_expand_mask_above_max(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=51)

    def test_expand_mask_below_min(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=-1)

    def test_edge_blending_false(self):
        assert RemoveRequest(use_edge_blending=False).use_edge_blending is False


@pytest.mark.unit
class TestRemoveMultipleRequest:
    def test_valid(self):
        req = RemoveMultipleRequest(bbox_ids=[1, 2, 3])
        assert len(req.bbox_ids) == 3

    def test_invalid_empty(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest(bbox_ids=[])

    def test_single_id(self):
        assert RemoveMultipleRequest(bbox_ids=[42]).bbox_ids == [42]

    def test_defaults(self):
        req = RemoveMultipleRequest(bbox_ids=[1])
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is True
        assert req.ldm.ldm_steps == 25

    def test_missing_bbox_ids(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest()  # type: ignore


@pytest.mark.unit
class TestReplaceRequest:
    def test_valid(self):
        req = ReplaceRequest(color_match_method="histogram")
        assert req.color_match_method == "histogram"

    def test_defaults(self):
        req = ReplaceRequest()
        assert req.expand_mask_pixels == 0
        assert req.use_color_matching is True
        assert req.color_match_method == 'mean_std'

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            ReplaceRequest(color_match_method="invalid")  # type: ignore

    def test_all_color_methods(self):
        for method in ['mean_std', 'histogram', 'color_transfer']:
            assert ReplaceRequest(color_match_method=method).color_match_method == method

    def test_no_color_matching(self):
        assert ReplaceRequest(use_color_matching=False).use_color_matching is False


@pytest.mark.unit
class TestSegmentRequest:
    def test_defaults(self):
        req = SegmentRequest()
        assert req.min_area == 500
        assert req.max_segments == 50

    def test_min_area_negative_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(min_area=-1)

    def test_max_segments_below_min_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=0)

    def test_max_segments_above_max_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=201)

    def test_max_segments_boundary(self):
        assert SegmentRequest(max_segments=200).max_segments == 200


@pytest.mark.unit
class TestSegmentWithPromptRequest:
    def test_all_optional_defaults(self):
        req = SegmentWithPromptRequest()
        assert req.point_coords is None
        assert req.point_labels is None
        assert req.bbox is None

    def test_with_points(self):
        req = SegmentWithPromptRequest(
            point_coords=[(10, 20), (30, 40)],
            point_labels=[1, 0],
        )
        assert req.point_coords == [(10, 20), (30, 40)]
        assert req.point_labels == [1, 0]

    def test_with_bbox(self):
        req = SegmentWithPromptRequest(bbox=BboxSchema(x1=0, y1=0, x2=50, y2=50))
        assert req.bbox.x2 == 50


@pytest.mark.unit
class TestSamRequests:
    def test_sam_remove_defaults(self):
        req = SamRemoveRequest()
        assert req.expand_mask_pixels == 12
        assert req.use_edge_blending is True

    def test_sam_remove_expand_bounds(self):
        with pytest.raises(ValidationError):
            SamRemoveRequest(expand_mask_pixels=51)

    def test_sam_replace_defaults(self):
        req = SamReplaceRequest()
        assert req.expand_mask_pixels == 8
        assert req.use_color_matching is True
        assert req.use_edge_blending is False
        assert req.color_match_method == 'color_transfer'

    def test_sam_replace_invalid_color_method(self):
        with pytest.raises(ValidationError):
            SamReplaceRequest(color_match_method="invalid")  # type: ignore


@pytest.mark.unit
class TestExtractPasteRequests:
    def test_extract_request_default(self):
        assert ExtractRequest().padding_pixels == 8

    def test_extract_request_bounds(self):
        with pytest.raises(ValidationError):
            ExtractRequest(padding_pixels=65)

    def test_extract_request_min_bound(self):
        assert ExtractRequest(padding_pixels=0).padding_pixels == 0

    def test_paste_request_valid(self):
        req = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )
        assert req.scale == 1.0
        assert req.color_match_method == 'color_transfer'

    def test_paste_request_scale_bounds(self):
        with pytest.raises(ValidationError):
            PasteRequest(
                extracted_url="s3://obj.png",
                target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
                scale=3.5,
            )

    def test_paste_request_scale_below_min(self):
        with pytest.raises(ValidationError):
            PasteRequest(
                extracted_url="s3://obj.png",
                target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
                scale=0.05,
            )

    def test_paste_request_missing_required(self):
        with pytest.raises(ValidationError):
            PasteRequest(extracted_url="s3://obj.png")  # type: ignore


@pytest.mark.unit
class TestMLResultResponse:
    def test_valid(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"time": 0.5},
            timestamp=datetime(2025, 1, 1),
        )
        assert res.result_url is not None

    def test_empty_metrics(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={},
            timestamp=datetime(2025, 1, 1),
        )
        assert res.metrics == {}

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            MLResultResponse(
                result_url="s3://result.jpg",
                presigned_url="http://url",
                metrics={},
            )  # type: ignore

    def test_metrics_nested(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"processing_time_ms": 250.5, "mask_size_pixels": 10000},
            timestamp=datetime(2025, 1, 1),
        )
        assert res.metrics["processing_time_ms"] == 250.5


@pytest.mark.unit
class TestSegmentResponses:
    def test_segment_info_valid(self):
        info = SegmentInfo(
            mask_id=1, bbox_id=2,
            bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
            area=100, stability_score=0.95,
        )
        assert info.area == 100
        assert info.stability_score == 0.95

    def test_segment_response_valid(self):
        resp = SegmentResponse(
            segments=[
                SegmentInfo(
                    mask_id=1, bbox_id=2,
                    bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
                    area=100, stability_score=0.9,
                )
            ],
            metrics={"count": 1},
            image_size=(640, 480),
            timestamp=datetime(2025, 1, 1),
        )
        assert len(resp.segments) == 1
        assert resp.image_size == (640, 480)

    def test_segment_response_empty_segments(self):
        resp = SegmentResponse(
            segments=[],
            metrics={},
            image_size=(100, 100),
            timestamp=datetime(2025, 1, 1),
        )
        assert resp.segments == []

    def test_extract_response_valid(self):
        resp = ExtractResponse(
            extracted_url="s3://obj.png",
            presigned_url="http://url",
            object_size=(50, 60),
            area_pixels=3000,
            cropped_bbox=BboxSchema(x1=0, y1=0, x2=50, y2=60),
            timestamp=datetime(2025, 1, 1),
        )
        assert resp.object_size == (50, 60)
        assert resp.area_pixels == 3000

    def test_paste_response_valid(self):
        resp = PasteResponse(
            result_url="s3://result.png",
            presigned_url="http://url",
            paste_bbox=BboxSchema(x1=0, y1=0, x2=20, y2=20),
            object_size=(20, 20),
            timestamp=datetime(2025, 1, 1),
        )
        assert resp.paste_bbox.x2 == 20

    def test_paste_response_missing_field(self):
        with pytest.raises(ValidationError):
            PasteResponse(
                result_url="s3://result.png",
                presigned_url="http://url",
                paste_bbox=BboxSchema(x1=0, y1=0, x2=20, y2=20),
                timestamp=datetime(2025, 1, 1),
            )  # type: ignore