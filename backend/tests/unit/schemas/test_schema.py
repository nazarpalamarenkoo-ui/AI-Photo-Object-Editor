import pytest
from pydantic import ValidationError
from app.db.schemas.user import UserCreate
from app.db.schemas.image import ImageCreate
from app.db.schemas.detection import DetectionCreate
from app.db.schemas.ml import (
    BboxSchema,
    LdmConfig,
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
        cfg = LdmConfig(ldm_steps=5)
        assert cfg.ldm_steps == 5

    def test_ldm_config_steps_max(self):
        cfg = LdmConfig(ldm_steps=50)
        assert cfg.ldm_steps == 50

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
        cfg = LdmConfig(ldm_sampler='plms')
        assert cfg.ldm_sampler == 'plms'

    def test_ldm_config_ddim_sampler(self):
        cfg = LdmConfig(ldm_sampler='ddim')
        assert cfg.ldm_sampler == 'ddim'


@pytest.mark.unit
class TestMLSchemas:

    def test_detect_request_valid(self):
        req = DetectRequest(conf_threshold=0.7, classes=["person"])
        assert req.conf_threshold == 0.7
        assert req.classes == ["person"]

    def test_detect_request_defaults(self):
        req = DetectRequest()
        assert req.conf_threshold == 0.5
        assert req.classes is None

    def test_detect_request_invalid_threshold(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=1.5)

    def test_detect_request_threshold_below_zero(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=-0.1)

    def test_detect_request_threshold_boundary_zero(self):
        req = DetectRequest(conf_threshold=0.0)
        assert req.conf_threshold == 0.0

    def test_detect_request_threshold_boundary_one(self):
        req = DetectRequest(conf_threshold=1.0)
        assert req.conf_threshold == 1.0

    def test_detect_request_empty_classes(self):
        req = DetectRequest(classes=[])
        assert req.classes == []

    # --- RemoveRequest ---

    def test_remove_request_defaults(self):
        req = RemoveRequest()
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is True
        assert req.scene_bboxes is None
        assert req.ldm.ldm_steps == 25
        assert req.ldm.ldm_sampler == 'plms'
        assert req.ldm.hd_strategy == 'CROP'

    def test_remove_request_with_ldm(self):
        req = RemoveRequest(ldm=LdmConfig(ldm_steps=10, ldm_sampler='ddim', hd_strategy='RESIZE'))
        assert req.ldm.ldm_steps == 10
        assert req.ldm.ldm_sampler == 'ddim'
        assert req.ldm.hd_strategy == 'RESIZE'

    def test_remove_request_ldm_default_factory(self):
        req1 = RemoveRequest()
        req2 = RemoveRequest()
        assert req1.ldm is not req2.ldm

    def test_remove_request_expand_mask_min(self):
        req = RemoveRequest(expand_mask_pixels=0)
        assert req.expand_mask_pixels == 0

    def test_remove_request_expand_mask_max(self):
        req = RemoveRequest(expand_mask_pixels=50)
        assert req.expand_mask_pixels == 50

    def test_remove_request_expand_mask_above_max(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=51)

    def test_remove_request_expand_mask_below_min(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=-1)

    def test_remove_request_with_scene_bboxes(self):
        req = RemoveRequest(scene_bboxes=[BboxSchema(x1=0, y1=0, x2=100, y2=100)])
        assert len(req.scene_bboxes) == 1
        assert req.scene_bboxes[0].x1 == 0

    def test_remove_request_edge_blending_false(self):
        req = RemoveRequest(use_edge_blending=False)
        assert req.use_edge_blending is False

    # --- RemoveMultipleRequest ---

    def test_remove_multiple_valid(self):
        req = RemoveMultipleRequest(bbox_ids=[1, 2, 3])
        assert len(req.bbox_ids) == 3

    def test_remove_multiple_invalid_empty(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest(bbox_ids=[])

    def test_remove_multiple_single_id(self):
        req = RemoveMultipleRequest(bbox_ids=[42])
        assert req.bbox_ids == [42]

    def test_remove_multiple_defaults(self):
        req = RemoveMultipleRequest(bbox_ids=[1])
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is True
        assert req.scene_bboxes is None
        assert req.ldm.ldm_steps == 25

    def test_remove_multiple_with_ldm(self):
        req = RemoveMultipleRequest(
            bbox_ids=[1, 2],
            ldm=LdmConfig(ldm_steps=10, ldm_sampler='ddim', hd_strategy='ORIGINAL')
        )
        assert req.ldm.ldm_steps == 10
        assert req.ldm.ldm_sampler == 'ddim'
        assert req.ldm.hd_strategy == 'ORIGINAL'

    def test_remove_multiple_with_scene_bboxes(self):
        req = RemoveMultipleRequest(
            bbox_ids=[1],
            scene_bboxes=[BboxSchema(x1=10, y1=10, x2=50, y2=50)]
        )
        assert len(req.scene_bboxes) == 1

    def test_remove_multiple_missing_bbox_ids(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest()  # type: ignore

    # --- ReplaceRequest ---

    def test_replace_request_valid(self):
        req = ReplaceRequest(color_match_method="histogram")
        assert req.color_match_method == "histogram"

    def test_replace_request_defaults(self):
        req = ReplaceRequest()
        assert req.expand_mask_pixels == 0
        assert req.use_color_matching is True
        assert req.use_edge_blending is True
        assert req.color_match_method == 'mean_std'
        assert req.scene_bboxes is None
        assert req.ldm.ldm_steps == 25

    def test_replace_request_invalid_method(self):
        with pytest.raises(ValidationError):
            ReplaceRequest(color_match_method="invalid")  # type: ignore

    def test_replace_request_all_color_methods(self):
        for method in ['mean_std', 'histogram', 'color_transfer']:
            req = ReplaceRequest(color_match_method=method)
            assert req.color_match_method == method

    def test_replace_request_with_ldm_fast(self):
        req = ReplaceRequest(ldm=LdmConfig(ldm_steps=10, ldm_sampler='plms', hd_strategy='CROP'))
        assert req.ldm.ldm_steps == 10

    def test_replace_request_with_ldm_custom(self):
        req = ReplaceRequest(ldm=LdmConfig(ldm_steps=30, ldm_sampler='ddim', hd_strategy='RESIZE'))
        assert req.ldm.ldm_sampler == 'ddim'
        assert req.ldm.hd_strategy == 'RESIZE'

    def test_replace_request_ldm_default_factory(self):
        req1 = ReplaceRequest()
        req2 = ReplaceRequest()
        assert req1.ldm is not req2.ldm

    def test_replace_request_with_scene_bboxes(self):
        req = ReplaceRequest(
            scene_bboxes=[
                BboxSchema(x1=0, y1=0, x2=100, y2=100),
                BboxSchema(x1=200, y1=200, x2=300, y2=300),
            ]
        )
        assert len(req.scene_bboxes) == 2

    def test_replace_request_no_color_matching(self):
        req = ReplaceRequest(use_color_matching=False)
        assert req.use_color_matching is False

    def test_replace_request_no_edge_blending(self):
        req = ReplaceRequest(use_edge_blending=False)
        assert req.use_edge_blending is False

    # --- MLResultResponse ---

    def test_ml_result_response_valid(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"time": 0.5},
            timestamp="2025-01-01T00:00:00"
        )
        assert res.result_url is not None

    def test_ml_result_response_empty_metrics(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={},
            timestamp="2025-01-01T00:00:00"
        )
        assert res.metrics == {}

    def test_ml_result_response_missing_field(self):
        with pytest.raises(ValidationError):
            MLResultResponse(
                result_url="s3://result.jpg",
                presigned_url="http://url",
                metrics={}
            )  # type: ignore

    def test_ml_result_response_metrics_nested(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"processing_time_ms": 250.5, "mask_size_pixels": 10000},
            timestamp="2025-01-01T00:00:00"
        )
        assert res.metrics["processing_time_ms"] == 250.5