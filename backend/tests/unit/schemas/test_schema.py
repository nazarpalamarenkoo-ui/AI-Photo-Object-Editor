import pytest
from datetime import datetime
from pydantic import ValidationError

from app.db.schemas.ml import (
    BboxSchema,
    LdmConfig,
    DetectRequest,
    RemoveRequest,
    RemoveMultipleRequest,
    ReplaceRequest,
    SegmentRequest,
    SegmentWithPromptRequest,
    SamRemoveRequest,
    SamReplaceRequest,
    ExtractRequest,
    PasteRequest,
    MLResultResponse,
    SegmentInfo,
    SegmentResponse,
    SegmentByPolygonRequest,
    ExtractResponse,
    PasteResponse,
    AssetResponse,
    RenameAssetRequest,
)


@pytest.mark.unit
class TestBboxSchema:
    def test_bbox_valid(self):
        bbox = BboxSchema(x1=10, y1=20, x2=100, y2=200)
        assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10, 20, 100, 200)

    def test_bbox_zero_coords(self):
        assert BboxSchema(x1=0, y1=0, x2=0, y2=0).x1 == 0

    def test_bbox_negative_coords_allowed(self):
        bbox = BboxSchema(x1=-10, y1=-20, x2=100, y2=200)
        assert bbox.x1 == -10

    def test_bbox_missing_field(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1=10, y1=20, x2=100)  # type: ignore

    def test_bbox_wrong_type(self):
        with pytest.raises(ValidationError):
            BboxSchema(x1="a", y1=20, x2=100, y2=200)  # type: ignore


@pytest.mark.unit
class TestLdmConfig:
    def test_defaults(self):
        cfg = LdmConfig()
        assert cfg.ldm_steps == 25
        assert cfg.ldm_sampler == "plms"
        assert cfg.hd_strategy == "CROP"

    def test_custom_valid(self):
        cfg = LdmConfig(ldm_steps=10, ldm_sampler="ddim", hd_strategy="RESIZE")
        assert (cfg.ldm_steps, cfg.ldm_sampler, cfg.hd_strategy) == (10, "ddim", "RESIZE")

    def test_hd_strategy_original(self):
        assert LdmConfig(hd_strategy="ORIGINAL").hd_strategy == "ORIGINAL"

    def test_steps_boundary_min(self):
        assert LdmConfig(ldm_steps=5).ldm_steps == 5

    def test_steps_boundary_max(self):
        assert LdmConfig(ldm_steps=50).ldm_steps == 50

    def test_steps_below_min_invalid(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_steps=4)

    def test_steps_above_max_invalid(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_steps=51)

    def test_invalid_sampler(self):
        with pytest.raises(ValidationError):
            LdmConfig(ldm_sampler="euler")  # type: ignore

    def test_invalid_hd_strategy(self):
        with pytest.raises(ValidationError):
            LdmConfig(hd_strategy="INVALID")  # type: ignore


@pytest.mark.unit
class TestDetectRequest:
    def test_valid(self):
        req = DetectRequest(conf_threshold=0.7, classes=["person"])
        assert req.conf_threshold == 0.7
        assert req.classes == ["person"]

    def test_defaults(self):
        req = DetectRequest()
        assert req.conf_threshold == 0.5
        assert req.classes is None

    def test_threshold_above_one_invalid(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=1.5)

    def test_threshold_below_zero_invalid(self):
        with pytest.raises(ValidationError):
            DetectRequest(conf_threshold=-0.1)

    def test_threshold_boundaries(self):
        assert DetectRequest(conf_threshold=0.0).conf_threshold == 0.0
        assert DetectRequest(conf_threshold=1.0).conf_threshold == 1.0

    def test_empty_classes_list(self):
        assert DetectRequest(classes=[]).classes == []


@pytest.mark.unit
class TestRemoveRequest:
    def test_defaults(self):
        req = RemoveRequest()
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is False
        assert req.ldm.ldm_steps == 25

    def test_with_custom_ldm_fields(self):
        req = RemoveRequest(ldm_steps=10, ldm_sampler="ddim", hd_strategy="RESIZE")
        assert req.ldm.ldm_steps == 10
        assert req.ldm.ldm_sampler == "ddim"
        assert req.ldm.hd_strategy == "RESIZE"

    def test_ldm_property_returns_new_instance_each_time(self):
        req = RemoveRequest()
        assert req.ldm is not req.ldm

    def test_expand_mask_pixels_boundaries(self):
        assert RemoveRequest(expand_mask_pixels=0).expand_mask_pixels == 0
        assert RemoveRequest(expand_mask_pixels=50).expand_mask_pixels == 50

    def test_expand_mask_pixels_above_max_invalid(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=51)

    def test_expand_mask_pixels_below_min_invalid(self):
        with pytest.raises(ValidationError):
            RemoveRequest(expand_mask_pixels=-1)

    def test_edge_blending_true(self):
        assert RemoveRequest(use_edge_blending=True).use_edge_blending is True

    def test_ldm_steps_below_min_invalid(self):
        with pytest.raises(ValidationError):
            RemoveRequest(ldm_steps=4)

    def test_invalid_sampler_invalid(self):
        with pytest.raises(ValidationError):
            RemoveRequest(ldm_sampler="euler")  # type: ignore


@pytest.mark.unit
class TestRemoveMultipleRequest:
    def test_valid(self):
        req = RemoveMultipleRequest(bbox_ids=[1, 2, 3])
        assert len(req.bbox_ids) == 3

    def test_empty_bbox_ids_invalid(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest(bbox_ids=[])

    def test_single_id(self):
        assert RemoveMultipleRequest(bbox_ids=[42]).bbox_ids == [42]

    def test_defaults(self):
        req = RemoveMultipleRequest(bbox_ids=[1])
        assert req.expand_mask_pixels == 5
        assert req.use_edge_blending is False
        assert req.ldm.ldm_steps == 25

    def test_missing_bbox_ids_invalid(self):
        with pytest.raises(ValidationError):
            RemoveMultipleRequest()  # type: ignore

    def test_custom_ldm_fields_reflected_in_property(self):
        req = RemoveMultipleRequest(bbox_ids=[1, 2], ldm_steps=15, ldm_sampler="ddim")
        assert req.ldm.ldm_steps == 15
        assert req.ldm.ldm_sampler == "ddim"


@pytest.mark.unit
class TestReplaceRequest:
    def test_defaults(self):
        req = ReplaceRequest()
        assert req.expand_mask_pixels == 0
        assert req.use_color_matching is False
        assert req.use_edge_blending is False
        assert req.color_match_method == "mean_std"

    def test_custom_method(self):
        assert ReplaceRequest(color_match_method="histogram").color_match_method == "histogram"

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            ReplaceRequest(color_match_method="invalid")  # type: ignore

    def test_all_valid_methods(self):
        for method in ["mean_std", "histogram", "color_transfer"]:
            assert ReplaceRequest(color_match_method=method).color_match_method == method

    def test_color_matching_true(self):
        assert ReplaceRequest(use_color_matching=True).use_color_matching is True

    def test_expand_mask_pixels_above_max_invalid(self):
        with pytest.raises(ValidationError):
            ReplaceRequest(expand_mask_pixels=51)


@pytest.mark.unit
class TestSegmentRequest:
    def test_defaults(self):
        req = SegmentRequest()
        assert req.min_area == 500
        assert req.max_segments == 50

    def test_min_area_negative_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(min_area=-1)

    def test_min_area_zero_valid(self):
        assert SegmentRequest(min_area=0).min_area == 0

    def test_max_segments_below_min_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=0)

    def test_max_segments_above_max_invalid(self):
        with pytest.raises(ValidationError):
            SegmentRequest(max_segments=201)

    def test_max_segments_boundaries(self):
        assert SegmentRequest(max_segments=1).max_segments == 1
        assert SegmentRequest(max_segments=200).max_segments == 200


@pytest.mark.unit
class TestSegmentWithPromptRequest:
    def test_all_optional_defaults(self):
        req = SegmentWithPromptRequest()
        assert req.point_coords is None
        assert req.point_labels is None
        assert req.bbox is None
        assert req.multimask_output is None

    def test_with_points(self):
        req = SegmentWithPromptRequest(point_coords=[(10, 20), (30, 40)], point_labels=[1, 0])
        assert req.point_coords == [(10, 20), (30, 40)]
        assert req.point_labels == [1, 0]

    def test_with_bbox(self):
        req = SegmentWithPromptRequest(bbox=BboxSchema(x1=0, y1=0, x2=50, y2=50))
        assert req.bbox.x2 == 50

    def test_multimask_output_flag(self):
        assert SegmentWithPromptRequest(multimask_output=True).multimask_output is True


@pytest.mark.unit
class TestSamRemoveRequest:
    def test_defaults(self):
        req = SamRemoveRequest()
        assert req.expand_mask_pixels == 12
        assert req.use_edge_blending is False
        assert req.ldm.ldm_steps == 25

    def test_expand_mask_pixels_above_max_invalid(self):
        with pytest.raises(ValidationError):
            SamRemoveRequest(expand_mask_pixels=51)

    def test_expand_mask_pixels_below_min_invalid(self):
        with pytest.raises(ValidationError):
            SamRemoveRequest(expand_mask_pixels=-1)

    def test_custom_ldm_reflected(self):
        req = SamRemoveRequest(ldm_steps=30, hd_strategy="ORIGINAL")
        assert req.ldm.ldm_steps == 30
        assert req.ldm.hd_strategy == "ORIGINAL"


@pytest.mark.unit
class TestSamReplaceRequest:
    def test_defaults(self):
        req = SamReplaceRequest()
        assert req.expand_mask_pixels == 8
        assert req.use_color_matching is False
        assert req.use_edge_blending is False
        assert req.color_match_method == "color_transfer"
        assert req.ldm_steps == 25
        assert req.ldm_sampler == "plms"
        assert req.hd_strategy == "CROP"

    def test_color_match_method_is_plain_str_no_validation(self):
        req = SamReplaceRequest(color_match_method="anything_goes")
        assert req.color_match_method == "anything_goes"

    def test_expand_mask_pixels_unconstrained(self):
        req = SamReplaceRequest(expand_mask_pixels=999)
        assert req.expand_mask_pixels == 999

    def test_expand_mask_pixels_negative_allowed(self):
        # plain int field with no ge/le constraint
        req = SamReplaceRequest(expand_mask_pixels=-50)
        assert req.expand_mask_pixels == -50

    def test_ldm_property_builds_ldm_config(self):
        req = SamReplaceRequest(ldm_steps=10, ldm_sampler="ddim", hd_strategy="RESIZE")
        cfg = req.ldm
        assert isinstance(cfg, LdmConfig)
        assert (cfg.ldm_steps, cfg.ldm_sampler, cfg.hd_strategy) == (10, "ddim", "RESIZE")

    def test_ldm_steps_field_itself_is_constrained(self):
        # ldm_steps field on SamReplaceRequest DOES have ge/le constraints
        with pytest.raises(ValidationError):
            SamReplaceRequest(ldm_steps=4)


@pytest.mark.unit
class TestExtractRequest:
    def test_default(self):
        req = ExtractRequest()
        assert req.padding_pixels == 8
        assert req.label is None
        assert req.persist_to_s3 is False

    def test_padding_pixels_unconstrained(self):
        req = ExtractRequest(padding_pixels=65)
        assert req.padding_pixels == 65

    def test_padding_pixels_negative_allowed(self):
        req = ExtractRequest(padding_pixels=-5)
        assert req.padding_pixels == -5

    def test_with_label_and_persist(self):
        req = ExtractRequest(label="cat", persist_to_s3=True)
        assert req.label == "cat"
        assert req.persist_to_s3 is True


@pytest.mark.unit
class TestPasteRequest:
    def test_valid_with_extracted_url(self):
        req = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )
        assert req.scale == 1.0
        assert req.color_match_method == "color_transfer"

    def test_valid_with_asset_id(self):
        req = PasteRequest(
            asset_id="abc-123",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )
        assert req.asset_id == "abc-123"

    def test_scale_unconstrained(self):
        req = PasteRequest(
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
            scale=3.5,
        )
        assert req.scale == 3.5

    def test_missing_target_bbox_invalid(self):
        with pytest.raises(ValidationError):
            PasteRequest(extracted_url="s3://obj.png")  # type: ignore

    def test_missing_both_source_fields_invalid(self):
        with pytest.raises(ValidationError):
            PasteRequest(target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10))

    def test_both_source_fields_present_is_valid(self):
        req = PasteRequest(
            asset_id="abc-123",
            extracted_url="s3://obj.png",
            target_bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
        )
        assert req.asset_id == "abc-123"
        assert req.extracted_url == "s3://obj.png"


@pytest.mark.unit
class TestMLResultResponse:
    def test_valid(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={"time": 0.5},
            timestamp=datetime(2025, 1, 1),
        )
        assert res.result_url == "s3://result.jpg"

    def test_empty_metrics(self):
        res = MLResultResponse(
            result_url="s3://result.jpg",
            presigned_url="http://url",
            metrics={},
            timestamp=datetime(2025, 1, 1),
        )
        assert res.metrics == {}

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            MLResultResponse(
                result_url="s3://result.jpg",
                presigned_url="http://url",
                metrics={},
            )  # type: ignore


@pytest.mark.unit
class TestSegmentInfoAndResponse:
    def test_segment_info_valid(self):
        info = SegmentInfo(
            mask_id=1, bbox_id=2,
            bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
            area=100, stability_score=0.95,
        )
        assert info.area == 100
        assert info.stability_score == 0.95

    def test_segment_info_stability_score_optional(self):
        info = SegmentInfo(
            mask_id=1, bbox_id=2,
            bbox=BboxSchema(x1=0, y1=0, x2=10, y2=10),
            area=100,
        )
        assert info.stability_score is None

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
            segments=[], metrics={}, image_size=(100, 100),
            timestamp=datetime(2025, 1, 1),
        )
        assert resp.segments == []


@pytest.mark.unit
class TestSegmentByPolygonRequest:
    def test_valid_triangle(self):
        req = SegmentByPolygonRequest(points=[(0, 0), (10, 0), (5, 10)])
        assert len(req.points) == 3

    def test_defaults(self):
        req = SegmentByPolygonRequest(points=[(0, 0), (10, 0), (5, 10)])
        assert req.smooth is True
        assert req.smoothing_factor == 0.0
        assert req.feather_px == 0

    def test_fewer_than_three_points_invalid(self):
        with pytest.raises(ValidationError):
            SegmentByPolygonRequest(points=[(0, 0), (10, 0)])

    def test_custom_values(self):
        req = SegmentByPolygonRequest(
            points=[(0, 0), (10, 0), (5, 10)],
            smooth=False, smoothing_factor=0.5, feather_px=3,
        )
        assert req.smooth is False
        assert req.smoothing_factor == 0.5
        assert req.feather_px == 3


@pytest.mark.unit
class TestExtractResponse:
    def test_valid_minimal(self):
        resp = ExtractResponse(
            asset_id="asset-1",
            object_size=(50, 60),
            area_pixels=3000,
            cropped_bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 60},
            timestamp="2025-01-01T00:00:00",
        )
        assert resp.object_size == (50, 60)
        assert resp.area_pixels == 3000
        assert resp.extracted_url is None
        assert resp.presigned_url is None

    def test_with_urls(self):
        resp = ExtractResponse(
            asset_id="asset-1",
            extracted_url="s3://obj.png",
            presigned_url="http://url",
            object_size=(50, 60),
            area_pixels=3000,
            cropped_bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 60},
            timestamp="2025-01-01T00:00:00",
        )
        assert resp.extracted_url == "s3://obj.png"
        assert resp.presigned_url == "http://url"

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            ExtractResponse(
                object_size=(50, 60),
                area_pixels=3000,
                cropped_bbox={},
                timestamp="2025-01-01T00:00:00",
            )  # type: ignore


@pytest.mark.unit
class TestPasteResponse:
    def test_valid(self):
        resp = PasteResponse(
            result_url="s3://result.png",
            presigned_url="http://url",
            paste_bbox=BboxSchema(x1=0, y1=0, x2=20, y2=20),
            object_size=(20, 20),
            timestamp=datetime(2025, 1, 1),
        )
        assert resp.paste_bbox.x2 == 20
        assert resp.object_size == (20, 20)

    def test_missing_field_invalid(self):
        with pytest.raises(ValidationError):
            PasteResponse(
                result_url="s3://result.png",
                presigned_url="http://url",
                paste_bbox=BboxSchema(x1=0, y1=0, x2=20, y2=20),
                timestamp=datetime(2025, 1, 1),
            )  # type: ignore


@pytest.mark.unit
class TestAssetResponse:
    def test_valid_minimal(self):
        resp = AssetResponse(
            asset_id="asset-1",
            source_image_id=1,
            object_size=(50, 60),
            area_pixels=3000,
            created_at="2025-01-01T00:00:00",
        )
        assert resp.label is None
        assert resp.s3_url is None

    def test_valid_full(self):
        resp = AssetResponse(
            asset_id="asset-1",
            source_image_id=1,
            object_size=(50, 60),
            area_pixels=3000,
            label="cat",
            s3_url="s3://obj.png",
            created_at="2025-01-01T00:00:00",
        )
        assert resp.label == "cat"
        assert resp.s3_url == "s3://obj.png"

    def test_missing_required_field_invalid(self):
        with pytest.raises(ValidationError):
            AssetResponse(
                asset_id="asset-1",
                source_image_id=1,
                object_size=(50, 60),
                area_pixels=3000,
            )  # type: ignore


@pytest.mark.unit
class TestRenameAssetRequest:
    def test_valid(self):
        req = RenameAssetRequest(label="new-name")
        assert req.label == "new-name"

    def test_missing_label_invalid(self):
        with pytest.raises(ValidationError):
            RenameAssetRequest()  # type: ignore