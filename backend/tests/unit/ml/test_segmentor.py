import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PILImage

pytestmark = pytest.mark.unit


def _bboxes(n):
    return [
        {"x1": 10 * i, "y1": 10 * i, "x2": 10 * i + 5, "y2": 10 * i + 5}
        for i in range(n)
    ]


def _auto_mask(idx, area, stability=0.9, iou=0.9, size=16):
    seg = np.zeros((size, size), dtype=bool)
    seg[: max(area // size, 1), :] = True
    return {
        "segmentation": seg,
        "bbox": [idx, idx, 5, 5],
        "area": area,
        "stability_score": stability,
        "predicted_iou": iou,
    }

class TestConstruction:
    @pytest.mark.asyncio
    async def test_builds_model_from_registry_with_checkpoint(
        self, segmentor, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["sam_model_registry"]["vit_t"].assert_called_once_with(
            checkpoint="fake_weights/mobile_sam.pt"
        )

    @pytest.mark.asyncio
    async def test_moves_model_to_device_and_sets_eval(
        self, segmentor, fake_mobile_sam_env
    ):
        model = fake_mobile_sam_env["model_instance"]
        model.to.assert_called_once_with(device="cpu")
        model.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_predictor_and_auto_generator_wrap_same_model(
        self, segmentor, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["SamPredictor"].assert_called_once_with(
            fake_mobile_sam_env["model_instance"]
        )
        _, kwargs = fake_mobile_sam_env["SamAutomaticMaskGenerator"].call_args
        assert kwargs["model"] is fake_mobile_sam_env["model_instance"]

    @pytest.mark.asyncio
    async def test_auto_generator_receives_configured_hyperparameters(
        self, fake_mobile_sam_env, tracker
    ):
        from importlib import import_module
        mod = import_module("app.ml.segmentor")

        mod.MobileSAMSegmentor(
            model_path="w.pt",
            device="cpu",
            tracker=tracker,
            points_per_side=12,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.6,
        )
        _, kwargs = fake_mobile_sam_env["SamAutomaticMaskGenerator"].call_args
        assert kwargs["points_per_side"] == 12
        assert kwargs["pred_iou_thresh"] == 0.5
        assert kwargs["stability_score_thresh"] == 0.6

    @pytest.mark.asyncio
    async def test_missing_mobile_sam_package_raises_runtime_error(
        self, monkeypatch, tracker
    ):
        from importlib import import_module, reload

        monkeypatch.setitem(sys.modules, "mobile_sam", None)
        mod = import_module("app.ml.segmentor")
        reload(mod)

        with pytest.raises(RuntimeError, match="mobile_sam not installed"):
            mod.MobileSAMSegmentor(model_path="w.pt", device="cpu", tracker=tracker)


class TestSegmentAuto:
    @pytest.mark.asyncio
    async def test_returns_segments_and_metrics_keys(self, segmentor, image_bytes):
        result = await segmentor.segment_auto(image_bytes)
        assert "segments" in result
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_calls_generate_with_rgb_array_of_image(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_auto(image_bytes)
        fake_mobile_sam_env["auto_generator_instance"].generate.assert_called_once()
        (arr,), _ = fake_mobile_sam_env["auto_generator_instance"].generate.call_args
        assert arr.shape[-1] == 3

    @pytest.mark.asyncio
    async def test_empty_masks_returns_no_segments(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["auto_generator_instance"].generate.return_value = []
        result = await segmentor.segment_auto(image_bytes)
        assert result["segments"] == []
        assert result["metrics"]["num_segments"] == 0

    @pytest.mark.asyncio
    async def test_segments_sorted_by_area_descending(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        masks = [_auto_mask(0, area=10), _auto_mask(1, area=100), _auto_mask(2, area=50)]
        fake_mobile_sam_env["auto_generator_instance"].generate.return_value = masks
        result = await segmentor.segment_auto(image_bytes)
        areas = [s["area"] for s in result["segments"]]
        assert areas == sorted(areas, reverse=True)

    @pytest.mark.asyncio
    async def test_bbox_converted_from_xywh_to_x1y1x2y2(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        mask = _auto_mask(0, area=10)
        mask["bbox"] = [5, 7, 20, 30]
        fake_mobile_sam_env["auto_generator_instance"].generate.return_value = [mask]
        result = await segmentor.segment_auto(image_bytes)
        bbox = result["segments"][0]["bbox"]
        assert bbox == {"x1": 5, "y1": 7, "x2": 25, "y2": 37}

    @pytest.mark.asyncio
    async def test_mask_bytes_decode_as_grayscale_png(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["auto_generator_instance"].generate.return_value = [
            _auto_mask(0, area=10)
        ]
        result = await segmentor.segment_auto(image_bytes)
        decoded = PILImage.open(BytesIO(result["segments"][0]["mask_bytes"]))
        assert decoded.mode == "L"

    @pytest.mark.asyncio
    async def test_tracks_metrics_by_default(
        self, segmentor, image_bytes, tracker, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["auto_generator_instance"].generate.return_value = [
            _auto_mask(0, area=10)
        ]
        await segmentor.segment_auto(image_bytes)
        tracker.log_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_track_metrics_when_disabled(
        self, segmentor, image_bytes, tracker
    ):
        await segmentor.segment_auto(image_bytes, track_metrics=False)
        tracker.log_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_inference_time_is_non_negative(self, segmentor, image_bytes):
        result = await segmentor.segment_auto(image_bytes)
        assert result["metrics"]["inference_time_ms"] >= 0


class TestSegmentWithPrompt:
    @pytest.mark.asyncio
    async def test_returns_segments_and_metrics_keys(self, segmentor, image_bytes):
        result = await segmentor.segment_with_prompt(
            image_bytes, point_coords=[(5, 5)], point_labels=[1]
        )
        assert "segments" in result
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_calls_set_image_once(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompt(
            image_bytes, point_coords=[(5, 5)], point_labels=[1]
        )
        fake_mobile_sam_env["predictor_instance"].set_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_point_defaults_to_multimask_true(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["predictor_instance"].predict.return_value = (
            np.array([np.ones((8, 8), dtype=bool)] * 3),
            np.array([0.9, 0.8, 0.7]),
            None,
        )
        await segmentor.segment_with_prompt(
            image_bytes, point_coords=[(5, 5)], point_labels=[1]
        )
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert kwargs["multimask_output"] is True

    @pytest.mark.asyncio
    async def test_bbox_prompt_defaults_to_multimask_false(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompt(
            image_bytes, bbox={"x1": 1, "y1": 1, "x2": 10, "y2": 10}
        )
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert kwargs["multimask_output"] is False

    @pytest.mark.asyncio
    async def test_multiple_points_default_to_multimask_false(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompt(
            image_bytes, point_coords=[(1, 1), (2, 2)], point_labels=[1, 0]
        )
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert kwargs["multimask_output"] is False

    @pytest.mark.asyncio
    async def test_explicit_multimask_output_overrides_default(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompt(
            image_bytes,
            bbox={"x1": 1, "y1": 1, "x2": 10, "y2": 10},
            multimask_output=True,
        )
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert kwargs["multimask_output"] is True

    @pytest.mark.asyncio
    async def test_forwards_box_array_from_bbox_dict(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        bbox = {"x1": 3, "y1": 4, "x2": 30, "y2": 40}
        await segmentor.segment_with_prompt(image_bytes, bbox=bbox)
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert np.array_equal(kwargs["box"], np.array([3, 4, 30, 40]))

    @pytest.mark.asyncio
    async def test_no_bbox_forwards_none_for_box(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompt(
            image_bytes, point_coords=[(1, 1)], point_labels=[1]
        )
        _, kwargs = fake_mobile_sam_env["predictor_instance"].predict.call_args
        assert kwargs["box"] is None

    @pytest.mark.asyncio
    async def test_segments_sorted_by_stability_score_descending(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["predictor_instance"].predict.return_value = (
            np.array([np.ones((8, 8), dtype=bool)] * 3),
            np.array([0.5, 0.9, 0.7]),
            None,
        )
        result = await segmentor.segment_with_prompt(
            image_bytes, bbox={"x1": 0, "y1": 0, "x2": 8, "y2": 8}
        )
        scores = [s["stability_score"] for s in result["segments"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_mask_is_skipped(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        fake_mobile_sam_env["predictor_instance"].predict.return_value = (
            np.array([np.zeros((8, 8), dtype=bool)]),
            np.array([0.5]),
            None,
        )
        result = await segmentor.segment_with_prompt(
            image_bytes, bbox={"x1": 0, "y1": 0, "x2": 8, "y2": 8}
        )
        assert result["segments"] == []

    @pytest.mark.asyncio
    async def test_tracks_metrics_by_default(
        self, segmentor, image_bytes, tracker
    ):
        await segmentor.segment_with_prompt(
            image_bytes, bbox={"x1": 0, "y1": 0, "x2": 8, "y2": 8}
        )
        tracker.log_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_track_metrics_when_disabled(
        self, segmentor, image_bytes, tracker
    ):
        await segmentor.segment_with_prompt(
            image_bytes,
            bbox={"x1": 0, "y1": 0, "x2": 8, "y2": 8},
            track_metrics=False,
        )
        tracker.log_run.assert_not_called()

class TestSegmentWithPromptsBatch:
    @pytest.mark.asyncio
    async def test_returns_segments_and_metrics_keys(self, segmentor, image_bytes):
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        assert "segments" in result
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_calls_encoder_exactly_once_regardless_of_bbox_count(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(5))
        assert fake_mobile_sam_env["predictor_instance"].set_image.call_count == 1

    @pytest.mark.asyncio
    async def test_calls_predict_once_per_bbox(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        bboxes = _bboxes(4)
        await segmentor.segment_with_prompts_batch(image_bytes, bboxes)
        assert fake_mobile_sam_env["predictor_instance"].predict.call_count == len(bboxes)

    @pytest.mark.asyncio
    async def test_uses_multimask_output_false_for_every_call(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))
        for call in fake_mobile_sam_env["predictor_instance"].predict.call_args_list:
            assert call.kwargs["multimask_output"] is False

    @pytest.mark.asyncio
    async def test_forwards_none_for_point_coords_and_labels(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        for call in fake_mobile_sam_env["predictor_instance"].predict.call_args_list:
            assert call.kwargs["point_coords"] is None
            assert call.kwargs["point_labels"] is None

    @pytest.mark.asyncio
    async def test_forwards_correct_box_per_prompt_in_order(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        bboxes = _bboxes(3)
        await segmentor.segment_with_prompts_batch(image_bytes, bboxes)
        calls = fake_mobile_sam_env["predictor_instance"].predict.call_args_list
        for bbox, call in zip(bboxes, calls):
            expected = np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            assert np.array_equal(call.kwargs["box"], expected)

    @pytest.mark.asyncio
    async def test_includes_prompt_bbox_reference_matching_input(
        self, segmentor, image_bytes
    ):
        bboxes = _bboxes(3)
        result = await segmentor.segment_with_prompts_batch(image_bytes, bboxes)
        for seg in result["segments"]:
            assert seg["prompt_bbox"] == bboxes[seg["mask_id"]]

    @pytest.mark.asyncio
    async def test_mask_id_matches_bbox_index_not_output_position(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        mask_zero = np.zeros((10, 10), dtype=bool)
        mask_full = np.ones((10, 10), dtype=bool)
        fake_mobile_sam_env["predictor_instance"].predict.side_effect = [
            (np.array([mask_full]), np.array([0.95]), None),
            (np.array([mask_zero]), np.array([0.10]), None),
            (np.array([mask_full]), np.array([0.80]), None),
        ]
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))
        mask_ids = [seg["mask_id"] for seg in result["segments"]]
        assert mask_ids == [0, 2]

    @pytest.mark.asyncio
    async def test_skips_prompts_whose_mask_is_entirely_empty(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        mask_zero = np.zeros((8, 8), dtype=bool)
        mask_full = np.ones((8, 8), dtype=bool)
        fake_mobile_sam_env["predictor_instance"].predict.side_effect = [
            (np.array([mask_zero]), np.array([0.5]), None),
            (np.array([mask_full]), np.array([0.9]), None),
        ]
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        assert len(result["segments"]) == 1
        assert result["segments"][0]["mask_id"] == 1

    @pytest.mark.asyncio
    async def test_area_matches_decoded_mask_pixel_count(self, segmentor, image_bytes):
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        for seg in result["segments"]:
            decoded = PILImage.open(BytesIO(seg["mask_bytes"]))
            arr = np.array(decoded)
            assert seg["area"] == int((arr > 0).sum())

    @pytest.mark.asyncio
    async def test_stability_score_equals_predicted_iou(self, segmentor, image_bytes):
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        for seg in result["segments"]:
            assert seg["stability_score"] == pytest.approx(seg["predicted_iou"])

    @pytest.mark.asyncio
    async def test_empty_bboxes_returns_no_segments_but_still_calls_set_image(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        result = await segmentor.segment_with_prompts_batch(image_bytes, [])
        assert result["segments"] == []
        assert result["metrics"]["num_segments"] == 0
        assert fake_mobile_sam_env["predictor_instance"].set_image.call_count == 1
        fake_mobile_sam_env["predictor_instance"].predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_results_are_independent_across_calls(self, segmentor, image_bytes):
        first = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        first["segments"].clear()
        second = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        assert len(second["segments"]) == 2

    @pytest.mark.asyncio
    async def test_tracks_metrics_by_default(self, segmentor, image_bytes, tracker):
        await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(2))
        tracker.log_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_track_metrics_when_disabled(
        self, segmentor, image_bytes, tracker
    ):
        await segmentor.segment_with_prompts_batch(
            image_bytes, _bboxes(2), track_metrics=False
        )
        tracker.log_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_metrics_num_segments_reflects_skipped_entries(
        self, segmentor, image_bytes, fake_mobile_sam_env
    ):
        mask_zero = np.zeros((8, 8), dtype=bool)
        mask_full = np.ones((8, 8), dtype=bool)
        fake_mobile_sam_env["predictor_instance"].predict.side_effect = [
            (np.array([mask_full]), np.array([0.9]), None),
            (np.array([mask_zero]), np.array([0.1]), None),
            (np.array([mask_full]), np.array([0.8]), None),
        ]
        result = await segmentor.segment_with_prompts_batch(image_bytes, _bboxes(3))
        assert result["metrics"]["num_segments"] == 2
        assert result["metrics"]["num_segments"] == len(result["segments"])


class TestCalculateMetrics:
    def test_empty_segments_returns_zeroed_metrics(self, segmentor):
        metrics = segmentor._calculate_metrics([], inference_time_ms=12.5)
        assert metrics["num_segments"] == 0
        assert metrics["avg_stability"] == 0.0
        assert metrics["inference_time_ms"] == 12.5
        assert metrics["inference_time_s"] == pytest.approx(0.0125)
        assert "total_area_px" not in metrics

    def test_non_empty_segments_computes_average_stability_and_total_area(
        self, segmentor
    ):
        segments = [
            {"stability_score": 0.8, "area": 100},
            {"stability_score": 0.6, "area": 50},
        ]
        metrics = segmentor._calculate_metrics(segments, inference_time_ms=40.0)
        assert metrics["num_segments"] == 2
        assert metrics["avg_stability"] == pytest.approx(0.7)
        assert metrics["total_area_px"] == 150
        assert metrics["inference_time_s"] == pytest.approx(0.04)


class TestGetSegmentorSingleton:
    def test_returns_same_instance_across_calls(
        self, fake_mobile_sam_env, monkeypatch
    ):
        from importlib import import_module, reload

        mod = import_module("app.ml.segmentor")
        reload(mod)
        monkeypatch.setattr(mod, "_segmentor_instance", None)
        monkeypatch.setattr(mod.DeviceManager, "get", lambda *_a, **_kw: "cpu")

        first = mod.get_segmentor()
        second = mod.get_segmentor()
        assert first is second
        # only constructed the underlying model registry entry once
        fake_mobile_sam_env["sam_model_registry"]["vit_t"].assert_called_once()