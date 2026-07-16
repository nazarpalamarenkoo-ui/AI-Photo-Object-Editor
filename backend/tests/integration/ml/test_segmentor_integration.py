import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

pytest.importorskip(
    "sam2",
    reason="sam2 package not installed; skipping integration tests",
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "sam2.1_hiera_s.pt"

WEIGHTS_PATH = Path(
    os.environ.get(
        "SAM2_INTEGRATION_WEIGHTS",
        str(DEFAULT_WEIGHTS),
    )
)

if not WEIGHTS_PATH.exists():
    pytest.skip(
        f"SAM2 weights not found at '{WEIGHTS_PATH}'. "
        "Set SAM2_INTEGRATION_WEIGHTS or download the model.",
        allow_module_level=True,
    )


@pytest.fixture
def real_segmentor(tracker):
    from app.ml.segmentor import SAM2Segmentor

    return SAM2Segmentor(
        model_path=str(WEIGHTS_PATH),
        tracker=tracker,
    )


@pytest.fixture(scope="module")
def two_squares_image_bytes():
    """One image containing two well-separated, distinctly colored
    squares so each bbox prompt targets a different real object."""
    arr = np.full((256, 256, 3), 30, dtype=np.uint8)
    arr[20:100, 20:100] = (220, 60, 60)      # top-left red square
    arr[150:230, 150:230] = (60, 60, 220)    # bottom-right blue square

    buffer = BytesIO()
    PILImage.fromarray(arr).save(buffer, format="PNG")

    return buffer.getvalue()


async def test_batch_recovers_each_of_two_distinct_squares(
    real_segmentor, two_squares_image_bytes
):
    bboxes = [
        {"x1": 15, "y1": 15, "x2": 105, "y2": 105},
        {"x1": 145, "y1": 145, "x2": 235, "y2": 235},
    ]

    result = await real_segmentor.segment_with_prompts_batch(
        two_squares_image_bytes, bboxes, track_metrics=False
    )

    assert len(result["segments"]) == 2

    for seg, bbox in zip(result["segments"], bboxes):
        assert seg["bbox"]["x1"] < bbox["x2"]
        assert seg["bbox"]["x2"] > bbox["x1"]
        assert seg["bbox"]["y1"] < bbox["y2"]
        assert seg["bbox"]["y2"] > bbox["y1"]
        assert seg["prompt_bbox"] == bbox


async def test_batch_segment_keys_match_single_prompt_shape(
    real_segmentor, two_squares_image_bytes
):
    bboxes = [{"x1": 15, "y1": 15, "x2": 105, "y2": 105}]

    result = await real_segmentor.segment_with_prompts_batch(
        two_squares_image_bytes, bboxes, track_metrics=False
    )

    assert len(result["segments"]) == 1
    segment = result["segments"][0]

    for key in (
        "mask_id",
        "bbox",
        "prompt_bbox",
        "area",
        "stability_score",
        "predicted_iou",
        "mask_bytes",
    ):
        assert key in segment

    assert segment["area"] > 0
    assert isinstance(segment["mask_bytes"], bytes)


async def test_batch_is_consistent_with_individual_segment_with_prompt_calls(
    real_segmentor, two_squares_image_bytes
):
    bbox = {"x1": 15, "y1": 15, "x2": 105, "y2": 105}

    batch_result = await real_segmentor.segment_with_prompts_batch(
        two_squares_image_bytes, [bbox], track_metrics=False
    )
    single_result = await real_segmentor.segment_with_prompt(
        two_squares_image_bytes, bbox=bbox, track_metrics=False
    )

    batch_area = batch_result["segments"][0]["area"]
    single_area = single_result["segments"][0]["area"]

    assert batch_area == pytest.approx(single_area, rel=0.15)


async def test_batch_tracks_metrics_end_to_end(
    real_segmentor, two_squares_image_bytes, tracker
):
    bboxes = [
        {"x1": 15, "y1": 15, "x2": 105, "y2": 105},
        {"x1": 145, "y1": 145, "x2": 235, "y2": 235},
    ]

    await real_segmentor.segment_with_prompts_batch(
        two_squares_image_bytes, bboxes, track_metrics=True
    )

    tracker.log_run.assert_called_once()
    _, kwargs = tracker.log_run.call_args
    assert kwargs["metrics"]["num_segments"] >= 1


async def test_batch_empty_bboxes_returns_no_segments_end_to_end(
    real_segmentor, two_squares_image_bytes
):
    result = await real_segmentor.segment_with_prompts_batch(
        two_squares_image_bytes, [], track_metrics=False
    )

    assert result["segments"] == []
    assert result["metrics"]["num_segments"] == 0