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
        device="cpu",
        tracker=tracker,
    )


@pytest.fixture(scope="module")
def real_image_bytes():
    arr = np.full((256, 256, 3), 30, dtype=np.uint8)
    arr[64:192, 64:192] = (220, 60, 60)

    buffer = BytesIO()
    PILImage.fromarray(arr).save(buffer, format="PNG")

    return buffer.getvalue()


async def test_segment_auto_finds_at_least_one_segment(
    real_segmentor,
    real_image_bytes,
):
    result = await real_segmentor.segment_auto(
        real_image_bytes,
        track_metrics=False,
    )

    assert len(result["segments"]) >= 1

    segment = result["segments"][0]

    for key in (
        "mask_id",
        "bbox",
        "area",
        "stability_score",
        "predicted_iou",
        "mask_bytes",
    ):
        assert key in segment

    assert segment["area"] > 0
    assert isinstance(segment["mask_bytes"], bytes)


async def test_segment_auto_segments_sorted_by_area_desc(
    real_segmentor,
    real_image_bytes,
):
    result = await real_segmentor.segment_auto(
        real_image_bytes,
        track_metrics=False,
    )

    areas = [segment["area"] for segment in result["segments"]]

    assert areas == sorted(areas, reverse=True)


async def test_segment_with_prompt_bbox_recovers_square(
    real_segmentor,
    real_image_bytes,
):
    bbox = {
        "x1": 60,
        "y1": 60,
        "x2": 196,
        "y2": 196,
    }

    result = await real_segmentor.segment_with_prompt(
        real_image_bytes,
        bbox=bbox,
        track_metrics=False,
    )

    assert len(result["segments"]) >= 1

    best = result["segments"][0]

    assert best["bbox"]["x1"] < bbox["x2"]
    assert best["bbox"]["x2"] > bbox["x1"]
    assert best["bbox"]["y1"] < bbox["y2"]
    assert best["bbox"]["y2"] > bbox["y1"]


async def test_segment_with_prompt_point_inside_square(
    real_segmentor,
    real_image_bytes,
):
    result = await real_segmentor.segment_with_prompt(
        real_image_bytes,
        point_coords=[(128, 128)],
        point_labels=[1],
        track_metrics=False,
    )

    assert len(result["segments"]) >= 1

    best = result["segments"][0]

    assert best["bbox"]["x1"] <= 128 <= best["bbox"]["x2"]
    assert best["bbox"]["y1"] <= 128 <= best["bbox"]["y2"]


async def test_segment_auto_tracks_metrics_end_to_end(
    real_segmentor,
    real_image_bytes,
    tracker,
):
    await real_segmentor.segment_auto(
        real_image_bytes,
        track_metrics=True,
    )

    tracker.log_metrics.assert_called_once()

    payload = tracker.log_metrics.call_args.args[0]

    assert payload["sam2_num_segments"] >= 1