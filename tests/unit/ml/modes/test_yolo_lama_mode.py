import pytest
from unittest.mock import MagicMock, AsyncMock
from PIL import Image
from io import BytesIO
import numpy as np

from app.ml.modes.yolo_lama_mode import YoloLamaMode, _normalize_size



def make_image(size=(128, 128), color=(255, 255, 255)):
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def make_bbox():
    return {"x1": 10, "y1": 10, "x2": 60, "y2": 60}


@pytest.mark.unit
def test_init_dependencies():
    detector = MagicMock()
    inpainter = MagicMock()

    mode = YoloLamaMode(
        detector=detector,
        inpainter=inpainter,
        edge_blender=MagicMock(),
        color_matcher=MagicMock(),
        background_remover=MagicMock(),
        device="cpu"
    )

    assert mode.detector is detector
    assert mode.inpainter is inpainter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_empty():
    detector = MagicMock()
    detector.detect = AsyncMock(return_value={"detections": [], "metrics": {}})

    mode = YoloLamaMode(detector, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    result = await mode.detect_objects(make_image())

    assert result["detections"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_objects_bbox_id_increment():
    detector = MagicMock()
    detector.detect = AsyncMock(return_value={
        "detections": [{"x1":0,"y1":0,"x2":10,"y2":10,"confidence":0.9} for _ in range(5)],
        "metrics": {}
    })

    mode = YoloLamaMode(detector, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    result = await mode.detect_objects(make_image())

    ids = [d["bbox_id"] for d in result["detections"]]
    assert ids == list(range(5))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mask_multiple_overlap():
    mode = YoloLamaMode(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

    img = make_image((100, 100))

    bboxes = [
        {"x1":10,"y1":10,"x2":50,"y2":50},
        {"x1":30,"y1":30,"x2":80,"y2":80},
    ]

    mask_bytes = await mode._create_combined_mask(img, bboxes)
    mask = np.array(Image.open(BytesIO(mask_bytes)))
    
    assert mask[40, 40] == 255
    assert mask[15, 15] == 255
    assert mask[90, 90] == 0

@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_size_changes_only_when_needed():
    a = make_image((50, 50))
    b = make_image((100, 100))

    out = await _normalize_size(a, b)
    assert Image.open(BytesIO(out)).size == (100, 100)

    out2 = await _normalize_size(b, b)
    assert Image.open(BytesIO(out2)).size == (100, 100)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_pipeline_full_flow():
    detector = MagicMock()
    inpainter = MagicMock()
    remover = MagicMock()
    color = MagicMock()

    inpainter.inpaint = AsyncMock(return_value={
        "result_bytes": make_image(),
        "metrics": {"ok": True}
    })

    remover.remove_and_resize = AsyncMock(return_value=make_image())
    color.match_against_original = AsyncMock(return_value=make_image())

    mode = YoloLamaMode(detector, inpainter, MagicMock(), color, remover)

    mode._create_remove_mask = AsyncMock(return_value=make_image())
    mode.compositor = MagicMock()
    mode.compositor.compose = MagicMock(return_value=make_image())

    res = await mode.replace_object(
        image_bytes=make_image(),
        selected_bbox=make_bbox(),
        replacement_image_bytes=make_image(),
        use_color_matching=True
    )

    assert res["metrics"]["ok"] is True

@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_without_color_matching():
    mode = YoloLamaMode(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

    mode.background_remover.remove_and_resize = AsyncMock(return_value=make_image())
    mode._create_remove_mask = AsyncMock(return_value=make_image())
    mode.inpainter.inpaint = AsyncMock(return_value={
        "result_bytes": make_image(),
        "metrics": {}
    })
    mode.compositor = MagicMock()
    mode.compositor.compose = MagicMock(return_value=make_image())

    res = await mode.replace_object(
        image_bytes=make_image(),
        selected_bbox=make_bbox(),
        replacement_image_bytes=make_image(),
        use_color_matching=False
    )

    assert "result_bytes" in res

@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_invalid_bbox():
    mode = YoloLamaMode(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

    with pytest.raises(Exception):
        await mode.replace_object(
            image_bytes=make_image(),
            selected_bbox=None,
            replacement_image_bytes=make_image()
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_object_inpainter_failure():
    inpainter = MagicMock()
    inpainter.inpaint = AsyncMock(side_effect=RuntimeError("fail"))

    mode = YoloLamaMode(MagicMock(), inpainter, MagicMock(), MagicMock(), MagicMock())

    mode.background_remover.remove_and_resize = AsyncMock(return_value=make_image())
    mode._create_remove_mask = AsyncMock(return_value=make_image())

    with pytest.raises(RuntimeError):
        await mode.replace_object(
            image_bytes=make_image(),
            selected_bbox=make_bbox(),
            replacement_image_bytes=make_image()
        )