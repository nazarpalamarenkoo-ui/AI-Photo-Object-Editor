import sys
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.diffuser_inpainter import DiffusionReplacer, get_diffusion_replacer


@pytest.fixture(autouse=True)
def mock_torch(monkeypatch):
    """Fresh, test-scoped `torch` mock. Reverted automatically after the
    test by monkeypatch — never leaks into other test files."""
    m = MagicMock()
    m.float32 = "float32"
    m.no_grad.return_value = nullcontext()
    m.isnan.return_value.any.return_value = False
    monkeypatch.setitem(sys.modules, 'torch', m)
    return m


@pytest.fixture(autouse=True)
def mock_diffusers(monkeypatch):
    """Fresh, test-scoped `diffusers` mock. Reverted automatically after
    the test by monkeypatch."""
    m = MagicMock()
    monkeypatch.setitem(sys.modules, 'diffusers', m)
    return m


def _make_image_bytes(width=200, height=200, color='blue', fmt='PNG'):
    img = Image.new('RGB', (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

@pytest.mark.unit
def test_init_defaults_when_settings_missing_attrs():
    """getattr(settings, "X", default) should fall back to the hardcoded
    defaults when the settings object doesn't define the attribute."""
    with patch('app.ml.diffuser_inpainter.settings', SimpleNamespace()):
        r = DiffusionReplacer(device="cpu", tracker=MagicMock())

    assert r.model_id == "stable-diffusion-v1-5/stable-diffusion-inpainting"
    assert r.ip_adapter_repo == "h94/IP-Adapter"
    assert r.ip_adapter_variant == "plus"
    assert r.ip_adapter_scale == 0.8
    assert r.default_steps == 30
    assert r.default_guidance_scale == 5.5
    assert r.default_strength == 0.7
    assert r.work_resolution == 640
    assert r.crop_padding_ratio == 0.35
    assert r.min_crop_size == 256
    assert r.mask_blur_radius == 6
    assert r.enable_cpu_offload is False


@pytest.mark.unit
def test_init_uses_settings_overrides_when_present():
    fake_settings = SimpleNamespace(
        DIFFUSION_WORK_RESOLUTION=512,
        DIFFUSION_STEPS=15,
        DIFFUSION_GUIDANCE_SCALE=4.0,
        IP_ADAPTER_VARIANT="regular",
        DIFFUSION_ENABLE_CPU_OFFLOAD=False,
    )
    with patch('app.ml.diffuser_inpainter.settings', fake_settings):
        r = DiffusionReplacer(device="cuda", tracker=MagicMock())

    assert r.work_resolution == 512
    assert r.default_steps == 15
    assert r.default_guidance_scale == 4.0
    assert r.ip_adapter_variant == "regular"
    assert r.enable_cpu_offload is False


@pytest.mark.unit
def test_init_stores_provided_tracker():
    tracker = MagicMock()
    r = DiffusionReplacer(device="cpu", tracker=tracker)
    assert r.tracker is tracker
    assert r.device == "cpu"
    assert r._pipe is None


@pytest.mark.unit
@pytest.mark.parametrize("x,expected", [
    (0, 8), (1, 8), (4, 8), (5, 8), (12, 16), (100, 96), (640, 640), (639, 640),
])
def test_round_to_multiple_of_8(x, expected):
    assert DiffusionReplacer._round_to_multiple_of_8(x) == expected


@pytest.mark.unit
def test_round_to_multiple_of_8_never_below_8():
    assert DiffusionReplacer._round_to_multiple_of_8(0) == 8
    assert DiffusionReplacer._round_to_multiple_of_8(2) == 8



@pytest.mark.unit
def test_get_crop_bbox_no_padding_no_min_size():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 150:250] = 255  # rows 100-199, cols 150-249
    bbox = DiffusionReplacer._get_crop_bbox(mask, padding_ratio=0.0, min_size=0)
    # w = 249-150 = 99, h = 199-100 = 99; pad = max(int(99*0), 1) = 1
    assert bbox == (149, 99, 250, 200)


@pytest.mark.unit
def test_get_crop_bbox_applies_padding():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 150:250] = 255  # w=99, h=99
    x0, y0, x1, y1 = DiffusionReplacer._get_crop_bbox(mask, padding_ratio=0.5, min_size=0)
    # pad = int(99 * 0.5) = 49
    assert (x0, y0, x1, y1) == (101, 51, 298, 248)


@pytest.mark.unit
def test_get_crop_bbox_enforces_min_crop_size():
    # Placed away from image edges so the final clamp-to-bounds step
    # doesn't shrink the crop back down after the min-size expansion.
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:210, 150:160] = 255  # tiny 10x10 region, centered vertically
    x0, y0, x1, y1 = DiffusionReplacer._get_crop_bbox(mask, padding_ratio=0.0, min_size=256)
    assert (x1 - x0) >= 256
    assert (y1 - y0) >= 256


@pytest.mark.unit
def test_get_crop_bbox_clamps_to_image_bounds():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 90:100] = 255  # near top-right corner
    x0, y0, x1, y1 = DiffusionReplacer._get_crop_bbox(mask, padding_ratio=0.35, min_size=256)
    assert 0 <= x0 <= x1 <= 100
    assert 0 <= y0 <= y1 <= 100


@pytest.mark.unit
def test_get_crop_bbox_empty_mask_raises():
    mask = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="Empty mask"):
        DiffusionReplacer._get_crop_bbox(mask, padding_ratio=0.35, min_size=256)


@pytest.mark.unit
def test_feather_mask_zero_radius_is_noop():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    result = DiffusionReplacer._feather_mask(mask, 0)
    np.testing.assert_array_equal(result, mask)


@pytest.mark.unit
def test_feather_mask_softens_hard_edges():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[15:45, 15:45] = 255
    result = DiffusionReplacer._feather_mask(mask, 6)
    assert result.shape == mask.shape
    assert not set(np.unique(result)).issubset({0, 255})


@pytest.mark.unit
def test_feather_mask_interior_and_exterior_stay_saturated():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[15:45, 15:45] = 255
    result = DiffusionReplacer._feather_mask(mask, 6)
    assert result[30, 30] == 255  # deep interior unaffected
    assert result[2, 2] == 0      # far exterior unaffected


@pytest.mark.unit
def test_load_reference_image_center_crops_to_square_and_resizes():
    img = Image.new('RGB', (300, 200), color='blue')
    buf = BytesIO()
    img.save(buf, format='PNG')
    result = DiffusionReplacer._load_reference_image(buf.getvalue())
    assert result.size == (224, 224)
    assert result.mode == 'RGB'


@pytest.mark.unit
def test_load_reference_image_respects_custom_target_size():
    img = Image.new('RGB', (100, 100), color='green')
    buf = BytesIO()
    img.save(buf, format='PNG')
    result = DiffusionReplacer._load_reference_image(buf.getvalue(), target_size=64)
    assert result.size == (64, 64)


@pytest.mark.unit
def test_load_reference_image_flattens_rgba_onto_white():
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    buf = BytesIO()
    img.save(buf, format='PNG')
    result = DiffusionReplacer._load_reference_image(buf.getvalue())
    assert result.mode == 'RGB'
    assert result.size == (224, 224)


@pytest.mark.unit
def test_load_reference_image_handles_palette_with_transparency():
    img = Image.new('P', (80, 80))
    img.info['transparency'] = 0
    buf = BytesIO()
    img.save(buf, format='PNG')
    result = DiffusionReplacer._load_reference_image(buf.getvalue())
    assert result.mode == 'RGB'


def _stub_diffusers_pipe(mock_diffusers):
    fake_pipe = MagicMock()
    mock_diffusers.AutoPipelineForInpainting.from_pretrained.return_value = fake_pipe
    mock_diffusers.DPMSolverMultistepScheduler.from_config.return_value = MagicMock()
    return fake_pipe


@pytest.mark.unit
def test_load_pipe_caches_instance_after_first_call(mock_diffusers):
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    _stub_diffusers_pipe(mock_diffusers)

    pipe1 = r._load_pipe()
    pipe2 = r._load_pipe()

    assert pipe1 is pipe2
    mock_diffusers.AutoPipelineForInpainting.from_pretrained.assert_called_once()


@pytest.mark.unit
def test_load_pipe_selects_plus_ip_adapter_weights(mock_diffusers):
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    r.ip_adapter_variant = "plus"
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    r._load_pipe()

    kwargs = fake_pipe.load_ip_adapter.call_args.kwargs
    assert kwargs['weight_name'] == 'ip-adapter-plus_sd15.bin'


@pytest.mark.unit
def test_load_pipe_selects_regular_ip_adapter_weights(mock_diffusers):
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    r.ip_adapter_variant = "regular"
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    r._load_pipe()

    kwargs = fake_pipe.load_ip_adapter.call_args.kwargs
    assert kwargs['weight_name'] == 'ip-adapter_sd15.bin'


@pytest.mark.unit
def test_load_pipe_sets_initial_ip_adapter_scale(mock_diffusers):
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    r.ip_adapter_scale = 0.42
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    r._load_pipe()

    fake_pipe.set_ip_adapter_scale.assert_called_once_with(0.42)


@pytest.mark.unit
def test_load_pipe_configures_dpm_scheduler(mock_diffusers):
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    original_scheduler_config = fake_pipe.scheduler.config
    fake_scheduler = mock_diffusers.DPMSolverMultistepScheduler.from_config.return_value

    r._load_pipe()

    assert fake_pipe.scheduler is fake_scheduler
    mock_diffusers.DPMSolverMultistepScheduler.from_config.assert_called_once_with(
        original_scheduler_config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )


@pytest.mark.unit
def test_load_pipe_uses_cpu_offload_when_enabled(mock_diffusers):
    r = DiffusionReplacer(device="cuda", tracker=MagicMock())
    r.enable_cpu_offload = True
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    r._load_pipe()

    fake_pipe.enable_model_cpu_offload.assert_called_once()
    fake_pipe.to.assert_not_called()


@pytest.mark.unit
def test_load_pipe_moves_to_device_when_offload_disabled(mock_diffusers):
    r = DiffusionReplacer(device="cuda", tracker=MagicMock())
    r.enable_cpu_offload = False
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)

    r._load_pipe()

    fake_pipe.to.assert_called_once_with("cuda")
    fake_pipe.enable_model_cpu_offload.assert_not_called()


@pytest.mark.unit
def test_load_pipe_enables_vae_slicing_on_cuda_only(mock_diffusers):
    r_cuda = DiffusionReplacer(device="cuda", tracker=MagicMock())
    fake_pipe_cuda = _stub_diffusers_pipe(mock_diffusers)
    r_cuda._load_pipe()
    fake_pipe_cuda.enable_vae_slicing.assert_called_once()


@pytest.mark.unit
def test_load_pipe_skips_vae_slicing_on_cpu(mock_diffusers):
    r_cpu = DiffusionReplacer(device="cpu", tracker=MagicMock())
    fake_pipe_cpu = _stub_diffusers_pipe(mock_diffusers)
    r_cpu._load_pipe()
    fake_pipe_cpu.enable_vae_slicing.assert_not_called()


@pytest.mark.unit
def test_load_pipe_xformers_failure_is_swallowed(mock_diffusers):
    r = DiffusionReplacer(device="cuda", tracker=MagicMock())
    fake_pipe = _stub_diffusers_pipe(mock_diffusers)
    fake_pipe.enable_xformers_memory_efficient_attention.side_effect = RuntimeError("no xformers")

    # Should not raise even though xformers isn't available.
    pipe = r._load_pipe()
    assert pipe is fake_pipe


@pytest.mark.unit
def test_load_pipe_raises_runtime_error_when_diffusers_missing():
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())
    with patch.dict(sys.modules, {'diffusers': None}):
        with pytest.raises(RuntimeError, match="diffusers/torch not installed"):
            r._load_pipe()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calculate_metrics_reports_expected_fields():
    r = DiffusionReplacer(device="cpu", tracker=MagicMock())

    img = Image.new('RGB', (640, 480), color='white')
    ibuf = BytesIO()
    img.save(ibuf, format='JPEG')

    mask = Image.new('L', (640, 480), color=0)
    ImageDraw.Draw(mask).rectangle([0, 0, 99, 99], fill=255)
    mbuf = BytesIO()
    mask.save(mbuf, format='PNG')

    metrics = await r._calculate_metrics(ibuf.getvalue(), mbuf.getvalue(), 250.0)

    assert metrics['processing_time_ms'] == 250.0
    assert metrics['processing_time_s'] == pytest.approx(0.25)
    assert metrics['mask_size_pixels'] == 100 * 100
    assert metrics['image_size'] == (640, 480)
    assert metrics['mode'] == 'replace_diffusion'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_track_metrics_logs_provided_overrides():
    tracker = MagicMock()
    r = DiffusionReplacer(device="cpu", tracker=tracker)
    metrics = {'processing_time_ms': 500.0, 'mask_size_pixels': 42, 'image_size': (640, 480)}

    await r._track_metrics(metrics, num_inference_steps=20, guidance_scale=8.0, ip_adapter_scale=0.7)

    tracker.log_run.assert_called_once()
    kwargs = tracker.log_run.call_args.kwargs
    assert kwargs['params']['steps'] == 20
    assert kwargs['params']['guidance_scale'] == 8.0
    assert kwargs['params']['ip_adapter_scale'] == 0.7
    assert kwargs['metrics']['mask_size_pixels'] == 42
    assert kwargs['tags']['operation'] == 'inpaint'
    assert kwargs['tags']['inpaint_model'] == 'diffusion_replace'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_track_metrics_falls_back_to_instance_defaults():
    tracker = MagicMock()
    r = DiffusionReplacer(device="cpu", tracker=tracker)
    metrics = {'processing_time_ms': 500.0, 'mask_size_pixels': 42, 'image_size': (640, 480)}

    await r._track_metrics(metrics, num_inference_steps=None, guidance_scale=None, ip_adapter_scale=None)

    kwargs = tracker.log_run.call_args.kwargs
    assert kwargs['params']['steps'] == r.default_steps
    assert kwargs['params']['guidance_scale'] == r.default_guidance_scale
    assert kwargs['params']['ip_adapter_scale'] == r.ip_adapter_scale


@pytest.mark.unit
def test_get_diffusion_replacer_returns_singleton():
    import app.ml.diffuser_inpainter as mod
    mod._replacer_instance = None
    with patch('app.ml.diffuser_inpainter.DeviceManager.get', return_value='cpu'):
        r1 = get_diffusion_replacer()
        r2 = get_diffusion_replacer()
    assert r1 is r2
    mod._replacer_instance = None