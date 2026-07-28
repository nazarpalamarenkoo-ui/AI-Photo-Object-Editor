import sys
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

with patch('app.ml.experiment_tracker.get_tracker'):
    from app.ml.diffuser_inpainter import DiffusionReplacer, get_diffusion_replacer


@pytest.fixture(autouse=True)
def mock_torch(monkeypatch):
    """Fresh, test-scoped `torch` mock, reverted automatically after each
    test so other test files' real torch import is never affected."""
    m = MagicMock()
    m.float32 = "float32"
    m.no_grad.return_value = nullcontext()
    m.isnan.return_value.any.return_value = False
    monkeypatch.setitem(sys.modules, 'torch', m)
    return m


def _make_image_bytes(width=640, height=480, color='white', fmt='JPEG'):
    img = Image.new('RGB', (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_mask_bytes(width=640, height=480, box=(200, 150, 400, 350)):
    mask = Image.new('L', (width, height), color=0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    buf = BytesIO()
    mask.save(buf, format='PNG')
    return buf.getvalue()


def _make_fake_pipe(mock_torch, nan_result=False):
    """Stand-in for the diffusers AutoPipelineForInpainting instance, wired
    so `_replace_sync` can run its real decode / NaN-check / composite path.
    The generated-crop *content* is random noise — these tests check
    plumbing (shapes, forwarded kwargs, error handling), not pixel fidelity.
    """
    fake_pipe = MagicMock()
    fake_output = MagicMock()
    fake_output.images = MagicMock()  # stand-in for the latents tensor
    fake_pipe.return_value = fake_output
    fake_pipe.vae.config.scaling_factor = 1.0

    decoded_tensor = MagicMock()
    fake_numpy = np.random.rand(1, 64, 64, 3).astype(np.float32)
    decoded_tensor.cpu.return_value.permute.return_value.float.return_value.numpy.return_value = fake_numpy
    decoded_tensor.__truediv__.return_value.__add__.return_value.clamp.return_value = decoded_tensor
    fake_pipe.vae.decode.return_value = (decoded_tensor,)

    mock_torch.isnan.return_value.any.return_value = nan_result
    return fake_pipe


@pytest.fixture
def replacer():
    return DiffusionReplacer(device="cpu", tracker=MagicMock())


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_returns_result_bytes_and_metrics(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200, color='red'),
            prompt="a wooden chair",
            track_metrics=False,
        )
    assert 'result_bytes' in result
    assert isinstance(result['result_bytes'], bytes)
    assert result['result_bytes'][:2] == b'\xff\xd8'  # valid JPEG signature
    assert 'metrics' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_preserves_full_frame_size(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(640, 480),
            mask_bytes=_make_mask_bytes(640, 480),
            reference_image_bytes=_make_image_bytes(200, 200, color='blue'),
            track_metrics=False,
        )
    out_img = Image.open(BytesIO(result['result_bytes']))
    assert out_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_resizes_mask_that_doesnt_match_image_size(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(640, 480),
            mask_bytes=_make_mask_bytes(320, 240, box=(100, 75, 200, 175)),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    out_img = Image.open(BytesIO(result['result_bytes']))
    assert out_img.size == (640, 480)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_small_mask_below_min_crop_size(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(640, 480),
            mask_bytes=_make_mask_bytes(640, 480, box=(300, 220, 320, 240)),  # 20x20
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_large_near_full_frame_mask(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(640, 480),
            mask_bytes=_make_mask_bytes(640, 480, box=(10, 10, 630, 470)),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_accepts_rgba_reference_image(replacer, mock_torch):
    """Reference images with alpha (e.g. product cutouts) shouldn't break
    the IP-Adapter preprocessing path."""
    fake_pipe = _make_fake_pipe(mock_torch)
    ref = Image.new('RGBA', (200, 200), color=(10, 20, 30, 100))
    buf = BytesIO()
    ref.save(buf, format='PNG')
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=buf.getvalue(),
            track_metrics=False,
        )
    assert 'result_bytes' in result


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_uses_fallback_prompt_when_blank(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            prompt="   ",
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['prompt'] == replacer.default_prompt_fallback


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_forwards_custom_prompt(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            prompt="a golden retriever",
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['prompt'] == "a golden retriever"


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_forwards_negative_prompt_override(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            negative_prompt="cartoon, sketch",
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['negative_prompt'] == "cartoon, sketch"


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_default_negative_prompt_used_when_not_given(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['negative_prompt'] == replacer.default_negative_prompt


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_forwards_steps_guidance_and_strength(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            num_inference_steps=12,
            guidance_scale=9.5,
            strength=0.8,
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['num_inference_steps'] == 12
    assert kwargs['guidance_scale'] == 9.5
    assert kwargs['strength'] == 0.8


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_defaults_steps_guidance_and_strength(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    kwargs = fake_pipe.call_args.kwargs
    assert kwargs['num_inference_steps'] == replacer.default_steps
    assert kwargs['guidance_scale'] == replacer.default_guidance_scale
    assert kwargs['strength'] == replacer.default_strength


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_ip_adapter_scale_override_applied_to_pipe(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            ip_adapter_scale=0.9,
            track_metrics=False,
        )
    fake_pipe.set_ip_adapter_scale.assert_called_with(0.9)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_seed_forwarded_to_generator(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            seed=42,
            track_metrics=False,
        )
    mock_torch.Generator.return_value.manual_seed.assert_called_with(42)


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_raises_runtime_error_on_nan_decode(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch, nan_result=True)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        with pytest.raises(RuntimeError, match="NaN in VAE output"):
            await replacer.replace(
                image_bytes=_make_image_bytes(),
                mask_bytes=_make_mask_bytes(),
                reference_image_bytes=_make_image_bytes(200, 200),
                track_metrics=False,
            )


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_metrics_contain_expected_fields(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        result = await replacer.replace(
            image_bytes=_make_image_bytes(640, 480),
            mask_bytes=_make_mask_bytes(640, 480),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    metrics = result['metrics']
    assert metrics['image_size'] == (640, 480)
    assert metrics['mask_size_pixels'] > 0
    assert metrics['mode'] == 'replace_diffusion'
    assert metrics['processing_time_ms'] > 0.0


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_track_metrics_true_logs_run(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=True,
        )
    replacer.tracker.log_run.assert_called_once()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_track_metrics_false_skips_logging(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            track_metrics=False,
        )
    replacer.tracker.log_run.assert_not_called()


@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.asyncio
async def test_replace_tracked_run_uses_overridden_params(replacer, mock_torch):
    fake_pipe = _make_fake_pipe(mock_torch)
    with patch.object(replacer, '_load_pipe', return_value=fake_pipe):
        await replacer.replace(
            image_bytes=_make_image_bytes(),
            mask_bytes=_make_mask_bytes(),
            reference_image_bytes=_make_image_bytes(200, 200),
            num_inference_steps=18,
            guidance_scale=6.5,
            ip_adapter_scale=0.3,
            track_metrics=True,
        )
    kwargs = replacer.tracker.log_run.call_args.kwargs
    assert kwargs['params']['steps'] == 18
    assert kwargs['params']['guidance_scale'] == 6.5
    assert kwargs['params']['ip_adapter_scale'] == 0.3


@pytest.mark.integration
@pytest.mark.ml
def test_get_diffusion_replacer_singleton_across_calls():
    import app.ml.diffuser_inpainter as mod
    mod._replacer_instance = None
    with patch('app.ml.diffuser_inpainter.DeviceManager.get', return_value='cpu'):
        r1 = get_diffusion_replacer()
        r2 = get_diffusion_replacer()
    assert r1 is r2
    assert isinstance(r1, DiffusionReplacer)
    mod._replacer_instance = None