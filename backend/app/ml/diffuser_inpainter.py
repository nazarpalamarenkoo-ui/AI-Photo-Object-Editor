import asyncio
import threading
import time
from io import BytesIO
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config.settings import settings
from app.config.device_manager import DeviceManager
from app.ml.experiment_tracker import ExperimentTracker, get_tracker
from app.core.logging import get_logger, log_ml_operation

logger = get_logger(__name__)


class DiffusionReplacer:
    """
    Diffusion-based object REPLACE.

    Pipeline (identical logic to the standalone prototype):
        1. Crop the frame to the mask bbox + padding (gives the model
           context beyond just the masked pixels)
        2. Upscale the crop to a fixed working resolution — otherwise a
           small mask relative to the full frame gets a tiny effective
           resolution and the generation turns to mush
        3. Run SD-inpainting + IP-Adapter on the crop with a *hard binary*
           mask (blurred/antialiased masks confuse the pipeline at the
           boundary — feathering happens only at composite time, below)
        4. Decode explicitly in fp32 and check for NaNs (fp16 UNet
           attention is unstable on some GPUs, e.g. GTX 16xx — cheaper to
           fail loudly than silently return a black square)
        5. Downscale the generated crop back to original crop size,
           feather-blend against the original crop, paste into the frame
    """

    def __init__(self, device: str = "cpu", tracker: Optional[ExperimentTracker] = None):
        self.device = device
        self.tracker = tracker or get_tracker()
        self._pipe = None
        self._pipe_lock = threading.Lock()

        self.model_id = getattr(
            settings, "DIFFUSION_INPAINT_MODEL_ID",
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
        )
        self.ip_adapter_repo = getattr(settings, "IP_ADAPTER_REPO", "h94/IP-Adapter")
        self.ip_adapter_subfolder = getattr(settings, "IP_ADAPTER_SUBFOLDER", "models")
        self.ip_adapter_image_encoder_subfolder = getattr(
            settings, "IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER", "models/image_encoder"
        )
        self.ip_adapter_variant = getattr(settings, "IP_ADAPTER_VARIANT", "plus")
        self.ip_adapter_weights = {
            "plus": "ip-adapter-plus_sd15.bin",
            "regular": "ip-adapter_sd15.bin",
        }
        self.ip_adapter_scale = getattr(settings, "IP_ADAPTER_SCALE", 0.8)

        self.default_steps = getattr(settings, "DIFFUSION_STEPS", 30)
        self.default_guidance_scale = getattr(settings, "DIFFUSION_GUIDANCE_SCALE", 5.5)
        self.default_strength = getattr(settings, "DIFFUSION_STRENGTH", 0.7)
        self.default_negative_prompt = getattr(
            settings, "DIFFUSION_NEGATIVE_PROMPT",
            "blurry, distorted, low quality, deformed, artifacts",
        )
        self.default_prompt_fallback = getattr(
            settings, "DIFFUSION_PROMPT_FALLBACK", "high quality, detailed, photorealistic"
        )

        self.work_resolution = getattr(settings, "DIFFUSION_WORK_RESOLUTION", 640)
        self.crop_padding_ratio = getattr(settings, "DIFFUSION_CROP_PADDING_RATIO", 0.35)
        self.min_crop_size = getattr(settings, "DIFFUSION_MIN_CROP_SIZE", 256)
        self.mask_blur_radius = getattr(settings, "DIFFUSION_MASK_BLUR_RADIUS", 6)
        self.enable_cpu_offload = getattr(settings, "DIFFUSION_ENABLE_CPU_OFFLOAD", False)
        self.model_name = "sd_inpaint_ip_adapter"
        self.model_version = f"{self.model_id}+{self.ip_adapter_variant}"
        logger.info(
            "diffusion_replacer_configured",
            model=self.model_id,
            device=device,
            ip_adapter_variant=self.ip_adapter_variant,
        )

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        with self._pipe_lock:
            if self._pipe is not None:
                return self._pipe

            try:
                import torch
                from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
            except ImportError as e:
                logger.error("diffusion_replacer_load_failed", exc_info=e)
                raise RuntimeError(
                    f"diffusers/torch not installed or incompatible: {e}. "
                    "pip install torch diffusers transformers accelerate safetensors"
                )

            dtype = torch.float32
            logger.info("diffusion_pipe_loading", model=self.model_id, device=self.device, dtype=str(dtype))

            pipe = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
            
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
            )

            weight_name = self.ip_adapter_weights[self.ip_adapter_variant]
            logger.info(
                "diffusion_ip_adapter_loading",
                variant=self.ip_adapter_variant, weight_name=weight_name,
            )
            pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder=self.ip_adapter_subfolder,
                weight_name=weight_name,
                image_encoder_folder=self.ip_adapter_image_encoder_subfolder,
            )
            pipe.set_ip_adapter_scale(self.ip_adapter_scale)

            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)

            if self.device == "cuda":
                pipe.enable_vae_slicing()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # no xformers, or torch SDPA is already efficient — not critical

            self._pipe = pipe
            logger.info("diffusion_pipe_loaded", model=self.model_id)
            return self._pipe

    async def replace(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        reference_image_bytes: bytes,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: int = 0,
        track_metrics: bool = True,
    ) -> Dict:
        """
        Generate new content inside `mask_bytes`, steered by
        `reference_image_bytes` (IP-Adapter) and `prompt`.

        Args:
            image_bytes:            Full input frame
            mask_bytes:             Binary mask (PNG, L) — white = area to regenerate.
                                    Does NOT need to be pre-feathered; feathering
                                    is applied only at the final composite step.
            reference_image_bytes:  Reference/asset image for IP-Adapter conditioning
            prompt:                 Text prompt (falls back to a generic quality
                                    prompt if empty — but an actual description
                                    of the target object gives much better results)
            negative_prompt:        Overrides the configured default
            num_inference_steps, guidance_scale, ip_adapter_scale, strength:
                                    Override diffusion defaults for this call
            seed:                   Generator seed (default: 0, deterministic)
            track_metrics:          Log this call to MLflow (default: True)

        Returns:
            Dict { result_bytes: JPEG bytes, metrics: Dict }

        Raises:
            RuntimeError: diffusers/torch not installed, or NaN in generation
                          (see message for what to check)
        """
        start_time = time.time()

        async with log_ml_operation(
            "diffusion_replace",
            model=f"sd_inpaint+ip_adapter_{self.ip_adapter_variant}",
            device=self.device,
            steps=num_inference_steps or self.default_steps,
            guidance_scale=guidance_scale or self.default_guidance_scale,
        ) as op:
            result_bytes = await asyncio.to_thread(
                self._replace_sync,
                image_bytes,
                mask_bytes,
                reference_image_bytes,
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                ip_adapter_scale,
                strength,
                seed,
            )

            processing_time_ms = (time.time() - start_time) * 1000
            metrics = await self._calculate_metrics(image_bytes, mask_bytes, processing_time_ms)

            op.set_output(
                mask_size_pixels=metrics["mask_size_pixels"],
                result_size_bytes=len(result_bytes),
            )

        if track_metrics:
            await self._track_metrics(metrics, num_inference_steps, guidance_scale, ip_adapter_scale)

        return {"result_bytes": result_bytes, "metrics": metrics}

    def _replace_sync(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        reference_image_bytes: bytes,
        prompt: str,
        negative_prompt: Optional[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        ip_adapter_scale: Optional[float],
        strength: Optional[float],
        seed: int,
    ) -> bytes:
        import torch

        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")
        mask = np.array(mask_pil)
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(
                mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask = (mask > 127).astype(np.uint8) * 255

        x0, y0, x1, y1 = self._get_crop_bbox(mask, self.crop_padding_ratio, self.min_crop_size)
        crop_w, crop_h = x1 - x0, y1 - y0
        logger.info("diffusion_replace_crop", bbox=(x0, y0, x1, y1), size=(crop_w, crop_h))

        img_crop_bgr = image_bgr[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]  # binary, at crop resolution

        # working resolution: longer crop side -> work_resolution
        scale = self.work_resolution / max(crop_w, crop_h)
        work_w = self._round_to_multiple_of_8(int(crop_w * scale))
        work_h = self._round_to_multiple_of_8(int(crop_h * scale))

        up_interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        img_work_bgr = cv2.resize(img_crop_bgr, (work_w, work_h), interpolation=up_interp)

        # mask fed to the pipeline must be HARD binary (nearest + threshold) —
        # feathering happens only at the final composite, further below.
        mask_work = cv2.resize(mask_crop, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
        mask_work = (mask_work > 127).astype(np.uint8) * 255

        img_work_pil = Image.fromarray(cv2.cvtColor(img_work_bgr, cv2.COLOR_BGR2RGB))
        mask_work_pil = Image.fromarray(mask_work).convert("L")

        ref_img = self._load_reference_image(reference_image_bytes)

        final_prompt = prompt.strip() if prompt and prompt.strip() else self.default_prompt_fallback
        final_negative = negative_prompt or self.default_negative_prompt
        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        ip_scale = ip_adapter_scale if ip_adapter_scale is not None else self.ip_adapter_scale
        strength_val = strength if strength is not None else self.default_strength

        pipe = self._load_pipe()
        pipe.set_ip_adapter_scale(ip_scale)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            "diffusion_replace_generating",
            work_size=(work_w, work_h),
            steps=steps,
            guidance=guidance,
            ip_scale=ip_scale,
            strength=strength_val,
            prompt=final_prompt,
        )

        # Whole pipeline is fp32 (see _load_pipe) — output_type="latent" is kept only so we can decode manually and assert no NaNs below,
        output = pipe(
            image=img_work_pil,
            mask_image=mask_work_pil,
            ip_adapter_image=ref_img,
            prompt=final_prompt,
            negative_prompt=final_negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength_val,
            generator=generator,
            output_type="latent",
        )

        latents = output.images
        with torch.no_grad():
            decoded = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False
            )[0]

        if torch.isnan(decoded).any():
            raise RuntimeError(
                "NaN in VAE output even in full fp32 — this isn't a precision "
                "issue. Check: torch/CUDA version compatible with the GPU "
                "driver, model weights not corrupted (try clearing "
                "~/.cache/huggingface and re-downloading), or retry with "
                "ip_adapter_image omitted to isolate whether IP-Adapter is "
                "the trigger."
            )

        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        image_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        gen_crop_pil = Image.fromarray((image_np * 255).round().astype(np.uint8))

        if gen_crop_pil.size != (work_w, work_h):
            gen_crop_pil = gen_crop_pil.resize((work_w, work_h), Image.Resampling.LANCZOS)

        # back to original crop size, then a careful composite into the full frame
        gen_crop_pil = gen_crop_pil.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        gen_crop_bgr = cv2.cvtColor(np.array(gen_crop_pil), cv2.COLOR_RGB2BGR).astype(np.float32)

        alpha = self._feather_mask(mask_crop, self.mask_blur_radius).astype(np.float32)[..., None] / 255.0
        orig_crop_bgr = img_crop_bgr.astype(np.float32)
        blended_crop = (orig_crop_bgr * (1 - alpha) + gen_crop_bgr * alpha).astype(np.uint8)

        result_bgr = image_bgr.copy()
        result_bgr[y0:y1, x0:x1] = blended_crop

        result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        result_img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


    @staticmethod
    def _load_reference_image(reference_image_bytes: bytes, target_size: int = 224) -> Image.Image:
        """Center-crop to square + resize — matches IP-Adapter's ViT-H
        image encoder input convention."""
        img = Image.open(BytesIO(reference_image_bytes))
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return img

    @staticmethod
    def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
        """Blurs the binary mask's edges. Used ONLY for the final composite
        (blending generated crop with original) — never fed into the
        diffusion pipeline itself."""
        if radius <= 0:
            return mask
        k = radius * 2 + 1
        return cv2.GaussianBlur(mask, (k, k), 0)

    @staticmethod
    def _round_to_multiple_of_8(x: int) -> int:
        return max(8, int(round(x / 8.0)) * 8)

    @staticmethod
    def _get_crop_bbox(mask: np.ndarray, padding_ratio: float, min_size: int) -> Tuple[int, int, int, int]:
        """Bbox around the mask + context padding, with a minimum crop size
        so small masks still get enough surrounding context."""
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            raise ValueError("Empty mask.")

        H, W = mask.shape[:2]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        w, h = x1 - x0, y1 - y0
        pad_w = max(int(w * padding_ratio), 1)
        pad_h = max(int(h * padding_ratio), 1)

        x0 -= pad_w
        x1 += pad_w
        y0 -= pad_h
        y1 += pad_h

        cw, ch = x1 - x0, y1 - y0
        if cw < min_size:
            extra = (min_size - cw) // 2 + 1
            x0 -= extra
            x1 += extra
        if ch < min_size:
            extra = (min_size - ch) // 2 + 1
            y0 -= extra
            y1 += extra

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)

        return x0, y0, x1, y1

    async def _calculate_metrics(
        self, image_bytes: bytes, mask_bytes: bytes, processing_time_ms: float
    ) -> Dict:
        def calc_sync():
            img = Image.open(BytesIO(image_bytes))
            image_size = img.size  # (width, height)

            mask = Image.open(BytesIO(mask_bytes)).convert("L")
            mask_array = np.array(mask)
            mask_size_pixels = int(np.sum(mask_array > 128))

            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "processing_time_ms": processing_time_ms,
                "processing_time_s": processing_time_ms / 1000,
                "mask_size_pixels": mask_size_pixels,
                "image_size": image_size,
                "mode": "replace_diffusion",
            }

        return await asyncio.to_thread(calc_sync)

    async def _track_metrics(
        self,
        metrics: Dict,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        ip_adapter_scale: Optional[float],
    ) -> None:
        def log_sync():
            self.tracker.log_run(
                run_name="inpaint_replace_diffusion",
                params={
                    "model": self.model_id,
                    "device": self.device,
                    "ip_adapter_variant": self.ip_adapter_variant,
                    "steps": num_inference_steps or self.default_steps,
                    "guidance_scale": guidance_scale or self.default_guidance_scale,
                    "ip_adapter_scale": ip_adapter_scale or self.ip_adapter_scale,
                },
                metrics={
                    "processing_time_ms": metrics["processing_time_ms"],
                    "processing_time_s": metrics["processing_time_ms"] / 1000,
                    "mask_size_pixels": metrics["mask_size_pixels"],
                    "image_width": metrics["image_size"][0],
                    "image_height": metrics["image_size"][1],
                },
                tags={"inpaint_model": "diffusion_replace", "operation": "inpaint_diffusion_replace"},
            )

        await asyncio.to_thread(log_sync)


import threading as _threading  # noqa: E402  (keep singleton block visually separate, like inpainter.py)

_replacer_instance = None
_replacer_lock = _threading.Lock()


def get_diffusion_replacer(tracker: Optional[ExperimentTracker] = None) -> DiffusionReplacer:
    """
    Singleton getter for DiffusionReplacer.

    Args:
        tracker: ExperimentTracker for MLflow (default: auto-created)

    Returns: DiffusionReplacer instance
    """
    global _replacer_instance
    if _replacer_instance is None:
        with _replacer_lock:
            if _replacer_instance is None:
                device = DeviceManager.get(getattr(settings, "DIFFUSION_DEVICE", "cuda"))
                _replacer_instance = DiffusionReplacer(device=device, tracker=tracker)
    return _replacer_instance