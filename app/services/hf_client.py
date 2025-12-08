from __future__ import annotations

import asyncio
import io
import time
from typing import Tuple

import torch
from diffusers import DiffusionPipeline
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import TextToImageRequest

_PIPELINE: DiffusionPipeline | None = None
_PIPELINE_LOCK = asyncio.Lock()
_PIPELINE_DEVICE: str | None = None


async def generate_image(payload: TextToImageRequest, settings: Settings) -> Tuple[bytes, int]:
    pipeline = await _get_pipeline(settings)
    device = _PIPELINE_DEVICE or pipeline._execution_device.type  # type: ignore[attr-defined]
    generator = None
    if payload.seed is not None:
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        image_bytes = await loop.run_in_executor(
            None,
            lambda: _run_pipeline(
                pipeline,
                payload,
                generator,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return image_bytes, elapsed_ms


def _run_pipeline(
    pipeline: DiffusionPipeline,
    payload: TextToImageRequest,
    generator: torch.Generator | None,
) -> bytes:
    result = pipeline(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        guidance_scale=payload.guidance_scale,
        num_inference_steps=payload.num_inference_steps,
        generator=generator,
    )
    image = result.images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def _get_pipeline(settings: Settings) -> DiffusionPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    async with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        loop = asyncio.get_running_loop()
        _PIPELINE = await loop.run_in_executor(None, lambda: _load_pipeline(settings))
    return _PIPELINE


def _load_pipeline(settings: Settings) -> DiffusionPipeline:
    global _PIPELINE_DEVICE
    device = _resolve_device(settings.torch_device)
    dtype = _select_dtype(settings.torch_dtype, device)

    load_kwargs: dict[str, object] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": settings.low_cpu_mem_usage,
        "ignore_mismatched_sizes": settings.ignore_mismatched_sizes,
    }
    if settings.pipeline_variant:
        load_kwargs["variant"] = settings.pipeline_variant
    if settings.model_revision:
        load_kwargs["revision"] = settings.model_revision
    if settings.hf_auth_token:
        load_kwargs["token"] = settings.hf_auth_token
    if settings.hf_cache_dir:
        load_kwargs["cache_dir"] = settings.hf_cache_dir

    pipeline = DiffusionPipeline.from_pretrained(settings.model_id, **load_kwargs)

    # Apply turbo LoRA adapter if configured (matches reference usage for 8-step turbo weights).
    if settings.lora_repo_id and settings.lora_weight_name:
        pipeline.load_lora_weights(
            settings.lora_repo_id,
            weight_name=settings.lora_weight_name,
            revision=settings.lora_revision,
            token=settings.hf_auth_token,
            cache_dir=settings.hf_cache_dir,
        )
        if settings.fuse_lora:
            pipeline.fuse_lora(lora_scale=settings.lora_scale)

    if device == "cuda":
        pipeline.to("cuda")
    elif device == "mps":
        pipeline.to("mps")
    else:
        pipeline.to("cpu")

    if settings.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif settings.enable_attention_slicing:
        pipeline.enable_attention_slicing()

    if settings.enable_xformers:
        try:  # pragma: no cover - optional optimization
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipeline.set_progress_bar_config(disable=True)
    _PIPELINE_DEVICE = device
    return pipeline


def _resolve_device(preferred: str) -> str:
    pref = preferred.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _select_dtype(preferred: str, device: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(preferred.lower(), torch.float16)
    if device == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype
