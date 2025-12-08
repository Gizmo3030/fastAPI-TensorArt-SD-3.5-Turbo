from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.routers import images
from app.schemas import TextToImageRequest
from app.services.hf_client import generate_image
from app.services.validators import ensure_dimensions

app = FastAPI(title="TensorArt Turbo UI", version="0.1.0")
app.include_router(images.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _build_form_state(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "prompt": "A cinematic photo of bioluminescent coral canyon",
        "negative_prompt": "low resolution, blurry",
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5.5,
        "num_inference_steps": 28,
        "seed": "",
    }
    defaults.update({k: v for k, v in data.items() if v is not None})
    return defaults


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    context = {"request": request, "result": None, "error": None, "form": _build_form_state({})}
    return templates.TemplateResponse("index.html", context)


@app.post("/", response_class=HTMLResponse)
async def generate_via_form(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(default=None),
    width: int = Form(1024),
    height: int = Form(1024),
    guidance_scale: float = Form(5.5),
    num_inference_steps: int = Form(28),
    seed: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    form_snapshot = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed or "",
    }

    try:
        raw_seed = int(seed) if seed not in (None, "") else None
    except ValueError:
        context = {
            "request": request,
            "result": None,
            "error": "Seed must be an integer",
            "form": _build_form_state(form_snapshot),
        }
        return templates.TemplateResponse("index.html", context, status_code=422)

    try:
        payload = TextToImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=raw_seed,
        )
    except ValidationError as exc:  # Rare UI level validation feedback
        context = {
            "request": request,
            "result": None,
            "error": exc.errors()[0]["msg"],
            "form": _build_form_state(form_snapshot),
        }
        return templates.TemplateResponse("index.html", context, status_code=422)

    try:
        ensure_dimensions(payload, settings)
        image_bytes, elapsed_ms = await generate_image(payload, settings)
    except HTTPException as exc:
        context = {
            "request": request,
            "result": None,
            "error": exc.detail,
            "form": _build_form_state(form_snapshot),
        }
        return templates.TemplateResponse("index.html", context, status_code=exc.status_code)

    encoded = base64.b64encode(image_bytes).decode("ascii")
    context = {
        "request": request,
        "result": {"image_base64": encoded, "elapsed_ms": elapsed_ms},
        "error": None,
        "form": _build_form_state(form_snapshot),
    }
    return templates.TemplateResponse("index.html", context)
