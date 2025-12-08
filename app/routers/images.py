from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.schemas import ErrorResponse, HealthResponse, ImageResponse, TextToImageRequest
from app.services.hf_client import generate_image
from app.services.validators import ensure_dimensions

router = APIRouter(prefix="/api", tags=["images"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/generate",
    response_model=ImageResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def generate_image_api(
    payload: TextToImageRequest,
    settings: Settings = Depends(get_settings),
) -> ImageResponse:
    ensure_dimensions(payload, settings)
    image_bytes, elapsed_ms = await generate_image(payload, settings)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return ImageResponse(image_base64=encoded, elapsed_ms=elapsed_ms, model_id=settings.model_id)
