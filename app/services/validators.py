from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import TextToImageRequest


def ensure_dimensions(payload: TextToImageRequest, settings: Settings) -> None:
    if payload.width > settings.image_max_width or payload.height > settings.image_max_height:
        message = (
            f"Requested size {payload.width}x{payload.height} exceeds allowed "
            f"{settings.image_max_width}x{settings.image_max_height}"
        )
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=message)
