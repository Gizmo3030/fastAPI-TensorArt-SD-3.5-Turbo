from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, validator


class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=600)
    negative_prompt: str | None = Field(default=None, max_length=600)
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)
    guidance_scale: float = Field(1.5, ge=0.0, le=20.0)
    num_inference_steps: int = Field(8, ge=1, le=80)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)

    @validator("width", "height")
    def _must_be_multiple_of_eight(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("value must be a multiple of 8")
        return value


class ImageResponse(BaseModel):
    image_base64: str
    elapsed_ms: int
    model_id: str

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
