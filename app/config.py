from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = Field(
        "stabilityai/stable-diffusion-3.5-medium",
        env="MODEL_ID",
        description="Base model repo used for local inference",
    )
    lora_repo_id: str = Field(
        "tensorart/stable-diffusion-3.5-medium-turbo",
        env="LORA_REPO_ID",
        description="Hugging Face repo containing the turbo LoRA weights",
    )
    lora_weight_name: str = Field(
        "lora_sd3.5m_turbo_8steps.safetensors",
        env="LORA_WEIGHT_NAME",
        description="Weight filename for the turbo LoRA",
    )
    lora_revision: str | None = Field(default=None, env="LORA_REVISION")
    lora_scale: float = Field(1.0, env="LORA_SCALE")
    fuse_lora: bool = Field(True, env="FUSE_LORA")
    model_revision: str | None = Field(default=None, env="MODEL_REVISION")
    hf_auth_token: str | None = Field(default=None, env="HF_AUTH_TOKEN")
    hf_cache_dir: str | None = Field(default=None, env="HF_CACHE_DIR")
    pipeline_variant: str | None = Field(default=None, env="PIPELINE_VARIANT")
    torch_device: str = Field("cuda", env="TORCH_DEVICE")
    torch_dtype: str = Field("float16", env="TORCH_DTYPE")
    enable_sequential_cpu_offload: bool = Field(False, env="ENABLE_SEQUENTIAL_CPU_OFFLOAD")
    enable_attention_slicing: bool = Field(True, env="ENABLE_ATTENTION_SLICING")
    enable_xformers: bool = Field(False, env="ENABLE_XFORMERS")
    low_cpu_mem_usage: bool = Field(False, env="LOW_CPU_MEM_USAGE")
    ignore_mismatched_sizes: bool = Field(True, env="IGNORE_MISMATCHED_SIZES")
    image_max_width: int = Field(1152, env="IMAGE_MAX_WIDTH")
    image_max_height: int = Field(1152, env="IMAGE_MAX_HEIGHT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
