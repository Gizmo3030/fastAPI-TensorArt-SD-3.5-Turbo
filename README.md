# TensorArt SD 3.5 Turbo · Local FastAPI Studio

FastAPI service plus UI for running the `tensorart/stable-diffusion-3.5-medium-turbo` model entirely on your own hardware. Hugging Face is only used to download model weights; all inference stays local. A glassmorphic UI ships with the backend so you can iterate on prompts without any extra tooling.

## Highlights
- 🚀 Local Stable Diffusion pipeline via `diffusers`, `torch`, and optional xFormers acceleration
- 🧠 REST + HTML form endpoints with shared validation and sane guardrails
- 🎨 Animated UI with live render timings
- 🐳 Dockerfile + compose stack with cache volume for model weights
- ✅ Pytest health check to ensure the service boots

## Prerequisites
- Python 3.12 (3.11 works too; 3.13 currently needs compiling `tokenizers` from source)
- GPU with CUDA 12.x drivers **or** Apple Silicon (MPS) **or** CPU fallback (slower)
- Optional Hugging Face access token if the model repo is gated

## Configuration
Duplicate the sample env file and tweak as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
| --- | --- |
| `MODEL_ID` | Hugging Face repo used for weights (defaults to TensorArt SD 3.5 Medium Turbo) |
| `TORCH_DEVICE` | `cuda`, `mps`, or `cpu`. Falls back automatically if unavailable. |
| `TORCH_DTYPE` | Preferred dtype (`float16`, `bfloat16`, `float32`). Automatically coerced to `float32` on CPU. |
| `ENABLE_SEQUENTIAL_CPU_OFFLOAD` | Keeps VRAM usage low on limited GPUs/CPUs. |
| `HF_CACHE_DIR` | Optional custom cache directory for downloaded weights. |

`IMAGE_MAX_WIDTH`/`HEIGHT` still gate incoming requests.

## Local Development
1. Remove any previous `.venv` created with another interpreter, then create a fresh Python 3.12 virtual environment (3.11 also works).
   ```bash
   rm -rf .venv
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. Install Rust (needed for `tokenizers` on Python 3.13+).
   ```bash
   curl https://sh.rustup.rs -sSf | sh -s -- -y
   source ~/.cargo/env
   ```
3. Install PyTorch first so you can pick the wheel that matches your hardware (example uses CUDA 12.4).
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```
4. Install the remaining dependencies.
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the app:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open `http://localhost:8000` for the UI or hit `POST /api/generate` with:
   ```json
   {
     "prompt": "A hyperreal photo of bioluminescent reefs",
     "negative_prompt": "low quality",
     "width": 1024,
     "height": 1024,
     "guidance_scale": 5.5,
     "num_inference_steps": 28
   }
   ```

The first request will download and initialize the pipeline; subsequent calls reuse the in-memory weights.

## Docker
The compose file builds a slim Python image and exposes port 8000. To persist the Hugging Face cache between runs, create a host directory and mount it:

```bash
mkdir -p .cache/huggingface
docker compose up --build
```

By default the container uses CPU inference. Override `TORCH_DEVICE`/`TORCH_DTYPE` in `.env` for GPU hosts.

## Testing
```bash
pytest
```

## Tips
- If VRAM is tight, keep `ENABLE_SEQUENTIAL_CPU_OFFLOAD=true` and run at 768px or less.
- Set `ENABLE_XFORMERS=true` after installing `xformers` to squeeze more throughput out of CUDA setups.
- Use `HF_CACHE_DIR` (or mount `~/.cache/huggingface`) so you don't redownload model weights inside ephemeral containers.
