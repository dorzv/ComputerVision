# Latent Diffusion Model (LDM) with Gradio

A custom Latent Diffusion Model implementation built on top of `stable-diffusion-v1-5/stable-diffusion-v1-5` and wrapped in a Gradio web application. This repository demonstrates step-by-step tokenization, text embedding generation, classifier-free guidance (CFG), and VAE latent decoding using PyTorch and Hugging Face `diffusers`.

---

## Overview

- **Custom Inference Pipeline**: Step-by-step latent diffusion decoding loop using raw pipeline components (Tokenizer, Text Encoder, UNet, Scheduler, and VAE).
- **Hardware Acceleration**: Automatic target selection for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), or fallback to `cpu`.
- **Interactive Web Interface**: Built with Gradio to easily adjust image resolution, CFG scale, and denoising steps.

---

## Project Structure

```text
├── main.py           # Entry point and device configuration
├── app.py            # Gradio UI layout and slider configurations
├── models.py         # LDM pipeline implementation and image utility methods
└── pyproject.toml    # Dependencies and platform specifications
```

## RequirementsPython: 
- **Python**: >=3.12, <3.15   
- **Package Manager**: uv (recommended) or pip

## Installation

### 1. Supported Platforms using uv (Fastest)
If you are running on macOS ARM64 (Apple Silicon) or Linux x86_64 with CUDA 12.8, the lockfile environment settings in pyproject.toml are fully supported:

```shell
uv sync
```

### 2. Unsupported Platforms using uv (Ignoring uv.lock)
If your platform is not covered by the environments list in pyproject.toml (for example, Windows or Linux ARM64), bypass the lockfile platform lock and install directly from pyproject.toml:   

```shell
uv sync --upgrade
```
### 3. Fallback using pip
If you do not have uv installed, you can set up a standard Python virtual environment and install dependencies via pip:

```shell
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
pip install .
```

## Quickstart
Run the application entry point:

```shell
# Using uv
uv run main.py

# Or using standard python
python main.py
```

Upon launching, the app will start a local Gradio server (typically accessible at http://127.0.0.1:7860)

If you want to get also public link (for example, for Google Colab), add the share flag:
```shell
# Using uv
uv run main.py --share

# Or using standard python
python main.py --share
```


## Application Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Prompt** | Text | N/A | Text description for image generation |
| **Height** | Slider | `512` | Image height in pixels ($256 \text{–} 1024$, step $64$) |
| **Width** | Slider | `512` | Image width in pixels ($256 \text{–} 1024$, step $64$) |
| **Inference Steps** | Slider | `50` | Number of denoising iterations ($1 \text{–} 150$) |
| **CFG Scale** | Slider | `7.5` | Classifier-Free Guidance weight ($1.0 \text{–} 20.0$) |
