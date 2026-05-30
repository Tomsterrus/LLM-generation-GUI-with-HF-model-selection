# HF Model Generator Environment

A simple desktop environment for loading and generating text using Hugging Face models, built with `customtkinter`. The application separates the frontend UI from the backend logic and monitors system resources (CPU and GPU memory).

## Prerequisites
- **Python**: 3.12.x (Recommended to avoid dependency issues with PyTorch)
- **CUDA Toolkit**: 12.4 or compatible (if using an NVIDIA GPU)
- **OS**: Windows

## Setup & Configuration

### 1. Hugging Face Token (Recommended)
To avoid API rate limits and access more models, it's recommended to use a Hugging Face token:
1. Create a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Set it as an environment variable:
   - Windows (PowerShell): `$env:HF_TOKEN="your_token_here"`
   - Or create a `.env` file in the project root.

### 2. Windows Symlinks
This application downloads large model files. To optimize disk space, it's recommended to enable **Developer Mode** in Windows Settings. This allows the Hugging Face library to use symbolic links instead of copying files.

## Installation

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/generation-with-model-selection.git
cd generation-with-model-selection

2. Create and activate a virtual environment (Python 3.12 required)

python -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

3. Install the dependencies:

pip install -r requirements.txt

# Note: The standard torch installation usually includes the default CUDA dependencies. If your GPU is not detected, ensure your NVIDIA drivers are up to date or install a specific PyTorch version from [PyTorch Get Started.](https://pytorch.org/get-started/locally/). On my setup (Acer Nitro 12th Gen Intel(R) Core(TM) i5-12500H (3.10 GHz); 16,0 GB; NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)), with Python 3.12.10, PyTorch available at https://download.pytorch.org/whl/cu124 was used.

# Usage
Run the application using the frontend script:

python frontend.py
