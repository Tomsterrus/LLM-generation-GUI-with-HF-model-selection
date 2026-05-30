# HF Model Generator Environment

A simple desktop environment for loading and generating text using Hugging Face models, built with `customtkinter`. The application separates the frontend UI from the backend logic and monitors system resources (CPU and GPU memory).

## Prerequisites
- **Python**: 3.12.x (Recommended to avoid dependency issues with PyTorch)
- **CUDA Toolkit**: 12.4 or compatible (if using an NVIDIA GPU)
- **OS**: Windows

## Installation

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/generation-with-model-selection.git
cd generation-with-model-selection

2. Create and activate a virtual environment (Python 3.12 required)

python -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

3. Install the dependencies:

pip install -r requirements.txt

# Note: The standard torch installation usually includes the default CUDA dependencies. If your GPU is not detected, ensure your NVIDIA drivers are up to date or install a specific PyTorch version from [PyTorch Get Started.](https://pytorch.org/get-started/locally/)

# Usage
Run the application using the frontend script:

python frontend.py
