# HF Model Generator Environment

A desktop GUI for discovering, loading, and interacting with Large Language Models (LLMs) from Hugging Face. Built with `customtkinter`, it provides a streamlined workflow for testing models while monitoring hardware constraints.

## Key Features

- Smart Model Filtering: Automatically filters models from Hugging Face Hub based on:
    - Hardware Compatibility: Calculates if a model fits into your available VRAM/RAM (with a 10% safety buffer).
    - Access Rights: Identifies and filters out "Gated" models requiring manual approval.
    - Format Support: Detects and skips quantized formats (GPTQ, AWQ, GGUF) incompatible with standard transformers loading.
- Live Hardware Monitoring: Real-time tracking of CPU and GPU (NVIDIA) memory usage.
- Security-First Validation: Automatically detects and skips models that require custom Python code execution (`trust_remote_code=True`). The application identifies these by scanning for the `auto_map` property in the model configuration, protecting the local environment from potentially unverified remote scripts.
- Streaming Interface: Real-time text generation response streaming.
- Robust Backend: Decoupled architecture separating UI logic from heavy model operations.


## Tech Stack

- Frontend: customtkinter (Modern UI)
- Backend: PyTorch, Hugging Face Transformers
- Hardware Info: psutil
- Concurrency: threading for non-blocking UI during model loading and inference.

## Prerequisites

- Python: 3.12.x (Recommended)
- CUDA Toolkit: 12.4 or compatible (for NVIDIA GPU acceleration)
- OS: Windows 11 (tested)

## Setup & Configuration

### 1. Hugging Face Token (Recommended)
To avoid API rate limits and access metadata for a wider range of models:
1. Create a token at hf.co/settings/tokens.
2. Set it as an environment variable:
   - PowerShell: $env:HF_TOKEN="your_token_here"
   - CMD: set HF_TOKEN=your_token_here
   - Or create a .env file in the project root.

### 2. Windows Symlinks
To optimize disk space and prevent file duplication, enable Developer Mode in Windows Settings. This allows the Hugging Face library to use symbolic links.

## Installation

1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/generation-with-model-selection.git
cd generation-with-model-selection

2. Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies:
pip install -r requirements.txt

Note on GPU: If your GPU is not detected, ensure NVIDIA drivers are up to date. This project was tested with PyTorch for CUDA 12.4. You can install it specifically via:
pip install torch --index-url https://download.pytorch.org/whl/cu124

## Usage

Run the application using the frontend script:
python frontend.py

## Hardware Compatibility Note
Tested on:
- CPU: Intel(R) Core(TM) i5-12500H
- RAM: 16.0 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)