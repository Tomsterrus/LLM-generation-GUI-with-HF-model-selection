# backend.py
import torch
import psutil
from huggingface_hub import HfApi

def get_system_memory():
    vm = psutil.virtual_memory()
    cpu_total = vm.total / (1024 ** 3)
    cpu_available = vm.available / (1024 ** 3)
    
    gpu_total = 0.0
    gpu_available = 0.0
    gpu_name = "No GPU"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        gpu_total = total_mem / (1024 ** 3)
        gpu_available = free_mem / (1024 ** 3)

    return {
        "cpu_total_gb": round(cpu_total, 2),
        "cpu_available_gb": round(cpu_available, 2),
        "gpu_name": gpu_name,
        "gpu_total_gb": round(gpu_total, 2),
        "gpu_available_gb": round(gpu_available, 2)
    }

def fetch_compatible_models(max_memory_gb, log_callback):
    api = HfApi()
    log_callback("Fetching model list from Hugging Face...")
    
    models = api.list_models(
        filter="text-generation",
        sort="downloads",
        direction=-1,
        limit=50,
        expand=["siblings"]
    )
    
    compatible_models = {} # Changed to dict to store model_id: required_gb
    
    for model in models:
        total_size_bytes = 0
        if hasattr(model, 'siblings') and model.siblings:
            for file in model.siblings:
                if file.rfilename.endswith(".safetensors"):
                    if hasattr(file, 'size') and file.size is not None:
                        total_size_bytes += file.size
            
        if total_size_bytes == 0:
            try:
                detailed_info = api.model_info(model.id, files_metadata=True)
                for file in detailed_info.siblings:
                    if file.rfilename.endswith(".safetensors") and file.size:
                        total_size_bytes += file.size
            except:
                continue

        if total_size_bytes == 0:
            continue
            
        size_gb = total_size_bytes / (1024 ** 3)
        required_mem = size_gb * 1.1
        
        log_callback(f"Checking {model.id}: {round(size_gb, 2)} GB (Req: {round(required_mem, 2)} GB)")
        
        if required_mem <= max_memory_gb:
            compatible_models[model.id] = round(required_mem, 2)
            log_callback(f"-> Match found: {model.id}")
            
    return compatible_models