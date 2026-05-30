# backend.py
import torch
import psutil
import gc
from threading import Thread
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Global variables to store the loaded model and tokenizer
current_model = None
current_tokenizer = None

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

def clear_memory():
    global current_model, current_tokenizer
    current_model = None
    current_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    compatible_models = {}
    
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
        
        log_callback(f"Model: {model.id}")
        log_callback(f"Safetensors size: {round(size_gb, 2)} GB")
        log_callback(f"Estimated required memory (x1.1): {round(required_mem, 2)} GB")
        
        if required_mem <= max_memory_gb:
            compatible_models[model.id] = round(required_mem, 2)
            log_callback("Status: COMPATIBLE")
        else:
            log_callback("Status: INSUFFICIENT MEMORY")
        
        log_callback("-" * 40)
            
    return compatible_models

def load_hf_model(model_id, device, log_callback, progress_callback):
    global current_model, current_tokenizer
    
    # Clear existing model from memory before loading a new one
    clear_memory()
    
    try:
        progress_callback(0.1)
        log_callback(f"Loading tokenizer for {model_id}...")
        current_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        progress_callback(0.3)
        log_callback(f"Initializing model on {device}...")
        
        device_map = "auto" if device == "cuda" else "cpu"
        
        progress_callback(0.5)
        current_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        
        progress_callback(1.0)
        log_callback("Model and tokenizer loaded successfully.")
        return True
    except Exception as e:
        log_callback(f"Error loading model: {str(e)}")
        return False

def generate_response_stream(user_input):
    global current_model, current_tokenizer
    
    if current_model is None or current_tokenizer is None:
        yield "Error: Model not loaded."
        return

    try:
        if hasattr(current_tokenizer, "chat_template") and current_tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": user_input}]
            prompt = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = user_input
            
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
        streamer = TextIteratorStreamer(current_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=current_tokenizer.eos_token_id
        )

        thread = Thread(target=current_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    except Exception as e:
        yield f"Error during generation: {str(e)}"