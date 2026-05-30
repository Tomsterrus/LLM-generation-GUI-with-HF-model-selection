# backend.py
import torch
import psutil
import gc
import os
from threading import Thread
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from tqdm import tqdm as original_tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_progress_callback = None

class _TqdmPatcher(original_tqdm):
    def update(self, n=1):
        result = super().update(n)
        if _progress_callback and self.total:
            _progress_callback(self.n / self.total)
        return result

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
    
    # Słowa kluczowe wskazujące na kwantyzację, których chcemy unikać
    QUANT_KEYWORDS = ["gptq", "awq", "gguf", "exl2", "quantized", "4bit", "8bit", "bnb"]

    try:
        models = api.list_models(
            filter="text-generation",
            sort="downloads",
            limit=50
        )
    except Exception as e:
        log_callback(f"Error fetching list: {e}")
        return {}
    
    compatible_models = {}
    
    for model in models:
        try:
            detailed_info = api.model_info(model.id, files_metadata=True)
            log_callback(f"Model: {model.id}")
            
            # 1. Sprawdzenie Gated
            is_gated = getattr(detailed_info, 'gated', False)
            if is_gated:
                log_callback("  - Gated: YES (Requires HF authentication)")
                log_callback("  - Status: SKIPPED")
                log_callback("-" * 40)
                continue

            # 2. Sprawdzenie kwantyzacji (Tagi i Nazwa)
            tags = [t.lower() for t in getattr(detailed_info, 'tags', [])]
            model_id_lower = model.id.lower()
            
            is_quantized = any(k in model_id_lower for k in QUANT_KEYWORDS) or \
                           any(k in tags for k in QUANT_KEYWORDS)
            
            # Sprawdzenie specyficznej sekcji w konfiguracji (jeśli dostępna)
            if hasattr(detailed_info, 'config') and detailed_info.config:
                if "quantization_config" in detailed_info.config:
                    is_quantized = True

            if is_quantized:
                log_callback("  - Quantized: YES (Unsupported format)")
                log_callback("  - Status: SKIPPED")
                log_callback("-" * 40)
                continue
            else:
                log_callback("  - Quantized: NO (Standard weights)")

            # 3. Liczenie rozmiaru safetensors
            total_size_bytes = 0
            if hasattr(detailed_info, 'siblings') and detailed_info.siblings:
                for file in detailed_info.siblings:
                    if file.rfilename.endswith(".safetensors") and file.size:
                        total_size_bytes += file.size
            
            if total_size_bytes == 0:
                log_callback("  - Status: SKIPPED (No .safetensors found)")
                log_callback("-" * 40)
                continue
                
            size_gb = total_size_bytes / (1024 ** 3)
            required_mem = size_gb * 1.1
            
            log_callback(f"  - Size: {round(size_gb, 2)} GB")
            log_callback(f"  - Estimated RAM/VRAM: {round(required_mem, 2)} GB")
            
            if required_mem <= max_memory_gb:
                compatible_models[model.id] = round(required_mem, 2)
                log_callback("  - Status: COMPATIBLE")
            else:
                log_callback("  - Status: INSUFFICIENT MEMORY")
            
        except Exception as e:
            log_callback(f"Model: {model.id} - Error: {str(e)}")
        
        log_callback("-" * 40)
            
    return compatible_models
   
def load_hf_model(model_id, device, log_callback, progress_callback):
    global current_model, current_tokenizer, _progress_callback
    clear_memory()

    try:
        import transformers
        import tqdm as tqdm_module
        
        _progress_callback = progress_callback
        original = tqdm_module.tqdm
        tqdm_module.tqdm = _TqdmPatcher
        transformers.utils.logging.disable_progress_bar()  # wyłącz domyślny pasek

        progress_callback(0.0)
        log_callback(f"Loading tokenizer for {model_id}...")
        current_tokenizer = AutoTokenizer.from_pretrained(model_id)

        log_callback(f"Initializing model on {device}...")
        device_map = "auto" if device == "cuda" else "cpu"

        current_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True
        )

        progress_callback(1.0)
        log_callback("Model loaded successfully.")
        return True

    except Exception as e:
        log_callback(f"Error loading model: {str(e)}")
        return False

    finally:
        tqdm_module.tqdm = original
        _progress_callback = None

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