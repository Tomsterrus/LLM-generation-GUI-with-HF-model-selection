# test_hf_size.py
from huggingface_hub import HfApi

def estimate_required_memory(model_id):
    api = HfApi()
    
    try:
        # Fetch model info with file metadata to get file sizes
        model_info = api.model_info(model_id, files_metadata=True)
        
        total_safetensors_size_bytes = 0
        
        for file in model_info.siblings:
            # Check for safetensors files (handles both single and sharded models)
            if file.rfilename.endswith(".safetensors"):
                if file.size is not None:
                    total_safetensors_size_bytes += file.size

        if total_safetensors_size_bytes == 0:
            print(f"[{model_id}] No .safetensors files found.")
            return None
            
        size_gb = total_safetensors_size_bytes / (1024 ** 3)
        required_memory_gb = size_gb * 1.1
        
        print(f"Model: {model_id}")
        print(f"Safetensors size: {round(size_gb, 2)} GB")
        print(f"Estimated required memory (x1.1): {round(required_memory_gb, 2)} GB")
        
        return round(required_memory_gb, 2)
        
    except Exception as e:
        print(f"Error fetching data for {model_id}: {e}")
        return None

if __name__ == "__main__":
    # Example models to test (single file and sharded)
    test_models = [
        "gpt2", 
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen1.5-0.5B"
    ]
    
    for model in test_models:
        estimate_required_memory(model)
        print("-" * 40)