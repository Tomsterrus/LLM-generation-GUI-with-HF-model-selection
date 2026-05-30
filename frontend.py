# frontend.py
import customtkinter as ctk
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HF Model Generator Environment")
        self.geometry("700x600")

        # System Info Frame
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.pack(pady=10, padx=20, fill="x")

        self.cpu_label = ctk.CTkLabel(self.info_frame, text="CPU Memory: Loading...")
        self.cpu_label.pack(pady=2, padx=10, anchor="w")

        self.gpu_label = ctk.CTkLabel(self.info_frame, text="GPU Memory: Loading...")
        self.gpu_label.pack(pady=2, padx=10, anchor="w")

        # Control Frame
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.scan_button = ctk.CTkButton(self.control_frame, text="Scan HF for Compatible Models", command=self.start_scan_thread)
        self.scan_button.pack(side="left", pady=10, padx=10)

        self.model_var = ctk.StringVar(value="No models loaded")
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=["No models loaded"], width=300)
        self.model_dropdown.pack(side="right", pady=10, padx=10)

        # Log Window
        self.log_label = ctk.CTkLabel(self, text="Processing Logs:")
        self.log_label.pack(pady=(10, 0), padx=20, anchor="w")
        
        self.log_textbox = ctk.CTkTextbox(self, height=300)
        self.log_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.update_memory_info()

    def update_memory_info(self):
        self.mem_data = backend.get_system_memory()
        cpu_text = f"CPU Memory: {self.mem_data['cpu_available_gb']} GB available / {self.mem_data['cpu_total_gb']} GB total"
        self.cpu_label.configure(text=cpu_text)
        
        if self.mem_data['gpu_name'] != "No GPU":
            gpu_text = f"GPU ({self.mem_data['gpu_name']}): {self.mem_data['gpu_available_gb']} GB available / {self.mem_data['gpu_total_gb']} GB total"
        else:
            gpu_text = "GPU: No compatible CUDA device detected"
        self.gpu_label.configure(text=gpu_text)

    def add_log(self, message):
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def start_scan_thread(self):
        self.scan_button.configure(state="disabled")
        self.log_textbox.delete("1.0", "end")
        
        # Determine max available memory (GPU preferred if available)
        if self.mem_data['gpu_total_gb'] > 0:
            max_mem = self.mem_data['gpu_available_gb']
            self.add_log(f"Scanning models for GPU ({max_mem} GB)...")
        else:
            max_mem = self.mem_data['cpu_available_gb']
            self.add_log(f"Scanning models for CPU ({max_mem} GB)...")

        thread = threading.Thread(target=self.run_scan, args=(max_mem,), daemon=True)
        thread.start()

    def run_scan(self, max_mem):
        try:
            models = backend.fetch_compatible_models(max_mem, self.add_log)
            if models:
                self.model_dropdown.configure(values=models)
                self.model_var.set(models[0])
                self.add_log(f"Done. Found {len(models)} compatible models.")
            else:
                self.add_log("No compatible models found within memory limits.")
        except Exception as e:
            self.add_log(f"Error during scan: {str(e)}")
        finally:
            self.scan_button.configure(state="normal")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()