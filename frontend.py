# frontend.py
import customtkinter as ctk
from tkinter import messagebox
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HF Model Generator Environment")
        self.geometry("700x650")

        self.found_models = {} # Stores {model_id: required_gb}

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

        self.scan_button = ctk.CTkButton(self.control_frame, text="Scan HF Models", command=self.start_scan_thread)
        self.scan_button.pack(side="left", pady=10, padx=10)

        self.model_var = ctk.StringVar(value="No models loaded")
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=["No models loaded"], width=300)
        self.model_dropdown.pack(side="left", pady=10, padx=10)

        self.load_button = ctk.CTkButton(self.control_frame, text="Load Model", command=self.handle_load_model, state="disabled")
        self.load_button.pack(side="left", pady=10, padx=10)

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
        self.load_button.configure(state="disabled")
        self.log_textbox.delete("1.0", "end")
        
        # Determine max available memory between CPU and GPU
        max_mem = max(self.mem_data['gpu_available_gb'], self.mem_data['cpu_available_gb'])
        self.add_log(f"Scanning models for max available resource ({max_mem} GB)...")

        thread = threading.Thread(target=self.run_scan, args=(max_mem,), daemon=True)
        thread.start()

    def run_scan(self, max_mem):
        try:
            self.found_models = backend.fetch_compatible_models(max_mem, self.add_log)
            if self.found_models:
                model_names = list(self.found_models.keys())
                self.model_dropdown.configure(values=model_names)
                self.model_var.set(model_names[0])
                self.load_button.configure(state="normal")
                self.add_log(f"Done. Found {len(self.found_models)} compatible models.")
            else:
                self.add_log("No compatible models found within memory limits.")
        except Exception as e:
            self.add_log(f"Error during scan: {str(e)}")
        finally:
            self.scan_button.configure(state="normal")

    def handle_load_model(self):
        selected_model = self.model_var.get()
        if selected_model not in self.found_models:
            return

        required_gb = self.found_models[selected_model]
        gpu_avail = self.mem_data['gpu_available_gb']

        if required_gb <= gpu_avail:
            self.add_log(f"Loading {selected_model} to GPU...")
            self.proceed_to_next_step(selected_model, "cuda")
        else:
            # Check if it fits in CPU
            if required_gb <= self.mem_data['cpu_available_gb']:
                answer = messagebox.askyesno("Warning", "Not enough GPU RAM available. Run the model on CPU instead? (Requires lots of patience...)")
                if answer:
                    self.add_log(f"Loading {selected_model} to CPU...")
                    self.proceed_to_next_step(selected_model, "cpu")
                else:
                    self.add_log("Load cancelled by user.")
            else:
                self.add_log("Error: Model no longer fits in available system memory.")

    def proceed_to_next_step(self, model_id, device):
        # Placeholder for next step
        self.add_log(f"Ready to initialize {model_id} on {device.upper()}.")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()