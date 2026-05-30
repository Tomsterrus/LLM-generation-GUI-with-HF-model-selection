# frontend.py
import customtkinter as ctk
from tkinter import messagebox
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HF Model Generator Environment")
        self.geometry("800x800")

        self.found_models = {}

        # System Info Frame
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.pack(pady=10, padx=20, fill="x")

        self.cpu_label = ctk.CTkLabel(self.info_frame, text="CPU Memory: Loading...")
        self.cpu_label.pack(pady=2, padx=10, anchor="w")

        self.gpu_label = ctk.CTkLabel(self.info_frame, text="GPU Memory: Loading...")
        self.gpu_label.pack(pady=2, padx=10, anchor="w")

        # Top Control Frame
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.scan_button = ctk.CTkButton(self.control_frame, text="Scan HF Models", command=self.start_scan_thread)
        self.scan_button.pack(side="left", padx=5)

        self.model_var = ctk.StringVar(value="No models loaded")
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=["No models loaded"], width=300)
        self.model_dropdown.pack(side="left", padx=5)

        self.load_button = ctk.CTkButton(self.control_frame, text="Load Model", command=self.handle_load_model, state="disabled")
        self.load_button.pack(side="left", padx=5)

        # Progress Frame (Hidden by default)
        self.progress_frame = ctk.CTkFrame(self)
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Loading Safetensors: 0%")
        self.progress_label.pack(side="left", padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)
        self.progress_bar.set(0)

        # View Container
        self.view_container = ctk.CTkFrame(self)
        self.view_container.pack(pady=10, padx=20, fill="both", expand=True)

        self.show_scan_logs_view()
        self.update_memory_info()

    def show_scan_logs_view(self):
        for widget in self.view_container.winfo_children():
            widget.destroy()
            
        self.log_label = ctk.CTkLabel(self.view_container, text="Model Scanning & System Logs:")
        self.log_label.pack(anchor="w", padx=10, pady=(5,0))
        
        self.log_textbox = ctk.CTkTextbox(self.view_container)
        self.log_textbox.pack(pady=10, padx=10, fill="both", expand=True)

    def show_generation_view(self, model_id):
        self.progress_frame.pack_forget()
        for widget in self.view_container.winfo_children():
            widget.destroy()

        self.gen_label = ctk.CTkLabel(self.view_container, text=f"Active Model: {model_id}", font=("Arial", 14, "bold"))
        self.gen_label.pack(pady=5)

        self.prompt_input = ctk.CTkTextbox(self.view_container, height=150)
        self.prompt_input.pack(pady=5, padx=10, fill="x")
        self.prompt_input.insert("1.0", "Enter your prompt here...")

        self.generate_button = ctk.CTkButton(self.view_container, text="Generate", command=self.handle_generation)
        self.generate_button.pack(pady=10)

        self.result_textbox = ctk.CTkTextbox(self.view_container, height=300)
        self.result_textbox.pack(pady=5, padx=10, fill="both", expand=True)

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
        if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.see("end")

    def set_progress(self, value):
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"Loading Safetensors: {int(value * 100)}%")

    def start_scan_thread(self):
        self.show_scan_logs_view()
        self.scan_button.configure(state="disabled")
        self.log_textbox.delete("1.0", "end")
        
        max_mem = max(self.mem_data['gpu_available_gb'], self.mem_data['cpu_available_gb'])
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
        finally:
            self.scan_button.configure(state="normal")

    def handle_load_model(self):
        selected_model = self.model_var.get()
        required_gb = self.found_models[selected_model]
        gpu_avail = self.mem_data['gpu_available_gb']

        device = "cuda"
        if required_gb > gpu_avail:
            if messagebox.askyesno("Warning", "Not enough GPU RAM available. Run the model on CPU instead? (Requires lots of patience...)"):
                device = "cpu"
            else:
                return

        self.load_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        
        # Show progress bar only for loading
        self.progress_frame.pack(after=self.control_frame, pady=5, padx=20, fill="x")
        self.set_progress(0)
        
        thread = threading.Thread(target=self.run_load, args=(selected_model, device), daemon=True)
        thread.start()

    def run_load(self, model_id, device):
        success = backend.load_hf_model(model_id, device, self.add_log, self.set_progress)
        if success:
            self.after(500, lambda: self.show_generation_view(model_id))
        else:
            self.after(0, lambda: self.progress_frame.pack_forget())
            self.load_button.configure(state="normal")
            self.scan_button.configure(state="normal")

    def handle_generation(self):
        self.result_textbox.insert("end", "Generation logic not yet connected.\n")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    app = App()
    app.mainloop()