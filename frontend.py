# frontend.py
import customtkinter as ctk
from tkinter import messagebox
import backend
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HF Model Generator Environment")
        self.geometry("800x850")

        self.found_models = {}
        self.is_scanning = False
        self.is_loading = False

        # System Info Frame
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.pack(pady=10, padx=20, fill="x")

        self.cpu_label = ctk.CTkLabel(self.info_frame, text="CPU: Loading...")
        self.cpu_label.pack(pady=2, padx=10, anchor="w")

        self.gpu_label = ctk.CTkLabel(self.info_frame, text="GPU: Loading...")
        self.gpu_label.pack(pady=2, padx=10, anchor="w")

        # Top Control Frame (Persistent)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.scan_button = ctk.CTkButton(self.control_frame, text="Scan HF Models", command=self.start_scan_thread)
        self.scan_button.pack(side="left", padx=5)

        self.model_var = ctk.StringVar(value="No models loaded")
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=["No models loaded"], width=300)
        self.model_dropdown.pack(side="left", padx=5)

        self.load_button = ctk.CTkButton(self.control_frame, text="Load Model", command=self.handle_load_model, state="disabled")
        self.load_button.pack(side="left", padx=5)

        # Progress Frame
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Loading Safetensors: 0%")
        self.progress_label.pack(side="left", padx=10)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)
        self.progress_bar.set(0)

        # Main View Container
        self.view_container = ctk.CTkFrame(self)
        self.view_container.pack(pady=10, padx=20, fill="both", expand=True)

        self.show_scan_logs_view()
        self.update_memory_info()

    def show_scan_logs_view(self):
        for widget in self.view_container.winfo_children():
            widget.destroy()
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

        self.generate_button = ctk.CTkButton(self.view_container, text="Generate", command=self.start_generation_thread)
        self.generate_button.pack(pady=10)

        self.result_textbox = ctk.CTkTextbox(self.view_container, height=350)
        self.result_textbox.pack(pady=5, padx=10, fill="both", expand=True)

    def update_memory_info(self):
        self.mem_data = backend.get_system_memory()
        cpu_text = f"CPU: {self.mem_data['cpu_available_gb']} GB / {self.mem_data['cpu_total_gb']} GB"
        self.cpu_label.configure(text=cpu_text)
        
        if self.mem_data['gpu_name'] != "No GPU":
            gpu_text = f"GPU ({self.mem_data['gpu_name']}): {self.mem_data['gpu_available_gb']} GB / {self.mem_data['gpu_total_gb']} GB"
        else:
            gpu_text = "GPU: Not detected"
        self.gpu_label.configure(text=gpu_text)
        
        # Schedule next update
        self.after(5000, self.update_memory_info)

    def add_log(self, message):
        if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.see("end")

    def set_progress(self, value):
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"Loading Safetensors: {int(value * 100)}%")

    def start_scan_thread(self):
        if self.is_scanning or self.is_loading: return
        self.show_scan_logs_view()
        self.is_scanning = True
        self.scan_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.log_textbox.delete("1.0", "end")
        max_mem = max(self.mem_data['gpu_available_gb'], self.mem_data['cpu_available_gb'])
        threading.Thread(target=self.run_scan, args=(max_mem,), daemon=True).start()

    def run_scan(self, max_mem):
        try:
            self.found_models = backend.fetch_compatible_models(max_mem, self.add_log)
            if self.found_models:
                model_names = list(self.found_models.keys())
                self.model_dropdown.configure(values=model_names)
                self.model_var.set(model_names[0])
                self.load_button.configure(state="normal")
        finally:
            self.is_scanning = False
            self.scan_button.configure(state="normal")

    def handle_load_model(self):
        if self.is_loading or self.is_scanning: return
        selected_model = self.model_var.get()
        if selected_model == "No models loaded": return
        
        required_gb = self.found_models[selected_model]
        device = "cuda" if required_gb <= self.mem_data['gpu_available_gb'] else "cpu"
        
        if device == "cpu" and not messagebox.askyesno("Warning", "Run on CPU?"):
            return

        self.is_loading = True
        self.load_button.configure(state="disabled")
        self.scan_button.configure(state="disabled")
        
        # Show logs before loading if we were in generation view
        self.show_scan_logs_view()
        self.progress_frame.pack(after=self.control_frame, pady=5, padx=20, fill="x")
        self.set_progress(0)
        threading.Thread(target=self.run_load, args=(selected_model, device), daemon=True).start()

    def run_load(self, model_id, device):
        success = backend.load_hf_model(model_id, device, self.add_log, self.set_progress)
        self.is_loading = False
        if success:
            self.after(500, lambda: self.show_generation_view(model_id))
        else:
            self.after(0, lambda: self.progress_frame.pack_forget())
        
        self.after(0, lambda: self.load_button.configure(state="normal"))
        self.after(0, lambda: self.scan_button.configure(state="normal"))

    def start_generation_thread(self):
        prompt = self.prompt_input.get("1.0", "end-1c").strip()
        if not prompt: return
        self.generate_button.configure(state="disabled", text="Generating...")
        self.result_textbox.delete("1.0", "end")
        threading.Thread(target=self.run_generation, args=(prompt,), daemon=True).start()

    def run_generation(self, prompt):
        for token in backend.generate_response_stream(prompt):
            self.after(0, lambda t=token: self.append_token(t))
        self.after(0, self.finish_generation)

    def append_token(self, token):
        if hasattr(self, 'result_textbox') and self.result_textbox.winfo_exists():
            self.result_textbox.insert("end", token)
            self.result_textbox.see("end")

    def finish_generation(self):
        if hasattr(self, 'generate_button') and self.generate_button.winfo_exists():
            self.generate_button.configure(state="normal", text="Generate")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    app = App()
    app.mainloop()