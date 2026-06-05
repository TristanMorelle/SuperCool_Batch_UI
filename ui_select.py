import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox


class SuperCoolUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SuperCool Image Upscaler Pipeline")
        self.root.geometry("550x420")
        self.root.resizable(False, False)

        self.checkpoint_dir = r"C:\Workspace\SuperCool\checkpoints"
        self.selected_files = []
        self.available_checkpoints = {}
        
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        main_frame.columnconfigure(1, weight=1)

        # --- SECTION 1: FILE SELECTION ---
        ttk.Label(main_frame, text="File Selection", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        self.browse_btn = ttk.Button(main_frame, text="Select Image Files...", command=self.browse_files)
        self.browse_btn.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        self.file_count_label = ttk.Label(main_frame, text="No files selected", foreground="gray")
        self.file_count_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=(0, 10))

        # --- SECTION 2: AUTO CHECKPOINT DROPDOWN ---
        ttk.Label(main_frame, text="Model Weights Checkpoint", font=("Segoe UI", 11, "bold")).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 10))
        
        ttk.Label(main_frame, text="Select Checkpoint:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(main_frame, textvariable=self.checkpoint_var, state="readonly", width=40)
        self.checkpoint_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self.on_checkpoint_changed)

        # --- SECTION 3: CONFIGURATIONS ---
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(main_frame, text="Manual Overrides (Used for Raw Weights / Safetensors)", font=("Segoe UI", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        ttk.Label(main_frame, text="Upscale Factor:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.upscale_var = tk.StringVar(value="4x")
        self.upscale_combo = ttk.Combobox(main_frame, textvariable=self.upscale_var, values=["2x", "4x", "8x"], state="readonly", width=15)
        self.upscale_combo.grid(row=6, column=1, sticky=tk.W, pady=5)

        ttk.Label(main_frame, text="Fallback Profile:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.profile_var = tk.StringVar(value="LARGE")
        self.profile_combo = ttk.Combobox(main_frame, textvariable=self.profile_var, values=["SMALL", "MEDIUM", "LARGE"], state="readonly", width=15)
        self.profile_combo.grid(row=7, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(main_frame, text="Execution Device:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ttk.Combobox(main_frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=15)
        self.device_combo.grid(row=8, column=1, sticky=tk.W, pady=5)

        # --- SECTION 4: ACTIONS ---
        ttk.Separator(main_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky="ew", pady=15)
        
        self.run_btn = ttk.Button(main_frame, text="START PIPELINE RUN", command=self.start_pipeline, state=tk.DISABLED)
        self.run_btn.grid(row=10, column=0, columnspan=2, sticky=(tk.E, tk.W), ipady=8)

        self.scan_checkpoints_folder()

    def scan_checkpoints_folder(self):
        if not os.path.exists(self.checkpoint_dir):
            try: os.makedirs(self.checkpoint_dir, exist_ok=True)
            except Exception: pass
                
        if os.path.exists(self.checkpoint_dir):
            files = os.listdir(self.checkpoint_dir)
            valid_extensions = (".safetensors", ".pt", ".pth")
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.available_checkpoints[file] = os.path.abspath(os.path.join(self.checkpoint_dir, file))
        
        if self.available_checkpoints:
            sorted_names = sorted(list(self.available_checkpoints.keys()))
            self.checkpoint_combo['values'] = sorted_names
            self.checkpoint_combo.current(0)
            self.on_checkpoint_changed(None)
        else:
            self.checkpoint_combo['values'] = ["No checkpoint files found (.safetensors/.pt/.pth)"]
            self.checkpoint_combo.current(0)

    def on_checkpoint_changed(self, event=None):
        selected_name = self.checkpoint_var.get()
        if not selected_name or selected_name.startswith("No checkpoint"):
            return

        name_lower = selected_name.lower()
        if "small" in name_lower:
            self.profile_var.set("SMALL")
        elif "medium" in name_lower:
            self.profile_var.set("MEDIUM")
        elif "large" in name_lower:
            self.profile_var.set("LARGE")
            
        self.check_ready_state()

    def browse_files(self):
        filetypes = [("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        files = filedialog.askopenfilenames(title="Select Images for Upscaling", filetypes=filetypes)
        if files:
            self.selected_files = list(files)
            self.file_count_label.config(text=f"Selected {len(self.selected_files)} file(s)", foreground="green")
            self.check_ready_state()
        else:
            self.selected_files = []
            self.file_count_label.config(text="No files selected", foreground="gray")
            self.run_btn.config(state=tk.DISABLED)

    def check_ready_state(self):
        selected_name = self.checkpoint_var.get()
        has_checkpoint = selected_name and not selected_name.startswith("No checkpoint")
        if self.selected_files and has_checkpoint:
            self.run_btn.config(state=tk.NORMAL)
        else:
            self.run_btn.config(state=tk.DISABLED)

    def start_pipeline(self):
        self.run_btn.config(text="PROCESSING BATCH...", state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.checkpoint_combo.config(state=tk.DISABLED)
        self.upscale_combo.config(state=tk.DISABLED)
        self.profile_combo.config(state=tk.DISABLED)
        self.device_combo.config(state=tk.DISABLED)

        threading.Thread(target=self._execute_batch_worker, daemon=True).start()

    def _execute_batch_worker(self):
        selected_name = self.checkpoint_var.get()
        chosen_checkpoint_path = self.available_checkpoints.get(selected_name)
        factor_str = self.upscale_var.get().replace("x", "") 
        profile = self.profile_var.get().lower()
        device = self.device_var.get()

        print("\n====================================================")
        print("Starting SuperCool Batch Processing Pipeline")
        print("====================================================")
        print(f"Using Checkpoint: {chosen_checkpoint_path}")
        
        unique_upscale_folders = set()
        
        try:
            for idx, img_path in enumerate(self.selected_files, 1):
                filename = os.path.basename(img_path)
                print(f"\nProcessing [{idx}/{len(self.selected_files)}]: {filename}")
                
                # Executing custom standalone upscale_pipeline entry-point
                cmd = [
                    sys.executable, "-u", "upscale_pipeline.py",
                    "--image_path", img_path,
                    "--device", device,
                    "--upscale_ratio", factor_str,
                    "--profile_override", profile,
                    "--checkpoint_path", chosen_checkpoint_path
                ]
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                if process.stdout:
                    for line in process.stdout:
                        print(line, end="")
                        if "TARGET_DIR_TRACKER_TOKEN:" in line:
                            extracted_folder = line.split("TARGET_DIR_TRACKER_TOKEN:")[-1].strip()
                            if extracted_folder:
                                unique_upscale_folders.add(extracted_folder)
                process.wait()
            
            print("\n====================================================")
            print("Batch Processing Finished Successfully!")
            print("====================================================")
            
        except Exception as e:
            print(f"\nWorker Thread Processing Error: {e}")
            
        self.root.after(0, self._finalize_pipeline_run, unique_upscale_folders)

    def _finalize_pipeline_run(self, unique_upscale_folders):
        self.browse_btn.config(state=tk.NORMAL)
        self.checkpoint_combo.config(state="readonly")
        self.upscale_combo.config(state="readonly")
        self.profile_combo.config(state="readonly")
        self.device_combo.config(state="readonly")
        self.run_btn.config(text="START PIPELINE RUN", state=tk.NORMAL)
        
        for upscale_folder in unique_upscale_folders:
            if os.path.exists(upscale_folder):
                os.startfile(upscale_folder)
                
        messagebox.showinfo("Success", f"Finished upscaling all {len(self.selected_files)} images completely!")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperCoolUI(root)
    root.mainloop()