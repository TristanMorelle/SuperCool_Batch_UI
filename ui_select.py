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
        self.root.geometry("600x820")
        self.root.resizable(False, False)

        self.checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
        self.selected_files = []
        self.available_checkpoints = {}
        
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        main_frame = ttk.Frame(root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        main_frame.columnconfigure(1, weight=1)

        # --- SECTION 1: FILE SELECTION ---
        ttk.Label(main_frame, text="File Selection", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        self.browse_btn = ttk.Button(main_frame, text="Select Image Files...", command=self.browse_files)
        self.browse_btn.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        self.file_count_label = ttk.Label(main_frame, text="No files selected", foreground="gray")
        self.file_count_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=(0, 10))

        # --- PIPELINE CONFIGURATION ---
        ttk.Label(main_frame, text="Pipeline Signal Flow", font=("Segoe UI", 11, "bold")).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 5))

        # STAGE 1: PRE-PROCESS
        self.pre_frame = ttk.LabelFrame(main_frame, text="Stage 1: Pre-Processing (Interpolation)", padding="10")
        self.pre_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.pre_active_var = tk.BooleanVar(value=False)
        self.pre_check = ttk.Checkbutton(self.pre_frame, text="Active", variable=self.pre_active_var, command=self.update_ui_states)
        self.pre_check.grid(row=0, column=0, sticky=tk.W, columnspan=4, pady=(0, 5))

        self.pre_mode_var = tk.StringVar(value="multiple")
        
        # Pre - Multiple Option
        self.pre_rad_multi = ttk.Radiobutton(self.pre_frame, text="Multiples:", variable=self.pre_mode_var, value="multiple", command=self.update_ui_states)
        self.pre_rad_multi.grid(row=1, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        
        self.pre_multi_var = tk.StringVar(value="0.5x")
        self.pre_multi_combo = ttk.Combobox(self.pre_frame, textvariable=self.pre_multi_var, values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"], state="readonly", width=10)
        self.pre_multi_combo.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Pre - Conform Option
        self.pre_rad_conform = ttk.Radiobutton(self.pre_frame, text="Conform to Max:", variable=self.pre_mode_var, value="conform", command=self.update_ui_states)
        self.pre_rad_conform.grid(row=2, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        
        ttk.Label(self.pre_frame, text="W:").grid(row=2, column=1, sticky=tk.E)
        self.pre_max_w_var = tk.StringVar(value="0")
        self.pre_max_w_entry = ttk.Entry(self.pre_frame, textvariable=self.pre_max_w_var, width=6)
        self.pre_max_w_entry.grid(row=2, column=2, sticky=tk.W, padx=(2, 10))
        
        ttk.Label(self.pre_frame, text="H:").grid(row=2, column=3, sticky=tk.E)
        self.pre_max_h_var = tk.StringVar(value="0")
        self.pre_max_h_entry = ttk.Entry(self.pre_frame, textvariable=self.pre_max_h_var, width=6)
        self.pre_max_h_entry.grid(row=2, column=4, sticky=tk.W, padx=(2, 0))

        # STAGE 2: AI MODEL
        self.ai_frame = ttk.LabelFrame(main_frame, text="Stage 2: AI Model Upscale", padding="10")
        self.ai_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.ai_active_var = tk.BooleanVar(value=True)
        self.ai_check = ttk.Checkbutton(self.ai_frame, text="Active", variable=self.ai_active_var, command=self.update_ui_states)
        self.ai_check.grid(row=0, column=0, sticky=tk.W)

        ttk.Label(self.ai_frame, text="Checkpoint:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(self.ai_frame, textvariable=self.checkpoint_var, state="readonly", width=45)
        self.checkpoint_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self.on_checkpoint_changed)

        ttk.Label(self.ai_frame, text="Profile fallback:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.profile_var = tk.StringVar(value="LARGE")
        self.profile_combo = ttk.Combobox(self.ai_frame, textvariable=self.profile_var, values=["SMALL", "MEDIUM", "LARGE"], state="readonly", width=15)
        self.profile_combo.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(5, 0))

        # STAGE 3: POST-PROCESS
        self.post_frame = ttk.LabelFrame(main_frame, text="Stage 3: Post-Processing (Interpolation)", padding="10")
        self.post_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.post_active_var = tk.BooleanVar(value=False)
        self.post_check = ttk.Checkbutton(self.post_frame, text="Active", variable=self.post_active_var, command=self.update_ui_states)
        self.post_check.grid(row=0, column=0, sticky=tk.W, columnspan=4, pady=(0, 5))

        self.post_mode_var = tk.StringVar(value="multiple")
        
        # Post - Multiple Option
        self.post_rad_multi = ttk.Radiobutton(self.post_frame, text="Multiples:", variable=self.post_mode_var, value="multiple", command=self.update_ui_states)
        self.post_rad_multi.grid(row=1, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        
        self.post_multi_var = tk.StringVar(value="1.0x")
        self.post_multi_combo = ttk.Combobox(self.post_frame, textvariable=self.post_multi_var, values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"], state="readonly", width=10)
        self.post_multi_combo.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Post - Conform Option
        self.post_rad_conform = ttk.Radiobutton(self.post_frame, text="Conform to Max:", variable=self.post_mode_var, value="conform", command=self.update_ui_states)
        self.post_rad_conform.grid(row=2, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        
        ttk.Label(self.post_frame, text="W:").grid(row=2, column=1, sticky=tk.E)
        self.post_max_w_var = tk.StringVar(value="0")
        self.post_max_w_entry = ttk.Entry(self.post_frame, textvariable=self.post_max_w_var, width=6)
        self.post_max_w_entry.grid(row=2, column=2, sticky=tk.W, padx=(2, 10))
        
        ttk.Label(self.post_frame, text="H:").grid(row=2, column=3, sticky=tk.E)
        self.post_max_h_var = tk.StringVar(value="0")
        self.post_max_h_entry = ttk.Entry(self.post_frame, textvariable=self.post_max_h_var, width=6)
        self.post_max_h_entry.grid(row=2, column=4, sticky=tk.W, padx=(2, 0))

        # --- SECTION 3: SYSTEM CONFIGURATIONS ---
        sys_frame = ttk.Frame(main_frame)
        sys_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(sys_frame, text="Device:").grid(row=0, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ttk.Combobox(sys_frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=10)
        self.device_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        self.skip_existing_var = tk.BooleanVar(value=True)
        self.skip_existing_check = ttk.Checkbutton(sys_frame, text="Skip existing files", variable=self.skip_existing_var)
        self.skip_existing_check.grid(row=0, column=2, sticky=tk.W, padx=20)

        # --- SECTION 4: ACTIONS ---
        ttk.Separator(main_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.run_btn = ttk.Button(main_frame, text="START PIPELINE RUN", command=self.start_pipeline, state=tk.DISABLED)
        self.run_btn.grid(row=8, column=0, columnspan=2, sticky=(tk.E, tk.W), ipady=8)

        self.scan_checkpoints_folder()
        self.update_ui_states()

    def update_ui_states(self, *args):
        if not self.pre_active_var.get():
            self.pre_rad_multi.config(state=tk.DISABLED)
            self.pre_rad_conform.config(state=tk.DISABLED)
            self.pre_multi_combo.config(state=tk.DISABLED)
            self.pre_max_w_entry.config(state=tk.DISABLED)
            self.pre_max_h_entry.config(state=tk.DISABLED)
        else:
            self.pre_rad_multi.config(state=tk.NORMAL)
            self.pre_rad_conform.config(state=tk.NORMAL)
            if self.pre_mode_var.get() == "multiple":
                self.pre_multi_combo.config(state="readonly")
                self.pre_max_w_entry.config(state=tk.DISABLED)
                self.pre_max_h_entry.config(state=tk.DISABLED)
            else:
                self.pre_multi_combo.config(state=tk.DISABLED)
                self.pre_max_w_entry.config(state=tk.NORMAL)
                self.pre_max_h_entry.config(state=tk.NORMAL)

        state = tk.NORMAL if self.ai_active_var.get() else tk.DISABLED
        self.checkpoint_combo.config(state="readonly" if self.ai_active_var.get() else tk.DISABLED)
        self.profile_combo.config(state="readonly" if self.ai_active_var.get() else tk.DISABLED)

        if not self.post_active_var.get():
            self.post_rad_multi.config(state=tk.DISABLED)
            self.post_rad_conform.config(state=tk.DISABLED)
            self.post_multi_combo.config(state=tk.DISABLED)
            self.post_max_w_entry.config(state=tk.DISABLED)
            self.post_max_h_entry.config(state=tk.DISABLED)
        else:
            self.post_rad_multi.config(state=tk.NORMAL)
            self.post_rad_conform.config(state=tk.NORMAL)
            if self.post_mode_var.get() == "multiple":
                self.post_multi_combo.config(state="readonly")
                self.post_max_w_entry.config(state=tk.DISABLED)
                self.post_max_h_entry.config(state=tk.DISABLED)
            else:
                self.post_multi_combo.config(state=tk.DISABLED)
                self.post_max_w_entry.config(state=tk.NORMAL)
                self.post_max_h_entry.config(state=tk.NORMAL)
                
        self.check_ready_state()

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
        if not selected_name or selected_name.startswith("No checkpoint"): return

        name_lower = selected_name.lower()
        if "small" in name_lower: self.profile_var.set("SMALL")
        elif "medium" in name_lower: self.profile_var.set("MEDIUM")
        elif "large" in name_lower: self.profile_var.set("LARGE")
            
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
        has_files = len(self.selected_files) > 0
        has_checkpoint = (not self.ai_active_var.get()) or (self.checkpoint_var.get() and not self.checkpoint_var.get().startswith("No checkpoint"))
        has_active_stage = self.pre_active_var.get() or self.ai_active_var.get() or self.post_active_var.get()
        
        if has_files and has_checkpoint and has_active_stage:
            self.run_btn.config(state=tk.NORMAL)
        else:
            self.run_btn.config(state=tk.DISABLED)

    def start_pipeline(self):
        self.run_btn.config(text="PROCESSING BATCH...", state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._execute_batch_worker, daemon=True).start()

    def _execute_batch_worker(self):
        selected_name = self.checkpoint_var.get()
        chosen_checkpoint_path = self.available_checkpoints.get(selected_name, "")
        checkpoint_base_name = os.path.splitext(os.path.basename(chosen_checkpoint_path))[0] if chosen_checkpoint_path else "NoAI"

        print("\n====================================================")
        print("Starting SuperCool Batch Processing Pipeline")
        print("====================================================")
        
        unique_upscale_folders = set()
        files_to_process = []

        # Identifier to keep files separate based on flow
        flow_id = ""
        if self.pre_active_var.get():
            if self.pre_mode_var.get() == "multiple": flow_id += f"Pre{self.pre_multi_var.get()}_"
            else: flow_id += f"PreConformW{self.pre_max_w_var.get()}H{self.pre_max_h_var.get()}_"
            
        if self.ai_active_var.get(): flow_id += f"{checkpoint_base_name}_"
        
        if self.post_active_var.get():
            if self.post_mode_var.get() == "multiple": flow_id += f"Post{self.post_multi_var.get()}"
            else: flow_id += f"PostConformW{self.post_max_w_var.get()}H{self.post_max_h_var.get()}"
            
        flow_id = flow_id.strip("_")

        if self.skip_existing_var.get():
            for img_path in self.selected_files:
                orig_dir = os.path.dirname(img_path)
                name_part, ext_part = os.path.splitext(os.path.basename(img_path))
                target_output_dir = os.path.abspath(os.path.join(orig_dir, "upscaled"))
                final_dest = os.path.join(target_output_dir, f"{name_part}_{flow_id}{ext_part}")
                
                if os.path.exists(final_dest):
                    print(f"Skipping existing: {os.path.basename(img_path)}")
                    unique_upscale_folders.add(target_output_dir)
                else:
                    files_to_process.append(img_path)
        else:
            files_to_process = self.selected_files

        if not files_to_process:
            print("\nNo new files to process.")
            self.root.after(0, self._finalize_pipeline_run, unique_upscale_folders)
            return
            
        # --- FIX: Write file paths to a temporary batch list text file ---
        batch_list_path = os.path.abspath("temp_batch_list.txt")
        with open(batch_list_path, "w", encoding="utf-8") as f:
            for img_path in files_to_process:
                f.write(f"{img_path}\n")
        
        try:
            print(f"\nProcessing {len(files_to_process)} files in a single batch...")
            
            # --- FIX: Send a single command pointing to the batch list ---
            cmd = [sys.executable, "-u", "upscale_pipeline.py", "--batch_list", batch_list_path, "--device", self.device_var.get()]
            
            if self.pre_active_var.get():
                cmd.extend(["--pre_active", "--pre_mode", self.pre_mode_var.get()])
                if self.pre_mode_var.get() == "multiple":
                    cmd.extend(["--pre_multiple", str(float(self.pre_multi_var.get().replace('x', '')))])
                else:
                    w_val = self.pre_max_w_var.get() if self.pre_max_w_var.get().isdigit() else "0"
                    h_val = self.pre_max_h_var.get() if self.pre_max_h_var.get().isdigit() else "0"
                    cmd.extend(["--pre_max_w", w_val, "--pre_max_h", h_val])
            
            if self.ai_active_var.get() and chosen_checkpoint_path:
                cmd.extend(["--ai_active", "--checkpoint_path", chosen_checkpoint_path, "--profile_override", self.profile_var.get().lower()])
            
            if self.post_active_var.get():
                cmd.extend(["--post_active", "--post_mode", self.post_mode_var.get()])
                if self.post_mode_var.get() == "multiple":
                    cmd.extend(["--post_multiple", str(float(self.post_multi_var.get().replace('x', '')))])
                else:
                    w_val = self.post_max_w_var.get() if self.post_max_w_var.get().isdigit() else "0"
                    h_val = self.post_max_h_var.get() if self.post_max_h_var.get().isdigit() else "0"
                    cmd.extend(["--post_max_w", w_val, "--post_max_h", h_val])
            
            cmd.extend(["--flow_id", flow_id])
            
            # Start the single process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            if process.stdout:
                for line in process.stdout:
                    print(line, end="")
                    if "TARGET_DIR_TRACKER_TOKEN:" in line:
                        extracted_folder = line.split("TARGET_DIR_TRACKER_TOKEN:")[-1].strip()
                        if extracted_folder: unique_upscale_folders.add(extracted_folder)
            process.wait()
            
            print("\n====================================================")
            print("Batch Processing Finished Successfully!")
            print("====================================================")
            
        except Exception as e:
            print(f"\nWorker Thread Processing Error: {e}")
        finally:
            # Clean up the temporary batch list file
            if os.path.exists(batch_list_path):
                os.remove(batch_list_path)
            
        self.root.after(0, self._finalize_pipeline_run, unique_upscale_folders)

    def _finalize_pipeline_run(self, unique_upscale_folders):
        self.browse_btn.config(state=tk.NORMAL)
        self.update_ui_states() 
        self.run_btn.config(text="START PIPELINE RUN")
        self.check_ready_state()
        
        for upscale_folder in unique_upscale_folders:
            if os.path.exists(upscale_folder):
                if sys.platform == "win32": os.startfile(upscale_folder)
                elif sys.platform == "darwin": subprocess.Popen(["open", upscale_folder])
                else: subprocess.Popen(["xdg-open", upscale_folder])
                
        messagebox.showinfo("Success", "Finished processing images!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SuperCoolUI(root)
    root.mainloop()