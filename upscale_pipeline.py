import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torchvision.io import decode_image, write_jpeg, write_png
from torchvision.transforms.v2 import ToDtype

from model import SuperCool

def apply_interpolation(tensor, scale_factor=None, target_size=None, mode="bicubic"):
    """Generalized interpolation function for tensors [B, C, H, W]"""
    return F.interpolate(
        tensor,
        scale_factor=scale_factor,
        size=target_size,
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None
    ).clamp(0.0, 1.0)

def main():
    parser = ArgumentParser(description="Super-resolution pipeline processor")

    # --- FIX: Replaced single image path with a batch text file ---
    parser.add_argument("--batch_list", type=str, required=True, help="Path to text file containing image paths")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--flow_id", default="output", type=str)
    
    # Pre-Process Stage Args
    parser.add_argument("--pre_active", action="store_true")
    parser.add_argument("--pre_mode", type=str, choices=["multiple", "conform"], default="multiple")
    parser.add_argument("--pre_multiple", type=float, default=1.0)
    parser.add_argument("--pre_max_w", type=int, default=0)
    parser.add_argument("--pre_max_h", type=int, default=0)
    
    # AI Model Stage Args
    parser.add_argument("--ai_active", action="store_true")
    parser.add_argument("--profile_override", default="large", type=str)
    parser.add_argument("--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str)
    
    # Post-Process Stage Args
    parser.add_argument("--post_active", action="store_true")
    parser.add_argument("--post_mode", type=str, choices=["multiple", "conform"], default="multiple")
    parser.add_argument("--post_multiple", type=float, default=1.0)
    parser.add_argument("--post_max_w", type=int, default=0)
    parser.add_argument("--post_max_h", type=int, default=0)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda driver target is not available.")

    # Read the list of files to process
    with open(args.batch_list, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if not image_paths:
        print("No images found in batch list to process.")
        return

    # ==========================================
    # SETUP: LOAD AI MODEL ONCE 
    # ==========================================
    model = None
    checkpoint_ratio = 4
    
    if args.ai_active:
        print(f"-> [System] Loading AI Model from {os.path.basename(args.checkpoint_path)}")
        ext = os.path.splitext(args.checkpoint_path.lower())[1]
        
        # 1. Load state_dict
        if ext == ".safetensors":
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(args.checkpoint_path)
        else:
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
            state_dict = checkpoint if not isinstance(checkpoint, dict) or "model" not in checkpoint else checkpoint["model"]

        # 2. Determine Architecture
        if "decoder.conv.bias" in state_dict:
            bias_size = state_dict["decoder.conv.bias"].shape[0]
            checkpoint_ratio = int((bias_size / 3) ** 0.5)

        # 3. Explicit Configuration
        profile_mode = args.profile_override.lower()
        model_kwargs = {
            "base_upscaler": "bicubic",
            "upscale_ratio": checkpoint_ratio,
            "hidden_ratio": 2,
        }

        if "small" in profile_mode:
            model_kwargs.update({"num_channels": 64, "num_layers": 4})
        elif "medium" in profile_mode:
            model_kwargs.update({"num_channels": 128, "num_layers": 32})
        else:
            model_kwargs.update({"num_channels": 256, "num_layers": 32})

        # 4. Strict Loading Logic
        print(f"-> [System] Initializing VRAM with: {model_kwargs}")
        model = SuperCool(**model_kwargs).to(args.device)
        model.remove_weight_norms()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("-> [System] Model loaded successfully. Starting batch process...\n")


    # ==========================================
    # BATCH PROCESSING LOOP
    # ==========================================
    image_to_tensor = ToDtype(torch.float32, scale=True)

    for idx, img_path in enumerate(image_paths, 1):
        print(f"Processing [{idx}/{len(image_paths)}]: {os.path.basename(img_path)}")
        
        # File setup
        orig_dir = os.path.dirname(img_path)
        orig_basename = os.path.basename(img_path)
        name_part, ext_part = os.path.splitext(orig_basename)
        
        target_output_dir = os.path.abspath(os.path.join(orig_dir, "upscaled"))
        output_filename = f"{name_part}_{args.flow_id}{ext_part}"
        final_destination_path = os.path.join(target_output_dir, output_filename)

        try:
            # Load initial image
            image = decode_image(img_path, mode="RGB")
            x = image_to_tensor(image).unsqueeze(0).to(args.device)

            # --- STAGE 1: PRE-PROCESS ---
            if args.pre_active:
                if args.pre_mode == "multiple":
                    x = apply_interpolation(x, scale_factor=args.pre_multiple)
                elif args.pre_mode == "conform":
                    _, _, h, w = x.shape
                    max_w, max_h = args.pre_max_w, args.pre_max_h
                    if max_w > 0 or max_h > 0:
                        scale_w = max_w / w if max_w > 0 else float('inf')
                        scale_h = max_h / h if max_h > 0 else float('inf')
                        scale = min(scale_w, scale_h)
                        if scale != float('inf'):
                            target_size = (int(h * scale), int(w * scale))
                            x = apply_interpolation(x, target_size=target_size)

            # --- STAGE 2: AI UPSCALING ---
            if args.ai_active and model is not None:
                with torch.no_grad():
                    x = model(x).clamp(0.0, 1.0)

            # --- STAGE 3: POST-PROCESS ---
            if args.post_active:
                if args.post_mode == "multiple":
                    x = apply_interpolation(x, scale_factor=args.post_multiple)
                elif args.post_mode == "conform":
                    _, _, h, w = x.shape
                    max_w, max_h = args.post_max_w, args.post_max_h
                    if max_w > 0 or max_h > 0:
                        scale_w = max_w / w if max_w > 0 else float('inf')
                        scale_h = max_h / h if max_h > 0 else float('inf')
                        scale = min(scale_w, scale_h)
                        if scale != float('inf'):
                            target_size = (int(h * scale), int(w * scale))
                            x = apply_interpolation(x, target_size=target_size)

            # --- FINAL EXPORT ---
            output_tensor = x.squeeze(0)
            
            if output_tensor.shape[0] == 3:
                pass
            elif output_tensor.shape[-1] == 3:
                output_tensor = output_tensor.permute(2, 0, 1)
            else:
                if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 3:
                    output_tensor = output_tensor.permute(1, 0, 2)

            output_tensor = output_tensor.contiguous()
            output_tensor = (output_tensor * 255.0).to(torch.uint8).cpu()
            
            os.makedirs(target_output_dir, exist_ok=True)

            if ext_part.lower() in [".jpg", ".jpeg"]: 
                write_jpeg(output_tensor, final_destination_path, quality=95)
            else: 
                write_png(output_tensor, final_destination_path)
                
            print(f"File processed successfully: {final_destination_path}")
            print(f"TARGET_DIR_TRACKER_TOKEN:{target_output_dir}\n")

        except Exception as e:
            print(f"ERROR processing {img_path}: {e}\n")

if __name__ == "__main__":
    main()