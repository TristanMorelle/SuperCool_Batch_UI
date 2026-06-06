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

    parser.add_argument("--image_path", type=str, required=True)
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

    # File setup
    orig_dir = os.path.dirname(args.image_path)
    orig_basename = os.path.basename(args.image_path)
    name_part, ext_part = os.path.splitext(orig_basename)
    
    target_output_dir = os.path.abspath(os.path.join(orig_dir, "upscaled"))
    output_filename = f"{name_part}_{args.flow_id}{ext_part}"
    final_destination_path = os.path.join(target_output_dir, output_filename)

    # Load initial image
    image = decode_image(args.image_path, mode="RGB")
    image_to_tensor = ToDtype(torch.float32, scale=True)
    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    # ==========================================
    # STAGE 1: PRE-PROCESS (INTERPOLATION)
    # ==========================================
    if args.pre_active:
        if args.pre_mode == "multiple":
            print(f"-> [Stage 1] Pre-processing: Scaling by multiple {args.pre_multiple}x")
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
                    print(f"-> [Stage 1] Pre-processing: Conforming aspect ratio to {target_size[1]}x{target_size[0]}")
                    x = apply_interpolation(x, target_size=target_size)
            else:
                print("-> [Stage 1] Pre-Conform skipped (both Max Width and Height were 0).")

    # ==========================================
    # STAGE 2: AI MODEL UPSCALING
    # ==========================================
    if args.ai_active:
        print(f"-> [Stage 2] AI Model Active: Initializing {os.path.basename(args.checkpoint_path)}")
        ext = os.path.splitext(args.checkpoint_path.lower())[1]
        
        # 1. Load state_dict (Do not pass to model yet)
        if ext == ".safetensors":
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(args.checkpoint_path)
        else:
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
            state_dict = checkpoint if not isinstance(checkpoint, dict) or "model" not in checkpoint else checkpoint["model"]

        # 2. Determine Architecture
        # Fallback to 4x ratio if not detectable
        checkpoint_ratio = 4 
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
            # Default to Large
            model_kwargs.update({"num_channels": 256, "num_layers": 32})

        # 4. Strict Loading Logic
        print(f"-> Initializing model with: {model_kwargs}")
        model = SuperCool(**model_kwargs).to(args.device)
        
        # Strip wrappers BEFORE loading weights
        model.remove_weight_norms()
        
        # Load weights with strict=True to catch structure mismatches immediately
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        print(f"-> Executing neural network pass at native {checkpoint_ratio}x.")
        with torch.no_grad():
            x = model(x).clamp(0.0, 1.0)

    # ==========================================
    # STAGE 3: POST-PROCESS (CONFORM/MULTIPLES)
    # ==========================================
    if args.post_active:
        if args.post_mode == "multiple":
            print(f"-> [Stage 3] Post-processing: Scaling by multiple {args.post_multiple}x")
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
                    print(f"-> [Stage 3] Post-processing: Conforming aspect ratio to {target_size[1]}x{target_size[0]}")
                    x = apply_interpolation(x, target_size=target_size)
            else:
                print("-> [Stage 3] Post-Conform skipped (both Max Width and Height were 0).")

    # ==========================================
    # FINAL EXPORT
    # ==========================================
    # 1. Remove the batch dimension [B, C, H, W] -> [C, H, W]
    output_tensor = x.squeeze(0)
    
    # 2. Debug Print (This will show you exactly what layout the tensor is in)
    print(f"-> Debug: Raw Output Tensor Shape before formatting: {list(output_tensor.shape)}")

    # 3. Handle shape configurations dynamically
    if output_tensor.shape[0] == 3:
        # Layout is already [C, H, W] (Planar) - Perfect for torchvision writers
        pass
    elif output_tensor.shape[-1] == 3:
        # Layout is [H, W, C] (Interleaved) - Convert to [C, H, W]
        output_tensor = output_tensor.permute(2, 0, 1)
    else:
        # Fallback safety: If it's a flattened or unusual array, force a reshape check
        # Some models require explicit memory layout continuity
        if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 3:
            output_tensor = output_tensor.permute(1, 0, 2)

    # 4. Ensure memory layout is completely contiguous in RAM before saving
    output_tensor = output_tensor.contiguous()

    # 5. Scale to 8-bit unsigned integers
    output_tensor = (output_tensor * 255.0).to(torch.uint8).cpu()
    
    # 6. Save to disk
    os.makedirs(target_output_dir, exist_ok=True)

    if ext_part.lower() in [".jpg", ".jpeg"]: 
        write_jpeg(output_tensor, final_destination_path, quality=95)
    else: 
        write_png(output_tensor, final_destination_path)
        
    print(f"File processed successfully: {final_destination_path}")
    print(f"TARGET_DIR_TRACKER_TOKEN:{target_output_dir}")

if __name__ == "__main__":
    main()