import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torchvision.io import decode_image, write_jpeg, write_png
from torchvision.transforms.v2 import ToDtype

from model import SuperCool


def main():
    parser = ArgumentParser(description="Super-resolution upscaling backend processor")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--upscale_ratio", default="4", type=str)
    parser.add_argument("--profile_override", default="large", type=str)
    parser.add_argument("--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda driver target is not available.")

    target_checkpoint = args.checkpoint_path
    if not os.path.exists(target_checkpoint):
        raise FileNotFoundError(f"Selected weights file path missing: {target_checkpoint}")

    checkpoint_base_name = os.path.splitext(os.path.basename(target_checkpoint))[0]

    # ---------------------------------------------------------
    # ADAPTIVE HYBRID CHECKPOINT ARCHITECTURE LOADING
    # ---------------------------------------------------------
    ext = os.path.splitext(target_checkpoint.lower())[1]
    
    if ext == ".safetensors":
        from safetensors.torch import load_file as load_safetensors
        print(f"Loading raw weights from Safetensors matrix: {target_checkpoint}")
        state_dict = load_safetensors(target_checkpoint)
    else:
        print(f"Loading PyTorch checkpoint archive: {target_checkpoint}")
        checkpoint = torch.load(target_checkpoint, map_location=args.device, weights_only=True)
        state_dict = checkpoint if not isinstance(checkpoint, dict) or "model" not in checkpoint else checkpoint["model"]

    # ---------------------------------------------------------
    # DYNAMIC METRIC CHECKPOINT RATIO VALIDATION
    # ---------------------------------------------------------
    requested_ratio = int(args.upscale_ratio)
    checkpoint_ratio = requested_ratio
    
    if "decoder.conv.bias" in state_dict:
        bias_size = state_dict["decoder.conv.bias"].shape[0]
        checkpoint_ratio = int((bias_size / 3) ** 0.5)
        if checkpoint_ratio != requested_ratio:
            print(f"-> Architecture mismatch detected: Checkpoint is baked for {checkpoint_ratio}x upscaling.")
            print(f"   The pipeline will run the neural network at {checkpoint_ratio}x and upscale the rest via high-quality interpolation.")
        else:
            print(f"-> Architecture verification: Checkpoint matches target {checkpoint_ratio}x upscaling factor.")

    profile_mode = args.profile_override.lower()
    if ext != ".safetensors" and isinstance(checkpoint, dict) and "model_args" in checkpoint:
        print("-> Success: Found embedded configuration parameters. Overriding manual UI layout profiles.")
        model_kwargs = checkpoint["model_args"]
        model_kwargs["upscale_ratio"] = checkpoint_ratio
    else:
        if "small" in profile_mode:
            model_kwargs = {"base_upscaler": "bicubic", "upscale_ratio": checkpoint_ratio, "num_channels": 64, "hidden_ratio": 2, "num_layers": 4}
        elif "medium" in profile_mode:
            model_kwargs = {"base_upscaler": "bicubic", "upscale_ratio": checkpoint_ratio, "num_channels": 128, "hidden_ratio": 2, "num_layers": 8}
        else:
            model_kwargs = {"base_upscaler": "bicubic", "upscale_ratio": checkpoint_ratio, "num_channels": 256, "hidden_ratio": 2, "num_layers": 32}

    model = SuperCool(**model_kwargs)

    # ---------------------------------------------------------
    # AUTOMATED CAPABILITY GATEKEEPER & COMPILATION ENGINE
    # ---------------------------------------------------------
    should_compile = True
    if "cuda" in args.device:
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            print(f"-> GPU Compute Capability {major}.{minor} detected (Pascal Architecture / Quadro P1000).")
            print("-> Triton compiler optimization bypassed to guarantee device execution safety.")
            should_compile = False

    if should_compile:
        print("Compiling model for performance optimization...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"-> Compilation optimization bypassed: {e}")
    else:
        print("Running pipeline via fallback native eager execution mode.")

    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=False)
    print("Model checkpoint loaded successfully")

    try:
        model.remove_weight_norms()
    except Exception:
        pass

    image = decode_image(args.image_path, mode="RGB")
    image_to_tensor = ToDtype(torch.float32, scale=True)
    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    model.eval()
    print("Upscaling...")
    with torch.no_grad():
        y_pred = model(x)
        
        if checkpoint_ratio != requested_ratio:
            extra_scale = requested_ratio / checkpoint_ratio
            upscale_mode = model_kwargs.get("base_upscaler", "bicubic")
            y_pred = F.interpolate(
                y_pred, 
                scale_factor=extra_scale, 
                mode=upscale_mode, 
                align_corners=False if upscale_mode in ["bilinear", "bicubic"] else None
            )

    output_tensor = y_pred.squeeze(0).clamp(0.0, 1.0)
    output_tensor = (output_tensor * 255.0).to(torch.uint8).cpu()

    # ---------------------------------------------------------
    # AUTOMATED OUTPUT PATH GENERATOR LOGIC WITH MODEL STEM
    # ---------------------------------------------------------
    orig_dir = os.path.dirname(args.image_path)
    orig_basename = os.path.basename(args.image_path)
    name_part, ext_part = os.path.splitext(orig_basename)
    
    target_output_dir = os.path.abspath(os.path.join(orig_dir, "upscaled"))
    os.makedirs(target_output_dir, exist_ok=True)
    
    output_filename = f"{name_part}_{checkpoint_base_name}_{requested_ratio}x{ext_part}"
    final_destination_path = os.path.join(target_output_dir, output_filename)

    ext_clean = ext_part.lower()
    if ext_clean in [".jpg", ".jpeg"]:
        write_jpeg(output_tensor, final_destination_path, quality=95)
    else:
        write_png(output_tensor, final_destination_path)
        
    print(f"File processed and saved successfully: {final_destination_path}")
    print(f"TARGET_DIR_TRACKER_TOKEN:{target_output_dir}")


if __name__ == "__main__":
    main()