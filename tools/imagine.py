#!/usr/bin/env python3
"""
Image generation toolkit using Diffusers.

Usage:
    imagine "a cyberpunk city at night"
    imagine "portrait of a warrior" --model stabilityai/sdxl-turbo
    imagine "a cat" --controlnet canny --input sketch.png
    imagine "a dog" --lora my_lora.safetensors --lora-scale 0.8
    imagine "scene" --steps 30 --cfg 7.5 --seed 42
"""
import argparse
import sys
import os
import torch
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Default models
DEFAULT_MODEL = "stabilityai/sdxl-turbo"
CONTROLNET_MODELS = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_pipeline(model_id, controlnet=None, lora_path=None, lora_scale=1.0):
    """Load the appropriate pipeline."""
    from diffusers import AutoPipelineForText2Image, ControlNetModel, StableDiffusionXLControlNetPipeline

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if controlnet:
        print(f"Loading ControlNet: {controlnet}")
        cn_model_id = CONTROLNET_MODELS.get(controlnet, controlnet)
        controlnet_model = ControlNetModel.from_pretrained(
            cn_model_id,
            torch_dtype=dtype,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id.replace("-turbo", "-base-1.0") if "turbo" in model_id else model_id,
            controlnet=controlnet_model,
            torch_dtype=dtype,
        )
    else:
        print(f"Loading model: {model_id}")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

    pipe = pipe.to(device)

    # Load LoRA if specified
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)

    # Optimizations
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()

    return pipe

def preprocess_controlnet(image_path, controlnet_type):
    """Preprocess image for ControlNet."""
    from PIL import Image
    from controlnet_aux import CannyDetector, OpenposeDetector
    import numpy as np

    image = Image.open(image_path).convert("RGB")

    if controlnet_type == "canny":
        detector = CannyDetector()
        return detector(image)
    elif controlnet_type == "pose":
        detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return detector(image)
    elif controlnet_type == "depth":
        # Use simple edge detection as fallback
        from controlnet_aux import MidasDetector
        detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        return detector(image)

    return image

def generate(
    prompt,
    negative_prompt="blurry, bad quality, distorted",
    model=DEFAULT_MODEL,
    steps=4,  # SDXL Turbo uses few steps
    cfg=0.0,  # SDXL Turbo uses cfg=0
    width=1024,
    height=1024,
    seed=None,
    controlnet=None,
    controlnet_input=None,
    controlnet_scale=0.8,
    lora_path=None,
    lora_scale=1.0,
    output=None,
):
    """Generate an image."""
    from PIL import Image
    import time

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Adjust defaults based on model
    if "turbo" not in model.lower():
        if steps == 4:
            steps = 30
        if cfg == 0.0:
            cfg = 7.5

    # Set seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=get_device()).manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=get_device()).manual_seed(seed)

    # Load pipeline
    pipe = load_pipeline(model, controlnet, lora_path, lora_scale)

    # Prepare kwargs
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "width": width,
        "height": height,
        "generator": generator,
    }

    # Add ControlNet image if specified
    if controlnet and controlnet_input:
        control_image = preprocess_controlnet(controlnet_input, controlnet)
        kwargs["image"] = control_image
        kwargs["controlnet_conditioning_scale"] = controlnet_scale

    print(f"Generating: '{prompt}'")
    print(f"Settings: {steps} steps, CFG {cfg}, {width}x{height}, seed {seed}")

    start = time.time()
    result = pipe(**kwargs)
    elapsed = time.time() - start

    image = result.images[0]

    # Save
    if output:
        output_path = Path(output)
    else:
        timestamp = int(time.time())
        output_path = OUTPUT_DIR / f"gen_{timestamp}_{seed}.png"

    image.save(output_path)
    print(f"Saved: {output_path} ({elapsed:.1f}s)")

    return image, output_path

def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("prompt", help="Text prompt for generation")
    parser.add_argument("-n", "--negative", default="blurry, bad quality, distorted",
                        help="Negative prompt")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=1024, help="Image height")
    parser.add_argument("-s", "--seed", type=int, help="Random seed")
    parser.add_argument("-o", "--output", help="Output file path")

    # ControlNet options
    parser.add_argument("--controlnet", choices=["canny", "depth", "pose"],
                        help="ControlNet type")
    parser.add_argument("--input", help="Input image for ControlNet")
    parser.add_argument("--cn-scale", type=float, default=0.8,
                        help="ControlNet conditioning scale")

    # LoRA options
    parser.add_argument("--lora", help="Path to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")

    args = parser.parse_args()

    if args.controlnet and not args.input:
        print("Error: --input required when using --controlnet")
        sys.exit(1)

    generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        model=args.model,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        seed=args.seed,
        controlnet=args.controlnet,
        controlnet_input=args.input,
        controlnet_scale=args.cn_scale,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        output=args.output,
    )

if __name__ == "__main__":
    main()
