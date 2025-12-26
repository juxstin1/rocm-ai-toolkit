"""Image generation with Stable Diffusion, ControlNet, and LoRA support."""

import torch
from pathlib import Path
from typing import Optional

from aikit.core import get_device, OUTPUT_DIR, print_status, print_download, require_file


class ImageGen:
    """Image generation with Stable Diffusion, ControlNet, and LoRA support."""

    CONTROLNET_MODELS = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
        "pose": "thibaud/controlnet-openpose-sdxl-1.0",
    }

    MODEL_SIZES = {
        "stabilityai/sdxl-turbo": "~6GB",
        "stabilityai/stable-diffusion-xl-base-1.0": "~7GB",
    }

    def __init__(self, model: str = "stabilityai/sdxl-turbo"):
        self.model_id = model
        self.pipe = None
        self.controlnet_pipe = None
        self.device = get_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._loaded_controlnet = None

    def load(self, controlnet: Optional[str] = None):
        """Load the pipeline."""
        from diffusers import AutoPipelineForText2Image, ControlNetModel, StableDiffusionXLControlNetPipeline

        # If requesting controlnet and we have a different one loaded, reset
        if controlnet and self._loaded_controlnet != controlnet:
            self.controlnet_pipe = None

        if controlnet:
            if self.controlnet_pipe is not None:
                return

            cn_model_id = self.CONTROLNET_MODELS.get(controlnet, controlnet)
            print_download("imagine", f"ControlNet ({controlnet})", "~2.5GB")
            controlnet_model = ControlNetModel.from_pretrained(
                cn_model_id,
                torch_dtype=self.dtype,
            )

            base_model = self.model_id.replace("-turbo", "-base-1.0") if "turbo" in self.model_id else self.model_id
            print_download("imagine", base_model, self.MODEL_SIZES.get(base_model, "~7GB"))
            self.controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model,
                controlnet=controlnet_model,
                torch_dtype=self.dtype,
            ).to(self.device)

            if hasattr(self.controlnet_pipe, 'enable_attention_slicing'):
                self.controlnet_pipe.enable_attention_slicing()

            self._loaded_controlnet = controlnet
        else:
            if self.pipe is not None:
                return

            size_hint = self.MODEL_SIZES.get(self.model_id, "~6GB")
            print_download("imagine", self.model_id, size_hint)
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(self.device)

            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality",
        steps: int = 4,
        cfg: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        controlnet: Optional[str] = None,
        control_image: Optional[str] = None,
        controlnet_scale: float = 0.8,
        lora: Optional[str] = None,
        lora_scale: float = 1.0,
        output: Optional[str] = None,
    ):
        """Generate an image.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: What to avoid in the image
            steps: Number of inference steps (4 for turbo, 30 for standard)
            cfg: Guidance scale (0 for turbo, 7.5 for standard)
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            controlnet: ControlNet type ("canny", "depth", "pose")
            control_image: Path to control image (required with controlnet)
            controlnet_scale: ControlNet conditioning scale
            lora: Path to LoRA weights
            lora_scale: LoRA influence scale
            output: Output file path (auto-generated if not specified)

        Returns:
            PIL Image and output path
        """
        import time
        from PIL import Image

        # Adjust defaults based on model
        if "turbo" not in self.model_id.lower():
            if steps == 4:
                steps = 30
            if cfg == 0.0:
                cfg = 7.5

        self.load(controlnet)
        pipe = self.controlnet_pipe if controlnet else self.pipe

        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Load LoRA if specified
        if lora:
            lora_path = require_file("imagine", lora, "LoRA file (.safetensors)")
            pipe.load_lora_weights(str(lora_path))
            pipe.fuse_lora(lora_scale=lora_scale)

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
        if controlnet and control_image:
            control_img = self._preprocess_controlnet(control_image, controlnet)
            kwargs["image"] = control_img
            kwargs["controlnet_conditioning_scale"] = controlnet_scale

        print_status("imagine", f"Generating: '{prompt}'")
        print_status("imagine", f"Settings: {steps} steps, CFG {cfg}, {width}x{height}, seed {seed}")

        start = time.time()
        result = pipe(**kwargs)
        elapsed = time.time() - start

        image = result.images[0]

        # Save
        OUTPUT_DIR.mkdir(exist_ok=True)
        if output:
            output_path = Path(output)
        else:
            timestamp = int(time.time())
            output_path = OUTPUT_DIR / f"gen_{timestamp}_{seed}.png"

        image.save(output_path)
        print_status("imagine", f"Saved: {output_path} ({elapsed:.1f}s)")

        return image, output_path

    def _preprocess_controlnet(self, image_path: str, controlnet_type: str):
        """Preprocess image for ControlNet."""
        from PIL import Image

        path = require_file("imagine", image_path, "Control image for ControlNet")
        image = Image.open(path).convert("RGB")

        if controlnet_type == "canny":
            from controlnet_aux import CannyDetector
            print_status("imagine", "Processing canny edges...")
            detector = CannyDetector()
            return detector(image)
        elif controlnet_type == "pose":
            from controlnet_aux import OpenposeDetector
            print_status("imagine", "Detecting pose...")
            print_download("imagine", "OpenPose model", "~200MB")
            detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            return detector(image)
        elif controlnet_type == "depth":
            from controlnet_aux import MidasDetector
            print_status("imagine", "Estimating depth...")
            print_download("imagine", "MiDaS model", "~400MB")
            detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            return detector(image)

        return image
