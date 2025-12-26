"""
Tests for aikit.generators module.

Tests cover:
- ImageGen initialization
- Pipeline loading
- Image generation
- ControlNet integration
- LoRA loading
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from PIL import Image


class TestImageGenInit:
    """Tests for ImageGen initialization."""

    def test_default_model(self):
        """Should use sdxl-turbo by default."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        assert gen.model_id == "stabilityai/sdxl-turbo"
        assert gen.pipe is None

    def test_custom_model(self):
        """Should accept custom model ID."""
        from aikit.generators import ImageGen

        gen = ImageGen(model="custom/model")
        assert gen.model_id == "custom/model"

    def test_device_detection(self, mock_torch_cuda):
        """Should detect available device."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        assert gen.device.type == "cuda"

    def test_dtype_selection(self, mock_torch_cuda):
        """Should use float16 on CUDA."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        assert gen.dtype == torch.float16

    def test_dtype_fallback_cpu(self, mock_torch_cpu):
        """Should use float32 on CPU."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        assert gen.dtype == torch.float32


class TestImageGenLoad:
    """Tests for ImageGen.load() method."""

    def test_loads_pipeline_once(self, mock_diffusers, mock_torch_cuda):
        """Should only load pipeline once."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.load()
        gen.load()  # Should be no-op

        assert gen.pipe is not None

    def test_loads_controlnet_pipeline(self, mock_diffusers, mock_torch_cuda):
        """Should load ControlNet pipeline when specified."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.load(controlnet="canny")

        assert gen.controlnet_pipe is not None
        assert gen._loaded_controlnet == "canny"

    def test_reloads_for_different_controlnet(self, mock_diffusers, mock_torch_cuda):
        """Should reload when switching ControlNet type."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.load(controlnet="canny")
        first_pipe = gen.controlnet_pipe

        gen.controlnet_pipe = None  # Reset for test
        gen.load(controlnet="depth")

        # Should have updated the controlnet type
        assert gen._loaded_controlnet == "depth"


class TestImageGenGenerate:
    """Tests for ImageGen.generate() method."""

    def test_generates_image(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should generate and return image."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen()
        image, path = gen.generate("a test prompt")

        assert isinstance(image, Image.Image)
        assert Path(path).exists()

    def test_respects_output_path(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should save to specified output path."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        output_path = tmp_path / "custom_output.png"

        gen = ImageGen()
        image, path = gen.generate("test", output=str(output_path))

        assert path == output_path
        assert output_path.exists()

    def test_uses_provided_seed(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should use provided random seed."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        with patch("torch.Generator") as mock_gen:
            mock_generator = MagicMock()
            mock_gen.return_value = mock_generator
            mock_generator.manual_seed.return_value = mock_generator

            from aikit.generators import ImageGen

            gen = ImageGen()
            gen.generate("test", seed=42)

            mock_generator.manual_seed.assert_called_with(42)

    def test_adjusts_steps_for_non_turbo(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should increase steps for non-turbo models."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen(model="stabilityai/stable-diffusion-xl-base-1.0")
        gen.load()

        # Check that generate would use 30 steps instead of 4
        with patch.object(gen.pipe, "__call__", return_value=MagicMock(images=[Image.new("RGB", (64, 64))])) as mock_call:
            gen.generate("test")
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["num_inference_steps"] == 30
            assert call_kwargs["guidance_scale"] == 7.5

    def test_prints_generation_info(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch, capsys):
        """Should print generation parameters."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.generate("test prompt", steps=10, cfg=5.0, width=512, height=512)

        captured = capsys.readouterr()
        assert "test prompt" in captured.out
        assert "10 steps" in captured.out
        assert "512x512" in captured.out


class TestImageGenLoRA:
    """Tests for LoRA loading functionality."""

    def test_loads_lora_weights(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should load LoRA weights when specified."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        # Create fake LoRA file
        lora_path = tmp_path / "style.safetensors"
        lora_path.write_bytes(b"fake lora data")

        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.load()

        # Mock the pipe methods
        gen.pipe.load_lora_weights = MagicMock()
        gen.pipe.fuse_lora = MagicMock()

        gen.generate("test", lora=str(lora_path))

        gen.pipe.load_lora_weights.assert_called_once_with(str(lora_path))
        gen.pipe.fuse_lora.assert_called_once()

    def test_exits_for_missing_lora(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should exit when LoRA file doesn't exist."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen()

        with pytest.raises(SystemExit):
            gen.generate("test", lora=str(tmp_path / "missing.safetensors"))


class TestImageGenControlNet:
    """Tests for ControlNet functionality."""

    def test_requires_control_image(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should use control image when ControlNet specified."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        # Create control image
        control_img = tmp_path / "control.png"
        img = Image.new("RGB", (64, 64), color="white")
        img.save(control_img)

        from aikit.generators import ImageGen

        gen = ImageGen()
        gen.load(controlnet="canny")

        with patch.object(gen, "_preprocess_controlnet", return_value=img) as mock_preprocess:
            gen.generate("test", controlnet="canny", control_image=str(control_img))
            mock_preprocess.assert_called_once()

    def test_controlnet_models_defined(self):
        """Should have predefined ControlNet model IDs."""
        from aikit.generators import ImageGen

        assert "canny" in ImageGen.CONTROLNET_MODELS
        assert "depth" in ImageGen.CONTROLNET_MODELS
        assert "pose" in ImageGen.CONTROLNET_MODELS


class TestImageGenPreprocessControlNet:
    """Tests for ControlNet image preprocessing."""

    def test_canny_preprocessing(self, tmp_image_file, mock_torch_cuda):
        """Should apply Canny edge detection."""
        with patch("controlnet_aux.CannyDetector") as mock_detector:
            mock_detector.return_value.return_value = Image.new("RGB", (64, 64))

            from aikit.generators import ImageGen

            gen = ImageGen()
            result = gen._preprocess_controlnet(str(tmp_image_file), "canny")

            mock_detector.assert_called_once()
            assert isinstance(result, Image.Image)

    def test_returns_original_for_unknown_type(self, tmp_image_file, mock_torch_cuda):
        """Should return original image for unknown ControlNet type."""
        from aikit.generators import ImageGen

        gen = ImageGen()
        result = gen._preprocess_controlnet(str(tmp_image_file), "unknown")

        assert isinstance(result, Image.Image)
