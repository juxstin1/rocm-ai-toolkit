"""
Pytest configuration and shared fixtures.

This module provides:
- Mock fixtures for heavy ML models (no GPU required)
- Temporary file/directory fixtures
- Common test utilities
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Temporary file fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, this is test content.\nLine two.")
    return file_path


@pytest.fixture
def tmp_image_file(tmp_path):
    """Create a valid PNG file for testing using PIL."""
    from PIL import Image

    file_path = tmp_path / "test_image.png"
    # Create a small 10x10 red image
    img = Image.new("RGB", (10, 10), color="red")
    img.save(file_path)
    return file_path


@pytest.fixture
def tmp_audio_file(tmp_path):
    """Create a minimal WAV file for testing (won't actually play)."""
    # Minimal WAV header with no audio data
    wav_data = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk1 size (16)
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Num channels (1)
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00,  # Data size (0)
    ])
    file_path = tmp_path / "test_audio.wav"
    file_path.write_bytes(wav_data)
    return file_path


@pytest.fixture
def tmp_folder_with_files(tmp_path):
    """Create a folder with various text files for embedding tests."""
    folder = tmp_path / "documents"
    folder.mkdir()

    (folder / "readme.md").write_text("# Project\n\nThis is a readme file.")
    (folder / "code.py").write_text("def hello():\n    print('Hello')")
    (folder / "notes.txt").write_text("Some notes about the project.")
    (folder / "ignored.json").write_text('{"key": "value"}')  # Not in default extensions

    return folder


# -----------------------------------------------------------------------------
# Mock fixtures for heavy ML models
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda to simulate GPU availability."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "AMD Radeon RX 9070 XT"
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"
    mock_torch.Generator.return_value = MagicMock()

    # Create a proper mock device that has a type attribute
    mock_cuda_device = MagicMock()
    mock_cuda_device.type = "cuda"
    mock_torch.device.return_value = mock_cuda_device

    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch


@pytest.fixture
def mock_torch_cpu():
    """Mock torch.cuda to simulate CPU-only environment."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"
    mock_torch.Generator.return_value = MagicMock()

    mock_cpu_device = MagicMock()
    mock_cpu_device.type = "cpu"
    mock_torch.device.return_value = mock_cpu_device

    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading real models."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_whisper_model():
    """Mock faster_whisper.WhisperModel."""
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " Hello world. "

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_rembg():
    """Mock rembg.remove function."""
    # Return a minimal PNG as output
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
        0x42, 0x60, 0x82,
    ])

    with patch("rembg.remove", return_value=png_data):
        yield


@pytest.fixture
def mock_sam_model():
    """Mock segment_anything models."""
    import numpy as np

    mock_mask = np.zeros((100, 100), dtype=bool)
    mock_mask[25:75, 25:75] = True

    mock_mask_dict = {
        "segmentation": mock_mask,
        "area": 2500,
        "bbox": [25, 25, 50, 50],
        "predicted_iou": 0.95,
        "stability_score": 0.98,
    }

    mock_generator = MagicMock()
    mock_generator.generate.return_value = [mock_mask_dict]

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (
        np.array([mock_mask]),
        np.array([0.95]),
        None,
    )

    with patch("segment_anything.sam_model_registry") as mock_registry:
        mock_model = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=lambda **kwargs: mock_model)
        with patch("segment_anything.SamAutomaticMaskGenerator", return_value=mock_generator):
            with patch("segment_anything.SamPredictor", return_value=mock_predictor):
                yield mock_model


@pytest.fixture
def mock_diffusers():
    """Mock diffusers pipelines."""
    from PIL import Image

    mock_image = Image.new("RGB", (64, 64), color="red")
    mock_result = MagicMock()
    mock_result.images = [mock_image]

    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_result
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_attention_slicing = MagicMock()

    with patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipe):
        with patch("diffusers.ControlNetModel.from_pretrained", return_value=MagicMock()):
            with patch("diffusers.StableDiffusionXLControlNetPipeline.from_pretrained", return_value=mock_pipe):
                yield mock_pipe


# -----------------------------------------------------------------------------
# Utility fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def capture_stdout(capsys):
    """Helper to capture and return stdout."""
    def _capture():
        return capsys.readouterr().out
    return _capture


@pytest.fixture
def isolated_dirs(tmp_path):
    """Create isolated directories for tests."""
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    models_dir.mkdir()
    output_dir.mkdir()
    return models_dir, output_dir


# -----------------------------------------------------------------------------
# Skip markers
# -----------------------------------------------------------------------------


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "security: security-related tests")
    config.addinivalue_line("markers", "gpu: tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests in CI environments."""
    if os.environ.get("CI"):
        skip_gpu = pytest.mark.skip(reason="GPU tests skipped in CI")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
