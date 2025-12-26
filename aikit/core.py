"""Core utilities and shared configuration."""

import torch
from pathlib import Path

# Directories
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_status(tool: str, message: str):
    """Print a status message with tool prefix."""
    print(f"[{tool}] {message}")


def print_download(tool: str, model_name: str, size_hint: str = None):
    """Print a download notification for first-run model downloads."""
    if size_hint:
        print(f"[{tool}] Downloading {model_name} ({size_hint}). This happens once.")
    else:
        print(f"[{tool}] Downloading {model_name}. This happens once.")


def resolve_path(file_path: str) -> Path:
    """Resolve a path to absolute, expanding user and resolving symlinks."""
    return Path(file_path).expanduser().resolve()


def require_file(tool: str, file_path: str, hint: str = None) -> Path:
    """Validate file exists and return resolved path. Exit with clear message if not."""
    import sys
    path = resolve_path(file_path)

    if not path.exists():
        print(f"[{tool}] File not found: {path}")
        if hint:
            print(f"[{tool}] {hint}")
        else:
            print(f"[{tool}] Use full path or cd into the folder first.")
        sys.exit(1)

    return path


def require_image(tool: str, file_path: str) -> "numpy.ndarray":
    """Load image with OpenCV, exit with clear message on failure."""
    import sys
    import cv2

    path = require_file(tool, file_path)
    print_status(tool, f"Loading: {path}")

    image = cv2.imread(str(path))
    if image is None:
        print(f"[{tool}] Could not read image: {path}")
        print(f"[{tool}] File may be corrupted or unsupported format.")
        sys.exit(1)

    return image


def require_audio(tool: str, file_path: str) -> Path:
    """Validate audio/video file exists. Return resolved path."""
    return require_file(tool, file_path, "Supported: mp3, wav, mp4, mkv, webm, etc.")
