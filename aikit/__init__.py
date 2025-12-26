"""
AI Toolkit - Building blocks for custom AI pipelines.

Install:
    pip install -e .

Import and use in your own projects:

    from aikit import ImageGen, Transcriber, Embedder, Segmenter, BackgroundRemover

Examples:

    # Generate images
    gen = ImageGen()
    image = gen.generate("a cyberpunk city")
    image = gen.generate("warrior", controlnet="canny", control_image="sketch.png")
    image = gen.generate("portrait", lora="path/to/lora.safetensors")

    # Transcribe audio
    transcriber = Transcriber()
    text = transcriber.transcribe("meeting.mp3")
    segments = transcriber.transcribe("video.mp4", timestamps=True)

    # Semantic search
    embedder = Embedder()
    embedder.index_folder("./docs")
    results = embedder.search("authentication flow")

    # Image segmentation
    segmenter = Segmenter()
    masks = segmenter.segment("photo.jpg")
    mask = segmenter.segment_point("photo.jpg", x=100, y=200)

    # Background removal
    rmbg = BackgroundRemover()
    image = rmbg.remove("photo.jpg")
"""

from aikit.generators import ImageGen
from aikit.transcription import Transcriber
from aikit.embeddings import Embedder
from aikit.segmentation import Segmenter
from aikit.background import BackgroundRemover
from aikit.core import get_device, MODELS_DIR, OUTPUT_DIR

__version__ = "0.1.0"

__all__ = [
    "ImageGen",
    "Transcriber",
    "Embedder",
    "Segmenter",
    "BackgroundRemover",
    "get_device",
    "MODELS_DIR",
    "OUTPUT_DIR",
]


# Convenience functions
def transcribe(audio_path: str, **kwargs) -> str:
    """Quick transcription."""
    return Transcriber().transcribe(audio_path, **kwargs)


def generate(prompt: str, **kwargs):
    """Quick image generation."""
    return ImageGen().generate(prompt, **kwargs)


def remove_bg(image_path: str, output_path=None):
    """Quick background removal."""
    return BackgroundRemover().remove(image_path, output_path)


def segment(image_path: str):
    """Quick segmentation."""
    return Segmenter().segment(image_path)
