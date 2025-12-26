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

__version__ = "0.1.0"

# Lazy imports - classes are loaded on first access
# This allows aikit.core to be imported without loading torch


def __getattr__(name):
    """Lazy import handler for heavy dependencies."""
    if name == "ImageGen":
        from aikit.generators import ImageGen
        return ImageGen
    elif name == "Transcriber":
        from aikit.transcription import Transcriber
        return Transcriber
    elif name == "Embedder":
        from aikit.embeddings import Embedder
        return Embedder
    elif name == "Segmenter":
        from aikit.segmentation import Segmenter
        return Segmenter
    elif name == "BackgroundRemover":
        from aikit.background import BackgroundRemover
        return BackgroundRemover
    elif name == "get_device":
        from aikit.core import get_device
        return get_device
    elif name == "MODELS_DIR":
        from aikit.core import MODELS_DIR
        return MODELS_DIR
    elif name == "OUTPUT_DIR":
        from aikit.core import OUTPUT_DIR
        return OUTPUT_DIR
    raise AttributeError(f"module 'aikit' has no attribute '{name}'")


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
    from aikit.transcription import Transcriber
    return Transcriber().transcribe(audio_path, **kwargs)


def generate(prompt: str, **kwargs):
    """Quick image generation."""
    from aikit.generators import ImageGen
    return ImageGen().generate(prompt, **kwargs)


def remove_bg(image_path: str, output_path=None):
    """Quick background removal."""
    from aikit.background import BackgroundRemover
    return BackgroundRemover().remove(image_path, output_path)


def segment(image_path: str):
    """Quick segmentation."""
    from aikit.segmentation import Segmenter
    return Segmenter().segment(image_path)
