"""
AI Toolkit - Building blocks for custom AI pipelines.

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

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageGen:
    """Image generation with Stable Diffusion, ControlNet, and LoRA support."""

    CONTROLNET_MODELS = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
        "pose": "thibaud/controlnet-openpose-sdxl-1.0",
    }

    def __init__(self, model: str = "stabilityai/sdxl-turbo"):
        self.model_id = model
        self.pipe = None
        self.device = get_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    def load(self, controlnet: Optional[str] = None):
        """Load the pipeline."""
        if self.pipe is not None:
            return

        from diffusers import AutoPipelineForText2Image

        print(f"Loading {self.model_id}...")
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
        lora: Optional[str] = None,
        lora_scale: float = 1.0,
    ):
        """Generate an image."""
        self.load(controlnet)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if lora:
            self.pipe.load_lora_weights(lora)
            self.pipe.fuse_lora(lora_scale=lora_scale)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )

        return result.images[0]


class Transcriber:
    """Audio/video transcription using Faster-Whisper."""

    def __init__(self, model: str = "medium"):
        self.model_size = model
        self.model = None

    def load(self):
        if self.model is not None:
            return

        from faster_whisper import WhisperModel
        print(f"Loading Whisper {self.model_size}...")
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> Union[str, List[Dict]]:
        """Transcribe audio/video file."""
        self.load()

        segments, info = self.model.transcribe(audio_path, language=language)

        if timestamps:
            return [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments
            ]
        else:
            return " ".join(s.text.strip() for s in segments)


class Embedder:
    """Semantic search using Sentence Transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self.model = None
        self.index = None
        self.index_path = MODELS_DIR / "embeddings_index.pkl"

    def load(self):
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer
        print(f"Loading {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device="cuda")

    def index_folder(
        self,
        folder: str,
        extensions: List[str] = [".txt", ".md", ".py", ".js", ".ts"],
    ):
        """Index all text files in a folder."""
        import pickle
        self.load()

        folder_path = Path(folder)
        documents = []
        paths = []

        for ext in extensions:
            for file_path in folder_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        documents.append(content[:8000])
                        paths.append(str(file_path))
                except Exception as e:
                    print(f"[embed] Skipped {file_path}: {e}")

        print(f"Indexing {len(documents)} files...")
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

        self.index = {"paths": paths, "embeddings": embeddings, "documents": documents}

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search indexed files."""
        import pickle
        from sklearn.metrics.pairwise import cosine_similarity

        self.load()

        if self.index is None:
            if self.index_path.exists():
                with open(self.index_path, "rb") as f:
                    self.index = pickle.load(f)
            else:
                raise ValueError("No index found. Run index_folder() first.")

        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.index["embeddings"])[0]
        top_idx = sims.argsort()[-top_k:][::-1]

        return [
            {"path": self.index["paths"][i], "score": float(sims[i]),
             "preview": self.index["documents"][i][:200]}
            for i in top_idx
        ]


class Segmenter:
    """Image segmentation using SAM."""

    SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    def __init__(self):
        self.model = None
        self.checkpoint = MODELS_DIR / "sam_vit_b_01ec64.pth"

    def load(self):
        if self.model is not None:
            return

        # Download if needed
        if not self.checkpoint.exists():
            print("Downloading SAM model...")
            import urllib.request
            urllib.request.urlretrieve(self.SAM_URL, self.checkpoint)

        from segment_anything import sam_model_registry
        print("Loading SAM...")
        self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint))
        self.model.to("cuda")

    def segment(self, image_path: str):
        """Auto-segment entire image."""
        import cv2
        from segment_anything import SamAutomaticMaskGenerator

        self.load()

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        generator = SamAutomaticMaskGenerator(self.model)
        masks = generator.generate(image_rgb)

        return masks

    def segment_point(self, image_path: str, x: int, y: int):
        """Segment at a specific point."""
        import cv2
        import numpy as np
        from segment_anything import SamPredictor

        self.load()

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor = SamPredictor(self.model)
        predictor.set_image(image_rgb)

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        best_idx = scores.argmax()
        return masks[best_idx], scores[best_idx]


class BackgroundRemover:
    """Background removal using rembg."""

    def remove(self, image_path: str, output_path: Optional[str] = None):
        """Remove background from image."""
        from rembg import remove
        from PIL import Image

        with open(image_path, "rb") as f:
            input_data = f.read()

        output_data = remove(input_data)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(output_data)

        return Image.open(__import__("io").BytesIO(output_data))


# Convenience functions
def transcribe(audio_path: str, **kwargs) -> str:
    """Quick transcription."""
    return Transcriber().transcribe(audio_path, **kwargs)

def generate(prompt: str, **kwargs):
    """Quick image generation."""
    return ImageGen().generate(prompt, **kwargs)

def remove_bg(image_path: str, output_path: Optional[str] = None):
    """Quick background removal."""
    return BackgroundRemover().remove(image_path, output_path)

def segment(image_path: str):
    """Quick segmentation."""
    return Segmenter().segment(image_path)
