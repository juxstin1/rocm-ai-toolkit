"""Image segmentation using SAM (Segment Anything)."""

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2

from aikit.core import MODELS_DIR, print_status, print_download, require_image


class Segmenter:
    """Image segmentation using SAM."""

    SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"

    def __init__(self):
        self.model = None
        self.checkpoint = MODELS_DIR / self.CHECKPOINT_NAME

    def load(self):
        """Load the SAM model."""
        if self.model is not None:
            return

        if not self.checkpoint.exists():
            print_download("segment", "SAM ViT-B", "~375MB")
            print_status("segment", f"Saving to {self.checkpoint}...")
            urllib.request.urlretrieve(self.SAM_URL, self.checkpoint)

        from segment_anything import sam_model_registry

        print_status("segment", "Loading SAM model...")
        self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint))
        self.model.to("cuda")

    def segment(self, image_path: str):
        """Auto-segment entire image."""
        from segment_anything import SamAutomaticMaskGenerator

        image = require_image("segment", image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.load()

        generator = SamAutomaticMaskGenerator(self.model)
        masks = generator.generate(image_rgb)

        print_status("segment", f"Found {len(masks)} segments")
        return masks

    def segment_point(self, image_path: str, x: int, y: int) -> Tuple[np.ndarray, float]:
        """Segment at a specific point."""
        from segment_anything import SamPredictor

        image = require_image("segment", image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.load()

        predictor = SamPredictor(self.model)
        predictor.set_image(image_rgb)

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        best_idx = scores.argmax()
        print_status("segment", f"Best mask confidence: {scores[best_idx]:.2f}")
        return masks[best_idx], scores[best_idx]
