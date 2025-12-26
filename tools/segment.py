#!/usr/bin/env python3
"""
Image segmentation using Segment Anything (SAM).

Usage:
    segment image.png                    # Auto-segment everything
    segment image.png --point 100,200    # Segment at specific point
    segment image.png --output mask.png  # Save mask
"""
import argparse
import sys
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

def download_model():
    """Download SAM model if not present."""
    if SAM_CHECKPOINT.exists():
        return

    print(f"Downloading SAM model (~375MB)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print(f"Saved to: {SAM_CHECKPOINT}")

def segment_image(image_path, point=None, output_path=None, show=True):
    """Segment an image using SAM."""
    import numpy as np
    import cv2
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    download_model()

    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to("cuda")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if point:
        # Point-based segmentation
        predictor = SamPredictor(sam)
        predictor.set_image(image_rgb)

        x, y = map(int, point.split(","))
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Use best mask
        best_idx = scores.argmax()
        mask = masks[best_idx]

        print(f"Segmented at point ({x}, {y}) with score: {scores[best_idx]:.3f}")

    else:
        # Auto-segment everything
        print("Auto-segmenting entire image...")
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks_data = mask_generator.generate(image_rgb)

        print(f"Found {len(masks_data)} segments")

        # Create colored visualization
        result = image.copy()
        for i, mask_info in enumerate(masks_data):
            mask = mask_info["segmentation"]
            color = np.random.randint(0, 255, 3).tolist()
            result[mask] = result[mask] * 0.5 + np.array(color) * 0.5

        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Saved segmented image: {output_path}")

        if show:
            cv2.imshow("Segments", result)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return

    # Single mask output
    if output_path:
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_image)
        print(f"Saved mask: {output_path}")

    if show:
        # Overlay mask on image
        overlay = image.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imshow("Segment", overlay)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Segment images using SAM")
    parser.add_argument("image", help="Input image file")
    parser.add_argument("-p", "--point", help="Point to segment at (x,y)")
    parser.add_argument("-o", "--output", help="Output mask/image file")
    parser.add_argument("--no-show", action="store_true", help="Don't display result")

    args = parser.parse_args()
    segment_image(args.image, args.point, args.output, show=not args.no_show)

if __name__ == "__main__":
    main()
