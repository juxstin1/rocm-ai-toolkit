#!/usr/bin/env python3
"""
Background removal using rembg.

Usage:
    rmbg photo.jpg                    # Output: photo_nobg.png
    rmbg photo.jpg --output clean.png # Custom output
    rmbg folder/ --batch              # Process all images in folder
"""
import argparse
import sys
import os
from pathlib import Path

def remove_background(input_path, output_path=None):
    """Remove background from a single image."""
    from rembg import remove
    from PIL import Image

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return False

    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_nobg.png"

    print(f"Processing: {input_path}")

    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)

    with open(output_path, "wb") as f:
        f.write(output_data)

    print(f"Saved: {output_path}")
    return True

def batch_process(folder_path, output_folder=None):
    """Process all images in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return

    if output_folder:
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = folder / "nobg"
        out.mkdir(exist_ok=True)

    extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    print(f"Found {len(images)} images")

    for img_path in images:
        output_path = out / f"{img_path.stem}_nobg.png"
        remove_background(str(img_path), str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Remove backgrounds from images")
    parser.add_argument("input", help="Input image or folder")
    parser.add_argument("-o", "--output", help="Output file or folder")
    parser.add_argument("--batch", action="store_true", help="Process folder of images")

    args = parser.parse_args()

    if args.batch:
        batch_process(args.input, args.output)
    else:
        remove_background(args.input, args.output)

if __name__ == "__main__":
    main()
