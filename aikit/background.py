"""Background removal using rembg."""

import io
import sys
from pathlib import Path
from typing import Optional, List

from PIL import Image

from aikit.core import print_status, print_download, require_file, resolve_path


class BackgroundRemover:
    """Background removal using rembg."""

    _model_loaded = False

    def remove(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """Remove background from image."""
        from rembg import remove

        path = require_file("rmbg", image_path)

        if not BackgroundRemover._model_loaded:
            print_download("rmbg", "U2Net", "~175MB")
            BackgroundRemover._model_loaded = True

        with open(path, "rb") as f:
            input_data = f.read()

        output_data = remove(input_data)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(output_data)
            print_status("rmbg", f"Saved: {output_path}")

        return Image.open(io.BytesIO(output_data))

    def batch(
        self,
        folder_path: str,
        output_folder: Optional[str] = None
    ) -> List[str]:
        """Process all images in a folder."""
        folder = resolve_path(folder_path)

        if not folder.exists():
            print(f"[rmbg] Folder not found: {folder}")
            sys.exit(1)

        if not folder.is_dir():
            print(f"[rmbg] Not a folder: {folder}")
            print(f"[rmbg] Use --batch only with folders, not single files.")
            sys.exit(1)

        if output_folder:
            out = Path(output_folder)
            out.mkdir(parents=True, exist_ok=True)
        else:
            out = folder / "nobg"
            out.mkdir(exist_ok=True)

        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

        if not images:
            print(f"[rmbg] No images found in: {folder}")
            sys.exit(1)

        print_status("rmbg", f"Found {len(images)} images in {folder}")

        results = []
        for img_path in images:
            output_path = out / f"{img_path.stem}_nobg.png"
            self.remove(str(img_path), str(output_path))
            results.append(str(output_path))

        return results
