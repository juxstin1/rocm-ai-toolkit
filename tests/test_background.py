"""
Tests for aikit.background module.

Tests cover:
- BackgroundRemover single image processing
- Batch folder processing
- Output file handling
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image


class TestBackgroundRemoverRemove:
    """Tests for BackgroundRemover.remove() method."""

    def test_removes_background(self, tmp_image_file, mock_rembg):
        """Should process image and return PIL Image."""
        from aikit.background import BackgroundRemover

        r = BackgroundRemover()
        result = r.remove(str(tmp_image_file))

        assert isinstance(result, Image.Image)

    def test_saves_output_when_path_provided(self, tmp_image_file, tmp_path, mock_rembg):
        """Should save to specified output path."""
        from aikit.background import BackgroundRemover

        output_path = tmp_path / "output.png"

        r = BackgroundRemover()
        r.remove(str(tmp_image_file), str(output_path))

        assert output_path.exists()

    def test_exits_for_missing_file(self, tmp_path, mock_rembg):
        """Should exit when input file doesn't exist."""
        from aikit.background import BackgroundRemover

        r = BackgroundRemover()

        with pytest.raises(SystemExit):
            r.remove(str(tmp_path / "missing.png"))

    def test_prints_status_on_save(self, tmp_image_file, tmp_path, mock_rembg, capsys):
        """Should print save confirmation."""
        from aikit.background import BackgroundRemover

        output_path = tmp_path / "output.png"

        r = BackgroundRemover()
        r.remove(str(tmp_image_file), str(output_path))

        captured = capsys.readouterr()
        assert "Saved" in captured.out


class TestBackgroundRemoverModelLoading:
    """Tests for model loading behavior."""

    def test_prints_download_on_first_load(self, tmp_image_file, mock_rembg, capsys):
        """Should print download message only once."""
        from aikit.background import BackgroundRemover

        # Reset class state
        BackgroundRemover._model_loaded = False

        r = BackgroundRemover()
        r.remove(str(tmp_image_file))

        captured = capsys.readouterr()
        assert "U2Net" in captured.out

    def test_no_download_message_on_subsequent_calls(self, tmp_image_file, mock_rembg, capsys):
        """Should not print download message after first load."""
        from aikit.background import BackgroundRemover

        # Simulate already loaded
        BackgroundRemover._model_loaded = True

        r = BackgroundRemover()
        r.remove(str(tmp_image_file))

        captured = capsys.readouterr()
        assert "U2Net" not in captured.out


class TestBackgroundRemoverBatch:
    """Tests for BackgroundRemover.batch() method."""

    def test_processes_all_images_in_folder(self, tmp_path, mock_rembg):
        """Should process all supported image formats."""
        from aikit.background import BackgroundRemover

        # Create test images
        input_folder = tmp_path / "input"
        input_folder.mkdir()

        # Create minimal PNG files
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])

        (input_folder / "image1.png").write_bytes(png_data)
        (input_folder / "image2.jpg").write_bytes(png_data)  # Fake jpg
        (input_folder / "ignore.txt").write_text("not an image")

        r = BackgroundRemover()
        results = r.batch(str(input_folder))

        assert len(results) == 2

    def test_creates_output_folder(self, tmp_path, mock_rembg):
        """Should create output folder if specified."""
        from aikit.background import BackgroundRemover

        input_folder = tmp_path / "input"
        input_folder.mkdir()
        output_folder = tmp_path / "output"

        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        (input_folder / "test.png").write_bytes(png_data)

        r = BackgroundRemover()
        r.batch(str(input_folder), str(output_folder))

        assert output_folder.exists()

    def test_creates_nobg_subfolder_by_default(self, tmp_path, mock_rembg):
        """Should create 'nobg' subfolder when no output specified."""
        from aikit.background import BackgroundRemover

        input_folder = tmp_path / "input"
        input_folder.mkdir()

        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        (input_folder / "test.png").write_bytes(png_data)

        r = BackgroundRemover()
        r.batch(str(input_folder))

        assert (input_folder / "nobg").exists()

    def test_exits_for_missing_folder(self, tmp_path, mock_rembg):
        """Should exit when input folder doesn't exist."""
        from aikit.background import BackgroundRemover

        r = BackgroundRemover()

        with pytest.raises(SystemExit):
            r.batch(str(tmp_path / "nonexistent"))

    def test_exits_for_file_instead_of_folder(self, tmp_text_file, mock_rembg):
        """Should exit when input is a file, not folder."""
        from aikit.background import BackgroundRemover

        r = BackgroundRemover()

        with pytest.raises(SystemExit):
            r.batch(str(tmp_text_file))

    def test_exits_when_no_images_found(self, tmp_path, mock_rembg):
        """Should exit when folder contains no images."""
        from aikit.background import BackgroundRemover

        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        (empty_folder / "readme.txt").write_text("no images here")

        r = BackgroundRemover()

        with pytest.raises(SystemExit):
            r.batch(str(empty_folder))

    def test_output_filenames_have_nobg_suffix(self, tmp_path, mock_rembg):
        """Should add _nobg suffix to output filenames."""
        from aikit.background import BackgroundRemover

        input_folder = tmp_path / "input"
        input_folder.mkdir()

        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        (input_folder / "photo.png").write_bytes(png_data)

        r = BackgroundRemover()
        results = r.batch(str(input_folder))

        assert any("photo_nobg.png" in r for r in results)
