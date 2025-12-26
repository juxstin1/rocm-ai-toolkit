"""
Tests for aikit.core module.

Tests cover:
- Device detection (GPU/CPU)
- Path resolution and validation
- File requirement helpers
- Status printing utilities
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestGetDevice:
    """Tests for get_device() function."""

    def test_returns_cuda_when_available(self, mock_torch_cuda):
        """Should return cuda device when GPU is available."""
        from aikit.core import get_device

        device = get_device()
        assert device.type == "cuda"

    def test_returns_cpu_when_no_gpu(self, mock_torch_cpu):
        """Should return cpu device when no GPU available."""
        from aikit.core import get_device

        device = get_device()
        assert device.type == "cpu"


class TestResolvePath:
    """Tests for resolve_path() function."""

    def test_resolves_absolute_path(self, tmp_path):
        """Should handle absolute paths."""
        from aikit.core import resolve_path

        file_path = tmp_path / "test.txt"
        file_path.touch()

        result = resolve_path(str(file_path))
        assert result == file_path
        assert result.is_absolute()

    def test_expands_user_home(self):
        """Should expand ~ to user home directory."""
        from aikit.core import resolve_path

        result = resolve_path("~/some/path")
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_resolves_relative_path(self, tmp_path, monkeypatch):
        """Should resolve relative paths to absolute."""
        from aikit.core import resolve_path

        monkeypatch.chdir(tmp_path)
        result = resolve_path("relative/path")
        assert result.is_absolute()
        assert str(tmp_path) in str(result)


class TestRequireFile:
    """Tests for require_file() function."""

    def test_returns_path_when_file_exists(self, tmp_text_file):
        """Should return resolved path for existing file."""
        from aikit.core import require_file

        result = require_file("test", str(tmp_text_file))
        assert result == tmp_text_file

    def test_exits_when_file_missing(self, tmp_path):
        """Should exit with code 1 when file doesn't exist."""
        from aikit.core import require_file

        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(SystemExit) as exc_info:
            require_file("test", str(missing))

        assert exc_info.value.code == 1

    def test_prints_error_message(self, tmp_path, capsys):
        """Should print helpful error message."""
        from aikit.core import require_file

        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(SystemExit):
            require_file("test", str(missing))

        captured = capsys.readouterr()
        assert "[test]" in captured.out
        assert "not found" in captured.out.lower()

    def test_prints_custom_hint(self, tmp_path, capsys):
        """Should print custom hint when provided."""
        from aikit.core import require_file

        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(SystemExit):
            require_file("test", str(missing), hint="Try this instead")

        captured = capsys.readouterr()
        assert "Try this instead" in captured.out


class TestRequireImage:
    """Tests for require_image() function."""

    def test_loads_valid_image(self, tmp_image_file):
        """Should load and return image array for valid image."""
        from aikit.core import require_image

        result = require_image("test", str(tmp_image_file))

        # Should be a numpy array with image data
        assert result is not None
        assert hasattr(result, "shape")

    def test_exits_when_file_missing(self, tmp_path):
        """Should exit when image file doesn't exist."""
        from aikit.core import require_image

        missing = tmp_path / "missing.png"

        with pytest.raises(SystemExit) as exc_info:
            require_image("test", str(missing))

        assert exc_info.value.code == 1

    def test_exits_for_invalid_image(self, tmp_text_file, capsys):
        """Should exit with error for non-image file."""
        from aikit.core import require_image

        with pytest.raises(SystemExit) as exc_info:
            require_image("test", str(tmp_text_file))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "could not read" in captured.out.lower() or "corrupted" in captured.out.lower()


class TestRequireAudio:
    """Tests for require_audio() function."""

    def test_returns_path_for_valid_audio(self, tmp_audio_file):
        """Should return path for existing audio file."""
        from aikit.core import require_audio

        result = require_audio("test", str(tmp_audio_file))
        assert result == tmp_audio_file

    def test_exits_when_file_missing(self, tmp_path):
        """Should exit when audio file doesn't exist."""
        from aikit.core import require_audio

        missing = tmp_path / "missing.mp3"

        with pytest.raises(SystemExit) as exc_info:
            require_audio("test", str(missing))

        assert exc_info.value.code == 1


class TestPrintStatus:
    """Tests for print_status() function."""

    def test_prints_with_tool_prefix(self, capsys):
        """Should print message with [tool] prefix."""
        from aikit.core import print_status

        print_status("mytool", "Hello world")

        captured = capsys.readouterr()
        assert "[mytool]" in captured.out
        assert "Hello world" in captured.out

    def test_handles_special_characters(self, capsys):
        """Should handle messages with special characters."""
        from aikit.core import print_status

        print_status("test", "Path: C:\\Users\\test & more")

        captured = capsys.readouterr()
        assert "C:\\Users\\test & more" in captured.out


class TestPrintDownload:
    """Tests for print_download() function."""

    def test_prints_with_size_hint(self, capsys):
        """Should include size hint when provided."""
        from aikit.core import print_download

        print_download("test", "ModelName", "~500MB")

        captured = capsys.readouterr()
        assert "[test]" in captured.out
        assert "ModelName" in captured.out
        assert "500MB" in captured.out
        assert "once" in captured.out.lower()

    def test_prints_without_size_hint(self, capsys):
        """Should work without size hint."""
        from aikit.core import print_download

        print_download("test", "ModelName")

        captured = capsys.readouterr()
        assert "[test]" in captured.out
        assert "ModelName" in captured.out


class TestDirectories:
    """Tests for MODELS_DIR and OUTPUT_DIR."""

    def test_models_dir_is_path(self):
        """MODELS_DIR should be a Path object."""
        from aikit.core import MODELS_DIR

        assert isinstance(MODELS_DIR, Path)

    def test_output_dir_is_path(self):
        """OUTPUT_DIR should be a Path object."""
        from aikit.core import OUTPUT_DIR

        assert isinstance(OUTPUT_DIR, Path)
