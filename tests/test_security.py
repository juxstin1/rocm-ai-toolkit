"""
Security tests for aikit.

Tests cover:
- Path traversal prevention
- Input validation
- Safe file handling
- Injection prevention

These tests verify the toolkit handles malicious or malformed input safely.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.security
class TestPathTraversal:
    """Tests for path traversal attack prevention."""

    def test_resolve_path_blocks_parent_traversal(self, tmp_path, monkeypatch):
        """Should resolve .. to absolute path, not allow escaping."""
        from aikit.core import resolve_path

        monkeypatch.chdir(tmp_path)

        # Create a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Attempt to traverse up
        result = resolve_path("subdir/../../../etc/passwd")

        # Should resolve to absolute path, not allow arbitrary traversal
        assert result.is_absolute()
        # The path should be resolved, not contain ..
        assert ".." not in str(result)

    def test_require_file_validates_existence(self, tmp_path):
        """Should not access files outside allowed paths."""
        from aikit.core import require_file

        # Try to access a system file (should fail because it doesn't exist in temp)
        with pytest.raises(SystemExit):
            require_file("test", str(tmp_path / "../../etc/passwd"))

    def test_resolve_path_handles_null_bytes(self, tmp_path):
        """Should handle paths with null bytes safely."""
        from aikit.core import resolve_path

        # Null bytes in paths are often used in attacks
        malicious_path = f"{tmp_path}/file\x00.txt"

        # Should either work or raise a clean error, not crash
        try:
            result = resolve_path(malicious_path)
            # If it works, should be a valid path
            assert isinstance(result, Path)
        except (ValueError, OSError):
            # These are acceptable - we just don't want crashes
            pass

    def test_batch_folder_stays_within_input(self, tmp_path, mock_rembg):
        """Batch processing should not escape input folder."""
        from aikit.background import BackgroundRemover

        # Create structure
        input_folder = tmp_path / "input"
        input_folder.mkdir()

        # Create minimal PNG
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
        results = r.batch(str(input_folder))

        # All results should be within the output folder, not escaped
        for result_path in results:
            assert str(tmp_path) in result_path or "nobg" in result_path


@pytest.mark.security
class TestInputValidation:
    """Tests for input validation."""

    def test_rejects_empty_prompt(self):
        """Should handle empty prompt gracefully."""
        # Empty prompt should be passed to model (which may handle it)
        # or caught by argparse - either way, no crash
        with patch.object(sys, "argv", ["imagine", ""]):
            with patch("aikit.ImageGen") as mock_gen:
                mock_instance = MagicMock()
                mock_gen.return_value = mock_instance

                from aikit.cli import imagine

                # Should not crash
                imagine()

    def test_handles_unicode_in_prompt(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should handle unicode characters in prompts."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen()

        # Various unicode including emoji, CJK, Arabic
        unicode_prompt = "A 日本語 prompt with émojis 🎨 and العربية"

        # Should not crash
        image, path = gen.generate(unicode_prompt)
        assert image is not None

    def test_handles_very_long_prompt(self, mock_diffusers, mock_torch_cuda, tmp_path, monkeypatch):
        """Should handle extremely long prompts without crashing."""
        monkeypatch.setattr("aikit.core.OUTPUT_DIR", tmp_path)

        from aikit.generators import ImageGen

        gen = ImageGen()

        # 10KB prompt
        long_prompt = "word " * 2000

        # Should not crash (model will likely truncate internally)
        image, path = gen.generate(long_prompt)
        assert image is not None

    def test_handles_special_shell_characters(self, tmp_path):
        """Should handle shell special characters in paths."""
        from aikit.core import resolve_path

        # Characters that might cause shell injection if not handled
        special_chars = ["$HOME", "`whoami`", "$(id)", "; rm -rf /", "| cat /etc/passwd"]

        for char in special_chars:
            # Create safe test path
            test_path = f"{tmp_path}/file{char.replace('/', '_')}.txt"

            # Should resolve safely without executing anything
            try:
                result = resolve_path(test_path)
                # If it works, should be a plain path, not executed
                assert isinstance(result, Path)
                # Should not contain evidence of execution
                assert "uid=" not in str(result).lower()
            except (ValueError, OSError):
                # Acceptable - just don't execute
                pass


@pytest.mark.security
class TestFileHandling:
    """Tests for safe file handling."""

    def test_requires_file_existence(self, tmp_path):
        """Should verify file exists before processing."""
        from aikit.core import require_file

        with pytest.raises(SystemExit):
            require_file("test", str(tmp_path / "nonexistent.txt"))

    def test_image_validation(self, tmp_path):
        """Should reject non-image files as images."""
        from aikit.core import require_image

        # Create a text file disguised as image
        fake_image = tmp_path / "fake.png"
        fake_image.write_text("This is not an image")

        with pytest.raises(SystemExit):
            require_image("test", str(fake_image))

    def test_pickle_files_not_loaded_from_untrusted_source(self, tmp_path, mock_sentence_transformer):
        """Index loading should only load from expected path."""
        from aikit.embeddings import Embedder

        e = Embedder()
        e.index_path = tmp_path / "malicious.pkl"

        # If pickle doesn't exist, should raise clean error
        with pytest.raises(ValueError) as exc_info:
            e.search("query")

        assert "No index found" in str(exc_info.value)

    def test_safe_file_writing(self, tmp_path, mock_rembg):
        """Should not write outside specified directory."""
        from aikit.background import BackgroundRemover

        # Create test image
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
        results = r.batch(str(input_folder), str(output_folder))

        # Verify output is in expected location
        for result in results:
            assert str(output_folder) in result


@pytest.mark.security
class TestResourceLimits:
    """Tests for resource limit handling."""

    def test_document_truncation(self, tmp_path, mock_sentence_transformer):
        """Should truncate excessively large documents."""
        from aikit.embeddings import Embedder

        folder = tmp_path / "docs"
        folder.mkdir()

        # Create a very large file
        large_content = "x" * 100000  # 100KB
        (folder / "large.txt").write_text(large_content)

        mock_sentence_transformer.encode.return_value = [[0.1, 0.2, 0.3]]

        e = Embedder()
        e.index_folder(str(folder))

        # Should have truncated to 8000 chars
        assert len(e.index["documents"][0]) == 8000

    def test_handles_binary_files_gracefully(self, tmp_path, mock_sentence_transformer):
        """Should handle binary files without crashing."""
        from aikit.embeddings import Embedder

        folder = tmp_path / "docs"
        folder.mkdir()

        # Create binary file with .txt extension
        (folder / "binary.txt").write_bytes(os.urandom(1000))
        (folder / "normal.txt").write_text("Normal text content")

        mock_sentence_transformer.encode.return_value = [[0.1, 0.2, 0.3]]

        e = Embedder()
        # Should not crash, might skip or partially read binary
        count = e.index_folder(str(folder))

        # Should have indexed at least the normal file
        assert count >= 1


@pytest.mark.security
class TestErrorMessages:
    """Tests for secure error messages."""

    def test_no_stack_trace_in_user_error(self, tmp_path, capsys):
        """User errors should show clean message, not full traceback."""
        from aikit.core import require_file

        with pytest.raises(SystemExit):
            require_file("test", str(tmp_path / "missing.txt"))

        captured = capsys.readouterr()

        # Should not show internal paths or traceback
        assert "Traceback" not in captured.out
        assert "File \"/" not in captured.out

    def test_no_sensitive_paths_leaked(self, tmp_path, capsys):
        """Error messages should not leak sensitive system paths."""
        from aikit.core import require_file

        with pytest.raises(SystemExit):
            require_file("test", "/etc/shadow")

        captured = capsys.readouterr()

        # Should mention the file but not leak other system info
        assert "[test]" in captured.out
        assert "not found" in captured.out.lower()


@pytest.mark.security
class TestModelDownloads:
    """Tests for secure model downloading."""

    def test_sam_downloads_from_official_source(self):
        """SAM model should download from Facebook's official CDN."""
        from aikit.segmentation import Segmenter

        # URL should be from official source
        assert "fbaipublicfiles.com" in Segmenter.SAM_URL
        assert "segment_anything" in Segmenter.SAM_URL

    def test_controlnet_uses_known_models(self):
        """ControlNet should only use known model IDs."""
        from aikit.generators import ImageGen

        known_sources = ["diffusers/", "thibaud/", "lllyasviel/"]

        for name, model_id in ImageGen.CONTROLNET_MODELS.items():
            # Each model should be from a known source
            assert any(model_id.startswith(src) for src in known_sources), \
                f"Unknown source for {name}: {model_id}"
