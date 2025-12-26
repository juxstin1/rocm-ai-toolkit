"""
Tests for aikit.cli module.

Tests cover:
- CLI argument parsing
- Command dispatch
- Help text and error messages
"""

import sys
from unittest.mock import patch, MagicMock

import pytest


class TestMainDispatcher:
    """Tests for main() command dispatcher."""

    def test_shows_help_with_no_args(self, capsys):
        """Should show help when no command given."""
        with patch.object(sys, "argv", ["aikit"]):
            from aikit.cli import main

            main()

            captured = capsys.readouterr()
            assert "AI Toolkit" in captured.out
            assert "imagine" in captured.out
            assert "transcribe" in captured.out

    def test_dispatches_to_imagine(self):
        """Should dispatch imagine command."""
        with patch.object(sys, "argv", ["aikit", "imagine", "test prompt"]):
            with patch("aikit.cli.imagine") as mock_imagine:
                from aikit.cli import main

                main()

                mock_imagine.assert_called_once()

    def test_dispatches_to_transcribe(self):
        """Should dispatch transcribe command."""
        with patch.object(sys, "argv", ["aikit", "transcribe", "audio.mp3"]):
            with patch("aikit.cli.transcribe") as mock_transcribe:
                from aikit.cli import main

                main()

                mock_transcribe.assert_called_once()


class TestImagineCommand:
    """Tests for imagine command."""

    def test_parses_prompt(self):
        """Should parse prompt argument."""
        mock_gen_class = MagicMock()
        mock_instance = MagicMock()
        mock_gen_class.return_value = mock_instance

        with patch.object(sys, "argv", ["imagine", "a beautiful sunset"]):
            with patch.dict("sys.modules", {"aikit.generators": MagicMock(ImageGen=mock_gen_class)}):
                # Force reimport of cli module
                import importlib
                import aikit.cli
                importlib.reload(aikit.cli)

                aikit.cli.imagine()

                mock_instance.generate.assert_called_once()
                call_kwargs = mock_instance.generate.call_args[1]
                assert call_kwargs["prompt"] == "a beautiful sunset"

    def test_parses_optional_args(self):
        """Should parse all optional arguments."""
        mock_gen_class = MagicMock()
        mock_instance = MagicMock()
        mock_gen_class.return_value = mock_instance

        with patch.object(sys, "argv", [
            "imagine", "prompt",
            "--steps", "20",
            "--cfg", "7.5",
            "-W", "512",
            "-H", "768",
            "-s", "42",
            "-o", "output.png",
        ]):
            with patch.dict("sys.modules", {"aikit.generators": MagicMock(ImageGen=mock_gen_class)}):
                import importlib
                import aikit.cli
                importlib.reload(aikit.cli)

                aikit.cli.imagine()

                call_kwargs = mock_instance.generate.call_args[1]
                assert call_kwargs["steps"] == 20
                assert call_kwargs["cfg"] == 7.5
                assert call_kwargs["width"] == 512
                assert call_kwargs["height"] == 768
                assert call_kwargs["seed"] == 42
                assert call_kwargs["output"] == "output.png"

    def test_requires_input_with_controlnet(self):
        """Should error when controlnet used without input."""
        with patch.object(sys, "argv", ["imagine", "prompt", "--controlnet", "canny"]):
            with pytest.raises(SystemExit) as exc_info:
                from aikit.cli import imagine

                imagine()

            assert exc_info.value.code == 1


class TestTranscribeCommand:
    """Tests for transcribe command."""

    def test_parses_input_file(self):
        """Should parse input file argument."""
        with patch.object(sys, "argv", ["transcribe", "audio.mp3"]):
            with patch("aikit.Transcriber") as mock_trans:
                mock_instance = MagicMock()
                mock_instance.transcribe.return_value = "transcribed text"
                mock_trans.return_value = mock_instance

                from aikit.cli import transcribe

                transcribe()

                mock_instance.transcribe.assert_called_once()
                assert mock_instance.transcribe.call_args[0][0] == "audio.mp3"

    def test_parses_model_choice(self):
        """Should parse model size argument."""
        with patch.object(sys, "argv", ["transcribe", "audio.mp3", "-m", "large-v3"]):
            with patch("aikit.Transcriber") as mock_trans:
                mock_instance = MagicMock()
                mock_instance.transcribe.return_value = "text"
                mock_trans.return_value = mock_instance

                from aikit.cli import transcribe

                transcribe()

                mock_trans.assert_called_with(model="large-v3")

    def test_timestamps_flag(self):
        """Should pass timestamps flag."""
        with patch.object(sys, "argv", ["transcribe", "audio.mp3", "--timestamps"]):
            with patch("aikit.Transcriber") as mock_trans:
                mock_instance = MagicMock()
                mock_instance.transcribe.return_value = []
                mock_trans.return_value = mock_instance

                from aikit.cli import transcribe

                transcribe()

                call_kwargs = mock_instance.transcribe.call_args[1]
                assert call_kwargs["timestamps"] is True


class TestEmbedCommand:
    """Tests for embed command."""

    def test_index_subcommand(self):
        """Should handle index subcommand."""
        with patch.object(sys, "argv", ["embed", "index", "/path/to/folder"]):
            with patch("aikit.Embedder") as mock_embed:
                mock_instance = MagicMock()
                mock_instance.index_folder.return_value = 10
                mock_embed.return_value = mock_instance

                from aikit.cli import embed

                embed()

                mock_instance.index_folder.assert_called_once()
                assert mock_instance.index_folder.call_args[0][0] == "/path/to/folder"

    def test_search_subcommand(self):
        """Should handle search subcommand."""
        with patch.object(sys, "argv", ["embed", "search", "query text", "-n", "10"]):
            with patch("aikit.Embedder") as mock_embed:
                mock_instance = MagicMock()
                mock_instance.search.return_value = []
                mock_embed.return_value = mock_instance

                from aikit.cli import embed

                embed()

                mock_instance.search.assert_called_once()
                call_kwargs = mock_instance.search.call_args
                assert call_kwargs[0][0] == "query text"
                assert call_kwargs[1]["top_k"] == 10


class TestSegmentCommand:
    """Tests for segment command."""

    def test_parses_image_path(self):
        """Should parse image path argument."""
        with patch.object(sys, "argv", ["segment", "image.jpg"]):
            with patch("aikit.Segmenter") as mock_seg:
                mock_instance = MagicMock()
                mock_instance.segment.return_value = []
                mock_seg.return_value = mock_instance

                from aikit.cli import segment

                segment()

                mock_instance.segment.assert_called_once()

    def test_point_segmentation(self):
        """Should handle point-based segmentation."""
        with patch.object(sys, "argv", ["segment", "image.jpg", "--point", "100", "200"]):
            with patch("aikit.Segmenter") as mock_seg:
                import numpy as np

                mock_instance = MagicMock()
                mock_instance.segment_point.return_value = (np.zeros((10, 10)), 0.95)
                mock_seg.return_value = mock_instance

                from aikit.cli import segment

                segment()

                mock_instance.segment_point.assert_called_once()
                call_args = mock_instance.segment_point.call_args[0]
                assert call_args[1] == 100  # x
                assert call_args[2] == 200  # y


class TestRmbgCommand:
    """Tests for rmbg command."""

    def test_parses_input(self):
        """Should parse input image path."""
        with patch.object(sys, "argv", ["rmbg", "photo.jpg"]):
            with patch("aikit.BackgroundRemover") as mock_rmbg:
                mock_instance = MagicMock()
                mock_rmbg.return_value = mock_instance

                from aikit.cli import rmbg

                rmbg()

                mock_instance.remove.assert_called_once()

    def test_batch_mode(self):
        """Should use batch processing when flag set."""
        with patch.object(sys, "argv", ["rmbg", "folder/", "--batch"]):
            with patch("aikit.BackgroundRemover") as mock_rmbg:
                mock_instance = MagicMock()
                mock_instance.batch.return_value = []
                mock_rmbg.return_value = mock_instance

                from aikit.cli import rmbg

                rmbg()

                mock_instance.batch.assert_called_once()

    def test_generates_output_name(self):
        """Should generate output name when not specified."""
        with patch.object(sys, "argv", ["rmbg", "photo.jpg"]):
            with patch("aikit.BackgroundRemover") as mock_rmbg:
                mock_instance = MagicMock()
                mock_rmbg.return_value = mock_instance

                from aikit.cli import rmbg

                rmbg()

                call_args = mock_instance.remove.call_args[0]
                assert "photo_nobg.png" in call_args[1]


class TestHelpText:
    """Tests for help text output."""

    def test_imagine_help(self, capsys):
        """Should show imagine help."""
        with patch.object(sys, "argv", ["imagine", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from aikit.cli import imagine

                imagine()

            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "Stable Diffusion" in captured.out

    def test_transcribe_help(self, capsys):
        """Should show transcribe help."""
        with patch.object(sys, "argv", ["transcribe", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from aikit.cli import transcribe

                transcribe()

            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "audio" in captured.out.lower()
