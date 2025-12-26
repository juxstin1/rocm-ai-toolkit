"""
Tests for aikit.transcription module.

Tests cover:
- Transcriber initialization
- Model loading
- Transcription with and without timestamps
"""

from unittest.mock import patch, MagicMock

import pytest


class TestTranscriberInit:
    """Tests for Transcriber initialization."""

    def test_default_model(self):
        """Should use medium model by default."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        assert t.model_size == "medium"
        assert t.model is None

    def test_custom_model(self):
        """Should accept custom model size."""
        from aikit.transcription import Transcriber

        t = Transcriber(model="large-v3")
        assert t.model_size == "large-v3"

    def test_model_sizes_dict(self):
        """Should have size hints for common models."""
        from aikit.transcription import Transcriber

        expected = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        for size in expected:
            assert size in Transcriber.MODEL_SIZES


class TestTranscriberLoad:
    """Tests for Transcriber.load() method."""

    def test_loads_model_once(self, mock_whisper_model):
        """Should only load model once."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        t.load()
        t.load()  # Should be no-op

        assert t.model is not None

    def test_uses_cpu_device(self):
        """Should use CPU since CTranslate2 doesn't support ROCm."""
        with patch("faster_whisper.WhisperModel") as mock_model:
            mock_model.return_value = MagicMock()

            from aikit.transcription import Transcriber

            t = Transcriber()
            t.load()

            mock_model.assert_called_once()
            call_kwargs = mock_model.call_args
            assert call_kwargs[1]["device"] == "cpu"
            assert call_kwargs[1]["compute_type"] == "int8"


class TestTranscriberTranscribe:
    """Tests for Transcriber.transcribe() method."""

    def test_transcribes_audio_file(self, tmp_audio_file, mock_whisper_model):
        """Should transcribe audio and return text."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        result = t.transcribe(str(tmp_audio_file))

        assert isinstance(result, str)
        assert "Hello world" in result

    def test_returns_timestamps_when_requested(self, tmp_audio_file, mock_whisper_model):
        """Should return segment list when timestamps=True."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        result = t.transcribe(str(tmp_audio_file), timestamps=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "start" in result[0]
        assert "end" in result[0]
        assert "text" in result[0]

    def test_exits_for_missing_file(self, tmp_path, mock_whisper_model):
        """Should exit when audio file doesn't exist."""
        from aikit.transcription import Transcriber

        t = Transcriber()

        with pytest.raises(SystemExit):
            t.transcribe(str(tmp_path / "missing.mp3"))

    def test_accepts_language_parameter(self, tmp_audio_file):
        """Should pass language to whisper model."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_segment = MagicMock()
            mock_segment.start = 0.0
            mock_segment.end = 1.0
            mock_segment.text = " Bonjour "

            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([mock_segment], MagicMock())
            mock_model_class.return_value = mock_model

            from aikit.transcription import Transcriber

            t = Transcriber()
            t.transcribe(str(tmp_audio_file), language="fr")

            mock_model.transcribe.assert_called_once()
            call_kwargs = mock_model.transcribe.call_args
            assert call_kwargs[1]["language"] == "fr"

    def test_strips_whitespace_from_segments(self, tmp_audio_file):
        """Should strip leading/trailing whitespace from text."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_segment = MagicMock()
            mock_segment.start = 0.0
            mock_segment.end = 1.0
            mock_segment.text = "   Extra spaces   "

            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([mock_segment], MagicMock())
            mock_model_class.return_value = mock_model

            from aikit.transcription import Transcriber

            t = Transcriber()
            result = t.transcribe(str(tmp_audio_file))

            assert result == "Extra spaces"


class TestTranscriberOutput:
    """Tests for Transcriber output formatting."""

    def test_prints_character_count(self, tmp_audio_file, mock_whisper_model, capsys):
        """Should print character count for plain text output."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        t.transcribe(str(tmp_audio_file))

        captured = capsys.readouterr()
        assert "characters" in captured.out.lower()

    def test_prints_segment_count(self, tmp_audio_file, mock_whisper_model, capsys):
        """Should print segment count for timestamp output."""
        from aikit.transcription import Transcriber

        t = Transcriber()
        t.transcribe(str(tmp_audio_file), timestamps=True)

        captured = capsys.readouterr()
        assert "segment" in captured.out.lower()
