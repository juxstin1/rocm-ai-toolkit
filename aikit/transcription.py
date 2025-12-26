"""Audio/video transcription using Faster-Whisper."""

from typing import Optional, List, Dict, Union

from aikit.core import print_status, print_download, require_audio


class Transcriber:
    """Audio/video transcription using Faster-Whisper."""

    MODEL_SIZES = {
        "tiny": "~75MB",
        "base": "~150MB",
        "small": "~500MB",
        "medium": "~1.5GB",
        "large-v2": "~3GB",
        "large-v3": "~3GB",
    }

    def __init__(self, model: str = "medium"):
        self.model_size = model
        self.model = None

    def load(self):
        """Load the Whisper model."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel
        from aikit.core import print_status

        size_hint = self.MODEL_SIZES.get(self.model_size, "")
        print_download("transcribe", f"Whisper {self.model_size}", size_hint)

        # CTranslate2 is CUDA-only, use CPU on AMD
        # Still fast - optimized with AVX2/INT8
        print_status("transcribe", "Using CPU (CTranslate2 doesn't support ROCm)")
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> Union[str, List[Dict]]:
        """Transcribe audio/video file.

        Args:
            audio_path: Path to audio or video file
            language: Language code (auto-detected if not specified)
            timestamps: If True, return list of segments with timestamps

        Returns:
            Either plain text or list of {start, end, text} dicts
        """
        path = require_audio("transcribe", audio_path)

        self.load()

        segments, info = self.model.transcribe(str(path), language=language)

        if timestamps:
            result = [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments
            ]
            print_status("transcribe", f"Found {len(result)} segments")
            return result
        else:
            text = " ".join(s.text.strip() for s in segments)
            print_status("transcribe", f"Transcribed {len(text)} characters")
            return text
