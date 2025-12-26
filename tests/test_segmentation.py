"""
Tests for aikit.segmentation module.

Tests cover:
- Segmenter initialization
- Model loading and checkpoint download
- Auto segmentation
- Point-based segmentation
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestSegmenterInit:
    """Tests for Segmenter initialization."""

    def test_default_state(self):
        """Should initialize with no model loaded."""
        from aikit.segmentation import Segmenter

        s = Segmenter()
        assert s.model is None

    def test_checkpoint_path(self, tmp_path, monkeypatch):
        """Should set checkpoint path in models directory."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_path)

        from aikit.segmentation import Segmenter

        s = Segmenter()
        assert s.checkpoint.parent == tmp_path
        assert "sam" in str(s.checkpoint).lower()


class TestSegmenterLoad:
    """Tests for Segmenter.load() method."""

    def test_downloads_checkpoint_if_missing(self, tmp_path, monkeypatch):
        """Should download checkpoint on first load."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_path)

        with patch("urllib.request.urlretrieve") as mock_download:
            with patch("segment_anything.sam_model_registry") as mock_registry:
                mock_model = MagicMock()
                mock_registry.__getitem__ = MagicMock(
                    return_value=lambda checkpoint: mock_model
                )

                from aikit.segmentation import Segmenter

                s = Segmenter()
                s.load()

                mock_download.assert_called_once()
                assert "sam" in mock_download.call_args[0][0].lower()

    def test_skips_download_if_checkpoint_exists(self, tmp_path, monkeypatch):
        """Should not download if checkpoint already exists."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_path)

        # Create fake checkpoint
        checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
        checkpoint.write_bytes(b"fake model data")

        with patch("urllib.request.urlretrieve") as mock_download:
            with patch("segment_anything.sam_model_registry") as mock_registry:
                mock_model = MagicMock()
                mock_registry.__getitem__ = MagicMock(
                    return_value=lambda checkpoint: mock_model
                )

                from aikit.segmentation import Segmenter

                s = Segmenter()
                s.checkpoint = checkpoint
                s.load()

                mock_download.assert_not_called()

    def test_loads_model_to_cuda(self, tmp_path, monkeypatch):
        """Should load model to cuda device (Windows ROCm)."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_path)
        checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
        checkpoint.write_bytes(b"fake")

        with patch("segment_anything.sam_model_registry") as mock_registry:
            mock_model = MagicMock()
            mock_registry.__getitem__ = MagicMock(
                return_value=lambda checkpoint: mock_model
            )

            from aikit.segmentation import Segmenter

            s = Segmenter()
            s.checkpoint = checkpoint
            s.load()

            mock_model.to.assert_called_once_with("cuda")


class TestSegmenterSegment:
    """Tests for Segmenter.segment() method."""

    def test_segments_image(self, tmp_image_file, mock_sam_model):
        """Should return list of mask dictionaries."""
        with patch("urllib.request.urlretrieve"):
            from aikit.segmentation import Segmenter

            s = Segmenter()
            # Bypass checkpoint check
            s.checkpoint = tmp_image_file.parent / "fake.pth"
            s.checkpoint.write_bytes(b"fake")

            masks = s.segment(str(tmp_image_file))

            assert isinstance(masks, list)
            assert len(masks) >= 1

    def test_exits_for_missing_image(self, tmp_path, mock_sam_model):
        """Should exit when image doesn't exist."""
        from aikit.segmentation import Segmenter

        s = Segmenter()

        with pytest.raises(SystemExit):
            s.segment(str(tmp_path / "missing.png"))

    def test_converts_bgr_to_rgb(self, tmp_image_file, monkeypatch):
        """Should convert image from BGR to RGB for SAM."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_image_file.parent)

        with patch("urllib.request.urlretrieve"):
            with patch("segment_anything.sam_model_registry") as mock_registry:
                with patch("segment_anything.SamAutomaticMaskGenerator") as mock_gen:
                    mock_model = MagicMock()
                    mock_registry.__getitem__ = MagicMock(
                        return_value=lambda checkpoint: mock_model
                    )
                    mock_gen.return_value.generate.return_value = []

                    with patch("cv2.cvtColor") as mock_cvt:
                        mock_cvt.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

                        from aikit.segmentation import Segmenter
                        import cv2

                        s = Segmenter()
                        s.checkpoint = tmp_image_file.parent / "fake.pth"
                        s.checkpoint.write_bytes(b"fake")
                        s.segment(str(tmp_image_file))

                        mock_cvt.assert_called()
                        # Should use COLOR_BGR2RGB
                        assert mock_cvt.call_args[0][1] == cv2.COLOR_BGR2RGB


class TestSegmenterSegmentPoint:
    """Tests for Segmenter.segment_point() method."""

    def test_segments_at_point(self, tmp_image_file, mock_sam_model):
        """Should return mask and score for point."""
        with patch("urllib.request.urlretrieve"):
            from aikit.segmentation import Segmenter

            s = Segmenter()
            s.checkpoint = tmp_image_file.parent / "fake.pth"
            s.checkpoint.write_bytes(b"fake")

            mask, score = s.segment_point(str(tmp_image_file), 50, 50)

            assert isinstance(mask, np.ndarray)
            assert isinstance(score, (int, float))

    def test_returns_best_mask(self, tmp_image_file, monkeypatch):
        """Should return the mask with highest score."""
        monkeypatch.setattr("aikit.core.MODELS_DIR", tmp_image_file.parent)

        with patch("urllib.request.urlretrieve"):
            with patch("segment_anything.sam_model_registry") as mock_registry:
                with patch("segment_anything.SamPredictor") as mock_pred_class:
                    mock_model = MagicMock()
                    mock_registry.__getitem__ = MagicMock(
                        return_value=lambda checkpoint: mock_model
                    )

                    # Three masks with different scores
                    masks = np.array([
                        np.zeros((10, 10), dtype=bool),
                        np.ones((10, 10), dtype=bool),
                        np.zeros((10, 10), dtype=bool),
                    ])
                    scores = np.array([0.5, 0.9, 0.3])

                    mock_predictor = MagicMock()
                    mock_predictor.predict.return_value = (masks, scores, None)
                    mock_pred_class.return_value = mock_predictor

                    from aikit.segmentation import Segmenter

                    s = Segmenter()
                    s.checkpoint = tmp_image_file.parent / "fake.pth"
                    s.checkpoint.write_bytes(b"fake")

                    mask, score = s.segment_point(str(tmp_image_file), 5, 5)

                    # Should return the mask with score 0.9
                    assert score == 0.9
                    assert mask.all()  # The one filled with ones
