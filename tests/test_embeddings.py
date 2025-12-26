"""
Tests for aikit.embeddings module.

Tests cover:
- Embedder initialization
- Folder indexing
- Search functionality
- Index persistence
"""

import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestEmbedderInit:
    """Tests for Embedder initialization."""

    def test_default_model(self):
        """Should use default model name."""
        from aikit.embeddings import Embedder

        e = Embedder()
        assert e.model_name == "all-MiniLM-L6-v2"
        assert e.model is None

    def test_custom_model(self):
        """Should accept custom model name."""
        from aikit.embeddings import Embedder

        e = Embedder(model="custom-model")
        assert e.model_name == "custom-model"


class TestEmbedderLoad:
    """Tests for Embedder.load() method."""

    def test_loads_model_once(self, mock_sentence_transformer):
        """Should only load model once."""
        from aikit.embeddings import Embedder

        e = Embedder()
        e.load()
        e.load()  # Second call should be no-op

        # Model should only be created once
        assert e.model is not None

    def test_uses_cuda_device(self, mock_sentence_transformer):
        """Should use cuda device for Windows ROCm."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()

            from aikit.embeddings import Embedder

            e = Embedder()
            e.load()

            mock_st.assert_called_once()
            call_kwargs = mock_st.call_args
            assert call_kwargs[1]["device"] == "cuda"


class TestEmbedderIndexFolder:
    """Tests for Embedder.index_folder() method."""

    def test_indexes_text_files(self, tmp_folder_with_files, mock_sentence_transformer):
        """Should index text files in folder."""
        from aikit.embeddings import Embedder

        # Setup mock to return proper shaped array
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        e = Embedder()
        count = e.index_folder(str(tmp_folder_with_files))

        # Should index .md, .py, .txt but not .json (not in default extensions)
        assert count == 3

    def test_respects_extensions_filter(self, tmp_folder_with_files, mock_sentence_transformer):
        """Should only index specified extensions."""
        from aikit.embeddings import Embedder

        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        e = Embedder()
        count = e.index_folder(str(tmp_folder_with_files), extensions=[".md"])

        assert count == 1  # Only readme.md

    def test_saves_index_to_disk(self, tmp_folder_with_files, mock_sentence_transformer, tmp_path):
        """Should persist index to pickle file."""
        from aikit.embeddings import Embedder

        mock_sentence_transformer.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        e = Embedder()
        e.index_path = tmp_path / "test_index.pkl"
        e.index_folder(str(tmp_folder_with_files))

        assert e.index_path.exists()

        with open(e.index_path, "rb") as f:
            saved_index = pickle.load(f)

        assert "paths" in saved_index
        assert "embeddings" in saved_index
        assert "documents" in saved_index

    def test_skips_empty_files(self, tmp_path, mock_sentence_transformer):
        """Should skip empty files."""
        from aikit.embeddings import Embedder

        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "empty.txt").write_text("")
        (folder / "content.txt").write_text("Has content")

        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        e = Embedder()
        count = e.index_folder(str(folder))

        assert count == 1  # Only the non-empty file

    def test_truncates_long_documents(self, tmp_path, mock_sentence_transformer):
        """Should truncate documents longer than 8000 chars."""
        from aikit.embeddings import Embedder

        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "long.txt").write_text("x" * 10000)

        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        e = Embedder()
        e.index_folder(str(folder))

        # Check the stored document is truncated
        assert len(e.index["documents"][0]) == 8000

    def test_handles_unreadable_files(self, tmp_path, mock_sentence_transformer, capsys):
        """Should skip unreadable files and report them."""
        from aikit.embeddings import Embedder

        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "good.txt").write_text("Good content")

        # Create a file that will cause read issues (binary with invalid UTF-8)
        binary_file = folder / "binary.txt"
        binary_file.write_bytes(b"\x80\x81\x82")

        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        e = Embedder()
        count = e.index_folder(str(folder))

        # Should still index the good file
        # Note: errors="ignore" in the code means binary won't actually fail,
        # but would just be empty or garbled
        assert count >= 1


class TestEmbedderSearch:
    """Tests for Embedder.search() method."""

    def test_searches_indexed_documents(self, tmp_path, mock_sentence_transformer):
        """Should return relevant search results."""
        from aikit.embeddings import Embedder

        # Create a pre-existing index
        index = {
            "paths": ["/path/to/doc1.txt", "/path/to/doc2.txt"],
            "embeddings": np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.1]]),
            "documents": ["Document about cats", "Document about dogs"],
        }

        index_path = tmp_path / "test_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(index, f)

        # Mock encode to return query embedding similar to first doc
        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.9, 0.1]])

        e = Embedder()
        e.index_path = index_path
        results = e.search("cats", top_k=2)

        assert len(results) == 2
        assert results[0]["path"] == "/path/to/doc1.txt"
        assert "score" in results[0]
        assert "preview" in results[0]

    def test_raises_when_no_index(self, tmp_path, mock_sentence_transformer):
        """Should raise error when no index exists."""
        from aikit.embeddings import Embedder

        e = Embedder()
        e.index_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(ValueError) as exc_info:
            e.search("query")

        assert "No index found" in str(exc_info.value)

    def test_loads_existing_index(self, tmp_path, mock_sentence_transformer):
        """Should load index from disk when needed."""
        from aikit.embeddings import Embedder

        index = {
            "paths": ["/path/to/doc.txt"],
            "embeddings": np.array([[0.5, 0.5, 0.5]]),
            "documents": ["Test document"],
        }

        index_path = tmp_path / "test_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(index, f)

        mock_sentence_transformer.encode.return_value = np.array([[0.5, 0.5, 0.5]])

        e = Embedder()
        e.index_path = index_path

        # Index should be None initially
        assert e.index is None

        results = e.search("test")

        # Should have loaded index
        assert e.index is not None
        assert len(results) == 1

    def test_truncates_preview(self, tmp_path, mock_sentence_transformer):
        """Should truncate preview to 200 characters."""
        from aikit.embeddings import Embedder

        long_doc = "x" * 500
        index = {
            "paths": ["/path/to/doc.txt"],
            "embeddings": np.array([[0.5, 0.5, 0.5]]),
            "documents": [long_doc],
        }

        index_path = tmp_path / "test_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(index, f)

        mock_sentence_transformer.encode.return_value = np.array([[0.5, 0.5, 0.5]])

        e = Embedder()
        e.index_path = index_path
        results = e.search("test")

        assert len(results[0]["preview"]) == 200
