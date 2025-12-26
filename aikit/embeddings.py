"""Semantic search using Sentence Transformers."""

import pickle
from pathlib import Path
from typing import List, Dict

from aikit.core import MODELS_DIR, print_status, print_download


class Embedder:
    """Semantic search using Sentence Transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self.model = None
        self.index = None
        self.index_path = MODELS_DIR / "embeddings_index.pkl"

    def load(self):
        """Load the embedding model."""
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer

        print_download("embed", self.model_name, "~90MB")
        self.model = SentenceTransformer(self.model_name, device="cuda")

    def index_folder(
        self,
        folder: str,
        extensions: List[str] = [".txt", ".md", ".py", ".js", ".ts"],
    ) -> int:
        """Index all text files in a folder.

        Args:
            folder: Path to folder to index
            extensions: File extensions to include

        Returns:
            Number of files indexed
        """
        self.load()

        folder_path = Path(folder)
        documents = []
        paths = []

        print_status("embed", f"Scanning {folder_path}...")

        for ext in extensions:
            for file_path in folder_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        documents.append(content[:8000])
                        paths.append(str(file_path))
                except Exception as e:
                    print_status("embed", f"Skipped {file_path}: {e}")

        print_status("embed", f"Indexing {len(documents)} files...")
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.index = {
            "paths": paths,
            "embeddings": embeddings,
            "documents": documents
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        print_status("embed", f"Index saved to {self.index_path}")
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search indexed files.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of {path, score, preview} dicts
        """
        from sklearn.metrics.pairwise import cosine_similarity

        self.load()

        if self.index is None:
            if self.index_path.exists():
                print_status("embed", "Loading index...")
                with open(self.index_path, "rb") as f:
                    self.index = pickle.load(f)
            else:
                raise ValueError("No index found. Run index_folder() first.")

        print_status("embed", f"Searching for: '{query}'")
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.index["embeddings"])[0]
        top_idx = sims.argsort()[-top_k:][::-1]

        results = [
            {
                "path": self.index["paths"][i],
                "score": float(sims[i]),
                "preview": self.index["documents"][i][:200]
            }
            for i in top_idx
        ]

        print_status("embed", f"Found {len(results)} results")
        return results
