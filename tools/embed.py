#!/usr/bin/env python3
"""
Semantic search and embedding tool using Sentence Transformers.

Usage:
    embed index ./notes/              # Index a folder of text files
    embed search "authentication flow" # Search indexed files
    embed similar file.txt            # Find similar files
"""
import argparse
import sys
import os
import json
import pickle
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "models" / "embeddings_index.pkl"

def get_model():
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model...")
    return SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

def index_folder(folder_path, extensions=[".txt", ".md", ".py", ".js", ".ts", ".json"]):
    """Index all text files in a folder."""
    model = get_model()
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    documents = []
    paths = []

    for ext in extensions:
        for file_path in folder.rglob(f"*{ext}"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    documents.append(content[:8000])  # Limit content length
                    paths.append(str(file_path))
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    if not documents:
        print("No documents found to index.")
        sys.exit(1)

    print(f"Indexing {len(documents)} files...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    index = {
        "paths": paths,
        "embeddings": embeddings,
        "documents": documents
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Indexed {len(documents)} files -> {INDEX_PATH}")

def search_query(query, top_k=5):
    """Search indexed files for a query."""
    if not INDEX_PATH.exists():
        print("No index found. Run: embed index <folder>")
        sys.exit(1)

    model = get_model()

    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True)

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, index["embeddings"])[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    print(f"\nTop {top_k} results for: '{query}'\n" + "="*50)
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        path = index["paths"][idx]
        preview = index["documents"][idx][:200].replace("\n", " ")
        print(f"\n{i+1}. [{score:.3f}] {path}")
        print(f"   {preview}...")

def find_similar(file_path, top_k=5):
    """Find files similar to a given file."""
    if not INDEX_PATH.exists():
        print("No index found. Run: embed index <folder>")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    model = get_model()

    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)

    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")[:8000]
    query_embedding = model.encode([content], convert_to_numpy=True)

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, index["embeddings"])[0]

    top_indices = similarities.argsort()[-top_k-1:][::-1]

    print(f"\nFiles similar to: {file_path}\n" + "="*50)
    for i, idx in enumerate(top_indices):
        if index["paths"][idx] == str(Path(file_path).resolve()):
            continue
        score = similarities[idx]
        path = index["paths"][idx]
        print(f"{i+1}. [{score:.3f}] {path}")

def main():
    parser = argparse.ArgumentParser(description="Semantic search tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a folder")
    index_parser.add_argument("folder", help="Folder to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed files")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--top", type=int, default=5, help="Number of results")

    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar files")
    similar_parser.add_argument("file", help="File to find similar matches for")
    similar_parser.add_argument("-n", "--top", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "index":
        index_folder(args.folder)
    elif args.command == "search":
        search_query(args.query, args.top)
    elif args.command == "similar":
        find_similar(args.file, args.top)

if __name__ == "__main__":
    main()
