import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_PATH = BASE_DIR / "data" / "processed" / "pages.jsonl"
INDEX_DIR = BASE_DIR / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
FAISS_PATH = INDEX_DIR / "faiss.index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pages():
    pages = []
    with open(PAGES_PATH, encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1 and newline_pos > start + size // 2:
                end = newline_pos
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(pages):

    chunks = []
    for page in pages:
        for piece in chunk_text(page["text"]):
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extracted pages...")
    pages = load_pages()
    print(f"  {len(pages)} pages loaded")

    print("Chunking...")
    chunks = build_chunks(pages)
    print(f"  {len(chunks)} chunks created")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks (this may take a minute)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True,normalize_embeddings=True,).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Indexed {len(chunks)} chunks (dim={dim}).")
    print(f"  Index saved to  {FAISS_PATH}")
    print(f"  Chunks saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
