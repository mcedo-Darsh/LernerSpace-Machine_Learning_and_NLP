import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
FAISS_PATH = INDEX_DIR / "faiss.index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 5

SYSTEM_PROMPT = """You are IITB Insti-Assist, an academic assistant for IIT Bombay students.

Answer the user's question using ONLY the information in the "Context" section below.
Rules:
- If the context does not contain enough information to answer confidently, say
  "I don't know based on the documents I have access to." Do not guess or use
  outside knowledge.
- Keep answers concise and directly address the question.
- When you state a specific fact (a date, a grade, a course code, a credit
  value, a room/slot, etc.), make sure it is drawn from the context.
- Do not mention "the context" or "the documents" explicitly in your answer;
  just answer naturally, as an assistant who knows this information.
"""


class RagPipeline:
    def __init__(self):
        if not FAISS_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python src/ingest.py` then "
                "`python src/build_index.py` first."
            )
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = genai.Client(http_options={'api_version': 'v1beta'})

    def retrieve(self, query: str, k: int = TOP_K):
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results

    def build_prompt(self, query: str, retrieved_chunks):
        context_blocks = []
        for i, c in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Source {i}: {c['source']}, page {c['page']}]\n{c['text']}"
            )
        context = "\n\n".join(context_blocks)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def answer(self, query: str, k: int = TOP_K):
        retrieved = self.retrieve(query, k=k)
        prompt = self.build_prompt(query, retrieved)

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )
        answer_text = response.text

        # de-duplicate sources for display (a doc/page may appear via multiple chunks)
        seen = set()
        sources = []
        for c in retrieved:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": c["source"], "page": c["page"], "score": c["score"]})

        return {"answer": answer_text, "sources": sources, "retrieved_chunks": retrieved}


if __name__ == "__main__":
    # quick manual test from the command line:
    #   python src/rag.py "What is the minimum passing grade at IIT Bombay?"
    import sys

    query = " ".join(sys.argv[1:]) or "What is the minimum passing grade?"
    pipeline = RagPipeline()
    result = pipeline.answer(query)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['source']} (page {s['page']}, score {s['score']:.3f})")
