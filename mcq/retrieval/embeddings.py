"""Gemini embeddings: documents and queries, returned as unit-length float32 rows."""

import time

import numpy as np
from google import genai
from google.genai import types

from mcq.config import get_api_key
from mcq.retrieval.config import DOC_FORMAT, EMBEDDING_DIM, EMBEDDING_MODEL, QUERY_FORMAT

BATCH = 50

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_api_key())
    return _client


def as_document(title: str, text: str) -> str:
    return DOC_FORMAT.format(title=title, text=text)


def as_query(text: str) -> str:
    return QUERY_FORMAT.format(text=text)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so FAISS inner product == cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed(texts: list[str], retries: int = 6) -> np.ndarray:
    """Embed already-formatted texts (use as_document / as_query first).

    Each text is sent as its own Content: gemini-embedding-2 is multimodal and
    MERGES a plain list of strings into a single embedding (verified 2026-10-03).
    Free tier: 100 texts per minute (each text in a batch counts), so on a 429 we
    wait for the minute window to pass instead of retrying immediately.
    """
    client = _get_client()
    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    rows: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
        for attempt in range(retries):
            try:
                result = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents, config=config)
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Embedding failed after {retries} attempts: {e}") from e
                rate_limited = "RESOURCE_EXHAUSTED" in str(e)
                if rate_limited and "PerDay" in str(e):
                    raise RuntimeError(f"Daily embedding quota used up: {e}") from e
                wait = 61 if rate_limited else 2 * 2**attempt
                print(f"    embedding {'rate limit' if rate_limited else 'error'}; waiting {wait}s")
                time.sleep(wait)
        if len(result.embeddings) != len(batch):   # guard against the merging behaviour above
            raise RuntimeError(f"Sent {len(batch)} texts but got {len(result.embeddings)} embeddings")
        rows.extend(e.values for e in result.embeddings)
    return normalize(np.array(rows))
