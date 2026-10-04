"""Build one FAISS index per subject from the processed chapters.

    python -m mcq.retrieval.build_index      (needs GOOGLE_API_KEY; run extract.py first)

Writes Knowledgebase/Index/<subject>/:
    index.faiss    chunk vectors (IndexFlatIP over unit vectors = exact cosine search)
    chunks.jsonl   one chunk per line; line i <-> vector i in index.faiss
    topics.json    the subject's question topics (derived from section headings)
    topics.npy     the topics' query vectors, precomputed so retrieval needs no API call
    manifest.json  how this index was built; checked when the retriever loads it

Topic embeddings are computed here, offline, on purpose: at quiz time the retriever only
does local FAISS searches, so it is fast, free, deterministic, and works without a key.
"""

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from mcq.retrieval.chunking import MAX_CHARS, chunk_markdown
from mcq.retrieval.config import (
    DOC_FORMAT, EMBEDDING_DIM, EMBEDDING_MODEL, INDEX_DIR, PROCESSED_DIR, QUERY_FORMAT, SOURCES,
)
from mcq.retrieval.embeddings import as_document, as_query

EmbedFn = Callable[[list[str]], np.ndarray]


def manifest_settings() -> dict:
    """The settings an index depends on; a mismatch at load time means: rebuild."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "doc_format": DOC_FORMAT,
        "query_format": QUERY_FORMAT,
        "chunk_max_chars": MAX_CHARS,
    }


def derive_topics(chunks) -> list[dict]:
    """One topic per section that has content, in chapter order. Introductions are skipped:
    they preview the chapter rather than teach something a question can test."""
    topics, seen = [], set()
    for c in chunks:
        number, _, title = c.section.partition(" ")
        if c.section in seen or not title or title.lower() == "introduction":
            continue
        seen.add(c.section)
        topics.append({"name": title, "section": c.section})
    return topics


def build_subject_index(subject: str, markdown: str, embed_fn: EmbedFn, out_dir: Path) -> dict:
    source = SOURCES[subject]
    file_stem = Path(source.pdf).stem
    chunks = chunk_markdown(markdown, subject, file_stem)
    if not chunks:
        raise ValueError(f"{subject}: no chunks produced; is the processed Markdown empty?")
    topics = derive_topics(chunks)

    bodies = [c.text.split("\n\n", 1)[-1] for c in chunks]
    chunk_vecs = embed_fn([as_document(c.section, b) for c, b in zip(chunks, bodies)])
    topic_vecs = embed_fn([as_query(f"{t['name']} ({source.chapter})") for t in topics])
    if chunk_vecs.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError(f"unexpected embedding shape {chunk_vecs.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.ascontiguousarray(chunk_vecs, dtype=np.float32))
    faiss.write_index(index, str(out_dir / "index.faiss"))

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id, "section": c.section, "text": c.text,
                "page_start": c.page_start, "page_end": c.page_end,
            }, ensure_ascii=False) + "\n")
    (out_dir / "topics.json").write_text(json.dumps(topics, indent=1, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "topics.npy", topic_vecs.astype(np.float32))

    manifest = {
        **manifest_settings(),
        "subject": subject,
        "source_pdf": source.pdf,
        "document": source.document,
        "chapter": source.chapter,
        "n_chunks": len(chunks),
        "n_topics": len(topics),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(subjects: list[str]) -> None:
    from mcq.retrieval.embeddings import embed   # imported here so tests never need the API
    for subject in subjects or list(SOURCES):
        md_path = PROCESSED_DIR / f"{subject}.md"
        if not md_path.exists():
            raise FileNotFoundError(f"{md_path} missing; run: python -m mcq.retrieval.extract {subject}")
        m = build_subject_index(subject, md_path.read_text(encoding="utf-8"), embed, INDEX_DIR / subject)
        print(f"{subject}: {m['n_chunks']} chunks, {m['n_topics']} topics -> {INDEX_DIR / subject}")


if __name__ == "__main__":
    main(sys.argv[1:])
