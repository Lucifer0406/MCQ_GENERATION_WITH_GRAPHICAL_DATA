"""Subject-aware retrieval for MCQ generation. This is the module Member 2 calls.

    from mcq.retrieval.retriever import retrieve_quiz_context
    quiz = retrieve_quiz_context("physics", "medium")          # -> QuizContext with 10 items

The app has no user question, so retrieval is topic-driven:
    1. pick a topic from the subject's catalog (derived from the chapter's section headings)
    2. use that topic's precomputed query vector  (no API call)
    3. exact cosine search in that subject's own FAISS index  (subjects can never mix)
    4. cap chunks per section, keep k by difficulty

A whole quiz (retrieve_quiz_context) uses "seed-and-expand" instead: each question gets a
different seed chunk as its focus, plus the seed's nearest neighbours as supporting context.
With one chapter there are only a few topics, so topic queries alone would repeat; seeds don't.
"""

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from mcq.retrieval.config import (
    CANDIDATES, CHUNKS_PER_QUESTION, INDEX_DIR, MAX_PER_SECTION, SOURCES, SUFFICIENT_SCORE,
)
from mcq.schemas import ContextChunk, QuizContext, RetrievedContext, SourceRef


@dataclass
class SubjectIndex:
    index: faiss.Index
    chunks: list[dict]
    topics: list[dict]
    topic_vecs: np.ndarray
    manifest: dict


@lru_cache(maxsize=None)   # load each subject once per process (Streamlit reruns reuse it)
def load_subject(subject: str, index_dir: Path = INDEX_DIR) -> SubjectIndex:
    from mcq.retrieval.build_index import manifest_settings

    d = index_dir / subject
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(
            f"No index for '{subject}' in {d}. Unzip the shared Knowledgebase folder, "
            f"or run: python -m mcq.retrieval.build_index {subject}"
        )
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest_settings()
    stale = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
    if stale:
        raise RuntimeError(f"Index for '{subject}' was built with different settings {stale}; rebuild it.")

    index = faiss.read_index(str(d / "index.faiss"))
    with open(d / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    topics = json.loads((d / "topics.json").read_text(encoding="utf-8"))
    topic_vecs = np.load(d / "topics.npy")
    if index.ntotal != len(chunks) or len(topic_vecs) != len(topics):
        raise RuntimeError(f"Index files for '{subject}' are inconsistent; rebuild it.")
    return SubjectIndex(index, chunks, topics, topic_vecs, manifest)


def _query_vector(si: SubjectIndex, topic: str) -> np.ndarray:
    for t, vec in zip(si.topics, si.topic_vecs):
        if t["name"] == topic:
            return vec[None, :]
    from mcq.retrieval.embeddings import as_query, embed   # custom topic: one live API call
    return embed([as_query(f"{topic} ({si.manifest['chapter']})")])


def _to_chunk(si: SubjectIndex, i: int, score: float) -> ContextChunk:
    c = si.chunks[i]
    return ContextChunk(
        chunk_id=c["chunk_id"],
        text=c["text"],
        source=SourceRef(
            document=si.manifest["document"],
            file=Path(si.manifest["source_pdf"]).name,
            chapter=si.manifest["chapter"],
            section=c["section"],
            page_start=c["page_start"],
            page_end=c["page_end"],
        ),
        score=float(np.clip(score, -1.0, 1.0)),   # float32 rounding can give 1.0000001
    )


def _search(
    si: SubjectIndex,
    query: np.ndarray,
    k: int,
    exclude: frozenset[str] | set[str],
    already_per_section: dict[str, int] | None = None,
) -> list[ContextChunk]:
    """Top-k chunks for a query vector, skipping `exclude`, at most MAX_PER_SECTION per section.

    already_per_section: chunks the caller has already picked, per section, so they count
    toward the cap too (the quiz path passes the focus chunk's section here).
    """
    scores, ids = si.index.search(np.ascontiguousarray(query, dtype=np.float32), min(CANDIDATES, si.index.ntotal))
    picked, per_section = [], dict(already_per_section or {})
    for score, i in zip(scores[0], ids[0]):
        if i < 0:   # FAISS pads with -1 when it has fewer results than asked for
            continue
        c = si.chunks[i]
        if c["chunk_id"] in exclude or per_section.get(c["section"], 0) >= MAX_PER_SECTION:
            continue
        per_section[c["section"]] = per_section.get(c["section"], 0) + 1
        picked.append(_to_chunk(si, int(i), score))
        if len(picked) == k:
            break
    return picked


def retrieve_context(
    subject: str,
    topic: str,
    difficulty: str = "medium",
    exclude_chunk_ids: frozenset[str] | set[str] = frozenset(),
    index_dir: Path = INDEX_DIR,
) -> RetrievedContext:
    """Grounding chunks for ONE question about a named topic (catalog topic, or any custom text)."""
    si = load_subject(subject, index_dir)
    picked = _search(si, _query_vector(si, topic), CHUNKS_PER_QUESTION[difficulty], exclude_chunk_ids)
    return RetrievedContext(
        subject=subject,
        topic=topic,
        chunks=picked,
        sufficient=bool(picked) and picked[0].score >= SUFFICIENT_SCORE,
        index_version=si.manifest["built_at"],
    )


def list_topics(subject: str, index_dir: Path = INDEX_DIR) -> list[str]:
    return [t["name"] for t in load_subject(subject, index_dir).topics]


def _seed_context(si: SubjectIndex, subject: str, seed_idx: int, difficulty: str) -> RetrievedContext:
    """Context for one question built around a seed chunk: the seed, then its nearest neighbours."""
    query = si.index.reconstruct(seed_idx)[None, :]   # the seed's own stored vector: no API call
    seed = si.chunks[seed_idx]
    k = CHUNKS_PER_QUESTION[difficulty]
    neighbours = _search(si, query, k - 1, exclude={seed["chunk_id"]},
                         already_per_section={seed["section"]: 1})   # the focus counts toward the cap
    chunks = [_to_chunk(si, seed_idx, 1.0)] + neighbours
    return RetrievedContext(
        subject=subject,
        topic=si.chunks[seed_idx]["section"].partition(" ")[2],
        chunks=chunks,
        sufficient=True,   # the seed is real chapter text about this topic by construction
        index_version=si.manifest["built_at"],
    )


def _pick_seeds(si: SubjectIndex, n: int, avoid: frozenset[str] | set[str], rng: random.Random) -> list[int]:
    """n seed chunks spread across sections: round-robin over sections, fresh chunks before avoided ones."""
    topic_sections = {t["section"] for t in si.topics}
    by_section: dict[str, list[int]] = {}
    for i, c in enumerate(si.chunks):
        if c["section"] in topic_sections:
            by_section.setdefault(c["section"], []).append(i)
    sections = sorted(by_section)
    rng.shuffle(sections)
    for ids in by_section.values():
        rng.shuffle(ids)

    def round_robin(pools: list[list[int]]) -> list[int]:
        order, pools = [], [list(p) for p in pools]
        while any(pools):
            for p in pools:
                if p:
                    order.append(p.pop())
        return order

    fresh = [[i for i in by_section[s] if si.chunks[i]["chunk_id"] not in avoid] for s in sections]
    stale = [[i for i in by_section[s] if si.chunks[i]["chunk_id"] in avoid] for s in sections]
    order = round_robin(fresh) + round_robin(stale)
    if not order:
        raise RuntimeError("index has no chunks under any topic section")
    return [order[q % len(order)] for q in range(n)]   # more questions than chunks: cycle


def retrieve_quiz_context(
    subject: str,
    difficulty: str,
    n_questions: int = 10,
    avoid_chunk_ids: set[str] | frozenset[str] = frozenset(),
    seed: int | None = None,
    index_dir: Path = INDEX_DIR,
) -> QuizContext:
    """Contexts for a whole quiz: one item per question, each with a DIFFERENT focus chunk.

    items[i].chunks[0] is the question's focus (the "seed"); the rest are its nearest
    neighbours, as supporting context. Seeds are unique within a quiz while possible and
    are spread across the chapter's sections; supporting chunks may repeat.

    avoid_chunk_ids: focus chunks to use last, e.g. the previous click's
        {item.chunks[0].chunk_id for item in previous_quiz.items}, so repeated clicks vary.
    seed: makes the selection reproducible (tests, debugging).
    """
    if subject not in SOURCES:
        raise ValueError(f"unknown subject {subject!r}; expected one of {list(SOURCES)}")
    si = load_subject(subject, index_dir)
    seeds = _pick_seeds(si, n_questions, avoid_chunk_ids, random.Random(seed))
    return QuizContext(
        subject=subject,
        difficulty=difficulty,
        items=[_seed_context(si, subject, i, difficulty) for i in seeds],
    )


def format_context_for_prompt(ctx: RetrievedContext) -> str:
    """Optional helper for Member 2: the chunks as prompt text, each tagged with its id."""
    parts = [f"Topic: {ctx.topic}"]
    for c in ctx.chunks:
        pages = f"p. {c.source.page_start}" if c.source.page_start == c.source.page_end else \
            f"pp. {c.source.page_start}-{c.source.page_end}"
        parts.append(f"[chunk_id: {c.chunk_id}] ({c.source.document}, §{c.source.section}, {pages})\n{c.text}")
    return "\n\n---\n\n".join(parts)
