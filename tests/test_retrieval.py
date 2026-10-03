"""Index + retrieval tests. Offline: a fake bag-of-words embedding replaces Gemini,
so these run without an API key and without NCERT text."""

import json
import re
import zlib

import numpy as np
import pytest

from mcq.retrieval import retriever
from mcq.retrieval.build_index import build_subject_index
from mcq.retrieval.config import CHUNKS_PER_QUESTION, EMBEDDING_DIM, MAX_PER_SECTION
from mcq.schemas import QuizContext

STOP = {"title", "text", "task", "search", "result", "query", "the", "of", "a", "is", "and", "in", "to"}


def fake_embed(texts: list[str]) -> np.ndarray:
    """Hash each word into one of EMBEDDING_DIM bins: texts sharing words get similar vectors."""
    out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for row, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            if w not in STOP:
                out[row, zlib.crc32(w.encode()) % EMBEDDING_DIM] += 1
        out[row, 0] += 0.01   # avoid all-zero rows
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def section(number, title, word, n_paragraphs):
    para = (f"{word} " * 60).strip()   # ~one chunk-sized paragraph about `word`
    return f"## {number} {title}\n\n" + "\n\n".join(f"{para} part{i}." for i in range(n_paragraphs))


PHYSICS_MD = "\n\n".join([
    "<!-- page 1 -->\n# MOTION",
    section("2.1", "Introduction", "overview", 1),
    section("2.2", "Velocity", "velocity", 4),
    "<!-- page 2 -->",
    section("2.3", "Acceleration", "acceleration", 6),
    section("2.4", "Relative motion", "relative", 2),
    "## EXERCISES\n\n2.1 velocity acceleration question that must not be indexed.",
])
BIOLOGY_MD = section("2.1", "Kingdom Monera", "bacteria", 3) + "\n\n" + section("2.2", "Kingdom Fungi", "fungi", 3)


@pytest.fixture
def index_dir(tmp_path):
    build_subject_index("physics", PHYSICS_MD, fake_embed, tmp_path / "physics")
    build_subject_index("biology", BIOLOGY_MD, fake_embed, tmp_path / "biology")
    retriever.load_subject.cache_clear()
    yield tmp_path
    retriever.load_subject.cache_clear()


# ── Index build ──


def test_build_writes_consistent_files(index_dir):
    d = index_dir / "physics"
    manifest = json.loads((d / "manifest.json").read_text())
    chunks = [json.loads(line) for line in (d / "chunks.jsonl").read_text().splitlines()]
    assert manifest["n_chunks"] == len(chunks) > 0
    assert manifest["embedding_dim"] == EMBEDDING_DIM
    assert not any("must not be indexed" in c["text"] for c in chunks)


def test_topics_come_from_sections_without_introduction(index_dir):
    assert retriever.list_topics("physics", index_dir) == ["Velocity", "Acceleration", "Relative motion"]


def test_index_reload_gives_identical_results(index_dir):
    a = retriever.retrieve_context("physics", "Velocity", index_dir=index_dir)
    retriever.load_subject.cache_clear()
    b = retriever.retrieve_context("physics", "Velocity", index_dir=index_dir)
    assert a == b


# ── Single-question retrieval ──


def test_topic_retrieves_its_own_section_first(index_dir):
    ctx = retriever.retrieve_context("physics", "Acceleration", index_dir=index_dir)
    assert ctx.chunks[0].source.section == "2.3 Acceleration"
    assert ctx.chunks[0].source.page_start == 2   # provenance survives the round trip


def test_sufficient_flag_follows_threshold(index_dir, monkeypatch):
    # The real threshold is calibrated for Gemini scores; fake-embedding scores top out near 0.5.
    monkeypatch.setattr(retriever, "SUFFICIENT_SCORE", 0.3)
    assert retriever.retrieve_context("physics", "Acceleration", index_dir=index_dir).sufficient
    monkeypatch.setattr(retriever, "SUFFICIENT_SCORE", 0.9)
    assert not retriever.retrieve_context("physics", "Acceleration", index_dir=index_dir).sufficient


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_number_of_chunks_follows_difficulty(index_dir, difficulty):
    ctx = retriever.retrieve_context("physics", "Acceleration", difficulty, index_dir=index_dir)
    assert len(ctx.chunks) == CHUNKS_PER_QUESTION[difficulty]


def test_at_most_max_per_section(index_dir):
    ctx = retriever.retrieve_context("physics", "Acceleration", "hard", index_dir=index_dir)
    counts = {}
    for c in ctx.chunks:
        counts[c.source.section] = counts.get(c.source.section, 0) + 1
    assert max(counts.values()) <= MAX_PER_SECTION


def test_excluded_chunks_are_not_returned(index_dir):
    first = retriever.retrieve_context("physics", "Velocity", index_dir=index_dir)
    used = frozenset(c.chunk_id for c in first.chunks)
    second = retriever.retrieve_context("physics", "Velocity", exclude_chunk_ids=used, index_dir=index_dir)
    assert used.isdisjoint(c.chunk_id for c in second.chunks)


def test_subjects_never_mix(index_dir):
    ctx = retriever.retrieve_context("biology", "Kingdom Fungi", "hard", index_dir=index_dir)
    assert all(c.chunk_id.startswith("biology/") for c in ctx.chunks)


def test_small_index_with_fewer_chunks_than_candidates(index_dir):
    ctx = retriever.retrieve_context("biology", "Kingdom Monera", "hard", index_dir=index_dir)
    assert 0 < len(ctx.chunks) <= 6   # biology fixture has only 6 chunks; FAISS -1 padding is ignored


# ── Whole-quiz retrieval (seed-and-expand) ──


def focus_ids(quiz):
    return [item.chunks[0].chunk_id for item in quiz.items]


def eligible(subject, index_dir):
    """Number of chunks that can be a question focus (chunks under a topic section)."""
    si = retriever.load_subject(subject, index_dir)
    sections = {t["section"] for t in si.topics}
    return sum(c["section"] in sections for c in si.chunks)


def test_quiz_has_one_valid_item_per_question(index_dir):
    quiz = retriever.retrieve_quiz_context("physics", "medium", n_questions=10, seed=1, index_dir=index_dir)
    assert isinstance(quiz, QuizContext) and len(quiz.items) == 10
    assert all(len(item.chunks) == CHUNKS_PER_QUESTION["medium"] for item in quiz.items)


def test_quiz_focus_chunks_are_unique_while_possible(index_dir):
    n = eligible("physics", index_dir)
    quiz = retriever.retrieve_quiz_context("physics", "medium", n_questions=n, seed=2, index_dir=index_dir)
    assert len(set(focus_ids(quiz))) == n


def test_focus_chunk_comes_first_and_matches_topic(index_dir):
    quiz = retriever.retrieve_quiz_context("physics", "easy", n_questions=5, seed=3, index_dir=index_dir)
    for item in quiz.items:
        assert item.chunks[0].score == 1.0
        assert item.chunks[0].source.section.endswith(item.topic)


def test_quiz_spreads_across_sections(index_dir):
    quiz = retriever.retrieve_quiz_context("physics", "easy", n_questions=3, seed=7, index_dir=index_dir)
    assert sorted(i.topic for i in quiz.items) == ["Acceleration", "Relative motion", "Velocity"]


def test_introduction_is_never_a_focus(index_dir):
    quiz = retriever.retrieve_quiz_context("physics", "easy", n_questions=12, seed=4, index_dir=index_dir)
    assert all("/2.1/" not in cid for cid in focus_ids(quiz))


def test_more_questions_than_chunks_cycles_instead_of_failing(index_dir):
    n = eligible("biology", index_dir)
    quiz = retriever.retrieve_quiz_context("biology", "easy", n_questions=n + 5, seed=1, index_dir=index_dir)
    assert len(quiz.items) == n + 5 and len(set(focus_ids(quiz))) == n


def test_avoided_focus_chunks_go_last(index_dir):
    n = eligible("physics", index_dir)
    first = retriever.retrieve_quiz_context("physics", "easy", n_questions=n // 2, seed=5, index_dir=index_dir)
    second = retriever.retrieve_quiz_context(
        "physics", "easy", n_questions=n - n // 2, avoid_chunk_ids=set(focus_ids(first)), seed=6, index_dir=index_dir)
    assert set(focus_ids(first)).isdisjoint(focus_ids(second))


def test_seed_makes_quiz_reproducible(index_dir):
    a = retriever.retrieve_quiz_context("physics", "medium", seed=42, index_dir=index_dir)
    b = retriever.retrieve_quiz_context("physics", "medium", seed=42, index_dir=index_dir)
    assert a == b


def test_unknown_subject_rejected(index_dir):
    with pytest.raises(ValueError, match="unknown subject"):
        retriever.retrieve_quiz_context("history", "easy", index_dir=index_dir)


# ── Failure modes ──


def test_missing_index_has_actionable_error(tmp_path):
    retriever.load_subject.cache_clear()
    with pytest.raises(FileNotFoundError, match="build_index"):
        retriever.load_subject("chemistry", tmp_path)


def test_stale_manifest_is_rejected(index_dir):
    path = index_dir / "physics" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["embedding_model"] = "some-other-model"
    path.write_text(json.dumps(manifest))
    retriever.load_subject.cache_clear()
    with pytest.raises(RuntimeError, match="rebuild"):
        retriever.load_subject("physics", index_dir)


def test_prompt_formatting_includes_ids_and_citation(index_dir):
    ctx = retriever.retrieve_context("physics", "Velocity", index_dir=index_dir)
    text = retriever.format_context_for_prompt(ctx)
    assert all(f"[chunk_id: {c.chunk_id}]" in text for c in ctx.chunks)
    assert "NCERT Physics Class XI Part 1" in text
