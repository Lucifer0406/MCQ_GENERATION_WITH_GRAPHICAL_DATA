"""Tests for the MCQ generation module.

Uses mocked Gemini responses so tests run without an API key and in < 1 second.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from mcq.schemas import MCQ, QuizContext, RetrievedContext
from mcq.generation import generate_mcqs, MCQGenerator
from mcq.generation.client import (
    _MCQBatch,
    _BATCH_SCHEMA,
    _check_all_citations,
    _try_fix_visuals,
    _validate_batch,
)
from mcq.generation.prompts import QUESTION_STYLES, SYSTEM_PROMPT, build_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_chunk(chunk_id="physics/keph102/2.4/000", score=0.72):
    return {
        "chunk_id": chunk_id,
        "text": "Acceleration is the rate of change of velocity, $a = dv/dt$.",
        "source": {
            "document": "NCERT Physics Class XI Part 1",
            "file": "keph102.pdf",
            "chapter": "Motion in a Straight Line",
            "section": "2.4 Acceleration",
            "page_start": 5,
        },
        "score": score,
    }


def _make_context(**overrides):
    data = {
        "subject": "physics",
        "topic": "Acceleration",
        "chunks": [_make_chunk()],
        "sufficient": True,
        "index_version": "test",
    }
    data.update(overrides)
    return data


def _make_mcq_dict(**overrides):
    data = {
        "subject": "physics",
        "difficulty": "medium",
        "topic": "Acceleration",
        "question": "A car's velocity changes from 10 m/s to 20 m/s in 5 s. Its acceleration is",
        "options": ["1 m/s²", "2 m/s²", "5 m/s²", "10 m/s²"],
        "correct_index": 1,
        "explanation": "$a = \\Delta v / \\Delta t = 10/5 = 2$ m/s².",
        "source_chunk_ids": ["physics/keph102/2.4/000"],
    }
    data.update(overrides)
    return data


def _make_batch_json(*mcq_dicts) -> str:
    """Create a JSON string mimicking Gemini's structured output."""
    return json.dumps({"mcqs": list(mcq_dicts)})


def _make_quiz_context(**overrides):
    data = {
        "subject": "physics",
        "difficulty": "medium",
        "items": [_make_context()],
    }
    data.update(overrides)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrompts:
    def test_system_prompt_is_nonempty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_visual_types(self):
        for vtype in ("formula", "data_graph", "function_plot", "chemical_structure"):
            assert vtype in SYSTEM_PROMPT

    def test_build_prompt_includes_chunk_ids(self):
        item = RetrievedContext.model_validate(_make_context())
        prompt = build_prompt(item, "medium", 3)
        for chunk in item.chunks:
            assert chunk.chunk_id in prompt

    def test_build_prompt_includes_chunk_text(self):
        item = RetrievedContext.model_validate(_make_context())
        prompt = build_prompt(item, "medium", 2)
        for chunk in item.chunks:
            assert chunk.text in prompt

    def test_build_prompt_includes_topic_subject_difficulty(self):
        item = RetrievedContext.model_validate(_make_context())
        prompt = build_prompt(item, "hard", 3)
        assert "Acceleration" in prompt
        assert "physics" in prompt
        assert "hard" in prompt

    def test_build_prompt_cycles_question_types(self):
        item = RetrievedContext.model_validate(_make_context())
        prompt = build_prompt(item, "medium", 7)
        # First 5 should each get a unique style; 6th and 7th cycle
        for style in QUESTION_STYLES:
            assert style["name"] in prompt

    def test_build_prompt_includes_page_range(self):
        chunk = _make_chunk()
        chunk["source"]["page_end"] = 7
        item = RetrievedContext.model_validate(_make_context(chunks=[chunk]))
        prompt = build_prompt(item, "easy", 1)
        assert "5" in prompt  # page_start
        assert "7" in prompt  # page_end


# ═══════════════════════════════════════════════════════════════════════════════
# Validation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_validate_batch_wrapper_format(self):
        raw = _make_batch_json(_make_mcq_dict())
        mcqs = _validate_batch(raw)
        assert len(mcqs) == 1
        assert isinstance(mcqs[0], MCQ)

    def test_validate_batch_bare_array_fallback(self):
        """Gemini might ignore the wrapper and return a bare array."""
        raw = json.dumps([_make_mcq_dict()])
        mcqs = _validate_batch(raw)
        assert len(mcqs) == 1

    def test_validate_batch_alternative_key_fallback(self):
        """Handle Gemini using 'questions' instead of 'mcqs'."""
        raw = json.dumps({"questions": [_make_mcq_dict()]})
        mcqs = _validate_batch(raw)
        assert len(mcqs) == 1

    def test_validate_batch_invalid_raises(self):
        raw = json.dumps({"mcqs": [{"bad": "data"}]})
        with pytest.raises(ValidationError):
            _validate_batch(raw)

    def test_batch_schema_is_valid_json_schema(self):
        assert "properties" in _BATCH_SCHEMA
        assert "mcqs" in _BATCH_SCHEMA["properties"]

    def test_mcq_batch_model_validates_correctly(self):
        batch = _MCQBatch.model_validate({"mcqs": [_make_mcq_dict()]})
        assert len(batch.mcqs) == 1
        assert batch.mcqs[0].topic == "Acceleration"


# ═══════════════════════════════════════════════════════════════════════════════
# Visual fix tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualFix:
    def _make_visual_error(self):
        """Create a ValidationError that's purely visual-related."""
        bad_mcq = _make_mcq_dict(visual={"type": "function_plot", "curves": []})
        try:
            _MCQBatch.model_validate({"mcqs": [bad_mcq]})
        except ValidationError as exc:
            return exc
        pytest.fail("Expected ValidationError")

    def test_try_fix_strips_bad_visual(self):
        bad_mcq = _make_mcq_dict(visual={"type": "function_plot", "curves": []})
        raw = _make_batch_json(bad_mcq)
        exc = self._make_visual_error()
        fixed = _try_fix_visuals(raw, exc)
        assert fixed is not None
        assert fixed[0]["visual"] is None

    def test_try_fix_returns_none_for_non_visual_errors(self):
        """Non-visual errors should not be auto-fixed."""
        bad_mcq = _make_mcq_dict(options=["a", "b", "c"])  # wrong count
        raw = _make_batch_json(bad_mcq)
        try:
            _MCQBatch.model_validate({"mcqs": [bad_mcq]})
        except ValidationError as exc:
            result = _try_fix_visuals(raw, exc)
            assert result is None

    def test_try_fix_returns_none_for_none_raw(self):
        exc = self._make_visual_error()
        assert _try_fix_visuals(None, exc) is None

    def test_try_fix_returns_none_for_invalid_json(self):
        exc = self._make_visual_error()
        assert _try_fix_visuals("not json at all", exc) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Citation check tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCitationCheck:
    def test_valid_citations_kept(self):
        item = RetrievedContext.model_validate(_make_context())
        mcqs = [MCQ.model_validate(_make_mcq_dict())]
        result = _check_all_citations(mcqs, item)
        assert len(result) == 1

    def test_invalid_citations_dropped(self):
        item = RetrievedContext.model_validate(_make_context())
        mcqs = [MCQ.model_validate(_make_mcq_dict(source_chunk_ids=["invented/id"]))]
        result = _check_all_citations(mcqs, item)
        assert len(result) == 0

    def test_mixed_citations_partial_keep(self):
        ctx = _make_context(chunks=[_make_chunk("physics/keph102/2.4/000"),
                                     _make_chunk("physics/keph102/2.4/001", score=0.5)])
        item = RetrievedContext.model_validate(ctx)
        good = MCQ.model_validate(_make_mcq_dict(
            source_chunk_ids=["physics/keph102/2.4/000"],
            question="Good question?",
        ))
        bad = MCQ.model_validate(_make_mcq_dict(
            source_chunk_ids=["invented/id"],
            question="Bad question?",
            options=["a", "b", "c", "d"],
        ))
        result = _check_all_citations([good, bad], item)
        assert len(result) == 1
        assert result[0].question == "Good question?"


# ═══════════════════════════════════════════════════════════════════════════════
# Generator integration tests (mocked Gemini)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCQGenerator:
    @pytest.fixture
    def generator(self):
        """MCQGenerator with a dummy API key (we mock the API calls)."""
        return MCQGenerator(api_key="test-key-not-real")

    @pytest.fixture
    def valid_response(self):
        return _make_batch_json(
            _make_mcq_dict(question="Q1: What is acceleration?"),
            _make_mcq_dict(
                question="Q2: Calculate the acceleration.",
                options=["1 m/s²", "3 m/s²", "5 m/s²", "10 m/s²"],
            ),
        )

    def test_generate_for_item_returns_mcqs(self, generator, valid_response):
        generator._call_gemini = AsyncMock(return_value=valid_response)
        item = RetrievedContext.model_validate(_make_context())
        mcqs = asyncio.run(generator.generate_for_item(item, "medium", num_mcqs=2))
        assert len(mcqs) == 2
        assert all(isinstance(m, MCQ) for m in mcqs)

    def test_generate_for_item_skips_insufficient(self, generator):
        generator._call_gemini = AsyncMock()
        item = RetrievedContext.model_validate(_make_context(sufficient=False))
        mcqs = asyncio.run(generator.generate_for_item(item, "medium"))
        assert mcqs == []
        generator._call_gemini.assert_not_called()

    def test_generate_for_item_skips_empty_chunks(self, generator):
        generator._call_gemini = AsyncMock()
        item = RetrievedContext.model_validate(
            _make_context(chunks=[], sufficient=False)
        )
        mcqs = asyncio.run(generator.generate_for_item(item, "medium"))
        assert mcqs == []

    def test_generate_for_item_retries_on_validation_error(self, generator, valid_response):
        bad_response = json.dumps({"mcqs": [{"bad": "data"}]})
        generator._call_gemini = AsyncMock(side_effect=[bad_response, valid_response])
        item = RetrievedContext.model_validate(_make_context())
        mcqs = asyncio.run(generator.generate_for_item(item, "medium", num_mcqs=2))
        assert len(mcqs) == 2
        assert generator._call_gemini.call_count == 2

    def test_generate_for_item_returns_empty_after_max_retries(self, generator):
        bad = json.dumps({"mcqs": [{"bad": "data"}]})
        generator._call_gemini = AsyncMock(return_value=bad)
        item = RetrievedContext.model_validate(_make_context())
        mcqs = asyncio.run(generator.generate_for_item(item, "medium"))
        assert mcqs == []

    def test_generate_for_item_recovers_bad_visual(self, generator):
        """If Gemini produces a valid MCQ with an invalid visual, strip it."""
        mcq_with_bad_visual = _make_mcq_dict(
            visual={"type": "function_plot", "curves": []}
        )
        bad_response = _make_batch_json(mcq_with_bad_visual)
        generator._call_gemini = AsyncMock(return_value=bad_response)
        item = RetrievedContext.model_validate(_make_context())
        mcqs = asyncio.run(generator.generate_for_item(item, "medium", num_mcqs=1))
        assert len(mcqs) == 1
        assert mcqs[0].visual is None  # stripped


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level generate_mcqs tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateMCQs:
    @pytest.fixture
    def mock_generator(self):
        """Patch MCQGenerator.generate_for_item to return predictable MCQs."""
        async def fake_generate(item, difficulty, num_mcqs=5):
            return [
                MCQ.model_validate(_make_mcq_dict(
                    question=f"Q{i + 1} about {item.topic}",
                    options=[f"A{i}", f"B{i}", f"C{i}", f"D{i}"],
                ))
                for i in range(num_mcqs)
            ]

        with patch.object(MCQGenerator, "generate_for_item", side_effect=fake_generate):
            yield

    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def test_distributes_across_items(self, mock_generator, mock_api_key):
        ctx = QuizContext.model_validate(_make_quiz_context(items=[
            _make_context(topic="Topic A"),
            _make_context(topic="Topic B"),
        ]))
        mcqs = asyncio.run(generate_mcqs(ctx, total_mcqs=10))
        topics = {m.topic for m in mcqs}
        assert "Acceleration" in topics  # both items have topic "Acceleration" by default
        assert len(mcqs) == 10  # 5 per item × 2 items

    def test_skips_insufficient_items(self, mock_generator, mock_api_key):
        ctx = QuizContext.model_validate(_make_quiz_context(items=[
            _make_context(topic="Good", sufficient=True),
            _make_context(topic="Bad", sufficient=False, chunks=[]),
        ]))
        mcqs = asyncio.run(generate_mcqs(ctx, total_mcqs=4))
        # Only one eligible item, so min 2 MCQs
        assert len(mcqs) >= 2

    def test_returns_empty_for_no_eligible_items(self, mock_generator, mock_api_key):
        ctx = QuizContext.model_validate(_make_quiz_context(items=[
            _make_context(sufficient=False, chunks=[]),
        ]))
        mcqs = asyncio.run(generate_mcqs(ctx, total_mcqs=10))
        assert mcqs == []

    def test_respects_custom_total(self, mock_generator, mock_api_key):
        ctx = QuizContext.model_validate(_make_quiz_context())
        # 1 item, total_mcqs=3 → per_item = max(2, min(5, ceil(3/1))) = 3
        mcqs = asyncio.run(generate_mcqs(ctx, total_mcqs=3))
        assert len(mcqs) == 3


class TestGroqGenerator:
    def test_groq_provider_selection(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
        gen = MCQGenerator()
        assert gen.provider == "groq"

    def test_explicit_groq_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
        gen = MCQGenerator(provider="groq")
        assert gen.provider == "groq"

    def test_groq_generate_for_item_returns_mcqs(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
        gen = MCQGenerator(provider="groq", api_key="gsk_test123")
        valid_response = _make_batch_json(
            _make_mcq_dict(question="Q1 from Groq?"),
            _make_mcq_dict(question="Q2 from Groq?"),
        )
        gen._call_groq = AsyncMock(return_value=valid_response)
        item = RetrievedContext.model_validate(_make_context())
        mcqs = asyncio.run(gen.generate_for_item(item, "medium", num_mcqs=2))
        assert len(mcqs) == 2
        assert mcqs[0].question == "Q1 from Groq?"

