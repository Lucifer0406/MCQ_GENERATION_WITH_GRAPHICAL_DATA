"""Contract tests. They also document the schemas by example: what passes, and what fails."""

import pytest
from pydantic import ValidationError

from mcq.schemas import (
    MCQ,
    ChemicalStructureVisual,
    ContextChunk,
    FunctionPlotVisual,
    QuizContext,
    RetrievedContext,
    check_citations,
)


def make_chunk(chunk_id="physics/keph102/2.4/000", score=0.72):
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


def make_context(**overrides):
    data = {
        "subject": "physics",
        "topic": "Acceleration",
        "chunks": [make_chunk()],
        "sufficient": True,
        "index_version": "test",
    }
    data.update(overrides)
    return data


def make_mcq(**overrides):
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


# ── Retrieval contracts ──


def test_context_parses_nested_dicts_into_models():
    ctx = RetrievedContext.model_validate(make_context())
    assert isinstance(ctx.chunks[0], ContextChunk)
    assert ctx.chunks[0].source.section == "2.4 Acceleration"
    assert ctx.chunks[0].source.page_end is None  # optional field defaults


def test_context_json_round_trip():
    ctx = RetrievedContext.model_validate(make_context())
    assert RetrievedContext.model_validate_json(ctx.model_dump_json()) == ctx


@pytest.mark.parametrize("subject", ["Physics", "maths", "history", ""])
def test_rejects_unknown_subject(subject):
    with pytest.raises(ValidationError):
        RetrievedContext.model_validate(make_context(subject=subject))


def test_rejects_out_of_range_score():
    with pytest.raises(ValidationError, match="score"):
        RetrievedContext.model_validate(make_context(chunks=[make_chunk(score=7)]))


def test_rejects_duplicate_chunk_ids():
    with pytest.raises(ValidationError, match="duplicate chunk_id"):
        RetrievedContext.model_validate(make_context(chunks=[make_chunk(), make_chunk()]))


def test_empty_insufficient_context_is_allowed():
    ctx = RetrievedContext.model_validate(make_context(chunks=[], sufficient=False))
    assert not ctx.sufficient


def test_rejects_unknown_field():
    with pytest.raises(ValidationError, match="extra"):
        RetrievedContext.model_validate(make_context(difficulty="hard"))


def test_quiz_context_requires_matching_subjects():
    other = make_context(subject="chemistry", topic="Ionic bond")
    with pytest.raises(ValidationError, match="another subject"):
        QuizContext.model_validate(
            {"subject": "physics", "difficulty": "easy", "items": [make_context(), other]}
        )


def test_quiz_context_requires_at_least_one_item():
    with pytest.raises(ValidationError):
        QuizContext.model_validate({"subject": "physics", "difficulty": "easy", "items": []})


# ── MCQ contract ──


def test_valid_mcq_without_visual():
    mcq = MCQ.model_validate(make_mcq())
    assert mcq.options[mcq.correct_index] == "2 m/s²"
    assert mcq.visual is None


@pytest.mark.parametrize("options", [["a", "b", "c"], ["a", "b", "c", "d", "e"]])
def test_mcq_needs_exactly_four_options(options):
    with pytest.raises(ValidationError):
        MCQ.model_validate(make_mcq(options=options))


def test_mcq_rejects_duplicate_options():
    with pytest.raises(ValidationError, match="distinct"):
        MCQ.model_validate(make_mcq(options=["2 m/s²", " 2 M/S² ", "c", "d"]))


def test_mcq_rejects_correct_index_out_of_range():
    with pytest.raises(ValidationError):
        MCQ.model_validate(make_mcq(correct_index=4))


def test_mcq_must_cite_a_source():
    with pytest.raises(ValidationError):
        MCQ.model_validate(make_mcq(source_chunk_ids=[]))


def test_mcq_json_round_trip_with_visual():
    mcq = MCQ.model_validate(make_mcq(visual={"type": "formula", "latex": "a = \\frac{dv}{dt}"}))
    assert MCQ.model_validate_json(mcq.model_dump_json()) == mcq


# ── Visuals: the discriminator picks the class from "type" ──


def test_visual_dispatches_on_type():
    mcq = MCQ.model_validate(make_mcq(visual={"type": "chemical_structure", "smiles": "O"}))
    assert isinstance(mcq.visual, ChemicalStructureVisual)


def test_parametric_curve_for_conic_sections():
    ellipse = {"kind": "parametric", "x_expr": "3*cos(t)", "y_expr": "2*sin(t)", "domain": [0, 6.2832]}
    mcq = MCQ.model_validate(make_mcq(visual={"type": "function_plot", "curves": [ellipse]}))
    assert isinstance(mcq.visual, FunctionPlotVisual)


def test_rejects_unknown_visual_type():
    with pytest.raises(ValidationError):
        MCQ.model_validate(make_mcq(visual={"type": "image", "url": "http://x"}))


@pytest.mark.parametrize(
    "curve",
    [
        {"kind": "explicit", "domain": [0, 1]},                         # missing expr
        {"kind": "parametric", "x_expr": "cos(t)", "domain": [0, 1]},   # missing y_expr
        {"kind": "explicit", "expr": "x**2", "domain": [5, 1]},         # reversed domain
    ],
)
def test_rejects_malformed_curves(curve):
    with pytest.raises(ValidationError):
        MCQ.model_validate(make_mcq(visual={"type": "function_plot", "curves": [curve]}))


def test_rejects_mismatched_series_lengths():
    graph = {
        "type": "data_graph", "chart": "line", "x_label": "t (s)", "y_label": "x (m)",
        "series": [{"name": "car", "x": [0, 1, 2], "y": [0, 5]}],
    }
    with pytest.raises(ValidationError, match="same length"):
        MCQ.model_validate(make_mcq(visual=graph))


# ── Grounding check across contracts ──


def test_check_citations_accepts_known_ids():
    check_citations(MCQ.model_validate(make_mcq()), RetrievedContext.model_validate(make_context()))


def test_check_citations_rejects_invented_ids():
    mcq = MCQ.model_validate(make_mcq(source_chunk_ids=["physics/keph102/9.9/999"]))
    with pytest.raises(ValueError, match="not in its context"):
        check_citations(mcq, RetrievedContext.model_validate(make_context()))
