"""Keeps the sample files teammates build against in sync with schemas.py."""

import json

from pydantic import TypeAdapter

from mcq.config import PROJECT_ROOT
from mcq.schemas import MCQ, QuizContext, check_citations

EXAMPLES = PROJECT_ROOT / "examples"


def test_sample_quiz_context_is_valid():
    QuizContext.model_validate_json((EXAMPLES / "sample_quiz_context.json").read_text(encoding="utf-8"))


def test_sample_mcqs_are_valid_and_cover_every_visual_type():
    raw = (EXAMPLES / "sample_mcqs.json").read_text(encoding="utf-8")
    mcqs = TypeAdapter(list[MCQ]).validate_json(raw)  # validates a list of models in one go
    types = {m.visual.type if m.visual else None for m in mcqs}
    assert types == {"data_graph", "formula", "function_plot", "chemical_structure", None}


def test_physics_sample_mcqs_cite_the_sample_context():
    ctx = QuizContext.model_validate_json((EXAMPLES / "sample_quiz_context.json").read_text(encoding="utf-8"))
    by_topic = {item.topic: item for item in ctx.items}
    mcqs = [MCQ.model_validate(m) for m in json.loads((EXAMPLES / "sample_mcqs.json").read_text(encoding="utf-8"))]
    for mcq in mcqs:
        if mcq.topic in by_topic:
            check_citations(mcq, by_topic[mcq.topic])
