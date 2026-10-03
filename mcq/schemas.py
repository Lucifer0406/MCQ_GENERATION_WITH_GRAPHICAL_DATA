"""Shared data contracts between retrieval, generation and rendering.

    Retrieval (Member 1)  ──QuizContext──►  Generation (Member 2)  ──list[MCQ]──►  UI/Rendering (Member 3)

Every hand-off between components is one of these models. Changing a field here
changes a contract for someone else: agree on it as a team and note the change
in TEAM_RAG_CONTEXT.md.

Pydantic in one paragraph: a `BaseModel` subclass declares fields with type
hints. Creating an instance (or calling `Model.model_validate(dict)` /
`Model.model_validate_json(str)`) checks every field and raises a
`ValidationError` naming the exact bad field. Valid data becomes a normal
object with attributes (`mcq.options[0]`); `model_dump()` / `model_dump_json()`
turn it back into a dict / JSON.
"""

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

# `Literal[...]` restricts a field to exactly these values. Anything else
# (e.g. "Physics", "maths") is a validation error, not a silent bug.
Subject = Literal["physics", "chemistry", "mathematics", "biology"]
Difficulty = Literal["easy", "medium", "hard"]

SUBJECTS: tuple[str, ...] = get_args(Subject)
DIFFICULTIES: tuple[str, ...] = get_args(Difficulty)


class Contract(BaseModel):
    """Base for every shared model.

    `extra="forbid"` rejects unknown fields, so a typo like `"optons"` or an
    LLM inventing a field fails loudly instead of being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────── Retrieval → Generation ───────────────────────────


class SourceRef(Contract):
    """Where a chunk came from: enough to cite it as "NCERT Physics XI, Ch 2, §2.4"."""

    document: str               # "NCERT Physics Class XI Part 1"
    file: str                   # "keph102.pdf"
    chapter: str                # "Motion in a Straight Line"
    section: str                # "2.4 Acceleration"
    page_start: int | None = None   # `X | None = None` means optional, default None
    page_end: int | None = None


class ContextChunk(Contract):
    chunk_id: str               # stable, readable id, e.g. "physics/keph102/2.4/001"
    text: str = Field(min_length=1)  # Markdown with LaTeX ($...$), as extracted
    source: SourceRef           # nested model: a dict here is validated as a SourceRef
    score: float = Field(ge=-1.0, le=1.0)  # cosine similarity to the topic; `ge`/`le` = range check


class RetrievedContext(Contract):
    """Grounding material for ONE question."""

    subject: Subject
    topic: str                  # what the question should be about, e.g. "Acceleration"
    chunks: list[ContextChunk]  # most relevant first; may be empty if retrieval found nothing usable
    sufficient: bool            # False → too little relevant text; generator should skip or flag it
    index_version: str          # which index build produced this (for debugging/reproducibility)

    # `@model_validator(mode="after")` runs after all fields are individually
    # valid, for rules that involve several fields at once.
    @model_validator(mode="after")
    def _unique_chunk_ids(self):
        ids = [c.chunk_id for c in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk_id in chunks")
        return self


class QuizContext(Contract):
    """What retrieval hands to generation for one Generate click (one item per question)."""

    subject: Subject
    difficulty: Difficulty
    items: list[RetrievedContext] = Field(min_length=1)

    @model_validator(mode="after")
    def _items_match_subject(self):
        wrong = [i.topic for i in self.items if i.subject != self.subject]
        if wrong:
            raise ValueError(f"items from another subject: {wrong}")
        return self


# ─────────────────────────── Visual specifications ────────────────────────────
# DRAFT, owned by Members 2 (produces) and 3 (renders). Retrieval never touches
# these. Each visual is DATA describing what to draw; renderers must never
# eval()/exec() anything from it (expressions are parsed with a safe parser).


class VisualBase(Contract):
    caption: str | None = None


class FormulaVisual(VisualBase):
    type: Literal["formula"]
    latex: str = Field(min_length=1)            # rendered with KaTeX


class ChemicalStructureVisual(VisualBase):
    type: Literal["chemical_structure"]
    smiles: str = Field(min_length=1)           # rendered with RDKit; renderer validates the SMILES
    name: str | None = None


class Curve(Contract):
    """One curve. Explicit y=f(x), or parametric (x(t), y(t)) for circles/ellipses/hyperbolas."""

    kind: Literal["explicit", "parametric"]
    expr: str | None = None                     # explicit: "x**2 - 4*x"
    x_expr: str | None = None                   # parametric: "3*cos(t)"
    y_expr: str | None = None                   # parametric: "2*sin(t)"
    domain: tuple[float, float]                 # x-range (explicit) or t-range (parametric)
    label: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.domain[0] >= self.domain[1]:
            raise ValueError("domain must be (low, high) with low < high")
        if self.kind == "explicit" and not self.expr:
            raise ValueError("explicit curve needs expr")
        if self.kind == "parametric" and not (self.x_expr and self.y_expr):
            raise ValueError("parametric curve needs x_expr and y_expr")
        return self


class FunctionPlotVisual(VisualBase):
    type: Literal["function_plot"]
    curves: list[Curve] = Field(min_length=1)   # rendered with Plotly
    x_label: str = "x"
    y_label: str = "y"


class Series(Contract):
    name: str
    x: list[float | str]                        # numbers, or category names for bar charts
    y: list[float]

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        return self


class DataGraphVisual(VisualBase):
    type: Literal["data_graph"]
    chart: Literal["line", "bar", "scatter"]    # e.g. position–time graph = "line"
    series: list[Series] = Field(min_length=1)
    x_label: str
    y_label: str


# Custom SVG physics/biology diagrams were dropped for the MVP (2026-10-03): the
# MVP chapters are covered by the four types above, and SVG templates were the
# most labour-intensive item. Questions that need no visual set visual=None.

# A "discriminated union": `visual` may be any one of these classes, and
# Pydantic reads the `type` key to decide which one to validate against,
# so {"type": "formula", "latex": "..."} becomes a FormulaVisual.
# `Annotated[X, Field(...)]` attaches that rule to the type itself.
Visual = Annotated[
    FormulaVisual | ChemicalStructureVisual | FunctionPlotVisual | DataGraphVisual,
    Field(discriminator="type"),
]


# ─────────────────────────── Generation → UI ──────────────────────────────────


class MCQ(Contract):
    subject: Subject
    difficulty: Difficulty
    topic: str
    question: str = Field(min_length=1)         # Markdown; inline math as $...$
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)      # index into options (avoids string-matching answers)
    explanation: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)  # chunks this MCQ is grounded in
    visual: Visual | None = None

    @model_validator(mode="after")
    def _distinct_options(self):
        normalized = [o.strip().lower() for o in self.options]
        if len(set(normalized)) != len(normalized):
            raise ValueError("options must be distinct")
        return self


def check_citations(mcq: MCQ, context: RetrievedContext) -> None:
    """Grounding check: an MCQ may only cite chunks it was actually given.

    Schema validation can't see the context, so the generator calls this after
    validating each MCQ. A failure usually means the LLM invented a chunk id.
    """
    known = {c.chunk_id for c in context.chunks}
    unknown = [cid for cid in mcq.source_chunk_ids if cid not in known]
    if unknown:
        raise ValueError(f"MCQ cites chunk ids not in its context: {unknown}")
