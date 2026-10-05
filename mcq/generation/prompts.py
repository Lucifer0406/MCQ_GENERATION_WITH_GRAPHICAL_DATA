"""Prompt templates for MCQ generation via Gemini.

The system prompt establishes the LLM's role and output rules.
``build_prompt`` constructs the user message for a single
:class:`~mcq.schemas.RetrievedContext` item, requesting a batch of varied
MCQs grounded in the provided chunks.
"""

from __future__ import annotations

from mcq.schemas import Difficulty, RetrievedContext

# ── Question styles — cycled so every MCQ in a batch is different ─────────────

QUESTION_STYLES: list[dict[str, str]] = [
    {
        "name": "Conceptual",
        "instruction": (
            "Test understanding of a definition, principle, or theorem. "
            "The student should NOT need to calculate anything."
        ),
    },
    {
        "name": "Numerical",
        "instruction": (
            "Require a mathematical calculation (substitution into a formula, "
            "unit conversion, etc.). Show concrete numbers in the stem."
        ),
    },
    {
        "name": "Application",
        "instruction": (
            "Present a real-world or novel scenario and ask the student to "
            "apply the concept to solve it."
        ),
    },
    {
        "name": "Analytical",
        "instruction": (
            "Ask the student to interpret, compare, or reason about data, "
            "graphs, or relationships described in the context."
        ),
    },
    {
        "name": "Diagram / Visual",
        "instruction": (
            "Create a question best answered with the help of a visual "
            "(graph, plot, formula, or chemical structure). You MUST include "
            "a non-null visual specification for this question."
        ),
    },
]


# ── System prompt (constant, sent once per API call) ─────────────────────────

SYSTEM_PROMPT = """\
You are an expert NCERT exam question-paper setter for Indian students \
(Class XI–XII, boards, JEE, NEET). You create high-quality multiple-choice \
questions (MCQs) that test genuine understanding, not rote memorisation.

━━━ OUTPUT FORMAT ━━━
Return a JSON object with a single key "mcqs" whose value is an array of MCQ \
objects. Each MCQ object has these fields (all required unless noted):

  subject          : string — the subject (lowercase: physics / chemistry / mathematics / biology)
  difficulty       : string — "easy" | "medium" | "hard"
  topic            : string — copy the topic name EXACTLY from the input
  question         : string — question stem in Markdown; inline math as $...$
  options          : array of exactly 4 DISTINCT strings (case-insensitive, trimmed)
  correct_index    : integer 0–3 — index of the correct option
  explanation      : string — full worked solution / reasoning in Markdown
  source_chunk_ids : array of strings — chunk_id values this MCQ is grounded in \
(MUST come from the provided chunks ONLY; NEVER invent IDs)
  visual           : object | null — a visual specification (see below), or null

━━━ VISUAL TYPES (set visual to null when none is needed) ━━━

1. Formula — highlight a key equation:
   {"type": "formula", "latex": "<LaTeX string>", "caption": "<optional string or omit>"}

2. Data graph — line / bar / scatter chart with numeric data:
   {"type": "data_graph", "chart": "line"|"bar"|"scatter",
    "x_label": "...", "y_label": "...",
    "series": [{"name": "...", "x": [<numbers or strings>], "y": [<numbers>]}],
    "caption": "<optional>"}
   IMPORTANT: x and y arrays in each series MUST have the same length.

3. Function plot — mathematical curves (explicit or parametric):
   {"type": "function_plot",
    "curves": [
      {"kind": "explicit", "expr": "x**2 - 4*x", "domain": [lo, hi], "label": "..."}
      OR
      {"kind": "parametric", "x_expr": "3*cos(t)", "y_expr": "2*sin(t)", \
"domain": [0, 6.2832], "label": "..."}
    ],
    "x_label": "x", "y_label": "y", "caption": "<optional>"}
   IMPORTANT: domain must be [low, high] with low < high. \
Use [0, 6.2832] for full closed curves (≈ 2π).

4. Chemical structure — SMILES notation for molecules:
   {"type": "chemical_structure", "smiles": "<valid SMILES>", \
"name": "<optional>", "caption": "<optional>"}

Subject-specific guidance:
• Physics     → prefer data_graph for kinematics/motion, formula for laws and derivations (e.g. v = u + at)
• Chemistry   → use chemical_structure for molecules/structures (e.g. C=C, c1ccccc1). For VSEPR / molecular geometry questions (e.g. PCl5, SF6, CH4, NH3, H2O, XeF4, BF3, SF4, ClF3, BeCl2), use the formula in smiles (e.g. "PCl5", "SF6") and name; formula for equilibria, kinetics, or bond order
• Mathematics → use function_plot for curves, conic sections, and areas under curves; formula for calculus & integration questions (e.g. \\int_a^b f(x)\\,dx), derivatives, and identities
• Biology     → visual is usually null; use formula only if a genetic ratio or numerical equation is central

━━━ DIFFICULTY CALIBRATION ━━━
• easy   → direct recall or single-step application
• medium → multi-step reasoning or moderate calculation
• hard   → complex problem-solving, multiple concepts, or tricky edge cases

━━━ HARD RULES ━━━
1. Ground EVERY question in the provided context chunks. Do NOT use outside knowledge.
2. source_chunk_ids must contain ONLY chunk_id values listed in the input. NEVER invent IDs.
3. All 4 options must be distinct (case-insensitive, ignoring whitespace).
4. Distractors should be plausible (common mistakes, nearby values) but clearly wrong.
5. The correct answer must be unambiguously right based on the context.
6. Use LaTeX for all math: inline $...$ in question, options, and explanation.
7. For parametric curves, domain [0, 6.2832] gives a full closed curve (≈ 2π).
8. In data_graph series, x and y arrays MUST have the same length.
9. Each question in the batch must test a DIFFERENT cognitive skill (see type hints below).
10. Do NOT add any extra keys beyond those listed above.
"""


def build_prompt(
    item: RetrievedContext,
    difficulty: Difficulty,
    num_mcqs: int,
) -> str:
    """Build the user-message prompt for one :class:`RetrievedContext` item.

    Cycles through :data:`QUESTION_STYLES` so each MCQ in the batch targets a
    different cognitive skill, ensuring the student sees varied question types
    even within the same topic.
    """
    # ── Format the context chunks ──────────────────────────────────────────
    chunks_block = ""
    for chunk in item.chunks:
        src = chunk.source
        location = f"{src.document}, {src.chapter}, {src.section}"
        if src.page_start is not None:
            location += f", p. {src.page_start}"
            if src.page_end is not None:
                location += f"–{src.page_end}"
        chunks_block += (
            f"--- chunk_id: {chunk.chunk_id} ---\n"
            f"Source: {location}\n"
            f"{chunk.text}\n\n"
        )

    # ── Assign question types (cycle through styles) ──────────────────────
    type_instructions = ""
    for i in range(num_mcqs):
        style = QUESTION_STYLES[i % len(QUESTION_STYLES)]
        type_instructions += f"  MCQ {i + 1} — {style['name']}: {style['instruction']}\n"

    # ── Build the available chunk IDs list for emphasis ────────────────────
    available_ids = ", ".join(f'"{c.chunk_id}"' for c in item.chunks)

    # ── Assemble ──────────────────────────────────────────────────────────
    return (
        f"SUBJECT: {item.subject}\n"
        f"DIFFICULTY: {difficulty}\n"
        f"TOPIC: {item.topic}\n"
        f"NUMBER OF MCQs TO GENERATE: {num_mcqs}\n\n"
        f"AVAILABLE CHUNK IDs (use ONLY these in source_chunk_ids): [{available_ids}]\n\n"
        f"CONTEXT CHUNKS (ground your questions in these ONLY):\n\n"
        f"{chunks_block}"
        f"QUESTION TYPE ASSIGNMENTS:\n{type_instructions}\n"
        f'Generate exactly {num_mcqs} MCQs as specified. '
        f'Return them inside a JSON object: {{"mcqs": [<MCQ>, ...]}}.'
    )
