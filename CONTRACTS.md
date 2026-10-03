# Team Contracts — how to use `mcq/schemas.py`

`mcq/schemas.py` defines every piece of data passed between our three parts.
The comments in that file explain each field; this page explains how to work with it.

```
Member 1: Retrieval ──QuizContext──► Member 2: Generation ──list[MCQ]──► Member 3: UI + visuals
```

## What each person uses

| Who | Receives | Produces | Start today with |
|---|---|---|---|
| Member 1 — Retrieval | subject, difficulty | `QuizContext` (one `RetrievedContext` per question) | — |
| Member 2 — Generation | `QuizContext` | `list[MCQ]` | `examples/sample_quiz_context.json` |
| Member 3 — UI + visuals | `list[MCQ]` | the Streamlit page | `examples/sample_mcqs.json` |

You do not need to wait for each other: build against the sample files now and swap in the real function later.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                           # then paste YOUR OWN key from https://aistudio.google.com/apikey
.venv/bin/python -m pytest -q                  # should be all green
```

## Loading the sample data

```python
from pathlib import Path
from pydantic import TypeAdapter
from mcq.schemas import MCQ, QuizContext

# Member 2: fake retrieval output
ctx = QuizContext.model_validate_json(Path("examples/sample_quiz_context.json").read_text(encoding="utf-8"))
for item in ctx.items:
    print(item.topic, [c.text for c in item.chunks])

# Member 3: fake generation output (covers every visual type + one with no visual)
mcqs = TypeAdapter(list[MCQ]).validate_json(Path("examples/sample_mcqs.json").read_text(encoding="utf-8"))
for m in mcqs:
    print(m.question, m.options[m.correct_index], m.visual.type if m.visual else None)
```

## Member 2: asking Gemini for an MCQ in our format (tested 2026-10-03)

```python
from google import genai
from google.genai import types
from mcq.config import get_api_key
from mcq.schemas import MCQ, check_citations

client = genai.Client(api_key=get_api_key())
response = client.models.generate_content(
    model="gemini-3.8-flash",          # gemini-2.5-flash is listed but returns 404 for new keys
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=MCQ.model_json_schema(),   # NOT response_schema=MCQ (rejects our visual union)
    ),
)
mcq = MCQ.model_validate_json(response.text)   # raises ValidationError if Gemini broke the format
check_citations(mcq, item)                     # the MCQ may only cite chunks it was given
```

Tips: ~7 s per call, so generate several MCQs per call (e.g. a list of 5) instead of 10 sequential calls.
On `ValidationError`, retry once; if a *visual* is the problem, keep the question and set `visual=None`.

## Member 3: visual types (MVP)

Dispatch on `mcq.visual.type`:

| `type` | Render with | Typical chapter |
|---|---|---|
| `formula` | `st.latex(visual.latex)` | all |
| `function_plot` | Plotly; curves are `explicit` (y = f(x)) or `parametric` (x(t), y(t)) | Conic Sections |
| `data_graph` | Plotly line / bar / scatter | Motion in a Straight Line |
| `chemical_structure` | RDKit → SVG from `visual.smiles` | Chemical Bonding |
| `None` | no visual | often Biology |

Expressions like `"3*cos(t)"` are data: parse them with a safe parser (e.g. `numexpr` or `sympy`), never `eval()`.
If a visual fails to render, show the question without it; never crash the page.

## Rules

1. **Do not change `schemas.py` alone.** Tell the group first; update the sample files and run `pytest`.
2. Never commit `.env` or anything in `Knowledgebase/Raw|Processed|Index` (NCERT is copyrighted; the repo is public).
3. Subject values are lowercase: `physics`, `chemistry`, `mathematics`, `biology`. Difficulty: `easy`, `medium`, `hard`.
4. `correct_index` is 0-based (0 = first option).

## Change log

- 2026-10-03 — Initial contracts. Removed `physics_diagram` / `biology_diagram` visual types for the MVP (not needed for our chapters; SVG templates too costly for the deadline).
