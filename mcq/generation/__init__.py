"""MCQ generation module (Member 2).

Consumes :class:`~mcq.schemas.QuizContext` from retrieval (Member 1) and
produces a ``list[MCQ]`` for the UI (Member 3).

Public API
----------
.. autofunction:: generate_mcqs
.. autofunction:: generate_mcqs_sync

Example
-------
::

    import asyncio
    from pathlib import Path
    from mcq.schemas import QuizContext
    from mcq.generation import generate_mcqs

    ctx = QuizContext.model_validate_json(
        Path("examples/sample_quiz_context.json").read_text(encoding="utf-8")
    )
    mcqs = asyncio.run(generate_mcqs(ctx))
    for m in mcqs:
        print(m.question, "→", m.options[m.correct_index])
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

from mcq.config import get_api_key
from mcq.schemas import MCQ

from .client import MCQGenerator

if TYPE_CHECKING:
    from mcq.schemas import QuizContext

__all__ = ["generate_mcqs", "generate_mcqs_sync", "MCQGenerator"]

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_TOTAL_MCQS = 10
MIN_PER_ITEM = 2   # ensure question-type variety even with many items
MAX_PER_ITEM = 5   # keep prompts manageable


# ── Async entry point ────────────────────────────────────────────────────────


async def generate_mcqs(
    context: QuizContext,
    *,
    total_mcqs: int = DEFAULT_TOTAL_MCQS,
    api_key: str | None = None,
    model: str | None = None,
) -> list[MCQ]:
    """Generate MCQs for every item in the *context*, concurrently.

    Each :class:`~mcq.schemas.RetrievedContext` item is sent to Gemini in a
    separate async call, requesting varied question types.  Results are
    validated with Pydantic and citation-checked before being returned.

    Parameters
    ----------
    context : QuizContext
        Retrieval output with one or more ``RetrievedContext`` items.
    total_mcqs : int
        Target number of MCQs (distributed evenly across items).
    api_key : str, optional
        Gemini API key.  Defaults to :func:`mcq.config.get_api_key`.
    model : str, optional
        Override the Gemini model (default ``gemini-3.8-flash``).

    Returns
    -------
    list[MCQ]
        Validated, citation-checked MCQs.  May be fewer than *total_mcqs*
        if items had insufficient context or validation/citation checks failed.
    """
    resolved_key = api_key or get_api_key()
    kwargs: dict = {"api_key": resolved_key}
    if model:
        kwargs["model"] = model
    generator = MCQGenerator(**kwargs)

    # ── Filter eligible items ─────────────────────────────────────────────
    eligible = [item for item in context.items if item.sufficient and item.chunks]
    skipped = len(context.items) - len(eligible)
    if skipped:
        logger.warning("Skipping %d item(s): insufficient context or no chunks.", skipped)
    if not eligible:
        logger.error("No items with sufficient context. Cannot generate MCQs.")
        return []

    # ── Distribute target MCQ count ───────────────────────────────────────
    per_item = max(MIN_PER_ITEM, min(MAX_PER_ITEM, math.ceil(total_mcqs / len(eligible))))
    logger.info(
        "Generating %d MCQ(s) × %d item(s)  (target total: %d).",
        per_item, len(eligible), total_mcqs,
    )

    # ── Fire all items concurrently ───────────────────────────────────────
    tasks = [
        generator.generate_for_item(item, context.difficulty, num_mcqs=per_item)
        for item in eligible
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Collect results ───────────────────────────────────────────────────
    all_mcqs: list[MCQ] = []
    for item, result in zip(eligible, results):
        if isinstance(result, BaseException):
            logger.error("Generation failed for topic %r: %s", item.topic, result)
            continue
        all_mcqs.extend(result)

    logger.info(
        "Generated %d MCQ(s) total across %d topic(s).", len(all_mcqs), len(eligible),
    )
    return all_mcqs


# ── Sync convenience wrapper ─────────────────────────────────────────────────


def generate_mcqs_sync(context: QuizContext, **kwargs) -> list[MCQ]:
    """Synchronous wrapper around :func:`generate_mcqs`.

    Calls ``asyncio.run()`` internally — do **not** call this from inside an
    already-running event loop.  Use ``await generate_mcqs(...)`` in async
    code instead.
    """
    return asyncio.run(generate_mcqs(context, **kwargs))
