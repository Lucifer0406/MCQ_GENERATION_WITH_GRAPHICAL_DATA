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
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[MCQ]:
    """Generate MCQs for every item in the *context*, concurrently.

    Each :class:`~mcq.schemas.RetrievedContext` item is sent to the LLM in a
    separate async call, requesting varied question types. Results are
    validated with Pydantic and citation-checked before being returned.

    Parameters
    ----------
    context : QuizContext
        Retrieval output with one or more ``RetrievedContext`` items.
    total_mcqs : int
        Target number of MCQs (distributed evenly across items).
    provider : str, optional
        'groq' (default when GROQ_API_KEY is set) or 'gemini'.
    api_key : str, optional
        API key override for the chosen provider.
    model : str, optional
        Model override (e.g. 'openai/gpt-oss-120b' for Groq).

    Returns
    -------
    list[MCQ]
        Validated, citation-checked MCQs.
    """
    kwargs: dict = {}
    if provider:
        kwargs["provider"] = provider
    if api_key:
        kwargs["api_key"] = api_key
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

    # ── Select diverse topics and batch 5-by-5 ────────────────────────────
    # Split requested questions across distinct topics in batches of up to 5
    # (per CONTRACTS.md: batch 3-5 to maximise throughput and respect rate limits).
    batch_size = 5
    n_batches = max(1, math.ceil(total_mcqs / batch_size))
    base_per_batch = total_mcqs // n_batches
    remainder = total_mcqs % n_batches
    batch_mcq_counts = [base_per_batch + (1 if i < remainder else 0) for i in range(n_batches)]

    items_per_batch = max(1, len(eligible) // n_batches)
    batches: list[tuple[list[RetrievedContext], int]] = []
    for i in range(n_batches):
        count = batch_mcq_counts[i]
        start_idx = (i * items_per_batch) % len(eligible)
        end_idx = min(len(eligible), start_idx + items_per_batch) if len(eligible) >= n_batches else len(eligible)
        batch_items = eligible[start_idx:end_idx]
        if not batch_items:
            batch_items = eligible[:1]
        batches.append((batch_items, count))

    logger.info(
        "Generating %d MCQ(s) across %d distinct topic(s) in %d batch(es) of 5 in parallel via %s.",
        total_mcqs, len(eligible), len(batches), generator.provider.upper(),
    )

    all_mcqs: list[MCQ] = []

    # Parallel concurrent execution across 5-5 batches with Groq
    tasks = [
        generator.generate_for_items(batch_items, context.difficulty, total_mcqs=count)
        for batch_items, count in batches
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error("Generation failed for batch %d: %s", i + 1, result)
            continue
        all_mcqs.extend(result)

    # Trim to exact requested total if we generated extra
    if len(all_mcqs) > total_mcqs:
        all_mcqs = all_mcqs[:total_mcqs]

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
