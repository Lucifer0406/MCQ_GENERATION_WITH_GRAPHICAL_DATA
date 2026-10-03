"""Gemini client wrapper with structured-output generation, validation, and retry.

This module is internal to ``mcq.generation``.  External callers should use
:func:`mcq.generation.generate_mcqs` instead of touching :class:`MCQGenerator`
directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from mcq.schemas import MCQ, Difficulty, RetrievedContext, check_citations

from .prompts import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)


# ── Internal batch wrapper (not part of team contract) ───────────────────────


class _MCQBatch(BaseModel):
    """Wrapper so Gemini returns ``{"mcqs": [...]}`` instead of a bare array.

    NOT a :class:`~mcq.schemas.Contract` subclass: it doesn't need
    ``extra="forbid"`` and is never serialised across team boundaries.
    """

    mcqs: list[MCQ] = Field(min_length=1)


_BATCH_SCHEMA: dict[str, Any] = _MCQBatch.model_json_schema()


# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "gemini-3.8-flash"
MAX_RETRIES = 2          # up to 2 retries on transient errors or validation issues
RETRY_DELAY_S = 1.5      # base seconds between retries


# ── Generator class ──────────────────────────────────────────────────────────


class MCQGenerator:
    """Async MCQ generator backed by the Gemini API.

    Typical usage is through :func:`mcq.generation.generate_mcqs`;  instantiate
    this class directly only when you need fine-grained control (custom model,
    per-item calls, etc.).
    """

    def __init__(self, *, api_key: str, model: str = MODEL) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)

    # ── Public ───────────────────────────────────────────────────────────

    async def generate_for_item(
        self,
        item: RetrievedContext,
        difficulty: Difficulty,
        num_mcqs: int = 5,
    ) -> list[MCQ]:
        """Generate *num_mcqs* varied MCQs for a single retrieval item.

        Handles validation errors with up to one retry.  Visual-only errors
        are fixed by stripping the visual; other errors trigger a full retry.

        Returns
        -------
        list[MCQ]
            Validated, citation-checked MCQs.  May be fewer than *num_mcqs*
            if some fail validation or citation checks.
        """
        if not item.sufficient:
            logger.warning(
                "Skipping topic %r: retrieval marked it as insufficient.", item.topic,
            )
            return []

        if not item.chunks:
            logger.warning("Skipping topic %r: no context chunks.", item.topic)
            return []

        prompt = build_prompt(item, difficulty, num_mcqs)
        raw_text: str | None = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                raw_text = await self._call_gemini(prompt)
                mcqs = _validate_batch(raw_text)
                return _check_all_citations(mcqs, item)

            except ValidationError as exc:
                if attempt < MAX_RETRIES:
                    # Try to salvage by stripping bad visuals
                    fixed = _try_fix_visuals(raw_text, exc)
                    if fixed is not None:
                        try:
                            mcqs = _MCQBatch.model_validate({"mcqs": fixed}).mcqs
                            logger.info(
                                "Recovered %d MCQ(s) after stripping bad visuals for %r.",
                                len(mcqs), item.topic,
                            )
                            return _check_all_citations(mcqs, item)
                        except ValidationError:
                            pass  # visual fix wasn't enough; fall through to full retry

                    logger.warning(
                        "Validation failed for %r (attempt %d/%d, %d error(s)), retrying…",
                        item.topic, attempt + 1, 1 + MAX_RETRIES, exc.error_count(),
                    )
                    await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
                else:
                    logger.error(
                        "Validation failed for %r after %d attempt(s): %s",
                        item.topic, 1 + MAX_RETRIES, exc.errors(),
                    )
                    return []

            except Exception as exc:
                logger.error(
                    "Gemini API error for %r (attempt %d/%d): %s",
                    item.topic, attempt + 1, 1 + MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
                else:
                    return []

        return []  # pragma: no cover — unreachable; satisfies type-checkers

    # ── Private ──────────────────────────────────────────────────────────

    async def _call_gemini(self, prompt: str) -> str:
        """Single async Gemini call with structured JSON output and 503 fallback."""
        models_to_try = [self._model]
        if self._model == "gemini-3.8-flash":
            models_to_try.append("gemini-3.5-flash")

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_json_schema=_BATCH_SCHEMA,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response.")
                return response.text
            except Exception as exc:
                last_error = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    logger.warning(
                        "Model %s unavailable (503). Trying fallback model...", model_name
                    )
                    continue
                raise exc

        if last_error:
            raise last_error
        raise RuntimeError("No model was available to generate content.")


# ── Module-level helpers (also used by tests) ────────────────────────────────


def _validate_batch(raw_json: str) -> list[MCQ]:
    """Parse and validate Gemini's JSON response into MCQ objects.

    Tries the expected ``{"mcqs": [...]}`` wrapper first, then falls back to
    a bare JSON array ``[...]`` in case the model ignores the wrapper.
    """
    first_error: ValidationError | None = None

    # Primary: wrapper object
    try:
        return _MCQBatch.model_validate_json(raw_json).mcqs
    except ValidationError as exc:
        first_error = exc

    # Fallback: bare array or dict with an unexpected key
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return _MCQBatch.model_validate({"mcqs": data}).mcqs
        if isinstance(data, dict):
            for key in ("questions", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return _MCQBatch.model_validate({"mcqs": data[key]}).mcqs
    except (json.JSONDecodeError, ValidationError):
        pass

    raise first_error


def _try_fix_visuals(
    raw_json: str | None,
    exc: ValidationError,
) -> list[dict[str, Any]] | None:
    """If ALL validation errors are visual-related, strip those visuals.

    Returns the fixed list of MCQ dicts (ready for re-validation), or ``None``
    if the errors aren't purely visual-related or JSON parsing fails.
    """
    if raw_json is None:
        return None

    # Classify errors
    has_visual = False
    has_other = False
    for err in exc.errors():
        loc_strs = [str(part) for part in err["loc"]]
        if "visual" in loc_strs:
            has_visual = True
        else:
            has_other = True

    if not has_visual or has_other:
        return None

    # Parse the raw JSON to get modifiable dicts
    try:
        data = json.loads(raw_json)
        raw_mcqs: list[dict[str, Any]]
        if isinstance(data, dict):
            raw_mcqs = data.get("mcqs", [])
        elif isinstance(data, list):
            raw_mcqs = data
        else:
            return None
        if not isinstance(raw_mcqs, list):
            return None
    except (json.JSONDecodeError, TypeError):
        return None

    # Identify which MCQ indices have bad visuals
    bad_indices: set[int] = set()
    for err in exc.errors():
        loc = err["loc"]
        for i, part in enumerate(loc):
            if part == "visual" and i >= 1:
                prev = loc[i - 1]
                if isinstance(prev, int):
                    bad_indices.add(prev)
                break

    if not bad_indices:
        return None

    for idx in bad_indices:
        if 0 <= idx < len(raw_mcqs):
            raw_mcqs[idx]["visual"] = None

    return raw_mcqs


def _check_all_citations(mcqs: list[MCQ], item: RetrievedContext) -> list[MCQ]:
    """Keep only MCQs whose ``source_chunk_ids`` are valid for *item*."""
    valid: list[MCQ] = []
    for mcq in mcqs:
        try:
            check_citations(mcq, item)
            valid.append(mcq)
        except ValueError as exc:
            logger.warning("Dropping MCQ %r — %s", mcq.question[:60], exc)

    if not valid:
        logger.error("ALL MCQs for topic %r failed citation checks.", item.topic)
    return valid
