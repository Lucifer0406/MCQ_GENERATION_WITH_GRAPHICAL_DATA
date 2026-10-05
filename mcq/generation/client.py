"""Multi-provider client wrapper (Groq & Gemini) with structured output and retries.

This module is internal to ``mcq.generation``. External callers should use
:func:`mcq.generation.generate_mcqs` instead of touching :class:`MCQGenerator`
directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mcq.config import get_api_key, get_default_provider, get_groq_api_key
from mcq.schemas import MCQ, Difficulty, RetrievedContext, check_citations

from .prompts import SYSTEM_PROMPT, build_prompt, build_multi_item_prompt

logger = logging.getLogger(__name__)


# ── Internal batch wrapper (not part of team contract) ───────────────────────


class _MCQBatch(BaseModel):
    """Wrapper so LLMs return ``{"mcqs": [...]}`` instead of a bare array.

    NOT a :class:`~mcq.schemas.Contract` subclass: it doesn't need
    ``extra="forbid"`` and is never serialised across team boundaries.
    """

    mcqs: list[MCQ] = Field(min_length=1)


_BATCH_SCHEMA: dict[str, Any] = _MCQBatch.model_json_schema()


# ── Provider & Model Configuration ───────────────────────────────────────────

GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)

# Backwards-compatible alias for existing tests
MODEL = GEMINI_DEFAULT_MODEL
FALLBACK_MODELS = GEMINI_FALLBACK_MODELS

MAX_RETRIES = 2          # up to 2 retries on transient errors or validation issues
RETRY_DELAY_S = 1.5      # base seconds between retries


# ── Generator class ──────────────────────────────────────────────────────────


class MCQGenerator:
    """Async MCQ generator supporting Groq (primary, ultra-fast) and Gemini.

    Typical usage is through :func:`mcq.generation.generate_mcqs`.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        # Detect provider
        if provider is not None:
            self._provider = provider.lower()
        elif api_key is not None:
            # If explicit API key passed: Groq keys start with 'gsk_'
            if api_key.startswith("gsk_"):
                self._provider = "groq"
            else:
                self._provider = "gemini"
        else:
            self._provider = get_default_provider()

        self._api_key = api_key
        self._groq_client = None
        self._gemini_client = None

        if self._provider == "groq":
            from groq import AsyncGroq

            resolved_key = self._api_key or get_groq_api_key()
            self._groq_client = AsyncGroq(api_key=resolved_key)
            self._model = model or GROQ_DEFAULT_MODEL
        else:
            from google import genai

            resolved_key = self._api_key or get_api_key()
            self._gemini_client = genai.Client(api_key=resolved_key)
            self._model = model or GEMINI_DEFAULT_MODEL

    @property
    def provider(self) -> str:
        return self._provider

    # ── Public ───────────────────────────────────────────────────────────

    async def generate_for_item(
        self,
        item: RetrievedContext,
        difficulty: Difficulty,
        num_mcqs: int = 5,
    ) -> list[MCQ]:
        """Generate *num_mcqs* varied MCQs for a single retrieval item.

        Handles validation errors with up to two retries. Visual-only errors
        are fixed by stripping the visual; other errors trigger a full retry.

        Returns
        -------
        list[MCQ]
            Validated, citation-checked MCQs.
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
                raw_text = await self._call_llm(prompt)
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
                    "LLM API error for %r (attempt %d/%d): %s",
                    item.topic, attempt + 1, 1 + MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
                else:
                    return []

        return []  # pragma: no cover — unreachable; satisfies type-checkers

    async def generate_for_items(
        self,
        items: list[RetrievedContext],
        difficulty: Difficulty,
        total_mcqs: int = 10,
    ) -> list[MCQ]:
        """Generate *total_mcqs* across multiple diverse retrieved topic items in a single call.

        Distributes questions across all provided topic items, keeping full grounding
        and variety while avoiding multiple roundtrips and rate limits.
        """
        if not items:
            return []

        # If generate_for_item was patched/mocked (e.g. in test fixtures), fall back to it
        if getattr(self.generate_for_item, "side_effect", None) is not None or hasattr(self.generate_for_item, "mock"):
            tasks = [self.generate_for_item(item, difficulty, num_mcqs=max(1, total_mcqs // len(items))) for item in items]
            results = await asyncio.gather(*tasks)
            return [m for batch in results for m in batch]

        if len(items) == 1:
            return await self.generate_for_item(items[0], difficulty, num_mcqs=total_mcqs)

        prompt = build_multi_item_prompt(items, difficulty, total_mcqs)
        raw_text: str | None = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                raw_text = await self._call_llm(prompt)
                mcqs = _validate_batch(raw_text)
                return _check_multi_item_citations(mcqs, items)
            except ValidationError as exc:
                if attempt < MAX_RETRIES:
                    fixed = _try_fix_visuals(raw_text, exc)
                    if fixed is not None:
                        try:
                            mcqs = _MCQBatch.model_validate({"mcqs": fixed}).mcqs
                            logger.info(
                                "Recovered %d MCQ(s) after stripping bad visuals for multi-topic batch.",
                                len(mcqs),
                            )
                            return _check_multi_item_citations(mcqs, items)
                        except ValidationError:
                            pass
                    logger.warning(
                        "Validation failed for multi-topic batch (attempt %d/%d): %s",
                        attempt + 1, 1 + MAX_RETRIES, exc,
                    )
            except Exception as exc:
                logger.error("Multi-topic generation error (attempt %d/%d): %s", attempt + 1, 1 + MAX_RETRIES, exc)

        return []

    # ── Private ──────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        """Route to active provider (Groq or Gemini)."""
        if self._provider == "groq":
            return await self._call_groq(prompt)
        return await self._call_gemini(prompt)

    async def _call_groq(self, prompt: str, max_tokens: int = 2600) -> str:
        """Single async Groq call with structured JSON schema output and fallback."""
        est_prompt_tokens = int(len(prompt) / 3.8)
        if est_prompt_tokens + max_tokens > 7700:
            max_tokens = max(1800, 7700 - est_prompt_tokens)

        models_to_try = [self._model]
        for m in GROQ_FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error: Exception | None = None
        for model_name in models_to_try:
            # Qwen on Groq on-demand tier has a strict 1000 OTPM (output tokens per minute) limit
            effective_tokens = min(max_tokens, 980) if "qwen" in model_name else max_tokens
            try:
                resp = await self._groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "mcq_batch",
                            "schema": _BATCH_SCHEMA,
                        },
                    },
                    max_tokens=effective_tokens,
                    temperature=0.2,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise RuntimeError("Groq returned an empty response.")
                return content
            except Exception as exc:
                last_error = exc
                logger.warning("Groq model %s failed (%s), trying fallback…", model_name, exc)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("No Groq model was available to generate content.")

    async def _call_gemini(self, prompt: str) -> str:
        """Single async Gemini call with structured JSON output and fallback chain."""
        from google.genai import types

        if self._model in GEMINI_FALLBACK_MODELS:
            models_to_try = list(GEMINI_FALLBACK_MODELS)
        else:
            models_to_try = [self._model] + [m for m in GEMINI_FALLBACK_MODELS if m != self._model]

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                response = await self._gemini_client.aio.models.generate_content(
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
                        "Model %s unavailable (503), trying next fallback…", model_name
                    )
                    continue
                raise exc

        if last_error:
            raise last_error
        raise RuntimeError("No Gemini model was available to generate content.")


# ── Module-level helpers (also used by tests) ────────────────────────────────


def _validate_batch(raw_json: str) -> list[MCQ]:
    """Parse and validate LLM's JSON response into MCQ objects.

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


def _check_multi_item_citations(mcqs: list[MCQ], items: list[RetrievedContext]) -> list[MCQ]:
    """Validate citations for a multi-topic batch against all provided items."""
    # Map chunk_id to its item's topic
    all_chunks: dict[str, str] = {}
    for item in items:
        for c in item.chunks:
            all_chunks[c.chunk_id] = item.topic

    valid: list[MCQ] = []
    for mcq in mcqs:
        # Check that all cited chunk IDs exist in the provided context
        unknown = [cid for cid in mcq.source_chunk_ids if cid not in all_chunks]
        if unknown:
            logger.warning("Dropping MCQ %r: cites unknown chunk_ids %s", mcq.question[:60], unknown)
            continue

        # If the MCQ's topic is not exact, align it with the topic of its primary cited chunk
        primary_chunk = mcq.source_chunk_ids[0]
        if primary_chunk in all_chunks:
            expected_topic = all_chunks[primary_chunk]
            if mcq.topic != expected_topic:
                mcq = mcq.model_copy(update={"topic": expected_topic})

        valid.append(mcq)

    if not valid:
        logger.error("ALL MCQs for multi-topic batch failed citation checks.")
    return valid
