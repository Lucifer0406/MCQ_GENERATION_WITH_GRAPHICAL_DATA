"""FastAPI Web API for NCERT MCQ Generation.

Exposes a single POST endpoint /generate that accepts subject & difficulty
and returns generated MCQs as JSON by reusing the existing RAG + LLM pipeline.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or from project root:
    python -m api.main
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# # ── Ensure project root is importable ────────────────────────────────────────
# ROOT_DIR = Path(__file__).resolve().parents[1]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

# api/main.py  →  parents[0]=api, parents[1]=mcq, parents[2]=PROJECT_ROOT
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mcq.schemas import MCQ

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NCERT MCQ Generator API",
    description=(
        "REST API that generates fresh NCERT-aligned MCQs using a RAG "
        "retrieval pipeline + LLM generation. Zero UI, pure JSON."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class QuizRequest(BaseModel):
    subject: str = Field(
        ...,
        description="Subject name (case-insensitive)",
        examples=["mathematics", "biology", "physics", "chemistry"],
    )
    difficulty: str = Field(
        ...,
        description="Difficulty level (case-insensitive)",
        examples=["easy", "medium", "hard"],
    )
    n_questions: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of MCQs to generate (1-30)",
    )
    avoid_chunk_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of previously used chunk IDs to avoid so that "
            "repeated calls produce fresh questions."
        ),
    )


class MCQItem(BaseModel):
    """Serializable shape of a single MCQ returned to the client."""

    id: Optional[str] = None
    question: str
    options: List[str]
    correct_index: int
    explanation: Optional[str] = None
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None


class QuizResponse(BaseModel):
    subject: str
    difficulty: str
    n_requested: int
    n_generated: int
    elapsed_seconds: float
    used_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs used to generate this batch. Send back in "
                    "`avoid_chunk_ids` on the next call for variety.",
    )
    mcqs: List[MCQItem]


# ── Helpers ──────────────────────────────────────────────────────────────────
VALID_SUBJECTS = {"mathematics", "biology", "physics", "chemistry"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _normalize_subject(subject: str) -> str:
    s = subject.strip().lower()
    if s not in VALID_SUBJECTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid subject '{subject}'. "
                f"Must be one of: {sorted(VALID_SUBJECTS)}"
            ),
        )
    return s


def _normalize_difficulty(difficulty: str) -> str:
    d = difficulty.strip().lower()
    if d not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid difficulty '{difficulty}'. "
                f"Must be one of: {sorted(VALID_DIFFICULTIES)}"
            ),
        )
    return d


def _mcq_to_item(mcq: MCQ) -> MCQItem:
    """Convert an internal MCQ pydantic model into the API response item.

    Uses ``model_dump`` (pydantic v2) so we only pull fields that exist;
    falls back to attribute access for safety.
    """
    try:
        data = mcq.model_dump()          # pydantic v2
    except AttributeError:               # pragma: no cover - pydantic v1
        data = mcq.dict()

    return MCQItem(
        id=str(data.get("id")) if data.get("id") is not None else None,
        question=data.get("question", ""),
        options=list(data.get("options", [])),
        correct_index=int(data.get("correct_index", 0)),
        explanation=data.get("explanation"),
        subject=data.get("subject"),
        difficulty=data.get("difficulty"),
        source_chunk_ids=data.get("source_chunk_ids"),
    )


# ── Core pipeline (reused logic, no Streamlit) ───────────────────────────────
async def _run_pipeline(
    subject: str,
    difficulty: str,
    n_questions: int,
    avoid_chunk_ids: Optional[List[str]] = None,
):
    """Retrieval → generation → returns (mcqs, used_chunk_ids).

    This is the FastAPI equivalent of the Streamlit `generate_quiz` helper,
    minus any session_state dependencies.
    """
    from mcq.retrieval.retriever import retrieve_quiz_context
    from mcq.generation import generate_mcqs

    avoid = set(avoid_chunk_ids or [])

    # Step 1 — Retrieval
    quiz_context = retrieve_quiz_context(
        subject,
        difficulty,
        n_questions=n_questions,
        avoid_chunk_ids=avoid,
    )

    # Step 2 — Generation (already async)
    mcqs = await generate_mcqs(quiz_context, total_mcqs=n_questions)

    # Collect the chunk IDs that were consumed, for caller-side diversity
    used_chunk_ids: List[str] = []
    for item in getattr(quiz_context, "items", []) or []:
        chunks = getattr(item, "chunks", None) or []
        if chunks:
            used_chunk_ids.append(str(chunks[0].chunk_id))

    return mcqs, used_chunk_ids


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "NCERT MCQ Generator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Generate MCQs for a subject & difficulty",
            "GET  /health":   "Health check",
            "GET  /subjects": "List supported subjects & difficulties",
        },
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/subjects", tags=["meta"])
async def subjects():
    return {
        "subjects": sorted(VALID_SUBJECTS),
        "difficulties": sorted(VALID_DIFFICULTIES),
    }


@app.post(
    "/generate",
    response_model=QuizResponse,
    status_code=status.HTTP_200_OK,
    tags=["quiz"],
    summary="Generate MCQs via the RAG + LLM pipeline",
)
async def generate(request: QuizRequest) -> QuizResponse:
    subject = _normalize_subject(request.subject)
    difficulty = _normalize_difficulty(request.difficulty)

    loop = asyncio.get_event_loop()
    t0 = loop.time()

    try:
        mcqs, used_chunk_ids = await _run_pipeline(
            subject=subject,
            difficulty=difficulty,
            n_questions=request.n_questions,
            avoid_chunk_ids=request.avoid_chunk_ids,
        )
    except FileNotFoundError as exc:
        logger.error("Index not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval index unavailable: {exc}",
        )
    except Exception as exc:
        logger.exception("Pipeline failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        )

    elapsed = loop.time() - t0

    if not mcqs:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The pipeline returned 0 MCQs. This usually means the LLM "
                "failed validation or was rate-limited. Try again."
            ),
        )

    return QuizResponse(
        subject=subject,
        difficulty=difficulty,
        n_requested=request.n_questions,
        n_generated=len(mcqs),
        elapsed_seconds=round(elapsed, 3),
        used_chunk_ids=used_chunk_ids,
        mcqs=[_mcq_to_item(m) for m in mcqs],
    )


# ── Dev entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )