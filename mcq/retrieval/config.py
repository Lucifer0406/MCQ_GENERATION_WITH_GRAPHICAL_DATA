"""Retrieval settings: which NCERT chapter backs each subject, and which models build the index."""

from dataclasses import dataclass

from mcq.config import INDEX_DIR, PROCESSED_DIR, RAW_DIR


@dataclass(frozen=True)
class Source:
    pdf: str        # path under Knowledgebase/Raw
    document: str   # human-readable book name, used in citations
    chapter: str


# MVP: one chapter per subject. Add a chapter = add an entry here + rerun extract and build_index.
SOURCES: dict[str, Source] = {
    "physics": Source("physics/keph102.pdf", "NCERT Physics Class XI Part 1", "Motion in a Straight Line"),
    "chemistry": Source("chemistry/kech104.pdf", "NCERT Chemistry Class XI Part 1", "Chemical Bonding and Molecular Structure"),
    "mathematics": Source("mathematics/kemh110.pdf", "NCERT Mathematics Class XI", "Conic Sections"),
    "biology": Source("biology/kebo102.pdf", "NCERT Biology Class XI", "Biological Classification"),
}

# One-time PDF page -> Markdown + LaTeX conversion (offline; see extract.py).
# A fallback chain, not one model: on 2026-10-03 individual Gemini models were
# intermittently returning 503 "high demand" while others answered fine.
# Each model also has its own free-tier daily quota (20 requests), so the chain spreads load.
EXTRACTION_MODELS = (
    "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",   # last resort; output still must pass the marker check
)

# Embeddings. gemini-embedding-2 has no task_type parameter: documents and queries are
# distinguished by these exact prefixes (from Google's docs). Changing any of these
# four values requires rebuilding the index; the manifest records them and the
# retriever refuses to load an index built with different settings.
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
DOC_FORMAT = "title: {title} | text: {text}"
QUERY_FORMAT = "task: search result | query: {text}"

# Retrieval. Difficulty only changes how much context each question gets, never the query.
CHUNKS_PER_QUESTION = {"easy": 3, "medium": 4, "hard": 5}
CANDIDATES = 20            # neighbours fetched from FAISS before filtering
MAX_PER_SECTION = 3        # diversity: at most this many chunks from one section per question
SUFFICIENT_SCORE = 0.6     # top chunk below this -> sufficient=False (unrelated text scored ~0.5 in testing)

__all__ = [
    "SOURCES", "Source", "EXTRACTION_MODELS", "RAW_DIR", "PROCESSED_DIR", "INDEX_DIR",
    "EMBEDDING_MODEL", "EMBEDDING_DIM", "DOC_FORMAT", "QUERY_FORMAT",
    "CHUNKS_PER_QUESTION", "CANDIDATES", "MAX_PER_SECTION", "SUFFICIENT_SCORE",
]
