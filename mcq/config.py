"""Project-wide paths and API-key loading, shared by all three components.

Paths are built from this file's location, never from the current working
directory, so they work no matter where `streamlit run` or `pytest` is launched.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = PROJECT_ROOT / "Knowledgebase"
RAW_DIR = KNOWLEDGE_DIR / "Raw"              # NCERT PDFs, one folder per subject (gitignored)
PROCESSED_DIR = KNOWLEDGE_DIR / "Processed"  # extracted Markdown (gitignored)
INDEX_DIR = KNOWLEDGE_DIR / "Index"          # FAISS indexes + chunk sidecars (gitignored)

ENV_FILE = PROJECT_ROOT / ".env"

# The SDK accepts either name; GOOGLE_API_KEY wins if both are set.
API_KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def get_api_key() -> str:
    """Return the Gemini API key from the environment or .env, or fail with a clear message."""
    load_dotenv(ENV_FILE)  # explicit path; does not override variables already set in the shell
    for var in API_KEY_VARS:
        key = os.environ.get(var, "").strip()
        if key:
            return key
    raise RuntimeError(
        f"No Gemini API key found. Copy .env.example to {ENV_FILE} and set "
        f"GOOGLE_API_KEY (get a key at https://aistudio.google.com/apikey)."
    )
