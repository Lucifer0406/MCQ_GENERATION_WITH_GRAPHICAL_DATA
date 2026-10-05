"""Root entry point for NCERT MCQ Generator Streamlit Web Application.

Run:
    streamlit run app.py
  or:
    streamlit run mcq/rendering/app.py
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mcq.rendering.app import main

if __name__ == "__main__":
    main()
