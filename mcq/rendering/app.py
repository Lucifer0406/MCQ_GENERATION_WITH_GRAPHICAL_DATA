"""Main Streamlit application — NCERT MCQ Renderer.

Run:
    streamlit run app.py
  or:
    streamlit run mcq/rendering/app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Literal

# ── Dynamic sys.path resolution ──────────────────────────────────────────────
_CURRENT = Path(__file__).resolve().parent
while _CURRENT != _CURRENT.parent:
    if (_CURRENT / "mcq").is_dir():
        if str(_CURRENT) not in sys.path:
            sys.path.insert(0, str(_CURRENT))
        break
    _CURRENT = _CURRENT.parent

import streamlit as st
import streamlit.components.v1 as components
from pydantic import TypeAdapter

from mcq.schemas import MCQ, SUBJECTS, DIFFICULTIES
from mcq.rendering.dispatch import render_visual
from mcq.rendering.katex_renderer import render_katex_text, normalize_latex

logger = logging.getLogger(__name__)


def _inject_katex() -> None:
    """Dynamically load KaTeX and auto-render math inside HTML cards and options."""
    components.html("""
    <script>
    (function() {
      var parentDoc = window.parent.document;
      function renderMath() {
        if (window.parent.renderMathInElement) {
          var targets = parentDoc.querySelectorAll('.opt-box, .exp-box, .katex-text-block, .katex-formula-block');
          targets.forEach(function(el) {
            window.parent.renderMathInElement(el, {
              delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
              ],
              throwOnError: false
            });
          });
        }
      }

      if (!parentDoc.getElementById('katex-css-dyn')) {
        var link = parentDoc.createElement('link');
        link.id = 'katex-css-dyn';
        link.rel = 'stylesheet';
        link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css';
        parentDoc.head.appendChild(link);

        var s1 = parentDoc.createElement('script');
        s1.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js';
        s1.onload = function() {
          var s2 = parentDoc.createElement('script');
          s2.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js';
          s2.onload = renderMath;
          parentDoc.head.appendChild(s2);
        };
        parentDoc.head.appendChild(s1);
      } else {
        setTimeout(renderMath, 150);
      }
    })();
    </script>
    """, height=0, width=0)

# ── Global Head & CSS Injection ───────────────────────────────────────────────
_HEAD_HTML = """
<!-- Fonts & Icons -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<!-- KaTeX CSS & Script -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>

<script>
(function() {
  if (window._katexInitialized) {
    if (typeof window.runKaTeX === 'function') window.runKaTeX();
    return;
  }
  window._katexInitialized = true;

  window.runKaTeX = function() {
    if (window.renderMathInElement) {
      // Scope KaTeX rendering to question containers only — NEVER touch React widgets/popovers
      var targets = document.querySelectorAll('.katex-text-block, .katex-formula-block, .opt-box, .exp-box');
      targets.forEach(function(el) {
        window.renderMathInElement(el, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\\\(', right: '\\\\)', display: false},
            {left: '\\\\[', right: '\\\\]', display: true}
          ],
          throwOnError: false
        });
      });
    }
  };

  var debounceTimer = null;
  var observer = new MutationObserver(function() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      window.runKaTeX();
    }, 60);
  });

  function start() {
    var container = document.querySelector('.main .block-container') || document.body;
    if (container) {
      observer.observe(container, { childList: true, subtree: true });
      window.runKaTeX();
    }
  }

  if (document.readyState === "complete" || document.readyState === "interactive") {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
</script>

<style>
/* ── Theme tokens ── */
:root {
  --cream:        #faf8f4;
  --parchment:    #f3efe8;
  --warm-white:   #ffffff;
  --border:       #e2dbd0;
  --border-light: #ede8e0;
  --maroon:       #8b1a1a;
  --maroon-dk:    #6b1414;
  --maroon-lt:    #b52a2a;
  --maroon-bg:    #f9f0f0;
  --charcoal:     #1e1a18;
  --ink:          #2c2825;
  --muted:        #6b6460;
  --label:        #4a4440;
  --shadow-sm:    0 1px 4px rgba(30,20,10,.07);
  --shadow-md:    0 4px 16px rgba(30,20,10,.10);
  --radius:       12px;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--cream) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--ink) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* Main container */
.main .block-container {
  max-width: 960px !important;
  padding: 1.5rem 1.5rem 4rem !important;
  margin: 0 auto !important;
}

/* Top header bar */
.app-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px 0 20px;
  border-bottom: 1.5px solid var(--border);
  margin-bottom: 24px;
}
.app-header-icon {
  width: 48px; height: 48px;
  background: var(--maroon);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 1.3rem;
  box-shadow: 0 3px 10px rgba(139,26,26,.25);
}
.app-header h1 {
  font-family: 'EB Garamond', Georgia, serif !important;
  font-size: 1.85rem !important;
  font-weight: 600 !important;
  color: var(--charcoal) !important;
  margin: 0 !important; padding: 0 !important;
  line-height: 1.2;
}
.app-header p {
  font-size: .85rem; color: var(--muted);
  margin: 4px 0 0; font-weight: 400;
}

/* Streamlit Selectbox override */
div[data-testid="stSelectbox"] label {
  font-family: 'Inter', sans-serif !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
  color: var(--label) !important;
}

/* Generate button */
.stButton > button {
  background: var(--maroon) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: .92rem !important;
  letter-spacing: .03em !important;
  padding: 10px 24px !important;
  cursor: pointer !important;
  transition: background .18s, transform .1s !important;
  box-shadow: 0 2px 8px rgba(139,26,26,.25) !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: var(--maroon-lt) !important;
}

/* Container border override for MCQ Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--warm-white) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
  margin-bottom: 20px !important;
  padding: 16px 20px !important;
}

/* MCQ Header inside Card */
.card-header-row {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 10px; margin-bottom: 12px;
  border-bottom: 1px solid var(--border-light);
}
.q-title {
  font-family: 'Inter', sans-serif;
  font-size: .75rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase;
  color: var(--maroon);
}
.q-topic {
  font-size: .8rem; color: var(--muted); font-style: italic;
  margin-left: 10px;
}
.diff-badge {
  background: var(--maroon-bg);
  border: 1px solid rgba(139,26,26,.18);
  border-radius: 14px; padding: 2px 10px;
  font-size: .72rem; font-weight: 600; color: var(--maroon);
  text-transform: capitalize;
}

/* Text blocks & KaTeX */
.katex-text-block {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 1.1rem;
  line-height: 1.65;
  color: var(--ink);
  margin-bottom: 12px;
}

.katex-formula-block {
  text-align: center;
  font-size: 1.25rem;
  margin: 12px 0;
}

/* Option box */
.opt-box {
  display: flex; align-items: center; gap: 12px;
  background: var(--parchment);
  border: 1px solid var(--border-light);
  border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px;
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 1.05rem; color: var(--ink);
}
.opt-box-correct {
  background: #f0faf0 !important;
  border: 1.5px solid #4a8a4a !important;
}
.opt-badge {
  width: 24px; height: 24px;
  border-radius: 50%; background: #ffffff;
  border: 1.5px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Inter', sans-serif; font-size: .75rem; font-weight: 700;
  color: var(--label); flex-shrink: 0;
}
.opt-badge-correct {
  background: #4a8a4a !important; border-color: #4a8a4a !important; color: #fff !important;
}

/* Explanation Panel */
.exp-box {
  background: var(--parchment);
  border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; margin-top: 10px;
}
.exp-title {
  font-family: 'Inter', sans-serif; font-size: .75rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .06em; color: var(--maroon);
  margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}

/* Expander styling */
[data-testid="stExpander"] {
  border: 1px solid var(--border-light) !important;
  border-radius: 8px !important;
  background: var(--cream) !important;
  margin-top: 12px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Inter', sans-serif !important;
  font-size: .82rem !important; font-weight: 600 !important;
  color: var(--maroon) !important;
}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
_OPTION_LETTERS = ["A", "B", "C", "D"]

_SUBJECT_ICONS = {
    "physics":     "fa-solid fa-atom",
    "chemistry":   "fa-solid fa-flask",
    "mathematics": "fa-solid fa-square-root-variable",
    "biology":     "fa-solid fa-dna",
}


def _load_sample() -> list[MCQ]:
    """Load sample MCQs fallback from examples json."""
    sample_path = _CURRENT / "examples" / "sample_mcqs.json"
    if not sample_path.exists():
        sample_path = Path(__file__).resolve().parents[2] / "examples" / "sample_mcqs.json"
    return TypeAdapter(list[MCQ]).validate_json(sample_path.read_text(encoding="utf-8"))


def _call_pipeline(subject: str, difficulty: str) -> list[MCQ]:
    """Run retrieval + generation pipeline dynamically."""
    try:
        from mcq.retrieval.retriever import retrieve_quiz_context
        from mcq.generation import generate_mcqs

        quiz_context = retrieve_quiz_context(
            subject=subject,
            difficulty=difficulty,
            n_questions=5,
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                mcqs = loop.run_until_complete(generate_mcqs(quiz_context, total_mcqs=5))
            else:
                mcqs = loop.run_until_complete(generate_mcqs(quiz_context, total_mcqs=5))
        except RuntimeError:
            mcqs = asyncio.run(generate_mcqs(quiz_context, total_mcqs=5))

        if mcqs:
            return mcqs
    except Exception as e:
        logger.warning("Pipeline error, falling back to sample dataset: %s", e)
        st.info(f"Using standard question dataset ({e}).")

    return _load_sample()


# ── Main Entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit execution block."""
    try:
        st.set_page_config(
            page_title="NCERT MCQ Generator",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        pass  # Page config already set

    st.markdown(_HEAD_HTML, unsafe_allow_html=True)

    # ── Session State ─────────────────────────────────────────────────────────
    if "mcqs" not in st.session_state:
        st.session_state.mcqs = []
    if "revealed" not in st.session_state:
        st.session_state.revealed = {}

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <div class="app-header-icon"><i class="fa-solid fa-graduation-cap"></i></div>
      <div>
        <h1>NCERT MCQ Generator</h1>
        <p>Physics &middot; Chemistry &middot; Mathematics &middot; Biology &nbsp;&mdash;&nbsp; Class XI &amp; XII</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Control Panel ─────────────────────────────────────────────────────────
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1.2], gap="medium")

        with col1:
            subject = st.selectbox(
                "Select Subject",
                options=list(SUBJECTS),
                format_func=lambda s: s.capitalize(),
                index=0,
                key="select_subject",
            )

        with col2:
            difficulty = st.selectbox(
                "Select Difficulty",
                options=list(DIFFICULTIES),
                format_func=lambda d: d.capitalize(),
                index=1,
                key="select_difficulty",
            )

        with col3:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            generate = st.button(
                "•  Generate",
                key="generate_btn",
                help="Call generation pipeline for questions",
            )

    # ── Pipeline Execution ────────────────────────────────────────────────────
    if generate:
        with st.spinner("Retrieving context & generating questions..."):
            st.session_state.mcqs = _call_pipeline(subject, difficulty)
            st.session_state.revealed = {}

    # ── Render Questions ──────────────────────────────────────────────────────
    mcqs: list[MCQ] = st.session_state.mcqs

    if mcqs:
        _inject_katex()
        try:
            with open("output.json", "w", encoding="utf-8") as f:
                f.write(TypeAdapter(list[MCQ]).dump_json(mcqs, indent=2).decode("utf-8"))
        except Exception as e:
            logger.debug("output.json write failed: %s", e)

        sub_icon = _SUBJECT_ICONS.get(mcqs[0].subject, "fa-solid fa-circle-question")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:20px 0 16px">
          <i class="{sub_icon}" style="color:var(--maroon);font-size:1.1rem"></i>
          <span style="font-size:.95rem;font-weight:700;color:var(--charcoal)">
            Generated Questions ({len(mcqs)})
          </span>
          <span class="diff-badge">{mcqs[0].subject.capitalize()} &middot; {mcqs[0].difficulty.capitalize()}</span>
        </div>
        """, unsafe_allow_html=True)

        for idx, mcq in enumerate(mcqs):
            card_key = f"q_{idx}"
            revealed = st.session_state.revealed.get(card_key, False)

            with st.container(border=True):
                # Card Header
                st.markdown(f"""
                <div class="card-header-row">
                  <div>
                    <span class="q-title">Question {idx + 1}</span>
                    <span class="q-topic"><i class="fa-regular fa-bookmark"></i> {mcq.topic}</span>
                  </div>
                  <span class="diff-badge">{mcq.difficulty}</span>
                </div>
                """, unsafe_allow_html=True)

                # Question Text
                render_katex_text(mcq.question)

                # Visual Component (if present)
                if mcq.visual:
                    st.markdown("<div style='margin:10px 0 14px'>", unsafe_allow_html=True)
                    render_visual(mcq.visual)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Options A, B, C, D
                for opt_idx, opt_text in enumerate(mcq.options):
                    is_correct = (opt_idx == mcq.correct_index)
                    letter = _OPTION_LETTERS[opt_idx]
                    norm_opt = normalize_latex(opt_text)

                    if is_correct and revealed:
                        render_katex_text(f"✅ **{letter})** {norm_opt}")
                    else:
                        render_katex_text(f"**{letter})** {norm_opt}")

                # Expander for Correct Answer & Explanation
                with st.expander("Show Answer & Explanation", expanded=revealed):
                    st.session_state.revealed[card_key] = True
                    correct_letter = _OPTION_LETTERS[mcq.correct_index]
                    correct_text = mcq.options[mcq.correct_index]

                    st.markdown(f"**Correct Answer: ({correct_letter})**")
                    render_katex_text(correct_text)
                    st.markdown("---")
                    st.markdown("**Explanation:**")
                    render_katex_text(mcq.explanation)

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:var(--muted)">
          <i class="fa-solid fa-book-open" style="font-size:2.4rem;color:var(--border);margin-bottom:14px;display:block"></i>
          <p style="font-size:.95rem;font-weight:600;color:var(--label);margin-bottom:4px">
            No Questions Loaded
          </p>
          <p style="font-size:.82rem">
            Choose a subject and difficulty level above, then click <strong>Generate</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
