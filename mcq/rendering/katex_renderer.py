r"""KaTeX renderer — outputs clean text with KaTeX math delimiters ($...$, $$...$$, \(...\)) directly without iframes."""

import html
import re
import streamlit as st


def normalize_latex(text: str) -> str:
    r"""Normalize LaTeX delimiters like \(...\) to $...$ and \[...\] to $$...$$."""
    if not text:
        return ""
    # Fix double-escaped LaTeX commands from LLM JSON (e.g. \\sigma -> \sigma, \\frac -> \frac)
    text = re.sub(r'\\+([a-zA-Z]+)', lambda m: '\\' + m.group(1), text)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    stripped = text.strip()
    if stripped.startswith(("\\", r"\frac", r"\sqrt")) and "$" not in stripped:
        return f"${stripped}$"
    return text


def render_katex_text(text: str, **kwargs) -> None:
    r"""Render plain text with inline LaTeX math directly to Streamlit DOM (no iframe)."""
    norm_text = normalize_latex(text)
    st.markdown(norm_text)


def render_formula(latex: str, **kwargs) -> None:
    """Render a standalone display-mode LaTeX formula directly to Streamlit DOM (no iframe)."""
    norm_latex = normalize_latex(latex).strip("$ ")
    st.latex(norm_latex)
