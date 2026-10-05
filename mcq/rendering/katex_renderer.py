r"""KaTeX renderer — outputs clean text with KaTeX math delimiters ($...$, $$...$$, \(...\)) directly without iframes."""

import html
import re
import streamlit as st


def normalize_latex(text: str) -> str:
    r"""Normalize LaTeX delimiters like \(...\) to $...$ and \[...\] to $$...$$."""
    if not text:
        return ""
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    return text


def render_katex_text(text: str, **kwargs) -> None:
    r"""Render plain text with inline LaTeX math directly to Streamlit DOM (no iframe)."""
    norm_text = normalize_latex(text)
    st.markdown(
        f'<div class="katex-text-block">{norm_text}</div>',
        unsafe_allow_html=True
    )


def render_formula(latex: str, **kwargs) -> None:
    """Render a standalone display-mode LaTeX formula directly to Streamlit DOM (no iframe)."""
    norm_latex = normalize_latex(latex).strip("$ ")
    st.markdown(
        f'<div class="katex-formula-block">$${norm_latex}$$</div>',
        unsafe_allow_html=True
    )
