"""Tests for mcq/rendering visual renderers and dispatcher."""

import json
from pathlib import Path
from pydantic import TypeAdapter
import pytest

from mcq.schemas import MCQ
from mcq.rendering.katex_renderer import normalize_latex
from mcq.rendering.vsepr_renderer import is_vsepr_molecule
from mcq.rendering.dispatch import render_visual


@pytest.fixture
def sample_mcqs():
    sample_path = Path("examples/sample_mcqs.json")
    if not sample_path.exists():
        pytest.skip("examples/sample_mcqs.json missing")
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    adapter = TypeAdapter(list[MCQ])
    return adapter.validate_python(data)


def test_katex_delimiters_normalized():
    assert normalize_latex(r"\(x^2\)") == "$x^2$"
    assert normalize_latex(r"\[x^2\]") == "$$x^2$$"
    # Over-escaped backslashes collapsed
    assert normalize_latex(r"\\sigma") == "$\\sigma$"


def test_vsepr_detection():
    assert is_vsepr_molecule("CH4")
    assert is_vsepr_molecule("PCl5")
    assert is_vsepr_molecule("SF6")
    assert not is_vsepr_molecule("c1ccccc1")  # benzene should go to SmilesDrawer


def test_dispatch_all_sample_visuals(sample_mcqs):
    # Guarantee dispatching over all sample visuals does not throw
    for mcq in sample_mcqs:
        render_visual(mcq.visual)


def test_dispatch_none():
    # Guarantee None visual doesn't raise
    render_visual(None)
