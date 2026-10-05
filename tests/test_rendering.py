"""Tests for mcq/rendering visual renderers and dispatcher."""

import json
from pathlib import Path
from pydantic import TypeAdapter
import pytest

from mcq.schemas import MCQ, FormulaVisual, ChemicalStructureVisual, FunctionPlotVisual, DataGraphVisual
from mcq.rendering.katex_renderer import build_mcq_card_html
from mcq.rendering.smiles_renderer import build_smiles_html
from mcq.rendering.plotly_renderer import build_function_plot_html, build_data_graph_html
from mcq.rendering.dispatch import render_visual


@pytest.fixture
def sample_mcqs():
    sample_path = Path("examples/sample_mcqs.json")
    if not sample_path.exists():
        pytest.skip("examples/sample_mcqs.json missing")
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    adapter = TypeAdapter(list[MCQ])
    return adapter.validate_python(data)


def test_katex_html_builder():
    html_out = build_mcq_card_html(1, "$v^2 = u^2 + 2as$", ["5 m/s", "10 m/s", "20 m/s", "50 m/s"], 1, "Explanation")
    assert "katex.min.js" in html_out
    assert "renderMathInElement" in html_out
    assert "throwOnError: false" in html_out


def test_smiles_html_builder():
    html_out = build_smiles_html("C=C", name="ethene")
    assert "smiles-drawer.min.js" in html_out
    assert "canvas data-smiles=\"C=C\"" in html_out
    assert "SmilesDrawer.apply" in html_out


def test_plotly_function_plot_builder(sample_mcqs):
    ellipse_mcq = [m for m in sample_mcqs if m.visual and m.visual.type == "function_plot"][0]
    assert isinstance(ellipse_mcq.visual, FunctionPlotVisual)
    html_out = build_function_plot_html(ellipse_mcq.visual)
    assert "plotly-2.35.2.min.js" in html_out
    assert "Plotly.newPlot" in html_out


def test_plotly_data_graph_builder(sample_mcqs):
    graph_mcq = [m for m in sample_mcqs if m.visual and m.visual.type == "data_graph"][0]
    assert isinstance(graph_mcq.visual, DataGraphVisual)
    html_out = build_data_graph_html(graph_mcq.visual)
    assert "plotly-2.35.2.min.js" in html_out
    assert "Plotly.newPlot" in html_out


def test_dispatch_none():
    # Guarantee None visual doesn't raise
    render_visual(None)
