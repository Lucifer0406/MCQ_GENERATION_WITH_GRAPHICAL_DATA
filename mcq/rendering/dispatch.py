"""Dispatch: route a Visual object to the correct renderer."""

from mcq.schemas import Visual, FormulaVisual, ChemicalStructureVisual, FunctionPlotVisual, DataGraphVisual
from mcq.rendering.katex_renderer import render_formula
from mcq.rendering.smiles_renderer import render_chemical_structure
from mcq.rendering.vsepr_renderer import render_vsepr, is_vsepr_molecule
from mcq.rendering.plotly_renderer import render_function_plot, render_data_graph


def render_visual(visual: Visual | None) -> None:
    """Call the appropriate renderer for the given visual. Silently no-ops on None."""
    if visual is None:
        return
    try:
        if isinstance(visual, FormulaVisual):
            render_formula(visual.latex)
        elif isinstance(visual, ChemicalStructureVisual):
            if is_vsepr_molecule(visual.smiles, visual.name):
                render_vsepr(visual.smiles, visual.name)
            else:
                render_chemical_structure(visual.smiles, visual.name)
        elif isinstance(visual, FunctionPlotVisual):
            render_function_plot(visual)
        elif isinstance(visual, DataGraphVisual):
            render_data_graph(visual)
    except Exception:
        pass  # never crash the page; just omit the visual
