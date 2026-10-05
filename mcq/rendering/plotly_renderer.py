"""Plotly renderer for function_plot and data_graph visuals."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from mcq.schemas import FunctionPlotVisual, DataGraphVisual


_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#faf8f5",
    font=dict(family="Inter, sans-serif", size=12, color="#2c2c2c"),
    margin=dict(l=48, r=20, t=28, b=44),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(showgrid=True, gridcolor="#e8e4de", zeroline=True, zerolinecolor="#b0a99a", linecolor="#c8c0b4"),
    yaxis=dict(showgrid=True, gridcolor="#e8e4de", zeroline=True, zerolinecolor="#b0a99a", linecolor="#c8c0b4"),
)

_LINE_COLORS = ["#8b1a1a", "#c46b2a", "#5a4a3a", "#a08060", "#3d2b1f"]


def _safe_eval(expr: str, var: np.ndarray, var_name: str) -> np.ndarray | None:
    """Safe expression evaluator — only numpy math, no builtins."""
    ns = {
        var_name: var,
        **{k: getattr(np, k) for k in dir(np) if not k.startswith("_")},
        "pi": np.pi, "e": np.e,
    }
    try:
        result = eval(compile(expr, "<expr>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
        return np.asarray(result, dtype=float)
    except Exception:
        return None


def render_function_plot(visual: FunctionPlotVisual, height: int = 340) -> None:
    fig = go.Figure()
    N = 600

    for i, curve in enumerate(visual.curves):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        lo, hi = curve.domain

        if curve.kind == "parametric":
            t = np.linspace(lo, hi, N)
            xs = _safe_eval(curve.x_expr, t, "t")
            ys = _safe_eval(curve.y_expr, t, "t")
        else:
            xs = np.linspace(lo, hi, N)
            ys = _safe_eval(curve.expr, xs, "x")

        if xs is None or ys is None:
            continue

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=curve.label or f"curve {i+1}",
            line=dict(color=color, width=2.2),
        ))

    layout = dict(**_LAYOUT)
    layout["xaxis"] = dict(**_LAYOUT["xaxis"], title=visual.x_label)
    layout["yaxis"] = dict(**_LAYOUT["yaxis"], title=visual.y_label)
    if visual.caption:
        layout["title"] = dict(text=visual.caption, font=dict(size=13), x=0.5)

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_data_graph(visual: DataGraphVisual, height: int = 300) -> None:
    fig = go.Figure()

    for i, series in enumerate(visual.series):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        if visual.chart == "line":
            fig.add_trace(go.Scatter(x=series.x, y=series.y, mode="lines+markers",
                                     name=series.name, line=dict(color=color, width=2.2),
                                     marker=dict(size=6, color=color)))
        elif visual.chart == "bar":
            fig.add_trace(go.Bar(x=series.x, y=series.y, name=series.name,
                                 marker_color=color, opacity=0.85))
        else:
            fig.add_trace(go.Scatter(x=series.x, y=series.y, mode="markers",
                                     name=series.name, marker=dict(size=7, color=color)))

    layout = dict(**_LAYOUT)
    layout["xaxis"] = dict(**_LAYOUT["xaxis"], title=visual.x_label)
    layout["yaxis"] = dict(**_LAYOUT["yaxis"], title=visual.y_label)
    if visual.caption:
        layout["title"] = dict(text=visual.caption, font=dict(size=13), x=0.5)

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
