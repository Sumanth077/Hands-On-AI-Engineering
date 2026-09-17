"""
create_chart(...) - renders a chart locally with Plotly.

Deliberately local. This project's scope is the Liner Model API only; it
does not call Liner's separate Visualization API. If you ever extend this
tool, keep it that way, wiring this up to the real Visualization API would
pull in a different product than the one this project is about.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px

_CHART_BUILDERS = {
    "bar": px.bar,
    "line": px.line,
    "scatter": px.scatter,
    "pie": px.pie,
    "histogram": px.histogram,
}


def create_chart(
    rows: list[dict[str, Any]],
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
):
    if chart_type not in _CHART_BUILDERS:
        return {"success": False, "error": f"Unsupported chart_type '{chart_type}'. Use one of {list(_CHART_BUILDERS)}."}

    df = pd.DataFrame(rows)
    if x not in df.columns:
        return {"success": False, "error": f"Column '{x}' not found in the provided rows."}

    kwargs: dict[str, Any] = {"x": x, "title": title}
    if y:
        kwargs["y"] = y
    if color:
        kwargs["color"] = color

    builder = _CHART_BUILDERS[chart_type]
    try:
        fig = builder(df, **kwargs)
    except Exception as exc:  # noqa: BLE001 - becomes model-facing data
        return {"success": False, "error": str(exc)}

    return {"success": True, "figure": fig}
