"""
run_python(code) - for statistics and calculations SQL isn't suited for.

SECURITY NOTE, read before using this outside a local/trusted demo:
This executes model-generated code with a restricted builtins list and no
filesystem/network access, which is a reasonable floor for a local demo but
is NOT a real security sandbox. A determined or simply buggy generation
could still find ways to consume excessive memory/CPU or exploit a gap in
the restricted namespace. For anything beyond your own machine or a trusted
demo, run this tool inside a real sandbox (e.g. E2B) instead of in-process.
"""

from __future__ import annotations

import contextlib
import io
import math
from typing import Any

import numpy as np
import pandas as pd

_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "print": print,
}


def run_python(code: str, dataframes: dict[str, pd.DataFrame]) -> dict:
    """Executes `code` with the loaded tables available as pandas DataFrames
    (by table name) plus pd/np/math, and returns whatever was printed plus
    the value of a `result` variable if the code set one."""

    local_vars: dict[str, Any] = {**dataframes}
    global_vars = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "math": math,
    }

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, global_vars, local_vars)  # noqa: S102 - restricted globals above
    except Exception as exc:  # noqa: BLE001 - becomes model-facing data, not a crash
        return {"success": False, "error": f"{type(exc).__name__}: {exc}", "stdout": stdout.getvalue()}

    result = local_vars.get("result")
    if isinstance(result, pd.DataFrame):
        result = result.head(50).to_dict(orient="records")
    elif isinstance(result, (pd.Series, np.ndarray)):
        result = list(result)[:50]

    return {"success": True, "stdout": stdout.getvalue(), "result": result}
