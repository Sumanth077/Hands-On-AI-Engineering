"""
Function-calling schemas for the tools, in the OpenAI/Liner Chat
Completions `tools` format.

Kept deliberately primitive, on purpose: there is no `find_root_cause()` or
similar shortcut. The model decides what to investigate; these tools only
execute one small, well-defined operation each.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": (
                "Returns schema, column types, row counts, and a few sample rows "
                "for every table currently loaded. Call this first, and again "
                "any time you're unsure what columns or tables exist."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Runs a SQL query (DuckDB syntax) against the loaded tables and "
                "returns the result rows. Use this for aggregations, filtering, "
                "grouping, and joins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A DuckDB SQL query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Runs Python code for statistics or calculations SQL isn't suited "
                "for. Loaded tables are available as pandas DataFrames by table "
                "name. pandas is available as pd, numpy as np. Set a variable "
                "named `result` to whatever you want returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Renders a chart from a list of rows (e.g. from a prior run_sql result).",
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Row objects to plot, e.g. the 'rows' field from a run_sql result.",
                    },
                    "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "pie", "histogram"]},
                    "x": {"type": "string", "description": "Column name for the x-axis."},
                    "y": {"type": "string", "description": "Column name for the y-axis, if applicable."},
                    "color": {"type": "string", "description": "Optional column name to color/group by."},
                    "title": {"type": "string", "description": "Chart title."},
                },
                "required": ["rows", "chart_type", "x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": (
                "Records a discovery into the investigation's persistent findings "
                "list, so it survives into the final report. Use this whenever "
                "you confirm something worth including in the final answer, not "
                "just for your own intermediate reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "importance": {"type": "string", "enum": ["low", "medium", "high"]},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short evidence notes supporting this finding.",
                    },
                },
                "required": ["finding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_hypothesis",
            "description": (
                "Records or updates a hypothesis about what's actually causing the "
                "pattern in the objective. Call this as you form, test, and resolve "
                "hypotheses during the investigation, not just for your own "
                "intermediate reasoning. Calling it again with the same hypothesis "
                "text updates its status/confidence/evidence in place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "status": {"type": "string", "enum": ["untested", "supported", "rejected"]},
                    "confidence": {"type": "number", "description": "Confidence from 0 to 1, if known."},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short evidence notes supporting or refuting this hypothesis.",
                    },
                },
                "required": ["hypothesis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_investigation",
            "description": (
                "Concludes the investigation and produces the final report. Call "
                "this once you genuinely have enough evidence to explain what's "
                "going on, instead of just replying with no more tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string", "description": "The single, specific root cause identified."},
                    "confidence": {"type": "number", "description": "Confidence in the root cause, from 0 to 1."},
                    "summary_markdown": {
                        "type": "string",
                        "description": "The full narrative report, in markdown, including findings, evidence, and conclusion.",
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The key findings that led to the conclusion.",
                    },
                    "recommended_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete next steps or actions recommended based on the root cause.",
                    },
                },
                "required": ["root_cause", "confidence", "summary_markdown"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
