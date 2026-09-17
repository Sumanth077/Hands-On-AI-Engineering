"""
Flags when the agent looks stuck: repeating near-identical SQL queries
without making progress. This does not stop the investigation, it injects
a message telling the model to reassess, and lets the model decide what to
do next.
"""

from __future__ import annotations

import re

from agent.state import InvestigationState


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def is_looping(state: InvestigationState, window: int = 4, min_repeats: int = 3) -> bool:
    recent = state.recent_sql_queries(n=window)
    if len(recent) < min_repeats:
        return False
    normalized = [_normalize(q) for q in recent]
    most_common = max(set(normalized), key=normalized.count)
    return normalized.count(most_common) >= min_repeats
