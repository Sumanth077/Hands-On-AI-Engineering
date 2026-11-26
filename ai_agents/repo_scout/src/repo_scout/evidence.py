"""Bounded verbatim evidence passages; archival content is never truncated."""

import re
from functools import lru_cache


@lru_cache(maxsize=256)
def passages(source_id: str, content: str) -> tuple[dict, ...]:
    output = []
    start = 0
    while start < len(content):
        end = min(len(content), start + 850)
        if end < len(content):
            boundary = content.rfind("\n", start + 350, end)
            if boundary > start:
                end = boundary + 1
        text = content[start:end]
        if text.strip():
            output.append({"id": f"{source_id}:p{len(output)}", "text": text})
        start = end
    return tuple(output)


GROUPS = (
    r"licen[cs]e|copyright|permission|apache|\bMIT\b|\bAGPL\b|\bGPL\b|model.weight",
    r"extract.{0,30}table|table.{0,30}extract|find_tables|read_pdf|tableformer|tabletransformer",
    r"install|command.line|localhost|local machine|run.locally|\.pdf\b|quick.start|\bCLI\b|self.host",
    r"commits on|\b20\d\d\b|latest release|released|maintenance|archived",
)


def select_passages(sources, budget: int, terms: str = "") -> tuple[list[dict], dict]:
    """Balance licence, capability, local use and dated activity, then fill by relevance.

    Selection indexes immutable original text. No rewritten quote or ellipsis enters a
    passage. Limits affect model context, not evidence retention or citation validation.
    """
    pool = [(s, p) for s in sources if s.outcome == "success" for p in passages(s.id, s.markdown)]
    chosen, used = {}, 0
    groups = list(GROUPS)
    words = set(re.findall(r"[a-z][a-z0-9_]{2,}", terms.lower())) - {
        "required",
        "project",
        "source",
        "open",
        "self",
        "hosted",
        "recent",
        "maintenance",
        "activity",
        "permissive",
        "licence",
        "license",
        "the",
        "and",
    }
    if words:
        groups[1] = r"\b(?:" + "|".join(re.escape(word) for word in sorted(words)) + r")\b"

    def add(source, passage):
        nonlocal used
        if passage["id"] not in chosen and used + len(passage["text"]) <= budget:
            chosen[passage["id"]] = (source, passage)
            used += len(passage["text"])

    # A group gets multiple opportunities across sources, rather than letting a long
    # README or licence consume every slot before local-use/maintenance passages.
    ranked_groups = []
    for pattern in groups:
        ranked_groups.append(
            sorted(
                pool,
                key=lambda pair: (
                    -min(8, len(re.findall(pattern, pair[1]["text"], re.I))),
                    pair[0].role == "repository",
                    pair[1]["id"],
                ),
            )
        )
    for rank in range(3):
        for pattern, ranked in zip(groups, ranked_groups, strict=True):
            matching = [(s, p) for s, p in ranked if re.search(pattern, p["text"], re.I)]
            if rank < len(matching):
                add(*matching[rank])
    ranked = sorted(
        pool,
        key=lambda pair: (
            -sum(bool(re.search(pattern, pair[1]["text"], re.I)) for pattern in groups)
        ),
    )
    for pair in ranked:
        add(*pair)
    context = []
    for source in sources:
        selected = [p for s, p in chosen.values() if s.id == source.id]
        context.append(
            {
                "id": source.id,
                "url": source.url,
                "role": source.role,
                "outcome": source.outcome,
                "passages": selected,
                "passages_available": len(passages(source.id, source.markdown)),
                "passages_shown": len(selected),
            }
        )
    return context, {
        "passages_available": len(pool),
        "passages_shown": len(chosen),
        "context_limited": len(chosen) < len(pool),
        "characters": used,
    }
