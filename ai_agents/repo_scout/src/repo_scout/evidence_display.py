"""Display-only Markdown repair. Never modify archived evidence or citation excerpts."""

import html
import re
from urllib.parse import quote, urljoin, urlsplit

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
REFERENCE = re.compile(r"^ {0,3}\[([^]\n]+)\]:\s*(<[^>]+>|\S+)", re.M)


def _balanced(text, start, opening, closing):
    depth = 0
    for i in range(start, len(text)):
        if i and text[i - 1] == "\\":
            continue
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    return None


def _links(text):
    """Locate complete inline links, including balanced parentheses in destinations."""
    i = 0
    while i < len(text):
        if text[i] != "[" or (i and text[i - 1] == "\\"):
            i += 1
            continue
        close = _balanced(text, i, "[", "]")
        if close is not None and text[close + 1 : close + 2] == "(":
            end = _balanced(text, close + 1, "(", ")")
            if end is not None:
                yield i, end + 1, text[i + 1 : close], text[close + 2 : end]
                i = end + 1
                continue
        i += 1


def _safe_url(destination, base):
    destination = html.unescape(destination.strip())
    if destination.startswith("<") and ">" in destination:
        destination = destination[1 : destination.index(">")]
    else:
        destination = destination.split()[0] if destination.split() else ""
    if not destination or any(ord(c) < 32 for c in destination):
        return None
    url = urljoin(base, destination)
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    return quote(url, safe=":/?#[]@!$&'()*+,;=%_~-")


def _inline(text, base, references):
    """Keep code literal; remove image embeds and restrict navigable link schemes."""
    output = []
    i = 0
    while i < len(text):
        if text[i] == "`":
            n = len(text[i:]) - len(text[i:].lstrip("`"))
            end = text.find("`" * n, i + n)
            if end >= 0:
                output.append(text[i : end + n])
                i = end + n
                continue
        image = text.startswith("![", i)
        start = i + 1 if image else i
        if text[start : start + 1] == "[":
            close = _balanced(text, start, "[", "]")
            if close is not None:
                label = text[start + 1 : close]
                end, destination = close + 1, None
                if text[end : end + 1] == "(":
                    finish = _balanced(text, end, "(", ")")
                    if finish is not None:
                        destination = text[end + 1 : finish]
                        end = finish + 1
                elif text[end : end + 1] == "[":
                    finish = text.find("]", end + 1)
                    if finish >= 0:
                        key = text[end + 1 : finish] or label
                        destination = references.get(key.casefold())
                        end = finish + 1
                elif label.casefold() in references:
                    destination = references[label.casefold()]
                if image:
                    output.append("*(image omitted)*")
                    i = end
                    continue
                if destination is not None:
                    safe = _safe_url(destination, base)
                    label = _inline(label, base, references)
                    output.append(f"[{label}](<{safe}>)" if safe else label)
                    i = end
                    continue
        if text[i] == "<":
            end = text.find(">", i + 1)
            if end >= 0:
                tag = text[i : end + 1]
                if re.match(r"<\s*(?:img|source|video|iframe|object|embed)\b", tag, re.I):
                    output.append("*(image or embed omitted)*")
                elif re.match(r"<https?://", tag, re.I):
                    safe = _safe_url(tag, base)
                    output.append(f"<{safe}>" if safe else html.escape(tag))
                else:
                    output.append(html.escape(tag))
                i = end + 1
                continue
        output.append(text[i])
        i += 1
    return "".join(output)


def _fence_state(text):
    active = None
    for line in text.splitlines():
        match = FENCE.match(line)
        if not match:
            continue
        marker, info = match.groups()
        if active is None:
            active = (marker, info)
        elif marker[0] == active[0][0] and len(marker) >= len(active[0]) and not info.strip():
            active = None
    return active


def evidence_preview(excerpt: str, source_text: str, source_url: str) -> str:
    """Recover boundary formatting from the source, returning a new display string.

    Only enclosing link syntax and a code-fence opener/closer are recovered. The
    original excerpt and source remain the inputs to citation validation and exports.
    """
    start = source_text.find(excerpt) if excerpt else -1
    initial_fence = None
    if start >= 0:
        end = start + len(excerpt)
        for left, right, _, _ in _links(source_text):
            if left < start < right:
                start = left
            if left < end < right:
                end = right
        # A passage boundary may even bisect a fence marker or its language tag.
        offset = 0
        for line in source_text.splitlines(keepends=True):
            right = offset + len(line)
            if FENCE.match(line.rstrip("\r\n")):
                if offset < start < right:
                    start = offset
                if offset < end < right:
                    end = right
            offset = right
        initial_fence = _fence_state(source_text[:start])
        text = source_text[start:end]
    else:
        text = excerpt
    references = {m[1].casefold(): m[2] for m in REFERENCE.finditer(source_text)}
    output = []
    prose = []

    def flush_prose():
        if prose:
            output.append(_inline("".join(prose), source_url, references))
            prose.clear()

    active = initial_fence
    if active:
        output.append(active[0] + active[1] + "\n")
    for line in text.splitlines(keepends=True):
        match = FENCE.match(line.rstrip("\r\n"))
        if match:
            flush_prose()
            marker, info = match.groups()
            if active is None:
                active = (marker, info)
            elif marker[0] == active[0][0] and len(marker) >= len(active[0]) and not info.strip():
                active = None
            output.append(line)
        elif active:
            output.append(line)
        elif not REFERENCE.match(line):
            prose.append(line)
    flush_prose()
    if active:
        output.append(("" if output and output[-1].endswith("\n") else "\n") + active[0] + "\n")
    return "".join(output)
