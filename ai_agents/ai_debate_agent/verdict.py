"""Helpers for reconciling judge scores with the declared verdict."""


def reconcile_scores(parsed: dict, debater_a: str, debater_b: str) -> dict:
    """Recompute totals and winner from per-argument scores when available."""
    totals = {debater_a: 0, debater_b: 0}
    for argument in parsed.get("arguments", []):
        speaker = argument.get("speaker")
        if speaker in totals:
            totals[speaker] += (
                int(argument.get("logic", 0))
                + int(argument.get("evidence", 0))
                + int(argument.get("persuasiveness", 0))
            )

    if any(totals.values()):
        parsed["totals"] = totals
        if totals[debater_a] > totals[debater_b]:
            parsed["winner"] = debater_a
        elif totals[debater_b] > totals[debater_a]:
            parsed["winner"] = debater_b
        else:
            parsed["winner"] = "Tie"

    parsed.setdefault("totals", totals)
    return parsed
