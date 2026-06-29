from typing import Any


def format_graph_context(graph_results: dict[str, Any]) -> str:
<<<<<<< HEAD
=======
    """Formats graph retrieval results into a plain-text block listing entities and relationships."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    entities = graph_results.get("entities", [])
    relationships = graph_results.get("relationships", [])

    if not entities and not relationships:
        return "No graph context available."

    lines = []
    if entities:
        lines.append("ENTITIES:")
        for e in entities:
            desc = e["description"] if e["description"] else "no description"
            lines.append(f"  • {e['name']} [{e['type']}]: {desc}")

    if relationships:
        lines.append("\nRELATIONSHIPS:")
        for r in relationships:
            rels = " | ".join(r["relations"])
            lines.append(f"  • {r['source']} --[{rels}]--> {r['target']}")
            if r["description"]:
                lines.append(f"    ({r['description']})")

    return "\n".join(lines)


def format_vector_context(vector_results: list[dict[str, Any]]) -> str:
<<<<<<< HEAD
=======
    """Formats vector retrieval results into a plain-text block of numbered chunks with similarity scores."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    if not vector_results:
        return "No vector context available."
    lines = []
    for i, chunk in enumerate(vector_results, 1):
        sim = chunk.get("similarity", 0.0)
        lines.append(f"[Chunk {i} | similarity: {sim:.3f}]")
        lines.append(chunk["text"].strip())
        lines.append("")
    return "\n".join(lines)


def fuse_context(
    graph_results: dict[str, Any],
    vector_results: list[dict[str, Any]],
) -> str:
<<<<<<< HEAD
=======
    """Combines formatted graph and vector contexts into a single prompt-ready string."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    graph_ctx = format_graph_context(graph_results)
    vector_ctx = format_vector_context(vector_results)
    return (
        "=== KNOWLEDGE GRAPH CONTEXT ===\n"
        f"{graph_ctx}\n\n"
        "=== DOCUMENT CHUNKS (Vector Retrieval) ===\n"
        f"{vector_ctx}"
    )
