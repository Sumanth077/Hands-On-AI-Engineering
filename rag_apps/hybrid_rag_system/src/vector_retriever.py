from typing import Any
import chromadb


def retrieve_from_vector(
    collection: chromadb.Collection,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    count = collection.count()
    if count == 0:
        return []

    n = min(top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity": round(1.0 - dist, 4),
        })

    return chunks
