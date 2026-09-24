"""Operator tool — NOT run automatically, and not called from
ensure_collection() (a rebuild has real cost).

Run this manually if search results ever seem degraded after heavy
re-ingestion churn against an already-open collection (as seen once during
step 9's fix-up debugging: repeated re-ingestion left manual_chunks
returning empty results for small top_k values until its index was
rebuilt). It rebuilds the HNSW index for the manual and incident
collections and prints before/after stats so you can see what changed.

Usage:
    python scripts/maintenance_rebuild_index.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.incident_store import _configured_incident_collection  # noqa: E402
from memory.manual import _configured_manual_collection  # noqa: E402
from memory.vectorai_client import get_client  # noqa: E402


def _print_stats(client, name: str, label: str) -> None:
    stats = client.vde.get_stats(name)
    print(
        f"  {label}: total_vectors={stats.total_vectors}, "
        f"indexed_vectors={stats.indexed_vectors}, "
        f"deleted_vectors={stats.deleted_vectors}, "
        f"storage_bytes={stats.storage_bytes}, "
        f"index_memory_bytes={stats.index_memory_bytes}"
    )


def rebuild(name: str) -> None:
    client = get_client()
    print(f"\n--- {name} ---")
    _print_stats(client, name, "before")

    client.vde.rebuild_index(name)
    print("  rebuild_index() called.")

    _print_stats(client, name, "after")


def main() -> None:
    collections = [_configured_manual_collection(), _configured_incident_collection()]
    print(f"Rebuilding HNSW index for: {collections}")
    for name in collections:
        rebuild(name)


if __name__ == "__main__":
    main()
