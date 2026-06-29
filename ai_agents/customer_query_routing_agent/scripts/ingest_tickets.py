"""
Ingest additional resolved ticket threads from a JSON export into VectorAI DB.

In a real deployment this file would be an export from your helpdesk platform
(Zendesk, Freshdesk, Salesforce Service Cloud, etc.). The format is a JSON
array where each object has the fields shown in data/sample_tickets.json.

All four data sources belong to the same organisation:
  product_faq       written by the support team
  product_docs      written by legal / ops
  resolved_tickets  exported from the helpdesk system (this script)
  resolved_queries  built live by the agent as it resolves queries

VectorAI DB provides the single retrieval layer across all of them.

Usage:
    uv run python scripts/ingest_tickets.py
    uv run python scripts/ingest_tickets.py --file path/to/your_export.json

The script is idempotent: ticket IDs already in the collection are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from actian_vectorai import PointStruct

from customer_query_routing_agent.config import TICKETS_COLLECTION
from customer_query_routing_agent.embedder import Embedder
from customer_query_routing_agent.vectorstore import get_client, setup_collections

DEFAULT_FILE = Path(__file__).parent.parent / "data" / "sample_tickets.json"


def load_tickets(path: Path) -> list[dict]:
    with open(path) as f:
        tickets = json.load(f)
    print(f"Loaded {len(tickets)} tickets from {path.name}")
    return tickets


def existing_ticket_ids(client) -> set[str]:
    """Return the set of ticket_id values already stored in VectorAI DB."""
    # Scroll all points and collect ticket_id payloads
    ids: set[str] = set()
    count = client.points.count(TICKETS_COLLECTION)
    if count == 0:
        return ids

    points, _ = client.points.scroll(TICKETS_COLLECTION, limit=count + 10)
    for p in points:
        tid = p.payload.get("ticket_id")
        if tid:
            ids.add(tid)
    return ids


def ingest(tickets: list[dict], client, embedder: Embedder) -> int:
    """
    Embed each ticket's summary, upsert into VectorAI DB.
    Skips tickets whose ticket_id is already present.
    Returns the number of tickets inserted.
    """
    existing = existing_ticket_ids(client)

    new_tickets = [t for t in tickets if t.get("ticket_id") not in existing]
    if not new_tickets:
        print("All tickets already ingested. Nothing to do.")
        return 0

    print(f"Inserting {len(new_tickets)} new tickets (skipping {len(tickets) - len(new_tickets)} already present)...")

    texts = [t["summary"] for t in new_tickets]
    vectors = embedder.embed_batch(texts)

    # Use high IDs to avoid colliding with the auto-seeded tickets (IDs 1-8)
    base_id = 20001
    points = [
        PointStruct(
            id=base_id + i,
            vector=vectors[i],
            payload=new_tickets[i],
        )
        for i in range(len(new_tickets))
    ]
    client.points.upsert(TICKETS_COLLECTION, points)
    return len(new_tickets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ticket threads into VectorAI DB")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to a JSON file containing an array of ticket objects",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    client = get_client()
    client.connect()
    setup_collections(client)

    before = client.points.count(TICKETS_COLLECTION)
    print(f"'{TICKETS_COLLECTION}' currently has {before} entries.")

    tickets = load_tickets(args.file)
    embedder = Embedder()

    inserted = ingest(tickets, client, embedder)

    after = client.points.count(TICKETS_COLLECTION)
    print(f"\nDone. Inserted {inserted} tickets. Collection now has {after} entries.")
    if inserted > 0:
        print("Restart the Streamlit app to see the updated Resolved tickets counter.")


if __name__ == "__main__":
    main()
