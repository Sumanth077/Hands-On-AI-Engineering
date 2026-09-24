import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.vectorai_client import (  # noqa: E402
    delete_collection,
    ensure_collection,
    health_check,
    search,
    upsert_point,
)

TEST_COLLECTION = "scratch_test_collection"
TEST_DIM = 8
TEST_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# actian-vectorai-client's PointStruct only accepts a non-negative int or a
# valid UUID string as an id — an arbitrary string like "scratch-point-1" is
# rejected with a ValidationError.
TEST_POINT_ID = str(uuid.uuid4())


def main() -> None:
    print("Checking local Actian VectorAI DB setup...")
    result = health_check()
    print(f"health_check result: {result}")

    if not result["reachable"]:
        print(
            "\nFAIL: could not reach Actian VectorAI DB. "
            "Make sure it is running (docker compose up -d) and try again."
        )
        return

    # Delete any leftover scratch collection from a previous run first: this
    # script always upserts the identical TEST_VECTOR, so a stale point from
    # an earlier run would tie with today's fresh point at cosine similarity
    # 1.0 and could win the top-1 slot non-deterministically (found during
    # the persistence-fix verification).
    print(f"\nClearing any leftover '{TEST_COLLECTION}' from a previous run...")
    delete_collection(TEST_COLLECTION)

    print(f"Ensuring collection '{TEST_COLLECTION}' exists (dim={TEST_DIM})...")
    ensure_collection(TEST_COLLECTION, dim=TEST_DIM)

    print(f"Upserting a dummy point (id={TEST_POINT_ID})...")
    upsert_point(
        TEST_COLLECTION,
        TEST_POINT_ID,
        TEST_VECTOR,
        {"note": "check_actian.py smoke test point"},
    )

    print("Searching for it with the same vector...")
    raw_top_result = search(TEST_COLLECTION, TEST_VECTOR, top_k=1)
    print(f"normalized search result: {raw_top_result}")

    if not raw_top_result:
        print("\nFAIL: search returned no results.")
        return

    top = raw_top_result[0]
    if str(top["id"]) != TEST_POINT_ID:
        print(f"\nFAIL: expected top result id {TEST_POINT_ID!r}, got {top['id']!r}.")
        return

    print("\nPASS: Actian VectorAI DB is reachable, and upsert/search round-tripped correctly.")


if __name__ == "__main__":
    main()
