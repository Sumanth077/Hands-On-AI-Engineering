import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .apify_client import ApifyClient, compare_content
from .config import Settings
from .exports import csv_download, json_download
from .models import FetchInput, SearchInput
from .service import configured_limits, run_research


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="RepoScout: explicit, billable Apify retrieval")
    root.add_argument("--allow-billable", action="store_true", help="Acknowledge live API charges")
    root.add_argument("--account-plan", help="Operator-supplied plan; defaults to env or unknown")
    root.add_argument(
        "--warm-idle",
        choices=["unknown", "warm", "first request after client idle"],
        default="unknown",
        help="Operator report; does not establish a cold server",
    )
    root.add_argument("--output-dir", type=Path, default=Path("outputs"))
    commands = root.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Search public GitHub repositories")
    search.add_argument("query")
    search.add_argument("--max-results", type=int, default=5)
    for name in ("fetch", "compare"):
        commands.add_parser(name).add_argument("url")
    many = commands.add_parser("fetch-many")
    many.add_argument("urls", nargs="+")
    many.add_argument("--concurrency", type=int, default=3)
    research = commands.add_parser(
        "research", help="Live research; requires --allow-billable and live configuration"
    )
    research.add_argument("requirement")
    research.add_argument("--candidate-limit", type=int, default=5)
    research.add_argument(
        "--live",
        action="store_true",
        help="Compatibility flag; research always uses live APIs with explicit enablement",
    )
    return root


async def run(args) -> int:
    if args.command == "research":
        limits = configured_limits(args.candidate_limit)
        result = await run_research(args.requirement, limits, offline=False)
        folder = args.output_dir / ("research-" + uuid4().hex[:8])
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "research.json").write_text(json_download(result), encoding="utf-8")
        (folder / "comparison.csv").write_text(csv_download(result), encoding="utf-8")
        (folder / "diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2), encoding="utf-8"
        )
        print(f"Live research: {len(result.projects)} projects; {result.status}")
        print(f"Saved evidence and comparison to {folder}")
        return 0 if result.projects else 1
    # Validate the entire workload before creating a client or starting any work.
    if args.command == "search":
        request = SearchInput(query=args.query, maxResults=args.max_results)
    elif args.command == "fetch-many":
        if not 1 <= args.concurrency <= 20 or len(args.urls) > 20:
            raise ValueError("Concurrency must be 1..20; at most 20 URLs")
        requests = [FetchInput(url=url) for url in args.urls]
    else:
        request = FetchInput(url=args.url)
    settings = Settings.from_env()
    if args.account_plan:
        settings.account_plan = args.account_plan
    # Reserve storage before incurring charges. Each invocation gets its own folder.
    folder = args.output_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    async with ApifyClient(settings, warm_idle_status=args.warm_idle) as client:
        if args.command == "search":
            outcomes = [await client.search(request)]
        elif args.command == "fetch-many":
            outcomes = await client.fetch_many(requests, concurrency=args.concurrency)
        elif args.command == "compare":
            outcomes = await asyncio.gather(
                client.fetch(request), client.fetch(request, classic=True)
            )
        else:
            outcomes = [await client.fetch(request)]
    records = {"measurements": [out.measurement.model_dump(mode="json") for out in outcomes]}
    if args.command == "compare":
        records["comparison"] = compare_content(outcomes)
    for index, out in enumerate(outcomes):
        (folder / f"raw-{index}-{out.measurement.interface}.json").write_text(
            json.dumps(out.raw, indent=2), encoding="utf-8"
        )
    (folder / "measurements.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(f"Saved sanitized responses to {folder}")
    return 0 if all(out.measurement.outcome == "success" for out in outcomes) else 1


def main() -> int:
    root = parser()
    args = root.parse_args()
    if not args.allow_billable:
        root.error("Live calls require --allow-billable; obtain operator approval first")
    try:
        return asyncio.run(run(args))
    except ValidationError:
        root.exit(2, "Invalid input; use public URLs and documented bounds. No values echoed.\n")
    except ValueError:
        root.exit(2, "Invalid configuration/input; check APIFY_TOKEN and command bounds.\n")
    except OSError:
        root.exit(
            2, "Local output could not be written. Check storage before repeating any live calls.\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
