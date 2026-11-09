#!/usr/bin/env python3

"""Daily data collection pipeline for the AI Research Agent.



Runs Devable Research Agent collectors, filters top items for digest generation, and logs

output to logs/pipeline.log. Designed to be invoked by Windows Task Scheduler

at 7:45am — OpenClaw reads digest_input_today.json at 8am for the Slack digest.

"""



import argparse

import json

import os

import subprocess

import sys

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Optional



PROJECT_ROOT = Path(

    os.environ.get("DEVABLE_PROJECT_ROOT", "")

).resolve() if os.environ.get("DEVABLE_PROJECT_ROOT") else Path(__file__).resolve().parent.parent

COLLECT_SCRIPT = PROJECT_ROOT / "skills" / "tracking-list" / "scripts" / "collect.py"

FILTER_SCRIPT = PROJECT_ROOT / "scripts" / "filter_top_items.py"

DATA_FILE = PROJECT_ROOT / "data_today.json"

DIGEST_FILE = PROJECT_ROOT / "digest_input_today.json"

LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

TZ_LAGOS = timezone(timedelta(hours=1))





def _log(message: str) -> None:

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(TZ_LAGOS).strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {message}"

    with LOG_FILE.open("a", encoding="utf-8") as handle:

        handle.write(line + "\n")

    print(line)





def _run_step(name: str, cmd: list) -> bool:

    _log(f"START {name}")

    _log(f"CMD {' '.join(cmd)}")

    try:

        result = subprocess.run(

            cmd,

            cwd=PROJECT_ROOT,

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace",

            env={**os.environ, "DEVABLE_PROJECT_ROOT": str(PROJECT_ROOT)},

        )

    except Exception as exc:

        _log(f"ERROR {name} failed to launch: {exc}")

        return False



    if result.stdout:

        for line in result.stdout.splitlines():

            _log(f"  {line}")

    if result.stderr:

        for line in result.stderr.splitlines():

            _log(f"  {line}")



    if result.returncode != 0:

        _log(f"ERROR {name} exited with code {result.returncode}")

        return False



    _log(f"DONE {name}")

    return True





def _parse_generated_at(value: str) -> Optional[datetime]:

    if not value:

        return None

    text = value.strip()

    if text.endswith("Z"):

        text = text[:-1] + "+00:00"

    try:

        parsed = datetime.fromisoformat(text)

    except ValueError:

        return None

    if parsed.tzinfo is None:

        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed





def _validate_digest_input() -> bool:

    """Ensure digest_input_today.json exists and is fresh (today Lagos or within 24h)."""

    if not DIGEST_FILE.exists():

        _log(

            "ERROR digest_input_today.json missing after filter step — "

            "OpenClaw cron will fail. Pipeline aborted."

        )

        return False



    try:

        data = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))

    except (json.JSONDecodeError, OSError) as exc:

        _log(f"ERROR digest_input_today.json unreadable: {exc}")

        return False



    items = data.get("items") or []

    if not items:

        _log("ERROR digest_input_today.json has zero items — pipeline aborted.")

        return False



    generated_at = _parse_generated_at(data.get("generated_at", ""))

    now_lagos = datetime.now(TZ_LAGOS)

    today_lagos = now_lagos.date()



    if generated_at is None:

        _log(

            "ERROR digest_input_today.json missing or invalid generated_at — "

            "run filter_top_items.py before OpenClaw cron."

        )

        return False



    generated_lagos = generated_at.astimezone(TZ_LAGOS)

    age = now_lagos - generated_lagos

    if generated_lagos > now_lagos:

        _log(

            f"ERROR digest_input_today.json has a future generated_at "

            f"(generated_at={generated_lagos.isoformat()}) — pipeline aborted."

        )

        return False



    if generated_lagos.date() != today_lagos and age > timedelta(hours=24):

        _log(

            f"ERROR digest_input_today.json is stale "

            f"(generated_at={generated_lagos.isoformat()}, age={age}) — pipeline aborted."

        )

        return False



    return True





def _write_status_line(data_path: Path, digest_path: Path) -> None:

    """Append one-line pipeline status summary to logs/pipeline.log."""

    try:

        data = json.loads(data_path.read_text(encoding="utf-8"))

        digest = json.loads(digest_path.read_text(encoding="utf-8"))

    except (json.JSONDecodeError, OSError) as exc:

        _log(f"WARNING could not write status line: {exc}")

        return



    by_source = data.get("stats", {}).get("by_source", {})

    if not by_source:

        counts: dict = {}

        for item in data.get("items", []):

            source = item.get("source", "unknown")

            counts[source] = counts.get(source, 0) + 1

        source_summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    else:

        source_summary = ", ".join(

            f"{name}={info.get('items', 0)}"

            for name, info in sorted(by_source.items())

        )



    digest_count = len(digest.get("items", []))

    categories = digest.get("stats", {}).get("by_category", {})

    if not categories and digest.get("items"):

        if str(FILTER_SCRIPT) not in sys.path:

            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

        try:

            from filter_top_items import _category_counts



            categories = _category_counts(digest.get("items", []))

        except Exception:

            categories = {}



    category_summary = ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))

    timestamp = datetime.now(TZ_LAGOS).strftime("%Y-%m-%d %H:%M:%S")

    _log(

        f"STATUS {timestamp} | collected: {source_summary} | "

        f"digest_items={digest_count} | categories: {category_summary}"

    )





def run_pipeline(date_str: str, depth: str = "default", top_n: int = 30) -> int:

    """Run collection and filtering. Returns process exit code."""

    _log("=" * 60)

    _log(f"Pipeline run started (date={date_str}, depth={depth}, top={top_n})")



    if not COLLECT_SCRIPT.exists():

        _log(f"ERROR collect script not found: {COLLECT_SCRIPT}")

        return 1

    if not FILTER_SCRIPT.exists():

        _log(f"ERROR filter script not found: {FILTER_SCRIPT}")

        return 1



    collect_cmd = [

        sys.executable,

        str(COLLECT_SCRIPT),

        "--date", date_str,

        "--depth", depth,

        "--no-cache",

        "--output", str(DATA_FILE),

    ]

    if not _run_step("collect", collect_cmd):

        _log("Pipeline aborted: collection failed")

        return 1



    if not DATA_FILE.exists():

        _log(f"ERROR expected output missing: {DATA_FILE}")

        return 1



    filter_cmd = [

        sys.executable,

        str(FILTER_SCRIPT),

        str(DATA_FILE),

        str(DIGEST_FILE),

        "--top", str(top_n),

    ]

    if not _run_step("filter", filter_cmd):

        _log("Pipeline aborted: filtering failed")

        return 1



    if not _validate_digest_input():

        return 1



    data_kb = DATA_FILE.stat().st_size / 1024

    digest_kb = DIGEST_FILE.stat().st_size / 1024

    _log(f"Pipeline complete: {DATA_FILE.name} ({data_kb:.1f}KB), {DIGEST_FILE.name} ({digest_kb:.1f}KB)")

    _write_status_line(DATA_FILE, DIGEST_FILE)

    return 0





def main() -> None:

    parser = argparse.ArgumentParser(description="Run Devable Research Agent daily collection pipeline")

    parser.add_argument(

        "--date",

        default=datetime.now(TZ_LAGOS).strftime("%Y-%m-%d"),

        help="Target collection date YYYY-MM-DD (default: today Lagos)",

    )

    parser.add_argument(

        "--depth",

        default="default",

        choices=["quick", "default", "deep"],

        help="Collector depth (default: default)",

    )

    parser.add_argument(

        "--top",

        type=int,

        default=30,

        help="Number of top items for digest input (default: 30)",

    )

    args = parser.parse_args()



    raise SystemExit(run_pipeline(args.date, args.depth, args.top))





if __name__ == "__main__":

    main()

