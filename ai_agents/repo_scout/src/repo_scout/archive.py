"""Append-only sanitized live response journal and final exports, never a fixture cache."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .apify_client import redact
from .exports import csv_download, json_download


class RunArchive:
    def __init__(self, secrets: list[str], *, root: Path = Path("outputs")):
        self.folder = (
            root
            / (
                "live-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid4().hex[:8]
            )
        ).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.secrets = secrets

    def record(self, event: str, data: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live",
            "event": event,
            "data": data,
        }
        for secret in self.secrets:
            entry = redact(entry, secret)
        with (self.folder / "responses.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save(self, result):
        result.diagnostics["saved_result_dir"] = str(self.folder)
        data = result.model_dump(mode="json")
        for secret in self.secrets:
            data = redact(data, secret)
        result = type(result).model_validate(data)
        (self.folder / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (self.folder / "research.json").write_text(json_download(result), encoding="utf-8")
        (self.folder / "comparison.csv").write_text(
            csv_download(result), encoding="utf-8", newline=""
        )
        (self.folder / "diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2), encoding="utf-8"
        )
        self.record("completed", {"status": result.status, "projects": len(result.projects)})
        return result
