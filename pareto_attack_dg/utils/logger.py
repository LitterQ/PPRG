from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_csv(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_gradient_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "step", "domain_i", "domain_j", "cosine"]
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def update_gradient_summary(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"epochs": []}
    payload.setdefault("epochs", []).append(summary)
    payload["latest"] = summary
    save_json(path, payload)
