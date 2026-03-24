"""Write per-paper JSON files and the consolidated results_summary.csv."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JSON_DIR = "json"
_CSV_NAME = "results_summary.csv"

_CSV_FIELDS = [
    "item_key",
    "item_id",
    "title",
    "year",
    "doi",
    "authors",
    "item_type",
    "retrieval_method",
    "no_text",
    "task_type",
    "driving_scenario",
    "level_of_automation",
    "visual_cue",
    "confidence",
]


def write_json(
    out_dir: Path,
    paper: dict[str, Any],
    extraction: dict[str, Any],
    retrieval_method: str,
    no_text: bool,
) -> Path:
    """Write per-paper JSON to ``out_dir/json/<key>.json``.

    Returns the path written.
    """
    json_dir = out_dir / _JSON_DIR
    json_dir.mkdir(parents=True, exist_ok=True)

    key = paper["key"]
    record = {
        "item_key": key,
        "item_id": paper["itemID"],
        "title": paper.get("title", ""),
        "year": paper.get("year", ""),
        "doi": paper.get("doi", ""),
        "authors": paper.get("authors", []),
        "item_type": paper.get("itemType", ""),
        "retrieval_method": retrieval_method,
        "no_text": no_text,
        **extraction,
    }

    out_path = json_dir / f"{key}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    logger.debug("wrote %s", out_path)
    return out_path


def write_summary_csv(
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write ``results_summary.csv`` to *out_dir*.

    Each element of *rows* should be a dict produced by ``build_row()``.

    Returns the path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / _CSV_NAME

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("wrote %s (%d rows)", csv_path, len(rows))
    return csv_path


def build_row(
    paper: dict[str, Any],
    extraction: dict[str, Any],
    retrieval_method: str,
    no_text: bool,
) -> dict[str, Any]:
    """Build a flat CSV row dict from paper + extraction data."""
    return {
        "item_key": paper["key"],
        "item_id": paper["itemID"],
        "title": paper.get("title", ""),
        "year": paper.get("year", ""),
        "doi": paper.get("doi", ""),
        "authors": "; ".join(paper.get("authors") or []),
        "item_type": paper.get("itemType", ""),
        "retrieval_method": retrieval_method,
        "no_text": no_text,
        **extraction,
    }
