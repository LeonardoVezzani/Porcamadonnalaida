"""CLI entry point for the Zotero metadata extraction pipeline.

Usage
-----
python -m zotero_extract \\
    --db     /path/to/zotero.sqlite \\
    --storage /path/to/Zotero/storage \\
    --out    out/

Optional flags
--------------
--limit N          Process at most N papers.
--since YYYY-MM-DD Only papers added on or after this date.
--log-level LEVEL  Logging verbosity (DEBUG | INFO | WARNING | ERROR).
                   Defaults to INFO.
--no-semantic-scholar
                   Skip Semantic Scholar fallback entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from .db import fetch_papers, open_db
from .extractor import extract
from .output import build_row, write_json, write_summary_csv
from .text_retrieval import retrieve_text


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zotero_extract",
        description="Extract driving-experiment metadata from a local Zotero library.",
    )
    p.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="Path to zotero.sqlite",
    )
    p.add_argument(
        "--storage",
        required=True,
        metavar="PATH",
        help="Path to the Zotero storage/ folder",
    )
    p.add_argument(
        "--out",
        default="out",
        metavar="PATH",
        help="Output directory (default: out/)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N papers",
    )
    p.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only papers added on or after this date",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        dest="no_semantic_scholar",
        help="Skip Semantic Scholar fallback",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    db_path = Path(args.db)
    storage_dir = Path(args.storage)
    out_dir = Path(args.out)

    if not db_path.is_file():
        log.error("zotero.sqlite not found: %s", db_path)
        return 1
    if not storage_dir.is_dir():
        log.error("storage directory not found: %s", storage_dir)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Opening database: %s", db_path)
    conn = open_db(db_path)

    log.info("Fetching paper list …")
    papers = fetch_papers(conn, limit=args.limit, since=args.since)
    log.info("Found %d paper(s) to process", len(papers))

    ss_session: requests.Session | None = None
    if not args.no_semantic_scholar:
        ss_session = requests.Session()
        ss_session.headers.update({"User-Agent": "zotero-extract/0.1 (research pipeline)"})

    rows = []
    no_text_count = 0

    for paper in tqdm(papers, desc="Processing", unit="paper"):
        result = retrieve_text(
            paper=paper,
            conn=conn,
            storage_dir=storage_dir,
            out_dir=out_dir,
            ss_session=ss_session,
        )

        extraction = extract(result.text, abstract=paper.get("abstract") or "")

        write_json(
            out_dir=out_dir,
            paper=paper,
            extraction=extraction,
            retrieval_method=result.method,
            no_text=result.no_text,
        )

        row = build_row(
            paper=paper,
            extraction=extraction,
            retrieval_method=result.method,
            no_text=result.no_text,
        )
        rows.append(row)

        if result.no_text:
            no_text_count += 1
            log.warning(
                "no text: %s  %s",
                paper["key"],
                paper.get("title", "")[:60],
            )

    write_summary_csv(out_dir, rows)

    log.info(
        "Done. %d processed, %d with no text. Results in: %s",
        len(rows),
        no_text_count,
        out_dir,
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
