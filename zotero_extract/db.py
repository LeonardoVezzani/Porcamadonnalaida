"""SQLite helpers for reading the local Zotero database.

Assumptions about the DB schema (Zotero ≥ 5.x):
- ``items``             – one row per library item (key, itemTypeID, …)
- ``itemTypes``         – maps itemTypeID → typeName
- ``itemData``          – (itemID, fieldID, valueID) pivot table
- ``fields``            – fieldID → fieldName
- ``itemDataValues``    – valueID → value (string)
- ``itemCreators``      – links items to creators (ordered)
- ``creators``          – firstName, lastName, creatorTypeID
- ``itemAttachments``   – child attachment rows (linkMode, contentType, path)
- ``fulltextItems``     – itemID → indexedChars / totalChars (coverage info)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection to the Zotero SQLite database.

    Parameters
    ----------
    db_path:
        Absolute path to ``zotero.sqlite``.

    Returns
    -------
    sqlite3.Connection
        The open connection (row_factory = sqlite3.Row).
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Research-item queries
# ---------------------------------------------------------------------------

#: Item types considered "research papers" for this pipeline.
PAPER_TYPES = (
    "journalArticle",
    "conferencePaper",
    "preprint",
    "report",
    "thesis",
    "manuscript",
    "bookSection",
    "book",
)


def fetch_papers(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Return a list of research-paper metadata dicts from the Zotero DB.

    Each dict contains at minimum:
    - ``itemID``, ``key``, ``itemType``
    - ``title``, ``year``, ``doi``, ``abstract``
    - ``authors`` (list of ``"Last, First"`` strings)

    Parameters
    ----------
    conn:
        Open SQLite connection.
    limit:
        If set, return at most *limit* items.
    since:
        If set, only items with ``dateAdded >= since`` (ISO 8601 date string,
        e.g. ``"2023-01-01"``).
    """
    placeholders = ",".join("?" * len(PAPER_TYPES))
    since_clause = "AND i.dateAdded >= ?" if since else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""
        SELECT i.itemID, i.key, it.typeName AS itemType,
               i.dateAdded
        FROM   items i
        JOIN   itemTypes it ON i.itemTypeID = it.itemTypeID
        WHERE  it.typeName IN ({placeholders})
               AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
               {since_clause}
        ORDER  BY i.itemID
        {limit_clause}
    """

    params: list[Any] = list(PAPER_TYPES)
    if since:
        params.append(since)

    rows = conn.execute(sql, params).fetchall()

    # Build a lookup: fieldName -> value for each itemID in one bulk query
    item_ids = [r["itemID"] for r in rows]
    if not item_ids:
        return []

    field_map = _fetch_field_map(conn, item_ids)
    creator_map = _fetch_creator_map(conn, item_ids)

    papers: list[dict[str, Any]] = []
    for row in rows:
        iid = row["itemID"]
        fmap = field_map.get(iid, {})
        papers.append(
            {
                "itemID": iid,
                "key": row["key"],
                "itemType": row["itemType"],
                "dateAdded": row["dateAdded"],
                "title": fmap.get("title", ""),
                "year": fmap.get("date", "")[:4] if fmap.get("date") else "",
                "doi": fmap.get("DOI", ""),
                "abstract": fmap.get("abstractNote", ""),
                "authors": creator_map.get(iid, []),
            }
        )
    return papers


def _fetch_field_map(
    conn: sqlite3.Connection, item_ids: list[int]
) -> dict[int, dict[str, str]]:
    """Return {itemID: {fieldName: value}} for all requested items."""
    if not item_ids:
        return {}
    placeholders = ",".join("?" * len(item_ids))
    sql = f"""
        SELECT id.itemID, f.fieldName, idv.value
        FROM   itemData id
        JOIN   fieldsCombined f   ON id.fieldID  = f.fieldID
        JOIN   itemDataValues idv ON id.valueID  = idv.valueID
        WHERE  id.itemID IN ({placeholders})
    """
    result: dict[int, dict[str, str]] = {}
    for r in conn.execute(sql, item_ids).fetchall():
        result.setdefault(r["itemID"], {})[r["fieldName"]] = r["value"]
    return result


def _fetch_creator_map(
    conn: sqlite3.Connection, item_ids: list[int]
) -> dict[int, list[str]]:
    """Return {itemID: ["Last, First", ...]} for all requested items."""
    if not item_ids:
        return {}
    placeholders = ",".join("?" * len(item_ids))
    sql = f"""
        SELECT ic.itemID, c.lastName, c.firstName
        FROM   itemCreators ic
        JOIN   creators c ON ic.creatorID = c.creatorID
        WHERE  ic.itemID IN ({placeholders})
        ORDER  BY ic.itemID, ic.orderIndex
    """
    result: dict[int, list[str]] = {}
    for r in conn.execute(sql, item_ids).fetchall():
        name = f"{r['lastName']}, {r['firstName']}".strip(", ")
        result.setdefault(r["itemID"], []).append(name)
    return result


# ---------------------------------------------------------------------------
# Attachment queries
# ---------------------------------------------------------------------------

def fetch_pdf_attachments(
    conn: sqlite3.Connection, item_id: int
) -> list[dict[str, Any]]:
    """Return PDF attachment rows for *item_id* ordered by preference.

    Preference: stored PDFs (linkMode 1 = imported_file, 2 = imported_url)
    over linked PDFs (linkMode 3 = linked_file).

    Each dict: ``itemID``, ``key``, ``path``, ``linkMode``.
    """
    sql = """
        SELECT ia.itemID, i.key, ia.path, ia.linkMode
        FROM   itemAttachments ia
        JOIN   items i ON ia.itemID = i.itemID
        WHERE  ia.parentItemID = ?
               AND ia.contentType = 'application/pdf'
               AND ia.itemID NOT IN (SELECT itemID FROM deletedItems)
        ORDER  BY CASE ia.linkMode WHEN 1 THEN 0 WHEN 2 THEN 1 ELSE 2 END
    """
    rows = conn.execute(sql, (item_id,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Full-text index query
# ---------------------------------------------------------------------------

def has_fulltext_index(conn: sqlite3.Connection, attachment_item_id: int) -> bool:
    """Return True if the attachment has a non-empty full-text index entry."""
    row = conn.execute(
        "SELECT indexedChars FROM fulltextItems WHERE itemID = ?",
        (attachment_item_id,),
    ).fetchone()
    return bool(row and row["indexedChars"] and row["indexedChars"] > 0)
