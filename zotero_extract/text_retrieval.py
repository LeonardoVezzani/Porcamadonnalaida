"""Text retrieval for Zotero items.

Priority order
--------------
1. Zotero full-text cache  – ``storage/<key>/.zotero-ft-cache``
2. PDF parsing             – ``storage/<key>/<filename>.pdf`` via PyMuPDF
3. Semantic Scholar        – abstract via DOI or title/author/year query

The module returns a ``TextResult`` namedtuple so callers always get the same
shape regardless of which source succeeded.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import NamedTuple

import requests

from .db import fetch_pdf_attachments, has_fulltext_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TextResult(NamedTuple):
    text: str
    method: str          # "cache" | "pdf" | "semanticscholar" | "none"
    no_text: bool        # True when no text could be retrieved


_EMPTY = TextResult(text="", method="none", no_text=True)

# Minimum character count to consider text usable
MIN_CHARS = 200

# ---------------------------------------------------------------------------
# 1. Zotero full-text cache
# ---------------------------------------------------------------------------

def _read_ft_cache(storage_dir: Path, attachment_key: str) -> str | None:
    """Read ``storage/<attachment_key>/.zotero-ft-cache`` and return its text.

    Returns None if the file does not exist or is too short.
    """
    cache_file = storage_dir / attachment_key / ".zotero-ft-cache"
    if not cache_file.is_file():
        return None
    text = cache_file.read_text(errors="replace")
    if len(text) < MIN_CHARS:
        return None
    return text


# ---------------------------------------------------------------------------
# 2. PDF parsing via PyMuPDF
# ---------------------------------------------------------------------------

def _parse_pdf(storage_dir: Path, attachment_key: str, path_field: str) -> str | None:
    """Extract text from a stored PDF using PyMuPDF (fitz).

    ``path_field`` is the ``itemAttachments.path`` value which has the form
    ``"storage:<filename>"`` for imported files.

    Returns None on failure or if the extracted text is too short.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF extraction")
        return None

    # Resolve the actual file path
    pdf_path = _resolve_pdf_path(storage_dir, attachment_key, path_field)
    if pdf_path is None or not pdf_path.is_file():
        # Try to find any PDF in the folder as a fallback
        pdfs = list((storage_dir / attachment_key).glob("*.pdf"))
        pdfs += list((storage_dir / attachment_key).glob("*.PDF"))
        if not pdfs:
            return None
        pdf_path = pdfs[0]

    try:
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n".join(pages)
    except Exception as exc:
        logger.debug("PDF parse error for %s: %s", pdf_path, exc)
        return None

    if len(text) < MIN_CHARS:
        return None
    return text


def _resolve_pdf_path(
    storage_dir: Path, attachment_key: str, path_field: str
) -> Path | None:
    """Convert a Zotero path field value to an absolute file path.

    Zotero stores paths as ``"storage:<filename>"`` for imported files
    (linkMode 1 or 2).  For linked files (linkMode 3) the path is absolute.
    """
    if path_field.startswith("storage:"):
        filename = path_field[len("storage:"):]
        return storage_dir / attachment_key / filename
    # Linked file – absolute path on disk
    p = Path(path_field)
    if p.is_file():
        return p
    return None


# ---------------------------------------------------------------------------
# 3. Semantic Scholar fallback
# ---------------------------------------------------------------------------

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "title,abstract,year,externalIds"
_SS_CACHE_FILE = ".ss_cache.json"


def _load_ss_cache(out_dir: Path) -> dict[str, str]:
    cache_path = out_dir / _SS_CACHE_FILE
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {}


def _save_ss_cache(out_dir: Path, cache: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _SS_CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False))


def _fetch_semantic_scholar(
    doi: str,
    title: str,
    authors: list[str],
    year: str,
    out_dir: Path,
    session: requests.Session | None = None,
) -> str | None:
    """Retrieve an abstract from Semantic Scholar; cache results in *out_dir*.

    Returns the abstract string or None.
    """
    cache_key = doi.strip() if doi.strip() else f"{title}|{year}"
    cache = _load_ss_cache(out_dir)
    if cache_key in cache:
        return cache[cache_key] or None

    sess = session or requests.Session()

    abstract: str | None = None
    try:
        if doi.strip():
            url = f"{_SS_BASE}/paper/DOI:{doi.strip()}"
            resp = sess.get(url, params={"fields": _SS_FIELDS}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("abstract") or None
            elif resp.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit; sleeping 10 s")
                time.sleep(10)
            elif resp.status_code != 404:
                logger.debug("S2 DOI lookup %s → %s", doi, resp.status_code)

        if abstract is None:
            query = title
            if authors:
                query += " " + authors[0].split(",")[0]
            url = f"{_SS_BASE}/paper/search"
            resp = sess.get(
                url,
                params={"query": query, "fields": _SS_FIELDS, "limit": 1},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("data", [])
                if hits:
                    abstract = hits[0].get("abstract") or None
            elif resp.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit; sleeping 10 s")
                time.sleep(10)
    except requests.RequestException as exc:
        logger.debug("Semantic Scholar request failed: %s", exc)

    cache[cache_key] = abstract or ""
    _save_ss_cache(out_dir, cache)
    return abstract


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve_text(
    paper: dict,
    conn,
    storage_dir: Path,
    out_dir: Path,
    ss_session: requests.Session | None = None,
) -> TextResult:
    """Retrieve text for *paper* following the priority chain.

    Parameters
    ----------
    paper:
        Dict as returned by ``db.fetch_papers()``.
    conn:
        Open SQLite connection to zotero.sqlite.
    storage_dir:
        Root of Zotero's ``storage/`` folder.
    out_dir:
        Output directory (used for Semantic Scholar cache).
    ss_session:
        Optional :class:`requests.Session` (allows sharing across calls).
    """
    item_id: int = paper["itemID"]
    attachments = fetch_pdf_attachments(conn, item_id)

    # -----------------------------------------------------------------------
    # 1. Zotero full-text cache
    # -----------------------------------------------------------------------
    for att in attachments:
        att_key: str = att["key"]
        if has_fulltext_index(conn, att["itemID"]):
            text = _read_ft_cache(storage_dir, att_key)
            if text:
                logger.debug("cache hit for item %s (attachment %s)", paper["key"], att_key)
                return TextResult(text=text, method="cache", no_text=False)
        # Also try the cache file even without a DB index entry
        text = _read_ft_cache(storage_dir, att_key)
        if text:
            logger.debug("cache file hit (no index) for item %s", paper["key"])
            return TextResult(text=text, method="cache", no_text=False)

    # -----------------------------------------------------------------------
    # 2. PDF parsing
    # -----------------------------------------------------------------------
    for att in attachments:
        att_key = att["key"]
        text = _parse_pdf(storage_dir, att_key, att.get("path") or "")
        if text:
            logger.debug("pdf parse for item %s (attachment %s)", paper["key"], att_key)
            return TextResult(text=text, method="pdf", no_text=False)

    # -----------------------------------------------------------------------
    # 3. Abstract from Zotero DB (may already be populated)
    # -----------------------------------------------------------------------
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) >= MIN_CHARS:
        logger.debug("using Zotero abstract for item %s", paper["key"])
        return TextResult(text=abstract, method="cache", no_text=False)

    # -----------------------------------------------------------------------
    # 4. Semantic Scholar
    # -----------------------------------------------------------------------
    ss_text = _fetch_semantic_scholar(
        doi=paper.get("doi") or "",
        title=paper.get("title") or "",
        authors=paper.get("authors") or [],
        year=paper.get("year") or "",
        out_dir=out_dir,
        session=ss_session,
    )
    if ss_text:
        logger.debug("semantic scholar hit for item %s", paper["key"])
        return TextResult(text=ss_text, method="semanticscholar", no_text=False)

    logger.info("no text found for item %s (%s)", paper["key"], paper.get("title", ""))
    return _EMPTY
