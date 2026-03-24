"""Tests for text retrieval helpers (PDF mock + Semantic Scholar mock)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zotero_extract.text_retrieval import (
    MIN_CHARS,
    TextResult,
    _read_ft_cache,
    _parse_pdf,
    _fetch_semantic_scholar,
    retrieve_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(**kwargs):
    defaults = {
        "itemID": 1,
        "key": "KEY00001",
        "title": "A Test Paper About Lane Keeping",
        "year": "2023",
        "doi": "",
        "abstract": "",
        "authors": ["Doe, Jane"],
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# 1. Zotero full-text cache
# ---------------------------------------------------------------------------

class TestReadFtCache:
    def test_returns_text_from_cache(self, tmp_path):
        att_dir = tmp_path / "ABCDEF"
        att_dir.mkdir()
        cache_file = att_dir / ".zotero-ft-cache"
        content = "x" * 300
        cache_file.write_text(content)
        result = _read_ft_cache(tmp_path, "ABCDEF")
        assert result == content

    def test_returns_none_when_file_missing(self, tmp_path):
        assert _read_ft_cache(tmp_path, "MISSING") is None

    def test_returns_none_when_too_short(self, tmp_path):
        att_dir = tmp_path / "SHORT"
        att_dir.mkdir()
        (att_dir / ".zotero-ft-cache").write_text("hi")
        assert _read_ft_cache(tmp_path, "SHORT") is None


# ---------------------------------------------------------------------------
# 2. PDF parsing
# ---------------------------------------------------------------------------

class TestParsePdf:
    def test_parse_pdf_success(self, tmp_path):
        att_dir = tmp_path / "ATT001"
        att_dir.mkdir()

        long_text = "This paper studies lane keeping on the highway. " * 10

        mock_page = MagicMock()
        mock_page.get_text.return_value = long_text
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        # Create a dummy PDF so the file exists
        pdf_path = att_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("fitz.open", return_value=mock_doc):
            result = _parse_pdf(tmp_path, "ATT001", "storage:test.pdf")

        assert result is not None
        assert "lane keeping" in result

    def test_parse_pdf_returns_none_on_too_short(self, tmp_path):
        att_dir = tmp_path / "ATT002"
        att_dir.mkdir()

        mock_page = MagicMock()
        mock_page.get_text.return_value = "short"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        pdf_path = att_dir / "small.pdf"
        pdf_path.write_bytes(b"%PDF")

        with patch("fitz.open", return_value=mock_doc):
            result = _parse_pdf(tmp_path, "ATT002", "storage:small.pdf")

        assert result is None

    def test_parse_pdf_missing_file(self, tmp_path):
        result = _parse_pdf(tmp_path, "NODIR", "storage:absent.pdf")
        assert result is None


# ---------------------------------------------------------------------------
# 3. Semantic Scholar
# ---------------------------------------------------------------------------

class TestFetchSemanticScholar:
    def test_doi_lookup_success(self, tmp_path, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/test",
            json={
                "title": "A Test Paper",
                "abstract": "This paper studies lane keeping on highway. " * 5,
            },
        )
        abstract = _fetch_semantic_scholar(
            doi="10.1/test",
            title="A Test Paper",
            authors=["Doe, Jane"],
            year="2023",
            out_dir=tmp_path,
        )
        assert abstract is not None
        assert "lane keeping" in abstract

    def test_title_query_fallback(self, tmp_path, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:",
            status_code=404,
        )
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            json={
                "data": [
                    {"abstract": "Studied car following on highway. " * 5}
                ]
            },
        )
        abstract = _fetch_semantic_scholar(
            doi="",
            title="Some Paper Title",
            authors=["Smith, John"],
            year="2021",
            out_dir=tmp_path,
        )
        assert abstract is not None
        assert "car following" in abstract

    def test_cache_prevents_second_request(self, tmp_path, requests_mock):
        abstract_text = "Cached abstract about takeover requests. " * 5
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.2/cached",
            json={"abstract": abstract_text},
        )
        # First call
        r1 = _fetch_semantic_scholar("10.2/cached", "T", [], "2020", tmp_path)
        # Second call should hit cache, not the network
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.2/cached",
            status_code=500,  # Would fail if called
        )
        r2 = _fetch_semantic_scholar("10.2/cached", "T", [], "2020", tmp_path)
        assert r1 == r2

    def test_returns_none_on_404(self, tmp_path, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.3/missing",
            status_code=404,
        )
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            json={"data": []},
        )
        result = _fetch_semantic_scholar(
            doi="10.3/missing", title="Unknown", authors=[], year="", out_dir=tmp_path
        )
        assert result is None


# ---------------------------------------------------------------------------
# 4. retrieve_text priority chain
# ---------------------------------------------------------------------------

class TestRetrieveText:
    def _mock_conn_no_attachments(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.fetchone.return_value = None
        return conn

    def test_uses_cache_first(self, tmp_path):
        paper = _make_paper(key="CACHEKEY")
        att_key = "ATTKEY01"
        att_dir = tmp_path / att_key
        att_dir.mkdir()
        (att_dir / ".zotero-ft-cache").write_text("x" * 500)

        import zotero_extract.text_retrieval as tr
        with patch.object(tr, "fetch_pdf_attachments", return_value=[
            {"itemID": 10, "key": att_key, "path": "storage:file.pdf", "linkMode": 1}
        ]):
            with patch.object(tr, "has_fulltext_index", return_value=True):
                result = retrieve_text(paper, MagicMock(), tmp_path, tmp_path)

        assert result.method == "cache"
        assert not result.no_text

    def test_falls_back_to_pdf(self, tmp_path):
        paper = _make_paper(key="PDFKEY00")
        att_key = "ATTKEY02"

        import zotero_extract.text_retrieval as tr
        with patch.object(tr, "fetch_pdf_attachments", return_value=[
            {"itemID": 11, "key": att_key, "path": "storage:paper.pdf", "linkMode": 1}
        ]):
            with patch.object(tr, "has_fulltext_index", return_value=False):
                with patch.object(tr, "_parse_pdf", return_value="long pdf text " * 20):
                    result = retrieve_text(paper, MagicMock(), tmp_path, tmp_path)

        assert result.method == "pdf"
        assert not result.no_text

    def test_returns_none_result_when_all_fail(self, tmp_path):
        paper = _make_paper(key="NOTEXT00")

        import zotero_extract.text_retrieval as tr
        with patch.object(tr, "fetch_pdf_attachments", return_value=[]):
            with patch.object(tr, "_fetch_semantic_scholar", return_value=None):
                result = retrieve_text(paper, MagicMock(), tmp_path, tmp_path)

        assert result.method == "none"
        assert result.no_text is True
