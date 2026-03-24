"""Tests for zotero_extract.db using a synthetic SQLite fixture."""

from __future__ import annotations

import sqlite3
import pytest
from zotero_extract.db import (
    fetch_papers,
    fetch_pdf_attachments,
    has_fulltext_index,
    open_db,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_minimal_db() -> sqlite3.Connection:
    """Build an in-memory Zotero-schema DB with minimal data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE itemTypes (
            itemTypeID INTEGER PRIMARY KEY,
            typeName   TEXT NOT NULL
        );
        CREATE TABLE libraries (
            libraryID INTEGER PRIMARY KEY,
            type      TEXT,
            editable  INT DEFAULT 1,
            filesEditable INT DEFAULT 1
        );
        CREATE TABLE items (
            itemID           INTEGER PRIMARY KEY,
            itemTypeID       INT NOT NULL,
            dateAdded        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dateModified     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            clientDateModified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            libraryID        INT NOT NULL DEFAULT 1,
            key              TEXT NOT NULL,
            version          INT NOT NULL DEFAULT 0,
            synced           INT NOT NULL DEFAULT 0
        );
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        CREATE TABLE fieldFormats (fieldFormatID INTEGER PRIMARY KEY, regex TEXT, isInteger INT);
        CREATE TABLE fields (
            fieldID    INTEGER PRIMARY KEY,
            fieldName  TEXT,
            fieldFormatID INT
        );
        CREATE TABLE fieldsCombined (
            fieldID    INTEGER PRIMARY KEY,
            fieldName  TEXT,
            label      TEXT,
            fieldFormatID INT,
            custom     INT NOT NULL DEFAULT 0
        );
        CREATE TABLE itemDataValues (
            valueID INTEGER PRIMARY KEY,
            value   TEXT UNIQUE
        );
        CREATE TABLE itemData (
            itemID   INT,
            fieldID  INT,
            valueID  INT,
            PRIMARY KEY (itemID, fieldID)
        );
        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        CREATE TABLE creators (
            creatorID  INTEGER PRIMARY KEY,
            firstName  TEXT,
            lastName   TEXT,
            fieldMode  INT
        );
        CREATE TABLE itemCreators (
            itemID      INT,
            creatorID   INT,
            creatorTypeID INT,
            orderIndex  INT DEFAULT 0,
            PRIMARY KEY (itemID, creatorID, creatorTypeID)
        );
        CREATE TABLE itemAttachments (
            itemID      INTEGER PRIMARY KEY,
            parentItemID INT,
            linkMode    INT,
            contentType TEXT,
            charsetID   INT,
            path        TEXT,
            syncState   INT DEFAULT 0,
            storageModTime INT,
            storageHash TEXT,
            lastProcessedModificationTime INT
        );
        CREATE TABLE fulltextItems (
            itemID       INTEGER PRIMARY KEY,
            indexedPages INT,
            totalPages   INT,
            indexedChars INT,
            totalChars   INT,
            version      INT NOT NULL DEFAULT 0,
            synced       INT NOT NULL DEFAULT 0
        );
    """)

    # Seed reference data
    conn.execute("INSERT INTO itemTypes VALUES (22, 'journalArticle')")
    conn.execute("INSERT INTO itemTypes VALUES (3, 'attachment')")
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")

    # Field IDs
    for fid, fname in [
        (1, "title"), (2, "date"), (3, "DOI"), (4, "abstractNote"),
    ]:
        conn.execute("INSERT INTO fields VALUES (?, ?, NULL)", (fid, fname))
        conn.execute(
            "INSERT INTO fieldsCombined VALUES (?, ?, ?, NULL, 0)",
            (fid, fname, fname),
        )

    # Parent item
    conn.execute(
        "INSERT INTO items (itemID, itemTypeID, libraryID, key) VALUES (1, 22, 1, 'ABC12345')"
    )
    for vid, val in [(1, "A Test Paper"), (2, "2023"), (3, "10.1/test"), (4, "Abstract text here.")]:
        conn.execute("INSERT INTO itemDataValues VALUES (?, ?)", (vid, val))
        conn.execute("INSERT INTO itemData VALUES (1, ?, ?)", (vid, vid))

    # Creator
    conn.execute("INSERT INTO creators VALUES (1, 'Jane', 'Doe', 0)")
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    conn.execute("INSERT INTO itemCreators VALUES (1, 1, 1, 0)")

    # Attachment item
    conn.execute(
        "INSERT INTO items (itemID, itemTypeID, libraryID, key) VALUES (2, 3, 1, 'ATT67890')"
    )
    conn.execute(
        "INSERT INTO itemAttachments VALUES (2, 1, 1, 'application/pdf', NULL, 'storage:test.pdf', 0, NULL, NULL, NULL)"
    )

    # Full-text index entry for attachment
    conn.execute(
        "INSERT INTO fulltextItems (itemID, indexedChars) VALUES (2, 5000)"
    )

    conn.commit()
    return conn


@pytest.fixture()
def db_conn():
    return _create_minimal_db()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOpenDb:
    def test_opens_real_file(self, tmp_path):
        db = tmp_path / "test.sqlite"
        c = sqlite3.connect(str(db))
        c.execute("CREATE TABLE t (x INT)")
        c.commit()
        c.close()
        conn = open_db(db)
        assert conn is not None
        conn.close()


class TestFetchPapers:
    def test_returns_journal_articles(self, db_conn):
        papers = fetch_papers(db_conn)
        assert len(papers) == 1
        p = papers[0]
        assert p["key"] == "ABC12345"
        assert p["title"] == "A Test Paper"
        assert p["year"] == "2023"
        assert p["doi"] == "10.1/test"
        assert p["authors"] == ["Doe, Jane"]

    def test_limit(self, db_conn):
        papers = fetch_papers(db_conn, limit=0)
        assert papers == []

    def test_since_filter(self, db_conn):
        papers = fetch_papers(db_conn, since="2099-01-01")
        assert papers == []

    def test_since_includes_past(self, db_conn):
        papers = fetch_papers(db_conn, since="2000-01-01")
        assert len(papers) == 1


class TestFetchPdfAttachments:
    def test_finds_pdf(self, db_conn):
        atts = fetch_pdf_attachments(db_conn, 1)
        assert len(atts) == 1
        assert atts[0]["key"] == "ATT67890"
        assert atts[0]["path"] == "storage:test.pdf"

    def test_no_attachments_for_unknown_item(self, db_conn):
        atts = fetch_pdf_attachments(db_conn, 9999)
        assert atts == []


class TestHasFulltextIndex:
    def test_true_for_indexed_attachment(self, db_conn):
        assert has_fulltext_index(db_conn, 2) is True

    def test_false_for_unknown(self, db_conn):
        assert has_fulltext_index(db_conn, 9999) is False
