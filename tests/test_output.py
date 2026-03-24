"""Tests for output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from zotero_extract.output import build_row, write_json, write_summary_csv


_PAPER = {
    "itemID": 42,
    "key": "OUTTEST1",
    "title": "A great driving study",
    "year": "2023",
    "doi": "10.1/great",
    "authors": ["Smith, John", "Doe, Jane"],
    "itemType": "journalArticle",
}

_EXTRACTION = {
    "task_type": "takeover_request",
    "driving_scenario": "highway",
    "level_of_automation": "SAE_L3",
    "visual_cue": "HUD",
    "confidence": 0.75,
}


class TestWriteJson:
    def test_creates_file(self, tmp_path):
        path = write_json(tmp_path, _PAPER, _EXTRACTION, "cache", False)
        assert path.exists()
        assert path.name == "OUTTEST1.json"

    def test_json_has_required_fields(self, tmp_path):
        path = write_json(tmp_path, _PAPER, _EXTRACTION, "cache", False)
        data = json.loads(path.read_text())
        for field in [
            "item_key", "item_id", "title", "year", "doi", "authors",
            "item_type", "retrieval_method", "no_text",
            "task_type", "driving_scenario", "level_of_automation",
            "visual_cue", "confidence",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_no_text_flag(self, tmp_path):
        path = write_json(tmp_path, _PAPER, _EXTRACTION, "none", True)
        data = json.loads(path.read_text())
        assert data["no_text"] is True


class TestWriteSummaryCsv:
    def test_creates_csv(self, tmp_path):
        row = build_row(_PAPER, _EXTRACTION, "cache", False)
        csv_path = write_summary_csv(tmp_path, [row])
        assert csv_path.exists()
        assert csv_path.name == "results_summary.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        row = build_row(_PAPER, _EXTRACTION, "cache", False)
        csv_path = write_summary_csv(tmp_path, [row])
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
        expected = [
            "item_key", "item_id", "title", "year", "doi", "authors",
            "item_type", "retrieval_method", "no_text",
            "task_type", "driving_scenario", "level_of_automation",
            "visual_cue", "confidence",
        ]
        for col in expected:
            assert col in cols, f"Missing column: {col}"

    def test_authors_joined_with_semicolon(self, tmp_path):
        row = build_row(_PAPER, _EXTRACTION, "cache", False)
        assert row["authors"] == "Smith, John; Doe, Jane"

    def test_empty_rows(self, tmp_path):
        csv_path = write_summary_csv(tmp_path, [])
        with csv_path.open() as fh:
            lines = fh.readlines()
        # Header only
        assert len(lines) == 1
