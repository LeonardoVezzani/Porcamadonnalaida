# Porcamadonnalaida

## Zotero Metadata Extraction Pipeline

A local-only Python pipeline that reads directly from your Zotero SQLite database and storage folder to extract driving-experiment metadata from scientific papers — no Zotero Web API key required.

### What it does

For each research paper in your Zotero library the pipeline:

1. **Retrieves text** in priority order:
   - Zotero's full-text cache (`storage/<key>/.zotero-ft-cache`)
   - PDF parsing via PyMuPDF (falls back to this when no cache exists)
   - Semantic Scholar abstract retrieval (for papers with no local text)
2. **Extracts** four driving-experiment attributes using keyword-based rules:
   - `task_type` — e.g. `takeover_request`, `lane_keeping`, `hazard_detection`
   - `driving_scenario` — e.g. `highway`, `urban`, `simulator`, `on_road`
   - `level_of_automation` — SAE levels `SAE_L0`–`SAE_L5` or `assisted`
   - `visual_cue` — e.g. `HUD`, `AR_display`, `ambient_light`, `auditory`
   - `confidence` — 0–1 numeric score
3. **Writes**:
   - `out/json/<itemKey>.json` — one JSON file per paper
   - `out/results_summary.csv` — one row per paper, including a `no_text` flag

---

### Setup

#### 1. Prerequisites

- Python 3.10 or later
- [pip](https://pip.pypa.io/)

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Locate your Zotero files

| Platform | Default path |
|----------|-------------|
| **macOS** | `~/Zotero/` |
| **Windows** | `%USERPROFILE%\Zotero\` |
| **Linux** | `~/Zotero/` |

Inside that folder you will find:
- `zotero.sqlite` — the main database
- `storage/` — attachment files and full-text caches

---

### Usage

```bash
python -m zotero_extract \
    --db      /path/to/zotero.sqlite \
    --storage /path/to/Zotero/storage \
    --out     out/
```

#### All options

| Flag | Description | Default |
|------|-------------|---------|
| `--db PATH` | Path to `zotero.sqlite` | *(required)* |
| `--storage PATH` | Path to Zotero `storage/` folder | *(required)* |
| `--out PATH` | Output directory | `out/` |
| `--limit N` | Process at most *N* papers | all |
| `--since YYYY-MM-DD` | Only papers added on or after this date | all |
| `--log-level LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |
| `--no-semantic-scholar` | Skip Semantic Scholar fallback | off |

#### macOS quick start

```bash
python -m zotero_extract \
    --db ~/Zotero/zotero.sqlite \
    --storage ~/Zotero/storage \
    --out out/
```

#### Windows quick start (PowerShell)

```powershell
python -m zotero_extract `
    --db "$env:USERPROFILE\Zotero\zotero.sqlite" `
    --storage "$env:USERPROFILE\Zotero\storage" `
    --out out/
```

---

### Output format

#### `out/json/<itemKey>.json`

```json
{
  "item_key": "K3EJWPQA",
  "item_id": 9,
  "title": "On the Road to Productivity …",
  "year": "2023",
  "doi": "10.1145/3626705.3627787",
  "authors": ["Patel, Shiv G", "Dufresne-Camaro, Charles-Olivier"],
  "item_type": "conferencePaper",
  "retrieval_method": "pdf",
  "no_text": false,
  "task_type": "takeover_request",
  "driving_scenario": "simulator",
  "level_of_automation": "SAE_L0",
  "visual_cue": "HUD",
  "confidence": 0.729
}
```

#### `out/results_summary.csv`

One row per paper with columns: `item_key`, `item_id`, `title`, `year`, `doi`,
`authors`, `item_type`, `retrieval_method`, `no_text`, `task_type`,
`driving_scenario`, `level_of_automation`, `visual_cue`, `confidence`.

Papers where no text could be retrieved have `no_text = True`.

---

### Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

### Project layout

```
zotero_extract/
  __init__.py        package metadata
  __main__.py        CLI entry point
  db.py              SQLite helpers (item queries, attachments, fulltext index)
  text_retrieval.py  text retrieval chain (cache → PDF → Semantic Scholar)
  extractor.py       keyword-based metadata extraction
  output.py          JSON + CSV writers
tests/
  test_db.py
  test_extractor.py
  test_text_retrieval.py
  test_output.py
requirements.txt
requirements-dev.txt
```
