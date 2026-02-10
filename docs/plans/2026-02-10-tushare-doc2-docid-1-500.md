# TuShare document/2 doc_id 1..500 PDF Export Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fetch `https://tushare.pro/document/2?doc_id={1..500}` (ignore 404s) and merge all existing pages into a single PDF.

**Architecture:** Add a scan mode to `tools/tushare_docs_to_pdf.py` that iterates doc_id 1..500, downloads each page to cache, skips 404, converts each to a cleaned section, then exports combined HTML and renders PDF with Chrome headless.

**Tech Stack:** Python (venv), requests, bs4+lxml, Chrome headless.

### Task 1: Add DocID Range Scan Mode

**Files:**
- Modify: `tools/tushare_docs_to_pdf.py`

**Step 1: Add CLI flags**
- `--no-default-urls`
- `--doc2-scan-range <min> <max>`
- `--ignore-http-status <code>` (repeatable; default includes 404 for scan)

**Step 2: Implement scan loop**
- For each doc_id in range: fetch with caching; if HTTP status is in ignore list, skip.
- Keep stable ordering by doc_id.

**Step 3: Quick verification (small range)**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --no-default-urls --doc2-scan-range 1 10 --out-html docs/tushare/doc2_1_10.html --out-pdf NUL`
Expected: HTML created; no crash if some doc_id are 404.

### Task 2: Add Convenience Wrapper

**Files:**
- Create: `scripts/export_tushare_document2_docid_1_500_pdf.ps1`

**Step 1: Wrapper**
- Calls python with `--no-default-urls --doc2-scan-range 1 500` and outputs into `docs/tushare/`.

**Step 2: Full export verification**

Run: `powershell -ExecutionPolicy Bypass -File scripts/export_tushare_document2_docid_1_500_pdf.ps1`
Expected: `docs/tushare/document2_docid_1_500.pdf` exists and size > 100KB.
