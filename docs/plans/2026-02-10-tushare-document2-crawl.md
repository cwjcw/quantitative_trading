# TuShare document/2 Crawl Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When exporting the combined TuShare permissions PDF, also crawl and include all child pages linked from `https://tushare.pro/document/2` (the `doc_id=...` pages).

**Architecture:** Add a small crawler that starts at `document/2`, discovers all `document/2?doc_id=...` links (from the page nav + content), downloads them with caching and de-duplication, then appends them as additional sections in the combined HTML before printing to PDF.

**Tech Stack:** Python (venv), requests, bs4+lxml, Chrome headless.

### Task 1: Add Crawler Options + URL Discovery

**Files:**
- Modify: `tools/tushare_docs_to_pdf.py`

**Step 1: Add CLI flags**
- Add `--crawl-doc2` (default on) and `--max-doc2-pages`.

**Step 2: Implement URL normalization/filtering**
- Only keep links on `tushare.pro` with path `/document/2`.
- Normalize by removing fragments and sorting query where needed.

**Step 3: Implement discovery**
- Parse raw HTML and collect `a[href]` links matching `/document/2?doc_id=...`.

**Step 4: Quick verify discovery**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --out-html docs/tushare/tushare_permissions.html --out-pdf NUL --max-doc2-pages 5`
Expected: HTML created and includes 5+ doc sections from document/2.

### Task 2: Crawl + Stable Ordering

**Files:**
- Modify: `tools/tushare_docs_to_pdf.py`

**Step 1: BFS crawl**
- Start queue with `https://tushare.pro/document/2`.
- For each visited page: fetch (cache) and discover more.

**Step 2: Stable ordering**
- Sort sections so the root `document/2` comes first, then by numeric `doc_id`.

**Step 3: Verify table(2) removal still works for doc_id=290**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --out-html docs/tushare/tushare_permissions.html --out-pdf NUL --max-doc2-pages 5`
Expected: Combined HTML contains `表（一）` and does NOT contain `表（二）` or `独立权限接口`.

### Task 3: Full Export Verification

**Files:**
- Modify: `scripts/export_tushare_permissions_pdf.ps1` (only if needed)

**Step 1: Export PDF**

Run: `powershell -ExecutionPolicy Bypass -File scripts/export_tushare_permissions_pdf.ps1`
Expected: `docs/tushare/tushare_permissions.pdf` exists and size increases vs prior run.
