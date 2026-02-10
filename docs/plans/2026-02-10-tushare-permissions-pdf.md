# TuShare Permissions Docs PDF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fetch the two TuShare permission documentation pages, remove the Table (2) independent-permission section from doc_id=290, and export a single combined PDF.

**Architecture:** Use Python (requests + BeautifulSoup) to download and clean HTML into one self-contained combined HTML file, then use locally installed Chrome headless `--print-to-pdf` to render the PDF.

**Tech Stack:** Python (venv), requests, bs4, lxml, Google Chrome headless.

### Task 1: Add The Export Script Skeleton

**Files:**
- Create: `tools/tushare_docs_to_pdf.py`
- Create: `docs/tushare/.gitkeep`

**Step 1: Create script with CLI args**
- Accept `--out-pdf`, `--out-html`, `--cache-dir`.
- Hardcode the two URLs as defaults (allow override via args).

**Step 2: Basic run scaffolding**
- Ensure cache dir exists.
- Download pages into cache.
- Write combined HTML to `--out-html`.
- Convert to PDF via Chrome to `--out-pdf`.

**Step 3: Smoke-run (expected to fail on missing Chrome path handling)**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --help`
Expected: exit 0.

### Task 2: Implement HTML Fetch + Clean + Combine

**Files:**
- Modify: `tools/tushare_docs_to_pdf.py`

**Step 1: Fetch HTML reliably**
- Use a browser-like `User-Agent`.
- Save raw HTML to cache (so runs are reproducible/offline).

**Step 2: Extract main article content**
- Parse with BeautifulSoup.
- Remove common chrome (nav/footer/scripts) and keep the content container.

**Step 3: Remove doc_id=290 Table (2) section**
- Locate the heading or label that matches `表（二）` or `独立权限接口`.
- Remove that heading and subsequent elements until the next same-level heading (or end).

**Step 4: Build combined HTML**
- Add a cover/title, date, and a simple TOC linking to each section.
- Add print-friendly CSS: `@page` margins, table borders, font-family with Chinese fonts.

**Step 5: Verify combined HTML renders**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --out-html docs/tushare/tushare_permissions.html --out-pdf NUL`
Expected: HTML created; script should skip PDF when out is `NUL`.

### Task 3: Implement Chrome Headless PDF Export

**Files:**
- Modify: `tools/tushare_docs_to_pdf.py`

**Step 1: Detect Chrome executable**
- Prefer `C:\Program Files\Google\Chrome\Application\chrome.exe`.
- Fallback to Edge `msedge.exe`.
- Allow override via `--chrome`.

**Step 2: Export PDF**
- Invoke Chrome with `--headless --disable-gpu --print-to-pdf=<path>`.
- Input should be the local combined HTML file via `file:///...`.

**Step 3: Verify PDF was created**

Run: `./venv/Scripts/python.exe tools/tushare_docs_to_pdf.py --out-html docs/tushare/tushare_permissions.html --out-pdf docs/tushare/tushare_permissions.pdf`
Expected: PDF exists and size > 10KB.

### Task 4: Add Convenience Wrapper (Optional)

**Files:**
- Create: `scripts/export_tushare_permissions_pdf.ps1`

**Step 1: Add a one-command PowerShell wrapper**
- Calls venv python and writes outputs into `docs/tushare/`.

**Step 2: Verify wrapper works**

Run: `powershell -ExecutionPolicy Bypass -File scripts/export_tushare_permissions_pdf.ps1`
Expected: PDF created.
