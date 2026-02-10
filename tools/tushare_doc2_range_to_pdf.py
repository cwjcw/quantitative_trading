#!/usr/bin/env python3
"""
Fetch https://tushare.pro/document/2?doc_id={MIN..MAX} and export a single PDF.

- Ignores 404 (page does not exist).
- Fails on other HTTP errors.
- Uses a local cache for reproducible/offline reruns.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


TUSHARE_DOC2 = "https://tushare.pro/document/2?doc_id={doc_id}"


@dataclass(frozen=True)
class Section:
    doc_id: int
    title: str
    url: str
    html_fragment: str


def resolve_chrome_executable(chrome_arg: str) -> str:
    p = Path(chrome_arg)
    if p.exists() and p.is_file():
        return str(p)

    w = shutil.which(chrome_arg)
    if w:
        return w

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    raise FileNotFoundError(
        "Could not find Chrome/Edge executable. Provide --chrome with a full path, e.g. "
        r"'C:\Program Files\Google\Chrome\Application\chrome.exe'."
    )


def cache_path_for_doc_id(cache_dir: Path, doc_id: int) -> Path:
    return cache_dir / f"document_2_doc_id_{doc_id}.html"


def fetch_to_cache(
    url: str,
    out_path: Path,
    *,
    force: bool,
    timeout_s: int,
    retries: int,
    sleep_s: float,
) -> Path | None:
    if out_path.exists() and not force:
        return out_path

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(r.text, encoding="utf-8")
            return out_path
        except Exception as e:
            last_exc = e
            time.sleep(min(8.0, 2 ** (attempt - 1)))
        finally:
            if sleep_s > 0:
                time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    return None


def make_links_absolute(container: BeautifulSoup, base_url: str) -> None:
    for a in container.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        a["href"] = urljoin(base_url, href)
    for img in container.select("img[src]"):
        src = img.get("src")
        if not src:
            continue
        img["src"] = urljoin(base_url, src)


def extract_main_content(html_text: str, *, base_url: str) -> BeautifulSoup:
    soup = BeautifulSoup(html_text, "lxml")
    main = soup.select_one("div.content") or soup.body or soup

    # Remove obvious chrome.
    for sel in ["nav", "footer", "script", "style", ".search-panel", ".search-container"]:
        for node in main.select(sel):
            node.decompose()

    make_links_absolute(main, base_url)
    return main


def title_from_main(main: BeautifulSoup, fallback: str) -> str:
    for sel in ["h1", "h2", "h3"]:
        h = main.select_one(sel)
        if h and h.get_text(strip=True):
            return h.get_text(" ", strip=True)
    return fallback


def build_section(doc_id: int, url: str, raw_html_path: Path) -> Section:
    raw_html = raw_html_path.read_text(encoding="utf-8")
    main = extract_main_content(raw_html, base_url=url)
    title = title_from_main(main, fallback=f"doc_id={doc_id}")
    fragment = "".join(str(x) for x in main.contents)
    return Section(doc_id=doc_id, title=title, url=url, html_fragment=fragment)


def build_combined_html(sections: list[Section], *, title: str) -> str:
    gen_time = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    toc_items = "\n".join(
        f"<li><a href='#doc_{s.doc_id}'>doc_id={s.doc_id} {s.title}</a></li>" for s in sections
    )
    blocks = []
    for s in sections:
        blocks.append(
            "<section class='doc'>"
            f"<h2 id='doc_{s.doc_id}'>doc_id={s.doc_id} {s.title}</h2>"
            f"<p class='source'>Source: <a href='{s.url}'>{s.url}</a></p>"
            f"{s.html_fragment}"
            "</section>"
        )

    css = r"""
    :root { --fg: #111; --muted: #666; --border: #cfcfcf; }
    @page { margin: 18mm; }
    body { color: var(--fg); font-family: "Microsoft YaHei", "PingFang SC", "SimSun", Arial, sans-serif; line-height: 1.4; }
    h1 { font-size: 22pt; margin: 0 0 8pt; }
    h2 { font-size: 12.5pt; margin: 14pt 0 6pt; }
    h3 { font-size: 11.5pt; margin: 12pt 0 6pt; }
    p, li { font-size: 10.5pt; }
    .meta { color: var(--muted); font-size: 9.5pt; margin: 0 0 12pt; }
    .source { color: var(--muted); font-size: 9.5pt; margin: 0 0 10pt; }
    .doc { page-break-before: always; }
    .doc:first-of-type { page-break-before: auto; }
    table { width: 100%; border-collapse: collapse; margin: 10pt 0; }
    th, td { border: 1px solid var(--border); padding: 6px 7px; vertical-align: top; }
    th { background: #f4f4f4; }
    a { color: #0b57d0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, pre { font-family: Consolas, "Courier New", monospace; }
    pre { white-space: pre-wrap; word-break: break-word; }
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        f"<title>{title}</title><style>{css}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='meta'>Generated: {gen_time}</p>"
        "<h2>Contents</h2>"
        f"<ul>{toc_items}</ul>"
        + "\n".join(blocks)
        + "</body></html>"
    )


def export_pdf(html_path: Path, pdf_path: Path, *, chrome: str) -> None:
    chrome_exe = resolve_chrome_executable(chrome)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome_exe,
        "--headless=new",
        "--disable-gpu",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={str(pdf_path.resolve())}",
        html_path.resolve().as_uri(),
    ]
    subprocess.check_call(cmd)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-id", type=int, default=1)
    ap.add_argument("--max-id", type=int, default=500)
    ap.add_argument("--cache-dir", default="docs/tushare/cache")
    ap.add_argument("--out-html", default="docs/tushare/document2_docid_1_500.html")
    ap.add_argument("--out-pdf", default="docs/tushare/document2_docid_1_500.pdf")
    ap.add_argument("--chrome", default="chrome")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.05, help="Polite sleep between requests")
    args = ap.parse_args(argv)

    if args.min_id < 1 or args.max_id < 1 or args.max_id < args.min_id:
        raise SystemExit("--min-id/--max-id must be positive and min<=max")

    cache_dir = Path(args.cache_dir)
    sections: list[Section] = []

    for doc_id in range(args.min_id, args.max_id + 1):
        url = TUSHARE_DOC2.format(doc_id=doc_id)
        out_path = cache_path_for_doc_id(cache_dir, doc_id)
        raw = fetch_to_cache(
            url,
            out_path,
            force=args.force,
            timeout_s=args.timeout,
            retries=args.retries,
            sleep_s=args.sleep,
        )
        if raw is None:
            continue
        sections.append(build_section(doc_id, url, raw))

    sections.sort(key=lambda s: s.doc_id)
    combined = build_combined_html(sections, title=f"TuShare document/2 doc_id {args.min_id}..{args.max_id}")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(combined, encoding="utf-8")

    export_pdf(out_html, Path(args.out_pdf), chrome=args.chrome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

