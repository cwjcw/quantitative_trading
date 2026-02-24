#!/usr/bin/env python3
"""
Crawl XtQuant Native API docs and export all pages to a single PDF.

Default scope:
- Host: dict.thinktrader.net
- Path prefix: /nativeApi/
- Entry URL: https://dict.thinktrader.net/nativeApi/start_now.html?id=5M2071
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


DEFAULT_START_URL = "https://dict.thinktrader.net/nativeApi/start_now.html?id=5M2071"


@dataclass(frozen=True)
class PageSection:
    order: int
    url: str
    title: str
    html_fragment: str


def resolve_chrome_executable(chrome_arg: str) -> str:
    p = Path(chrome_arg)
    if p.exists() and p.is_file():
        return str(p)

    found = shutil.which(chrome_arg)
    if found:
        return found

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    raise FileNotFoundError(
        "Could not find Chrome/Edge executable. Provide --chrome with full path."
    )


def canonicalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    netloc = p.netloc.lower()
    path = p.path or "/"

    # Drop tracking-style query param "id", keep all others in stable order.
    query_items = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() != "id"]
    query_items.sort()
    query = urlencode(query_items, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def is_allowed_doc_url(url: str, *, allow_host: str, allow_prefix: str) -> bool:
    p = urlparse(url)
    if p.netloc.lower() != allow_host.lower():
        return False
    if not p.path.startswith(allow_prefix):
        return False
    if not p.path.endswith(".html"):
        return False
    return True


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    p = urlparse(url)
    base = p.path.strip("/").replace("/", "__") or "index"
    if not base.endswith(".html"):
        base += ".html"
    if p.query:
        qhash = hashlib.sha1(p.query.encode("utf-8")).hexdigest()[:10]
        base = base[:-5] + f"__q_{qhash}.html"
    return cache_dir / base


def fetch_to_cache(
    url: str,
    out_path: Path,
    *,
    force: bool,
    timeout_s: int,
    retries: int,
    sleep_s: float,
) -> Path:
    if out_path.exists() and not force:
        return out_path

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout_s)
            resp.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(resp.text, encoding="utf-8")
            return out_path
        except Exception as exc:
            last_exc = exc
            time.sleep(min(8.0, 2 ** (attempt - 1)))
        finally:
            if sleep_s > 0:
                time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to fetch: {url}")


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


def clean_heading_text(text: str) -> str:
    cleaned = re.sub(r"^\s*#+\s*", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_main_content(html_text: str, *, base_url: str) -> BeautifulSoup:
    soup = BeautifulSoup(html_text, "lxml")
    main = soup.select_one("main.page") or soup.select_one("main") or soup.body or soup

    # Remove site chrome while preserving article content.
    for sel in [
        "script",
        "style",
        "nav",
        "footer",
        ".navbar",
        ".sidebar",
        ".page-meta",
        ".page-nav",
        ".back-to-top",
        ".search-box",
    ]:
        for node in main.select(sel):
            node.decompose()

    make_links_absolute(main, base_url)
    return main


def title_from_page(main: BeautifulSoup, raw_html: str, fallback: str) -> str:
    for sel in ["h1", "h2", "h3"]:
        h = main.select_one(sel)
        if h and h.get_text(strip=True):
            t = clean_heading_text(h.get_text(" ", strip=True))
            if t:
                return t

    soup = BeautifulSoup(raw_html, "lxml")
    if soup.title and soup.title.get_text(strip=True):
        t = soup.title.get_text(" ", strip=True)
        t = t.split("|")[0].strip()
        if t:
            return t

    return fallback


def build_section(order: int, url: str, raw_html_path: Path) -> PageSection:
    raw_html = raw_html_path.read_text(encoding="utf-8")
    main = extract_main_content(raw_html, base_url=url)
    title = title_from_page(main, raw_html, fallback=urlparse(url).path)
    fragment = "".join(str(x) for x in main.contents)
    return PageSection(order=order, url=url, title=title, html_fragment=fragment)


def discover_links(
    *,
    page_url: str,
    html_text: str,
    allow_host: str,
    allow_prefix: str,
) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    found: list[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        absolute = canonicalize_url(urljoin(page_url, href))
        if is_allowed_doc_url(absolute, allow_host=allow_host, allow_prefix=allow_prefix):
            found.append(absolute)
    # Keep stable order and de-duplicate.
    return list(dict.fromkeys(found))


def crawl_sections(
    *,
    start_url: str,
    allow_host: str,
    allow_prefix: str,
    cache_dir: Path,
    max_pages: int,
    force: bool,
    timeout_s: int,
    retries: int,
    sleep_s: float,
) -> list[PageSection]:
    start = canonicalize_url(start_url)
    if not is_allowed_doc_url(start, allow_host=allow_host, allow_prefix=allow_prefix):
        raise ValueError("Start URL is outside the allowed crawl scope.")

    queue: deque[str] = deque([start])
    seen: set[str] = {start}
    sections: list[PageSection] = []

    while queue and len(sections) < max_pages:
        url = queue.popleft()
        raw_path = cache_path_for_url(cache_dir, url)
        fetch_to_cache(
            url,
            raw_path,
            force=force,
            timeout_s=timeout_s,
            retries=retries,
            sleep_s=sleep_s,
        )
        raw_html = raw_path.read_text(encoding="utf-8")
        section = build_section(len(sections) + 1, url, raw_path)
        sections.append(section)

        for nxt in discover_links(
            page_url=url,
            html_text=raw_html,
            allow_host=allow_host,
            allow_prefix=allow_prefix,
        ):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    # Keep first page as the explicit start, sort remaining by path for stable output.
    if not sections:
        return sections
    head = sections[0]
    tail = sorted(sections[1:], key=lambda s: urlparse(s.url).path)
    return [head] + tail


def build_combined_html(sections: list[PageSection], *, title: str, start_url: str) -> str:
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    toc_items = "\n".join(
        f"<li><a href='#page_{i}'>{i}. {s.title}</a></li>"
        for i, s in enumerate(sections, start=1)
    )
    blocks = []
    for i, s in enumerate(sections, start=1):
        blocks.append(
            "<section class='doc-page'>"
            f"<h2 id='page_{i}'>{i}. {s.title}</h2>"
            f"<p class='source'>Source: <a href='{s.url}'>{s.url}</a></p>"
            f"{s.html_fragment}"
            "</section>"
        )

    css = r"""
    :root { --fg: #111; --muted: #666; --border: #d0d0d0; }
    @page { margin: 16mm; }
    body {
      color: var(--fg);
      font-family: "Microsoft YaHei", "PingFang SC", "SimSun", Arial, sans-serif;
      line-height: 1.45;
    }
    h1 { font-size: 22pt; margin: 0 0 8pt; }
    h2 { font-size: 13pt; margin: 14pt 0 8pt; }
    h3 { font-size: 11.5pt; margin: 10pt 0 6pt; }
    p, li { font-size: 10.5pt; }
    .meta, .source { color: var(--muted); font-size: 9.5pt; }
    .doc-page { page-break-before: always; }
    .doc-page:first-of-type { page-break-before: auto; }
    table { width: 100%; border-collapse: collapse; margin: 10pt 0; }
    th, td { border: 1px solid var(--border); padding: 6px 7px; vertical-align: top; }
    th { background: #f4f4f4; }
    pre, code {
      font-family: Consolas, "Courier New", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    a { color: #0b57d0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        f"<title>{title}</title><style>{css}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='meta'>Generated: {generated}</p>"
        f"<p class='meta'>Entry: <a href='{start_url}'>{start_url}</a></p>"
        f"<p class='meta'>Pages crawled: {len(sections)}</p>"
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
    ap.add_argument("--start-url", default=DEFAULT_START_URL)
    ap.add_argument("--allow-host", default="dict.thinktrader.net")
    ap.add_argument("--allow-prefix", default="/nativeApi/")
    ap.add_argument("--max-pages", type=int, default=300)
    ap.add_argument("--cache-dir", default="cache/xtquant_nativeapi/raw")
    ap.add_argument("--out-html", default="cache/xtquant_nativeapi_docs.html")
    ap.add_argument("--out-pdf", default="cache/xtquant_nativeapi_docs.pdf")
    ap.add_argument("--chrome", default="chrome")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args(argv)

    sections = crawl_sections(
        start_url=args.start_url,
        allow_host=args.allow_host,
        allow_prefix=args.allow_prefix,
        cache_dir=Path(args.cache_dir),
        max_pages=args.max_pages,
        force=args.force,
        timeout_s=args.timeout,
        retries=args.retries,
        sleep_s=args.sleep,
    )
    if not sections:
        raise RuntimeError("No pages crawled. Check URL/scope settings.")

    title = "XtQuant Native API Documentation (Merged)"
    merged_html = build_combined_html(sections, title=title, start_url=args.start_url)

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(merged_html, encoding="utf-8")

    export_pdf(out_html, Path(args.out_pdf), chrome=args.chrome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
