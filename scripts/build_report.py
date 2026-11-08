"""Render reports/final_report.md to HTML and PDF (headless Chrome's print-to-PDF).

Usage: python -m scripts.build_report [--md reports/final_report.md]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 15mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #0b0b0b; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #ddd; padding-bottom: 1pt; }
h3 { font-size: 10pt; margin: 6pt 0 2pt 0; }
p, li { margin: 2pt 0; }
ul { margin: 2pt 0 2pt 0; padding-left: 15pt; }
table { border-collapse: collapse; margin: 4pt 0 6pt 0; font-size: 8pt; }
th, td { border: 1px solid #ccc; padding: 1.5pt 3.5pt; text-align: right; white-space: nowrap; }
th { background: #f3f2ee; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; display: block; margin: 4pt auto; }
code { font-size: 8.6pt; background: #f3f2ee; padding: 0 2px; }
.meta { color: #52514e; margin-bottom: 4pt; }
"""

CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]


def main(md_path: Path) -> None:
    html_body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code", "attr_list"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html)
    pdf_path = md_path.with_suffix(".pdf")
    chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        print(f"HTML written to {html_path}; no Chrome found - print it to PDF from a browser.")
        return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="reports/final_report.md")
    main(Path(ap.parse_args().md))
