#!/usr/bin/env python3
"""Build the collaborator report DOCX from research-narrative.md via Pandoc.

Source of truth: research-narrative.md (+ figures/).
Style reference: the manually revised DOCX export (kept as reference-doc).
Figures 4–7 can be regenerated with generate_figures.py before building.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "research-narrative.md"
REFERENCE = ROOT / (
    "From Stalling to Coordinated Preparation–Content Control in "
    "Gemma 3 270M — Revised for Collaborators.docx"
)
BUILD_DIR = ROOT / "build"
OUTPUT = ROOT / "collaborator-report-v3.docx"
BUILD_COPY = BUILD_DIR / "collaborator-report-v3.docx"


def build() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    BUILD_DIR.mkdir(exist_ok=True)
    cmd = [
        "pandoc",
        str(SOURCE),
        "--from",
        "markdown+tex_math_single_backslash+tex_math_dollars",
        "--to",
        "docx",
        "--resource-path",
        str(ROOT),
        "-o",
        str(OUTPUT),
    ]
    if REFERENCE.exists():
        cmd.extend(["--reference-doc", str(REFERENCE)])
    subprocess.run(cmd, check=True, cwd=ROOT)
    BUILD_DIR.mkdir(exist_ok=True)
    shutil.copy2(OUTPUT, BUILD_COPY)

    # Verify Word math landed for display equations.
    with zipfile.ZipFile(OUTPUT) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    has_omath = "<m:oMath" in xml or "<m:oMathPara" in xml
    raw_dollars = "$$" in xml
    print(OUTPUT)
    print(f"oMath_present={has_omath} raw_dollar_delimiters={raw_dollars}")
    if not has_omath:
        print(
            "WARNING: no Word math tags found; equations may be plain text. "
            "Check Pandoc/math setup."
        )
    return OUTPUT


if __name__ == "__main__":
    build()
