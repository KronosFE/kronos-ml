"""Shared helpers for the KODEX draft-first deposits.

Builds a clean release archive and defines the attach list + metadata. Imported by
zenodo_draft.py / osf_draft.py / figshare_draft.py. NONE of those scripts ever call
a venue's publish/complete endpoint — they create a DRAFT and upload files, then
print the draft URL for the founder to review and publish herself.
"""
from __future__ import annotations
import json, os, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUB = REPO / "publish"
VERSION = json.load(open(PUB / "_scaffolding_summary.json"))["version"]
ARCHIVE = PUB / f"kronos-ml-v{VERSION}-source.zip"

# individual human-readable artifacts attached alongside the source archive
DOCS = ["README.md", "FAMILY.md", "CODES.md", "BENCHMARKS.md", "benchmarks/benchmarks.json",
        "demo.ipynb", "CITATION.cff", "LICENSE", "codemeta.json", ".zenodo.json",
        "publish/PROVENANCE.md", "publish/MANIFEST.sha256"]

_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", "dist"}
_SKIP_SUFFIX = {".pyc", ".pyo"}


def build_archive() -> Path:
    """Zip the whole package (source + tests + docs + manifest), minus cruft."""
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fn in files:
                p = Path(root) / fn
                if p.suffix in _SKIP_SUFFIX or p == ARCHIVE:
                    continue
                if p.name.endswith(".parquet"):
                    continue
                z.write(p, p.relative_to(REPO))
    return ARCHIVE


def attach_paths() -> list[Path]:
    """Full attach set: the source archive + the individual docs f7 asked for."""
    out = [build_archive()]
    for rel in DOCS:
        p = REPO / rel
        if p.exists():
            out.append(p)
    return out


def zenodo_metadata() -> dict:
    return json.load(open(REPO / ".zenodo.json"))


def require_env(var: str) -> str:
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"[STOP] environment variable {var} is not set. "
                         f"Export your token first (see PUBLISH_RUNBOOK.md).")
    return v


if __name__ == "__main__":
    paths = attach_paths()
    print(f"archive: {ARCHIVE.name}")
    print(f"attach set ({len(paths)} files):")
    for p in paths:
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size:,} B)")
