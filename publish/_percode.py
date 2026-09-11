"""Per-code deposit engine for KODEX (26 BUILT codes -> 26 records per data venue).

Generates publish/records/<kname>/ (metadata + per-code attach set) and provides the
shared helpers the venue scripts use to loop. State (created draft ids/DOIs) is written
to publish/state/<venue>_drafts.json so a run can resume and the batch-publish step
knows exactly what to act on.

DRAFT-FIRST: nothing here publishes. The *_draft.py scripts create drafts; the separate
*_publish_all.py scripts publish only drafts listed in the state file, only with --yes.
"""
from __future__ import annotations
import hashlib, html, inspect, json, os, random, re, shutil, sys, time
from datetime import date
from pathlib import Path

_RETRY_CODES = {403, 408, 425, 429, 500, 502, 503, 504}


def http_retry(fn, *, tries=6, base=0.8, label=""):
    """Call fn() -> requests.Response; retry transient/rate-limit codes with expo backoff.
    Returns the final Response (caller still checks .ok / raises). 2xx returns immediately."""
    r = None
    for i in range(tries):
        r = fn()
        if 200 <= r.status_code < 300 or r.status_code not in _RETRY_CODES or i == tries - 1:
            return r
        wait = base * (2 ** i) + random.random() * 0.4
        sys.stderr.write(f"  [retry] {label} {r.status_code}; sleep {wait:.1f}s ({i+1}/{tries})\n")
        time.sleep(wait)
    return r

REPO = Path(__file__).resolve().parent.parent
PUB = REPO / "publish"
REC_ROOT = PUB / "records"
STATE_ROOT = PUB / "state"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO.parent / "kronos-toolkit"))
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings; warnings.filterwarnings("ignore")
import kronos_ml as K

TODAY = date.today().isoformat()
VERSION = K.__version__
ORCID = "0000-0003-0395-1752"
AFFIL = "Kronos Fusion Energy"
KODEX = "https://kronosfusionenergy.com/kodex"
REPO_URL = "https://github.com/KronosFE/kronos-ml"
TOOLKIT_URL = "https://github.com/KronosFE/kronos-toolkit"
LICENSE = "Apache-2.0"

DOI_REGISTER = "10.5281/zenodo.22645689"
DOI_CGYRO = "10.5281/zenodo.22136279"
DOI_AIQ = "10.5281/zenodo.21842371"

KEYWORDS_BASE = ["fusion energy", "spherical tokamak", "D-3He", "AI/ML surrogate model",
                 "uncertainty quantification", "digital twin", "Kronos Fusion Energy"]


def built():
    """The codes a publish run acts on: every BUILT code, unless KODEX_PUBLISH_ONLY
    restricts it to an explicit allow-list (comma/space-separated code names).

    The allow-list is how an operator deposits a deliberate SUBSET (e.g. only the
    v0.2.0 additions) without touching the rest of the fleet — it does NOT change any
    code's real status, so the package stays honest. Unset = all BUILT (default). A
    named code that is not currently BUILT is a hard error, so a typo can never
    silently narrow (or widen) the set.
    """
    names = sorted(n for n in K.list_surrogates() if K.get(n).status == "BUILT")
    only = os.environ.get("KODEX_PUBLISH_ONLY", "").strip()
    if only:
        want = {c.strip().upper() for c in re.split(r"[,\s]+", only) if c.strip()}
        unknown = want - set(names)
        if unknown:
            raise SystemExit(f"[STOP] KODEX_PUBLISH_ONLY names not in the BUILT set: "
                             f"{sorted(unknown)} (BUILT: {names})")
        names = [n for n in names if n in want]
        sys.stderr.write(f"[scope] KODEX_PUBLISH_ONLY -> {len(names)} codes: {names}\n")
    return names


def physics_source(s):
    """(relation, identifier, note) list tracing a code to its real physics source."""
    p = (s.provenance or "").upper()
    if "CGYRO" in p:
        return [("isDerivedFrom", DOI_CGYRO, "CGYRO A1e nonlinear-kinetic turbulence map (mu=400)")]
    if p == "REAL-ANCHOR":
        return [("isDerivedFrom", "https://mastapp.site", "FAIR-MAST real-device data (UKAEA)")]
    if s.name in ("KQUBIT", "KQROSS", "KSENSE", "KTENSOR"):
        return [("isDerivedFrom", DOI_AIQ, "Kronos AI + Quantum deposit")]
    if s.name in ("KBREED", "KFLUX"):
        return [("references", DOI_REGISTER, "De-Risking Register neutronics gates")]
    if s.name == "KISO":
        return [("isDerivedFrom", "https://www-nds.iaea.org/fendl/", "FENDL-3.2 activation data (IAEA)"),
                ("references", DOI_REGISTER, "De-Risking Register")]
    return [("references", DOI_AIQ, "Kronos AI + Quantum corpus")]


def _headline(name, b):
    try:
        from benchmarks.run_all import _headline as hl
        return hl(name, b)
    except Exception:
        return (b or {}).get("note", "")


def record_meta(name, b=None):
    s = K.get(name); kn = name.lower()
    if b is None:
        b = s.benchmark()
    page = f"{KODEX}/{kn}"
    src = physics_source(s)
    related = [{"relation": "isDocumentedBy", "identifier": page, "resource_type": "publication-other"},
               {"relation": "isPartOf", "identifier": KODEX, "resource_type": "publication-other"},
               {"relation": "isSupplementTo", "identifier": REPO_URL, "resource_type": "software"},
               {"relation": "references", "identifier": DOI_REGISTER, "resource_type": "publication"},
               {"relation": "isCompiledBy", "identifier": TOOLKIT_URL, "resource_type": "software"}]
    for rel, ident, _ in src:
        rtype = "dataset" if ident.startswith("10.") or "mast" in ident or "fendl" in ident else "other"
        related.append({"relation": rel, "identifier": ident, "resource_type": rtype})
    # dedupe by (relation, identifier), keeping first (e.g. register DOI appended by both
    # the base list and physics_source for KBREED/KFLUX/KISO)
    _seen, _ded = set(), []
    for r in related:
        k = (r["relation"], r["identifier"])
        if k not in _seen:
            _seen.add(k); _ded.append(r)
    related = _ded
    desc = (f"<p><strong>KODEX &mdash; {name}</strong> ({s.function}). {(s.note or '').strip()}</p>"
            f"<p><strong>Benchmark:</strong> {_headline(name, b)}</p>"
            f"<p>Part of KODEX, the Kronos Family of Codes &mdash; a benchmarked AI/ML fusion "
            f"surrogate suite with calibrated uncertainty and an abstention gate "
            f"(<code>predict(x) &rarr; (y, uncertainty, in_domain)</code>). Provenance: "
            f"{'; '.join(f'{r} {i}' for r, i, _ in src)}. Home: <a href=\"{page}\">{page}</a> &middot; "
            f"suite: <a href=\"{REPO_URL}\">{REPO_URL}</a>. Licensed under Apache-2.0. "
            f"Honest card preserved verbatim.</p>")
    # plaintext variant for venues whose description field does NOT render HTML (OSF)
    desc_plain = html.unescape(re.sub(r"<[^>]+>", "", desc)).strip()
    return {
        "kname": name, "page": page,
        "title": f"KODEX — {name}: {s.function}",
        "version": VERSION, "publication_date": TODAY, "language": "eng",
        "upload_type": "software",
        "creators": [{"name": "Ford, P. I.", "orcid": ORCID, "affiliation": AFFIL}],
        "license": {"id": LICENSE},
        "keywords": KEYWORDS_BASE + [s.function, f"KODEX:{name}"],
        "related_identifiers": related,
        "description": desc,
        "description_plain": desc_plain,
        "notes": (f"Draft-first per-code deposit. Full package: {REPO_URL}. "
                  f"Provenance chain in card.md + MANIFEST.sha256."),
    }


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def build_records():
    REC_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    names = built()
    for name in names:
        d = REC_ROOT / name.lower()
        if (d / "metadata.json").exists() and (d / "MANIFEST.sha256").exists():
            continue                       # resume: already generated
        s = K.get(name); kn = name.lower()
        b = s.benchmark()                  # compute ONCE, reuse
        d.mkdir(exist_ok=True)
        # 1) source excerpt (the class), readable
        try:
            src_code = inspect.getsource(type(s))
        except Exception:
            src_code = f"# source not extractable for {name}\n"
        (d / f"{kn}.py").write_text(
            f'"""KODEX {name} — source excerpt (class). Full runnable package: {REPO_URL}"""\n\n'
            + src_code)
        # 2) benchmark
        (d / "benchmark.json").write_text(json.dumps(b, default=str, indent=2) + "\n")
        # 3) card
        c = s.card()
        (d / "card.md").write_text(
            f"# KODEX {name} — card\n\n"
            + "\n".join(f"- **{k}:** {v}" for k, v in c.items()) + "\n\n"
            + f"Benchmark headline: {_headline(name, b)}\n")
        # 4) per-code CITATION.cff
        (d / "CITATION.cff").write_text("\n".join([
            "cff-version: 1.2.0",
            f"title: \"KODEX — {name}: {s.function} (Kronos Family of Codes)\"",
            f"version: \"{VERSION}\"", f"date-released: \"{TODAY}\"",
            f"license: {LICENSE}", f"url: \"{KODEX}/{kn}\"",
            f"repository-code: \"{REPO_URL}\"", "type: software", "authors:",
            "  - family-names: Ford", "    given-names: \"P. I.\"",
            f"    orcid: \"https://orcid.org/{ORCID}\"", f"    affiliation: \"{AFFIL}\"",
        ]) + "\n")
        # 5) LICENSE
        shutil.copyfile(REPO / "LICENSE", d / "LICENSE")
        # 6) metadata
        (d / "metadata.json").write_text(json.dumps(record_meta(name, b), indent=2) + "\n")
        # 7) manifest
        files = [p for p in sorted(d.iterdir()) if p.name != "MANIFEST.sha256"]
        (d / "MANIFEST.sha256").write_text(
            f"# KODEX {name} SHA-256 ({VERSION}, {TODAY})\n"
            + "\n".join(f"{_sha(p)}  {p.name}" for p in files) + "\n")
    (PUB / "_percode_summary.json").write_text(json.dumps(
        {"version": VERSION, "date": TODAY, "n_records": len(names), "codes": names}, indent=2) + "\n")
    return names


def record_dir(kname):
    return REC_ROOT / kname.lower()


def attach_paths(kname):
    d = record_dir(kname)
    # metadata.json is included so the uploaded file set exactly equals the set the
    # MANIFEST.sha256 hashes (sha256sum -c then verifies clean; -c skips the un-self-listed manifest)
    order = [f"{kname.lower()}.py", "benchmark.json", "card.md",
             "CITATION.cff", "LICENSE", "metadata.json", "MANIFEST.sha256"]
    return [d / f for f in order if (d / f).exists()]


def load_meta(kname):
    return json.loads((record_dir(kname) / "metadata.json").read_text())


def state_path(venue):
    return STATE_ROOT / f"{venue}_drafts.json"


def read_state(venue):
    p = state_path(venue)
    return json.loads(p.read_text()) if p.exists() else {}


def write_state(venue, st):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path(venue).write_text(json.dumps(st, indent=2) + "\n")


def require_env(var):
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"[STOP] environment variable {var} is not set (see PUBLISH_RUNBOOK.md).")
    return v


if __name__ == "__main__":
    names = build_records()
    print(f"built {len(names)} per-code records under publish/records/:")
    for n in names:
        ap = attach_paths(n)
        print(f"  {n.lower():9s} {len(ap)} files, meta title={load_meta(n)['title']!r}")
