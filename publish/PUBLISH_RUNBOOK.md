# KODEX — publish runbook (PER-CODE, DRAFT-FIRST)

**Release:** KODEX v0.1.0 — the Kronos Family of Codes.
**Model:** **26 per-code records** on each data repository (Zenodo / OSF / Figshare / Harvard Dataverse /
Internet Archive) — each BUILT code gets its own record + DOI. **GitHub = ONE repo** (`kronos-ml`, 26
modules). **PyPI = ONE package** (`kronos-ml`, 26 importable codes). **Held as ROADMAP (not published):**
KHEAT, KTENSOR, KEDGE, KRAD.
**Author on every record:** Ford, P. I. · ORCID `0000-0003-0395-1752` · Kronos Fusion Energy · Apache-2.0.

> **Draft-first, two-step per data venue.** Step 1 `*_draft.py` creates all 26 records as **private/
> unpublished** and uploads each code's files — nothing is public. Step 2 `*_publish_all.py` is a **dry run
> by default** and publishes only with an explicit **`--yes`** — that `--yes` is your single deliberate
> authorization for that venue's 26. The draft scripts NEVER publish; the publish scripts do nothing
> without `--yes`. (Internet Archive is one step — items are public on upload — so `ia_upload_all.py`
> requires `--yes` and IS the publish.)

Per-code metadata lives in `publish/records/<code>/metadata.json`; each record's homepage is
`kronosfusionenergy.com/kodex/<code>`, with its physics-source + the shared Kronos anchors
(register `10.5281/zenodo.22645689`, CGYRO `10.5281/zenodo.22136279`, AI+Quantum `10.5281/zenodo.21842371`,
toolkit) as related-identifiers, honest card preserved. Per-code attach set: `<code>.py` (source excerpt),
`benchmark.json`, `card.md`, `CITATION.cff`, `LICENSE`, `MANIFEST.sha256`. The repo-root `.zenodo.json`,
`CITATION.cff`, `codemeta.json` describe the **GitHub/PyPI package** (the one-repo/one-package side).

## Token map

| Venue | Credential | Env var | Status |
|---|---|---|---|
| GitHub | SSH key | — | ✅ ready |
| Zenodo | token `deposit:write` | `ZENODO_TOKEN` | ✅ ready |
| OSF | token `osf.full_write` | `OSF_TOKEN` | ✅ ready |
| Figshare | personal token | `FIGSHARE_TOKEN` | ✅ ready |
| Internet Archive | `ia configure` (S3 keys) | — | ✅ ready |
| Harvard Dataverse | API token (115-paper/books token) | `DATAVERSE_TOKEN` + `DATAVERSE_CONTACT_EMAIL` | ⏳ token needed |
| PyPI | API token | `TWINE_PASSWORD` (`TWINE_USERNAME=__token__`) | ⏳ fast-follow |
| Hugging Face | write token | `HF_TOKEN` | ⏳ fast-follow |

## Preflight (once)

```bash
cd "/Users/pford/Desktop/Kronos Fusion Energy/04 - IT INFRASTRUCTURE/07 - IT Infrastructure & Code Repos/Code Repos (live)/kronos-ml"
PY=~/kronos-hpc-build/ml311_venv/bin/python
$PY -m pytest tests/ -q                 # expect 9 passed
$PY publish/_percode.py                  # (re)builds publish/records/<code>/ for all 26 (resume-safe)
```

> ⚠ **numpy pin:** this venv must keep **numpy<2** (torch 2.2.2 needs it; numpy 2.x breaks KORE). If a
> tool ever pulls numpy 2.x back in, run `$PY -m pip install "numpy==1.26.4"` before benchmarking.

---

## 1) GitHub — ONE repo (git@github.com:KronosFE/kronos-ml.git)

```bash
git init -b main
git add .
git commit -m "KODEX v0.1.0 — Kronos Family of Codes (26 BUILT surrogates)"
git tag -a v0.1.0 -m "KODEX v0.1.0"
# create the repo PRIVATE on GitHub, then:
git remote add origin git@github.com:KronosFE/kronos-ml.git
git push -u origin main --tags
# → review; when ready, make public + draft a Release from tag v0.1.0.
# Do NOT enable the GitHub↔Zenodo webhook (we deposit Zenodo per-code below).
```

## 2) Zenodo — 26 per-code records

```bash
export ZENODO_TOKEN=...                  # rehearse first: export ZENODO_BASE=https://sandbox.zenodo.org
$PY publish/zenodo_draft.py              # creates 26 DRAFTS (resume-safe), nothing public
$PY publish/zenodo_publish_all.py        # DRY RUN — lists the 26 drafts
$PY publish/zenodo_publish_all.py --yes  # ← your one authorization: publishes all 26, mints DOIs
```

## 3) OSF — 26 private projects → public

```bash
export OSF_TOKEN=...
$PY publish/osf_draft.py                 # 26 PRIVATE projects
$PY publish/osf_publish_all.py           # DRY RUN
$PY publish/osf_publish_all.py --yes     # ← make all 26 public
```

## 4) Figshare — 26 draft articles → published

```bash
export FIGSHARE_TOKEN=...
$PY publish/figshare_draft.py            # 26 DRAFT articles (DOIs reserved, not published)
$PY publish/figshare_publish_all.py      # DRY RUN
$PY publish/figshare_publish_all.py --yes
```

## 5) Harvard Dataverse — 26 datasets (published by CURATORS, not by --yes)

STATUS 2026-09-10: the 26 drafts are created; **25 were submitted for Harvard curator review**
(`review:true` in state) and publish when a Harvard **curator approves** — NOT via a depositor
`--yes`. `dataverse_publish_all.py` now refuses to `:publish` review-queued datasets (a direct
publish on an In-Review dataset errors), so `--yes` is a no-op for the 25. Two open items before
curators approve (the published version is immutable):

1. **KLAW** (`doi:10.7910/DVN/ANAFXK`) was NOT submitted for review — submit it so it isn't stranded:
   ```bash
   export DATAVERSE_TOKEN=...
   curl -H "X-Dataverse-key:$DATAVERSE_TOKEN" -X POST \
     "https://dataverse.harvard.edu/api/datasets/:persistentId/submitForReview?persistentId=doi:10.7910/DVN/ANAFXK"
   # then set "review": true for KLAW in publish/state/dataverse_drafts.json
   ```
2. **License field**: the drafts carry no dataset-level license, so Harvard will publish them as
   **CC0** unless Apache-2.0 is set. Check + set before approval:
   ```bash
   curl -s -H "X-Dataverse-key:$DATAVERSE_TOKEN" https://dataverse.harvard.edu/api/licenses | grep -i apache
   # if Apache-2.0 is installed, set it on each draft version (API) before curator approval;
   # otherwise the Apache-2.0 statement in the description + bundled LICENSE stand, and you accept CC0.
   ```
(Re-running `dataverse_draft.py` is safe — it skips the 26 already-created datasets.)

## 6) Internet Archive — 26 items (public on upload; one step)

```bash
ia configure                             # once, with your IA S3 keys
$PY publish/ia_upload_all.py             # DRY RUN — lists the 26 item identifiers
$PY publish/ia_upload_all.py --yes       # ← uploads 26 items (PUBLIC immediately)
```

---

## Fast-follow (when the tokens land)

### PyPI — ONE package
```bash
$PY -m build
export TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...
$PY -m twine upload --repository testpypi dist/*      # rehearse first
$PY -m twine upload dist/*
```

### Hugging Face — per-code model repos (or one)
```bash
export HF_TOKEN=...
$PY -m pip install -q huggingface_hub
# one suite repo:
huggingface-cli upload KronosFE/kronos-ml . . --repo-type=model --token "$HF_TOKEN"
# (per-code repos KronosFE/kodex-<code> can be scripted the same way if preferred)
```

---

## Rollback (discard drafts — nothing was public)
- **Zenodo / Figshare / Dataverse:** delete the unpublished drafts in each UI (or via each API).
- **OSF:** projects are private until `--yes`; delete in the UI.
- **GitHub:** repo stays private until you flip it.
- **Internet Archive:** items are public on upload — only run `--yes` when you're sure; ask IA support to
  dark an item if needed.
- Re-running any `*_draft.py` is safe: it skips codes already in `publish/state/<venue>_drafts.json`.

## Provenance recorded in every record
code → real benchmark (`benchmark.json`) → on-disk data/method → Kronos corpus DOI → `/kodex/<code>` page.
Honest cards intact: μ=400 representative-mass (CGYRO codes); validated null result (quantum codes);
KWARD heuristic labels + independent-precursor AUC 0.975 (n=1 Mirnov + P_rad only); KECON financial firewall.
