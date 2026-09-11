#!/usr/bin/env python
"""Publish the 26 per-code Zenodo DRAFTS created by zenodo_draft.py.

THIS IS THE IRREVERSIBLE STEP. It is a DRY RUN by default — it only lists what would
be published. It publishes ONLY when you pass --yes, and only the drafts recorded in
publish/state/zenodo_drafts.json. Publishing mints DOIs and makes records public.

    export ZENODO_TOKEN=...
    python publish/zenodo_publish_all.py           # dry run (lists drafts)
    python publish/zenodo_publish_all.py --yes      # actually publish
"""
from __future__ import annotations
import argparse, os, sys
import requests
from _percode import read_state, write_state, require_env

BASE = os.environ.get("ZENODO_BASE", "https://zenodo.org").rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually publish (irreversible)")
    args = ap.parse_args()
    state = read_state("zenodo")
    pending = {n: r for n, r in state.items() if r.get("id") and not r.get("published")}
    if not pending:
        print("[zenodo] no unpublished drafts in state (run zenodo_draft.py first)."); return
    print(f"[zenodo] {len(pending)} draft(s) to publish on {BASE}:")
    for n, r in pending.items():
        print(f"    {n:9s} id={r['id']}  {r.get('edit_url','')}")
    if not args.yes:
        print("\n[zenodo] DRY RUN — nothing published. Re-run with --yes to publish these.")
        return
    p = {"access_token": require_env("ZENODO_TOKEN")}
    s = requests.Session()
    for n, r in pending.items():
        resp = s.post(f"{BASE}/api/deposit/depositions/{r['id']}/actions/publish", params=p)
        resp.raise_for_status()
        d = resp.json()
        r["published"] = True
        r["doi"] = d.get("doi") or d.get("metadata", {}).get("doi")
        write_state("zenodo", state)
        print(f"[zenodo] PUBLISHED {n}: DOI {r.get('doi')}")
    print(f"\n[zenodo] published {len(pending)} records.")


if __name__ == "__main__":
    sys.exit(main())
