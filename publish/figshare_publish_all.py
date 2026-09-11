#!/usr/bin/env python
"""Publish the 26 per-code Figshare DRAFTS created by figshare_draft.py.

IRREVERSIBLE STEP. DRY RUN by default; publishes only with --yes, only the articles
recorded in publish/state/figshare_drafts.json.

    export FIGSHARE_TOKEN=...
    python publish/figshare_publish_all.py           # dry run
    python publish/figshare_publish_all.py --yes      # publish
"""
from __future__ import annotations
import argparse, sys
import requests
from _percode import read_state, write_state, require_env

API = "https://api.figshare.com/v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    state = read_state("figshare")
    pending = {n: r for n, r in state.items() if r.get("id") and not r.get("published")}
    if not pending:
        print("[figshare] no unpublished drafts in state (run figshare_draft.py first)."); return
    print(f"[figshare] {len(pending)} draft(s) to publish:")
    for n, r in pending.items():
        print(f"    {n:9s} id={r['id']}")
    if not args.yes:
        print("\n[figshare] DRY RUN — nothing published. Re-run with --yes.")
        return
    h = {"Authorization": f"token {require_env('FIGSHARE_TOKEN')}"}
    s = requests.Session()
    for n, r in pending.items():
        resp = s.post(f"{API}/account/articles/{r['id']}/publish", headers=h)
        resp.raise_for_status()
        loc = resp.json().get("location", "")
        r["published"] = True
        r["public_url"] = loc
        write_state("figshare", state)
        print(f"[figshare] PUBLISHED {n}: {loc}")
    print(f"\n[figshare] published {len(pending)} records.")


if __name__ == "__main__":
    sys.exit(main())
