#!/usr/bin/env python
"""Make the 26 per-code OSF projects PUBLIC (created private by osf_draft.py).

IRREVERSIBLE STEP. DRY RUN by default; flips projects to public only with --yes,
and only those recorded in publish/state/osf_drafts.json.

    export OSF_TOKEN=...
    python publish/osf_publish_all.py           # dry run
    python publish/osf_publish_all.py --yes      # make public
"""
from __future__ import annotations
import argparse, sys
import requests
from _percode import read_state, write_state, require_env

API = "https://api.osf.io/v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    state = read_state("osf")
    pending = {n: r for n, r in state.items() if r.get("id") and not r.get("public")}
    if not pending:
        print("[osf] no private projects in state (run osf_draft.py first)."); return
    print(f"[osf] {len(pending)} project(s) to make public:")
    for n, r in pending.items():
        print(f"    {n:9s} {r.get('url','')}")
    if not args.yes:
        print("\n[osf] DRY RUN — nothing made public. Re-run with --yes.")
        return
    hj = {"Authorization": f"Bearer {require_env('OSF_TOKEN')}",
          "Content-Type": "application/vnd.api+json"}
    s = requests.Session()
    for n, r in pending.items():
        payload = {"data": {"type": "nodes", "id": r["id"], "attributes": {"public": True}}}
        resp = s.patch(f"{API}/nodes/{r['id']}/", headers=hj, json=payload)
        resp.raise_for_status()
        r["public"] = True
        write_state("osf", state)
        print(f"[osf] PUBLIC {n}: {r.get('url')}")
    print(f"\n[osf] made {len(pending)} projects public.")


if __name__ == "__main__":
    sys.exit(main())
