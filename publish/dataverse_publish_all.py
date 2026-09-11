#!/usr/bin/env python
"""Publish the 26 per-code Harvard Dataverse DRAFTS created by dataverse_draft.py.

IRREVERSIBLE STEP. DRY RUN by default; publishes (major version) only with --yes,
only the datasets recorded in publish/state/dataverse_drafts.json.

    export DATAVERSE_TOKEN=...
    python publish/dataverse_publish_all.py           # dry run
    python publish/dataverse_publish_all.py --yes      # publish
"""
from __future__ import annotations
import argparse, os, sys
import requests
from _percode import read_state, write_state, require_env

BASE = os.environ.get("DATAVERSE_BASE", "https://dataverse.harvard.edu").rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    state = read_state("dataverse")
    # records submitted for Harvard CURATOR review (review:true) are published by the
    # curators on approval, NOT by a depositor :publish call (which errors on an In-Review
    # dataset). Only ever direct-publish records that are NOT in the review queue.
    in_review = {n: r for n, r in state.items() if r.get("review") and not r.get("published")}
    pending = {n: r for n, r in state.items()
               if r.get("pid") and not r.get("published") and not r.get("review")}
    if in_review:
        print(f"[dataverse] {len(in_review)} dataset(s) are in Harvard curator review — "
              f"published on curator APPROVAL, not here. Skipping: {', '.join(sorted(in_review))}")
    if not pending:
        print("[dataverse] no directly-publishable drafts (all are curator-review or already "
              "published). Nothing to do — await Harvard curators."); return
    print(f"[dataverse] {len(pending)} NON-review draft(s) directly publishable on {BASE}:")
    for n, r in pending.items():
        print(f"    {n:9s} {r['pid']}")
    if not args.yes:
        print("\n[dataverse] DRY RUN — nothing published. Re-run with --yes.")
        return
    h = {"X-Dataverse-key": require_env("DATAVERSE_TOKEN")}
    s = requests.Session()
    ok = 0
    for n, r in pending.items():
        resp = s.post(f"{BASE}/api/datasets/:persistentId/actions/:publish",
                      headers=h, params={"persistentId": r["pid"], "type": "major"})
        if not resp.ok:                      # don't let one failure abort the batch
            print(f"[dataverse] {n}: publish FAILED {resp.status_code}: {resp.text[:200]}")
            continue
        r["published"] = True
        write_state("dataverse", state)
        ok += 1
        print(f"[dataverse] PUBLISHED {n}: {r['pid']}")
    print(f"\n[dataverse] published {ok}/{len(pending)} non-review records.")


if __name__ == "__main__":
    sys.exit(main())
