#!/usr/bin/env python
"""Create 26 per-code DRAFT Zenodo depositions (one per BUILT KODEX code).

DRAFT-FIRST: creates unpublished depositions and uploads each code's attach set.
NEVER publishes. Resume-safe: records already in publish/state/zenodo_drafts.json are
skipped. After review, run zenodo_publish_all.py --yes to publish them in one go.

    export ZENODO_TOKEN=...            # optional: export ZENODO_BASE=https://sandbox.zenodo.org
    python publish/zenodo_draft.py
"""
from __future__ import annotations
import os, sys
import requests
from _percode import built, attach_paths, load_meta, read_state, write_state, require_env

BASE = os.environ.get("ZENODO_BASE", "https://zenodo.org").rstrip("/")


def _api_meta(m):
    lic = m["license"]; lic = lic["id"] if isinstance(lic, dict) else lic
    out = {"title": m["title"], "upload_type": "software", "description": m["description"],
           "creators": m["creators"], "license": lic, "keywords": m["keywords"],
           "related_identifiers": m["related_identifiers"], "version": m["version"],
           "language": "eng", "notes": m.get("notes")}
    return {k: v for k, v in out.items() if v is not None}


def main():
    token = require_env("ZENODO_TOKEN")
    p = {"access_token": token}
    s = requests.Session()
    state = read_state("zenodo")
    names = built()
    for i, name in enumerate(names, 1):
        if state.get(name, {}).get("id"):
            print(f"[zenodo] ({i}/{len(names)}) {name}: already drafted (skip)"); continue
        m = load_meta(name)
        r = s.post(f"{BASE}/api/deposit/depositions", params=p, json={}); r.raise_for_status()
        dep = r.json(); dep_id, bucket = dep["id"], dep["links"]["bucket"]
        for path in attach_paths(name):
            with open(path, "rb") as fh:
                s.put(f"{bucket}/{path.name}", data=fh, params=p).raise_for_status()
        s.put(f"{BASE}/api/deposit/depositions/{dep_id}", params=p,
              json={"metadata": _api_meta(m)}).raise_for_status()
        state[name] = {"id": dep_id, "edit_url": f"{BASE}/deposit/{dep_id}"}
        write_state("zenodo", state)
        print(f"[zenodo] ({i}/{len(names)}) {name}: DRAFT {dep_id}")
    print(f"\n[zenodo] {len(state)} drafts ready (NONE published). Review, then:")
    print("         python publish/zenodo_publish_all.py --yes")


if __name__ == "__main__":
    sys.exit(main())
