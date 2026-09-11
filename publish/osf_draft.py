#!/usr/bin/env python
"""Create 26 PRIVATE OSF projects (one per BUILT KODEX code).

DRAFT-FIRST: each project is created private (public=false) and never made public here.
Resume-safe via publish/state/osf_drafts.json. Throttled + retried against OSF's
rate-limiting (403/429). After review, run osf_publish_all.py --yes.

    export OSF_TOKEN=...
    python publish/osf_draft.py
"""
from __future__ import annotations
import sys, time
import requests
from _percode import (built, attach_paths, load_meta, read_state, write_state,
                      require_env, http_retry)

API = "https://api.osf.io/v2"
WB = "https://files.osf.io/v1"
PACE_FILE = 0.6      # seconds between file uploads
PACE_PROJECT = 1.2   # seconds between projects


def main():
    token = require_env("OSF_TOKEN")
    h = {"Authorization": f"Bearer {token}"}
    hj = {**h, "Content-Type": "application/vnd.api+json"}
    s = requests.Session()
    state = read_state("osf")
    names = built()
    for i, name in enumerate(names, 1):
        rec = state.get(name, {})
        node = rec.get("id")
        if rec.get("done"):
            print(f"[osf] ({i}/{len(names)}) {name}: complete (skip)"); continue
        m = load_meta(name)
        if not node:            # create the project (retry rate-limits)
            payload = {"data": {"type": "nodes", "attributes": {
                "title": m["title"], "category": "software",
                # OSF renders the description as PLAIN TEXT (no HTML) -> use the stripped variant
                "description": m.get("description_plain", m["description"]),
                "public": False, "tags": m["keywords"][:25]}}}
            r = http_retry(lambda: s.post(f"{API}/nodes/", headers=hj, json=payload),
                           label=f"osf-create {name}")
            if not r.ok:
                print(f"[osf] {name}: create FAILED {r.status_code}: {r.text[:300]}"); return 1
            node = r.json()["data"]["id"]
            state[name] = {"id": node, "url": f"https://osf.io/{node}/"}
            write_state("osf", state)
            time.sleep(PACE_PROJECT)
        # upload files (throttled; 409 = already there = fine)
        for path in attach_paths(name):
            def _put():
                with open(path, "rb") as fh:
                    return s.put(f"{WB}/resources/{node}/providers/osfstorage/", headers=h,
                                 params={"kind": "file", "name": path.name}, data=fh)
            r = http_retry(_put, label=f"osf-upload {name}/{path.name}")
            if r.status_code == 409:
                pass  # file already exists (resumed) -> treat as done
            elif not r.ok:
                print(f"[osf] {name}/{path.name}: upload FAILED {r.status_code}: {r.text[:200]}")
                return 1
            time.sleep(PACE_FILE)
        state[name]["done"] = True
        write_state("osf", state)
        print(f"[osf] ({i}/{len(names)}) {name}: PRIVATE {node} ({len(attach_paths(name))} files)")
    print(f"\n[osf] {sum(1 for r in state.values() if r.get('done'))} private projects (NONE public). Then:")
    print("      python publish/osf_publish_all.py --yes")


if __name__ == "__main__":
    sys.exit(main())
