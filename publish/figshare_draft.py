#!/usr/bin/env python
"""Create 26 DRAFT Figshare articles (one per BUILT KODEX code).

DRAFT-FIRST: never calls /publish. Creates private articles, reserves DOIs (a
reservation is not a publication), uploads each code's attach set via the multipart
flow. Resume-safe via publish/state/figshare_drafts.json (article id recorded the
moment it is created, so a mid-upload failure never orphans it). After review, run
figshare_publish_all.py --yes.

    export FIGSHARE_TOKEN=...
    python publish/figshare_draft.py
"""
from __future__ import annotations
import hashlib, sys
import requests
from _percode import built, attach_paths, load_meta, read_state, write_state, require_env

API = "https://api.figshare.com/v2"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def _refs(meta):
    """Figshare 'references' must be valid URLs — DOIs -> https://doi.org/..., keep only
    http(s), dedupe, preserve order."""
    out, seen = [], set()
    for r in meta["related_identifiers"]:
        ident = r["identifier"].strip()
        url = f"https://doi.org/{ident}" if ident.startswith("10.") else ident
        if url.startswith("http") and url not in seen:
            seen.add(url); out.append(url)
    return out


def _tags(meta):
    """Figshare tags: each >= 3 chars; keep it modest."""
    out, seen = [], set()
    for k in meta["keywords"]:
        k = k.strip()
        if len(k) >= 3 and k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:10]


def _existing_files(s, h, article_id):
    try:
        r = s.get(f"{API}/account/articles/{article_id}/files", headers=h)
        return {f["name"] for f in r.json()} if r.ok else set()
    except Exception:
        return set()


def _upload(s, h, article_id, path):
    size = path.stat().st_size
    r = s.post(f"{API}/account/articles/{article_id}/files", headers=h,
               json={"name": path.name, "md5": _md5(path), "size": size}); r.raise_for_status()
    file_url = r.json()["location"]
    info = s.get(file_url, headers=h).json()
    up = s.get(info["upload_url"], headers=h).json()
    with open(path, "rb") as fh:
        for part in up["parts"]:
            fh.seek(part["startOffset"])
            chunk = fh.read(part["endOffset"] - part["startOffset"] + 1)
            s.put(f"{info['upload_url']}/{part['partNo']}", headers=h, data=chunk).raise_for_status()
    s.post(file_url, headers=h).raise_for_status()


def main():
    token = require_env("FIGSHARE_TOKEN")
    h = {"Authorization": f"token {token}"}
    s = requests.Session()
    state = read_state("figshare")
    names = built()
    for i, name in enumerate(names, 1):
        rec = state.get(name, {})
        if rec.get("done"):
            print(f"[figshare] ({i}/{len(names)}) {name}: complete (skip)"); continue
        m = load_meta(name)
        aid = rec.get("id")
        if not aid:
            body = {"title": m["title"], "description": m["description"],
                    "defined_type": "software", "tags": _tags(m), "references": _refs(m),
                    # explicit author (ORCID is linked via the uploading Figshare account)
                    "authors": [{"name": "Ford, P. I."}]}
            r = s.post(f"{API}/account/articles", headers=h, json=body)
            if not r.ok:
                print(f"[figshare] {name}: create FAILED {r.status_code}: {r.text[:400]}")
                print(f"           body.tags={body['tags']}")
                print(f"           body.references={body['references']}")
                return 1
            aid = int(r.json()["location"].rstrip("/").split("/")[-1])
            state[name] = {"id": aid, "url": f"https://figshare.com/account/articles/{aid}"}
            write_state("figshare", state)
            try:
                s.post(f"{API}/account/articles/{aid}/reserve_doi", headers=h).raise_for_status()
            except Exception:
                pass
        have = _existing_files(s, h, aid)
        for path in attach_paths(name):
            if path.name in have:
                continue                      # already uploaded (resumed) -> idempotent
            _upload(s, h, aid, path)
        state[name]["done"] = True
        write_state("figshare", state)
        print(f"[figshare] ({i}/{len(names)}) {name}: DRAFT {aid}")
    print(f"\n[figshare] {sum(1 for r in state.values() if r.get('done'))} drafts (NONE published). Then:")
    print("           python publish/figshare_publish_all.py --yes")


if __name__ == "__main__":
    sys.exit(main())
