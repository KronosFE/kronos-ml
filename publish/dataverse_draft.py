#!/usr/bin/env python
"""Create 26 DRAFT Harvard Dataverse datasets (one per BUILT KODEX code) in the
kronos-fusion-energy collection.

DRAFT-FIRST: Dataverse datasets are created UNPUBLISHED. Never calls the publish
action. Resume-safe via publish/state/dataverse_drafts.json (pid recorded the moment
the dataset is created). Prints the API error body on failure so a 400 names its field.
After review, run dataverse_publish_all.py --yes.

    export DATAVERSE_TOKEN=...
    export DATAVERSE_CONTACT_EMAIL=...     # shown PUBLICLY — use a public-safe address
    # optional: DATAVERSE_COLLECTION=kronos-fusion-energy  DATAVERSE_BASE=https://dataverse.harvard.edu
    python publish/dataverse_draft.py
"""
from __future__ import annotations
import os, sys
import requests
from _percode import built, attach_paths, load_meta, read_state, write_state, require_env, ORCID

BASE = os.environ.get("DATAVERSE_BASE", "https://dataverse.harvard.edu").rstrip("/")
COLLECTION = os.environ.get("DATAVERSE_COLLECTION", "kronos-fusion-energy")


def _prim(name, value, multiple=False, cls="primitive"):
    return {"typeName": name, "multiple": multiple, "typeClass": cls, "value": value}


def _dataset_json(m, contact_email):
    author = {"authorName": _prim("authorName", "Ford, P. I."),
              "authorAffiliation": _prim("authorAffiliation", "Kronos Fusion Energy"),
              "authorIdentifierScheme": {"typeName": "authorIdentifierScheme", "multiple": False,
                                         "typeClass": "controlledVocabulary", "value": "ORCID"},
              "authorIdentifier": _prim("authorIdentifier", ORCID)}
    contact = {"datasetContactName": _prim("datasetContactName", "Ford, P. I."),
               "datasetContactAffiliation": _prim("datasetContactAffiliation", "Kronos Fusion Energy"),
               "datasetContactEmail": _prim("datasetContactEmail", contact_email)}
    desc = {"dsDescriptionValue": _prim("dsDescriptionValue", m["description"])}
    kws = [{"keywordValue": _prim("keywordValue", k)} for k in m["keywords"]]
    # only well-formed DOI publications (bare-URL entries live in the description instead)
    pubs = []
    for r in m["related_identifiers"]:
        ident = r["identifier"]
        if ident.startswith("10."):
            pubs.append({"publicationIDType": {"typeName": "publicationIDType", "multiple": False,
                                               "typeClass": "controlledVocabulary", "value": "doi"},
                         "publicationIDNumber": _prim("publicationIDNumber", ident),
                         "publicationURL": _prim("publicationURL", f"https://doi.org/{ident}")})
    fields = [_prim("title", m["title"]),
              _prim("author", [author], True, "compound"),
              _prim("datasetContact", [contact], True, "compound"),
              _prim("dsDescription", [desc], True, "compound"),
              _prim("subject", ["Physics"], True, "controlledVocabulary"),
              _prim("keyword", kws, True, "compound")]
    if pubs:
        fields.append(_prim("publication", pubs, True, "compound"))
    # NOTE: no dataset-level `license` block. Harvard defaults license-less datasets to CC0,
    # so Apache-2.0 is asserted in the description (above), CITATION.cff and the LICENSE file.
    # To set the dataset LICENSE FIELD to Apache-2.0: GET {BASE}/api/licenses to confirm it is
    # installed, then PUT the license onto each draft version before curator approval (the
    # published version is immutable). Left out of create because it is not guaranteed installed.
    return {"datasetVersion": {"metadataBlocks": {"citation": {"fields": fields}}}}


def main():
    token = require_env("DATAVERSE_TOKEN")
    contact = os.environ.get("DATAVERSE_CONTACT_EMAIL")
    if not contact:
        raise SystemExit("[STOP] set DATAVERSE_CONTACT_EMAIL (public on the record) first.")
    h = {"X-Dataverse-key": token}
    s = requests.Session()
    state = read_state("dataverse")
    names = built()
    for i, name in enumerate(names, 1):
        rec = state.get(name, {})
        if rec.get("done"):
            print(f"[dataverse] ({i}/{len(names)}) {name}: complete (skip)"); continue
        m = load_meta(name)
        pid = rec.get("pid")
        if not pid:
            r = s.post(f"{BASE}/api/dataverses/{COLLECTION}/datasets", headers=h,
                       json=_dataset_json(m, contact))
            if not r.ok:
                print(f"[dataverse] {name}: create FAILED {r.status_code}: {r.text[:500]}")
                return 1
            pid = r.json()["data"].get("persistentId")
            state[name] = {"pid": pid,
                           "url": f"{BASE}/dataset.xhtml?persistentId={pid}&version=DRAFT"}
            write_state("dataverse", state)
        for path in attach_paths(name):
            with open(path, "rb") as fh:
                r = s.post(f"{BASE}/api/datasets/:persistentId/add", headers=h,
                           params={"persistentId": pid}, files={"file": (path.name, fh)})
            if not r.ok and "duplicate" not in r.text.lower() and "exists" not in r.text.lower():
                print(f"[dataverse] {name}/{path.name}: add FAILED {r.status_code}: {r.text[:300]}")
                return 1
        state[name]["done"] = True
        write_state("dataverse", state)
        print(f"[dataverse] ({i}/{len(names)}) {name}: DRAFT {pid}")
    print(f"\n[dataverse] {sum(1 for r in state.values() if r.get('done'))} drafts (NONE published). Then:")
    print("            python publish/dataverse_publish_all.py --yes")


if __name__ == "__main__":
    sys.exit(main())
