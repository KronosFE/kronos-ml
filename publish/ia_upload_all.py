#!/usr/bin/env python
"""Upload the 26 per-code KODEX records to the Internet Archive (one item per code).

NOTE: Internet Archive items become PUBLIC on upload — there is no private-draft state —
so this IS the publish step for IA. DRY RUN by default; uploads only with --yes.
Requires the `ia` CLI configured (`ia configure` with your S3 keys) once.

    python publish/ia_upload_all.py            # dry run (lists items + files)
    python publish/ia_upload_all.py --yes       # upload (public)
"""
from __future__ import annotations
import argparse, subprocess, sys
from _percode import built, attach_paths, load_meta, VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    names = built()
    print(f"[ia] {len(names)} items to upload (identifier = kronos-ml-kodex-<code>-v{VERSION}):")
    plans = []
    for name in names:
        m = load_meta(name)
        ident = f"kronos-ml-kodex-{name.lower()}-v{VERSION}"
        files = [str(p) for p in attach_paths(name)]
        plans.append((ident, m, files))
        print(f"    {ident}  ({len(files)} files)")
    if not args.yes:
        print("\n[ia] DRY RUN — nothing uploaded. Re-run with --yes (items go PUBLIC on upload).")
        return
    for ident, m, files in plans:
        cmd = ["ia", "upload", ident, *files,
               f"--metadata=title:{m['title']}",
               "--metadata=creator:Ford, P. I.",
               "--metadata=licenseurl:https://spdx.org/licenses/Apache-2.0",
               f"--metadata=external-identifier:{m['page']}",
               "--metadata=subject:fusion; spherical tokamak; AI/ML surrogate; Kronos Fusion Energy"]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[ia] FAILED {ident} (rc={r.returncode})"); return r.returncode
        print(f"[ia] uploaded https://archive.org/details/{ident}")
    print(f"\n[ia] uploaded {len(plans)} items.")


if __name__ == "__main__":
    sys.exit(main())
