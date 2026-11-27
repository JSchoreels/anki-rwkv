"""Fetch a Fushi donation batch (format "fushi-srs-donation-batch") into the Fushi donation folder.

usage: fetch_fushi_batch.py MANIFEST_URL [--folder DIR] [--apply-withdrawals]

The manifest lists one .apkg per donor (stable anonymous donorId, updatedAt). A file is downloaded when it is new or
its updatedAt is newer than the copy we hold (a re-upload replaces the old file); it is written to a temporary name
first and checked to be a zip before it replaces anything. Donors in the manifest's "withdrawn" list are reported;
with --apply-withdrawals their files are deleted from the folder (the donations watcher then drops them from the
stats). Prints one line per action. Never stores anything in the repository.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_FOLDER = "D:/Dropbox/Fushi Collections"
STATE = "_fushi_batches.json"  # donorId -> updatedAt of the copy we hold (not an Anki file: ignored by the inventory)


def get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "anki-rwkv-donations"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_url")
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--apply-withdrawals", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(get(args.manifest_url))
    if manifest.get("format") != "fushi-srs-donation-batch":
        sys.exit(f"unexpected manifest format: {manifest.get('format')!r}")
    state_path = folder / STATE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    for f in manifest.get("files", []):
        donor, updated = str(f["donorId"]), int(f["updatedAt"])
        if not donor.isalnum():
            print(f"SKIP bad donorId {donor!r}")
            continue
        dest = folder / f"{donor}.apkg"
        if dest.exists() and state.get(donor, 0) >= updated:
            print(f"HAVE {donor}")
            continue
        data = get(f["url"])
        if data[:2] != b"PK":
            print(f"FAIL {donor}: not a zip")
            continue
        tmp = folder / f".{donor}.part"
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        print(f"{'UPDATED' if donor in state else 'NEW'} {donor} {len(data)} bytes, {f.get('reviews')} reviews")
        state[donor] = updated
    for donor in manifest.get("withdrawn", []):
        donor = str(donor)
        dest = folder / f"{donor}.apkg"
        if not dest.exists():
            continue
        if args.apply_withdrawals:
            dest.unlink()
            state.pop(donor, None)
            print(f"WITHDRAWN-REMOVED {donor}")
        else:
            print(f"WITHDRAWN {donor} (rerun with --apply-withdrawals to delete it)")
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
