"""Sync the Fushi donation feed into the Fushi donation folder (one .apkg per anonymous donor).

usage: fetch_fushi_batch.py [FEED_URL] [--folder DIR] [--dry-run]

The feed URL is private: it is never stored in the repository. Without an argument it is read from
%LOCALAPPDATA%/anki-rwkv-donations/fushi_feed_url.txt (one line). The donations watcher runs this every 30 minutes.

Protocol (Fushi format doc, section 1; agreed with Fushi on 2026-11-27):
- The manifest ("fushi-srs-donation-feed", or the older "fushi-srs-donation-batch") is requested with If-None-Match;
  a 304 means nothing changed.
- A donor file is downloaded only when its etag (or updatedAt) differs from the copy we hold; it is written to a
  temporary name, checked to be a zip, then replaces the old file.
- A donor listed in "withdrawn", or no longer present in "files", is deleted from the folder (the donor withdrew;
  this is the removal Fushi's users were promised). Files not named like a donor id are never touched.
Prints one line per action and a summary; exits 1 on failure.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_FOLDER = "D:/Dropbox/Fushi Collections"
HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "anki-rwkv-donations"
URL_FILE = HOME / "fushi_feed_url.txt"
STATE = "_fushi_batches.json"  # manifest etag + donorId -> etag of the copy we hold (ignored by the inventory)
DONOR = re.compile(r"^[0-9a-f]{16,64}$")
FORMATS = ("fushi-srs-donation-feed", "fushi-srs-donation-batch")


def request(url, etag=None, timeout=600):
    headers = {"User-Agent": "anki-rwkv-donations"}
    if etag:
        headers["If-None-Match"] = etag
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status, r.headers.get("ETag"), r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return 304, etag, b""
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feed_url", nargs="?")
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, change nothing")
    args = ap.parse_args()
    url = args.feed_url or (URL_FILE.read_text(encoding="utf-8").strip() if URL_FILE.exists() else "")
    if not url:
        print(f"FAIL no feed URL (pass it, or put it in {URL_FILE})")
        return 1
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / STATE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    donors = state.setdefault("donors", {k: v for k, v in state.items() if DONOR.match(k)})
    for k in [k for k in state if DONOR.match(k)]:
        state.pop(k)
    try:
        status, etag, body = request(url, state.get("manifest_etag"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL manifest: {exc}")
        return 1
    if status == 304:
        print("UNCHANGED (304)")
        return 0
    manifest = json.loads(body)
    if manifest.get("format") not in FORMATS:
        print(f"FAIL unexpected manifest format {manifest.get('format')!r}")
        return 1
    new = upd = rem = 0
    listed = set()
    for f in manifest.get("files", []):
        donor = str(f.get("donorId", ""))
        if not DONOR.match(donor):
            print(f"SKIP bad donorId {donor!r}")
            continue
        listed.add(donor)
        version = str(f.get("etag") or f.get("updatedAt"))
        dest = folder / f"{donor}.apkg"
        if dest.exists() and donors.get(donor) == version:
            continue
        kind = "UPDATED" if dest.exists() else "NEW"
        if args.dry_run:
            print(f"{kind} {donor} (dry run)")
            continue
        try:
            _, _, data = request(f["url"])
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {donor}: {exc}")
            continue
        if data[:2] != b"PK":
            print(f"FAIL {donor}: not a zip")
            continue
        tmp = folder / f".{donor}.part"
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        donors[donor] = version
        new += kind == "NEW"
        upd += kind == "UPDATED"
        print(f"{kind} {donor} {len(data)} bytes, {f.get('reviews')} reviews")
    gone = {str(d) for d in manifest.get("withdrawn", [])} | {p.stem for p in folder.glob("*.apkg") if DONOR.match(p.stem)} - listed
    for donor in sorted(gone):
        dest = folder / f"{donor}.apkg"
        if not DONOR.match(donor) or not dest.exists():
            continue
        if args.dry_run:
            print(f"REMOVE {donor} (dry run)")
            continue
        dest.unlink()
        donors.pop(donor, None)
        rem += 1
        print(f"REMOVED {donor} (withdrawn or no longer in the feed)")
    if not args.dry_run:
        state["manifest_etag"] = etag
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"SYNCED {len(listed)} donors in feed: {new} new, {upd} updated, {rem} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
