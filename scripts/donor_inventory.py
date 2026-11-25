"""Inventory of donated Anki collections: reviews, cards, date range, duplicates and likely-same-donor groups.

Reads .apkg / .colpkg / .anki2 / .anki21 files (and .zip archives of them) from a donation folder, read-only.
Packages are extracted to a scratch directory outside the repository; zstd-compressed collections
(collection.anki21b) are decompressed with the libzstd.dll that ships with Git for Windows (ctypes, no install).

Outputs:
  --report  local JSON with one row per collection (includes file names = donor labels: keep it local, never commit)
  --aggregate  aggregate-only JSON (counts, no names, no ids) that may be published

Privacy: never commit the --report output or any extracted collection; only the aggregate may be published.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ZSTD_DLL = r"C:\Program Files\Git\mingw64\bin\libzstd.dll"
COLLECTION_NAMES = ("collection.anki21b", "collection.anki21", "collection.anki2")


class _Buf(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("size", ctypes.c_size_t), ("pos", ctypes.c_size_t)]


def zstd_decompress_stream(src, dst, chunk=1 << 20):
    lib = ctypes.CDLL(ZSTD_DLL)
    lib.ZSTD_createDStream.restype = ctypes.c_void_p
    lib.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
    lib.ZSTD_decompressStream.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Buf), ctypes.POINTER(_Buf)]
    lib.ZSTD_decompressStream.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
    ds = lib.ZSTD_createDStream()
    lib.ZSTD_initDStream(ds)
    out_mem = ctypes.create_string_buffer(chunk * 4)
    try:
        while True:
            data = src.read(chunk)
            if not data:
                break
            in_mem = ctypes.create_string_buffer(data, len(data))
            inb = _Buf(ctypes.cast(in_mem, ctypes.c_void_p), len(data), 0)
            while inb.pos < inb.size:
                outb = _Buf(ctypes.cast(out_mem, ctypes.c_void_p), len(out_mem), 0)
                ret = lib.ZSTD_decompressStream(ds, ctypes.byref(outb), ctypes.byref(inb))
                if lib.ZSTD_isError(ret):
                    raise RuntimeError("zstd error")
                dst.write(out_mem.raw[: outb.pos])
            # flush remaining output for this input
            while True:
                outb = _Buf(ctypes.cast(out_mem, ctypes.c_void_p), len(out_mem), 0)
                ret = lib.ZSTD_decompressStream(ds, ctypes.byref(outb), ctypes.byref(_Buf(None, 0, 0)))
                if lib.ZSTD_isError(ret):
                    raise RuntimeError("zstd error")
                dst.write(out_mem.raw[: outb.pos])
                if outb.pos < len(out_mem):
                    break
    finally:
        lib.ZSTD_freeDStream(ds)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_collection(zf, scratch, tag):
    names = {i.filename: i for i in zf.infolist()}
    for name in COLLECTION_NAMES:
        info = names.get(name)
        if info is None:
            continue
        if name == "collection.anki2" and info.file_size < 200_000 and "collection.anki21b" in names:
            continue  # legacy stub next to the real collection
        out = scratch / f"{tag}.sqlite"
        with zf.open(info) as src, open(out, "wb") as dst:
            if name.endswith("anki21b"):
                zstd_decompress_stream(src, dst)
            else:
                while block := src.read(1 << 20):
                    dst.write(block)
        return out, name
    return None, None


def iter_sources(folder, scratch):
    """Yield (label, path_of_sqlite, container_kind, file_sha256)."""
    n = 0
    for entry in sorted(Path(folder).iterdir()):
        if not entry.is_file():
            continue
        sha = sha256_file(entry)
        low = entry.name.lower()
        if low.endswith((".anki2", ".anki21")):
            yield entry.name, entry, "sqlite", sha
        elif low.endswith((".apkg", ".colpkg")):
            n += 1
            with zipfile.ZipFile(entry) as zf:
                path, inner = extract_collection(zf, scratch, f"c{n:04d}")
            if path:
                yield entry.name, path, inner, sha
        elif low.endswith(".zip"):
            with zipfile.ZipFile(entry) as outer:
                for info in outer.infolist():
                    if info.filename.lower().endswith((".apkg", ".colpkg")):
                        n += 1
                        nested = scratch / f"n{n:04d}.pkg"
                        with outer.open(info) as src, open(nested, "wb") as dst:
                            while block := src.read(1 << 20):
                                dst.write(block)
                        nested_sha = sha256_file(nested)
                        with zipfile.ZipFile(nested) as zf:
                            path, inner = extract_collection(zf, scratch, f"c{n:04d}")
                        nested.unlink()
                        if path:
                            yield f"{entry.name}::{info.filename}", path, inner, nested_sha


def read_collection(path):
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro&immutable=1", uri=True)
    cur = con.cursor()
    q = lambda sql: cur.execute(sql).fetchone()  # noqa: E731
    row = {}
    row["revlog_rows"] = q("select count(*) from revlog")[0]
    # reviews as in the benchmark: answered with a button (ease 1-4), not manual / rescheduled entries
    row["reviews"] = q("select count(*) from revlog where ease between 1 and 4 and type in (0,1,2,3)")[0]
    row["cards_reviewed"] = q("select count(distinct cid) from revlog where ease between 1 and 4")[0]
    row["cards"] = q("select count(*) from cards")[0]
    row["notes"] = q("select count(*) from notes")[0]
    lo, hi = q("select min(id), max(id) from revlog where ease between 1 and 4")
    row["first_review"] = datetime.fromtimestamp(lo / 1000, timezone.utc).date().isoformat() if lo else None
    row["last_review"] = datetime.fromtimestamp(hi / 1000, timezone.utc).date().isoformat() if hi else None
    row["review_days"] = q("select count(distinct id/86400000) from revlog where ease between 1 and 4")[0]
    ids = [r[0] for r in cur.execute("select id from revlog where ease between 1 and 4")]
    # deck-config ids other than the default one are timestamps unique to the source collection
    try:
        presets = [r[0] for r in cur.execute("select id from deck_config where id != 1")]
    except sqlite3.OperationalError:
        presets = []
    con.close()
    return row, set(ids), set(presets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--scratch", required=True, help="directory OUTSIDE the repository for extracted collections")
    ap.add_argument("--report", required=True, help="local per-collection JSON (has donor labels: never commit)")
    ap.add_argument("--aggregate", required=True, help="aggregate-only JSON (publishable)")
    ap.add_argument("--keep", action="store_true", help="keep extracted collections")
    args = ap.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    rows, idsets, presetsets = [], [], []
    for label, path, kind, sha in iter_sources(args.folder, scratch):
        try:
            row, ids, presets = read_collection(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the inventory
            row, ids, presets = {"error": str(exc)[:200]}, set(), set()
        row.update(label=label, kind=kind, file_sha256=sha)
        rows.append(row)
        idsets.append(ids)
        presetsets.append(presets)
        if not args.keep and Path(path).parent == scratch:
            Path(path).unlink()
        print(f"[inventory] {len(rows):3d} {row.get('reviews', 'ERR')!s:>10} reviews  {kind}", file=sys.stderr)

    n = len(rows)
    # exact duplicates: same file, or same set of review ids
    for i in range(n):
        rows[i]["duplicate_of"] = None
        rows[i]["contained_in"] = None
        for j in range(i):
            if rows[i]["file_sha256"] == rows[j]["file_sha256"] or (idsets[i] and idsets[i] == idsets[j]):
                rows[i]["duplicate_of"] = rows[j]["label"]
                break
    # partial overlap: share of this collection's review ids found in a bigger collection
    for i in range(n):
        if rows[i]["duplicate_of"] or not idsets[i]:
            continue
        best = (0.0, None)
        for j in range(n):
            if i == j or not idsets[j] or len(idsets[j]) <= len(idsets[i]):
                continue
            share = len(idsets[i] & idsets[j]) / len(idsets[i])
            if share > best[0]:
                best = (share, rows[j]["label"])
        if best[0] >= 0.5:
            rows[i]["contained_in"] = {"label": best[1], "share": round(best[0], 4)}
    # same-donor groups: union-find over shared review ids (review ids are millisecond timestamps, so two
    # collections sharing 100+ of them come from the same person; preset ids are not used: shared decks
    # and default presets make them collide across people)
    parent = list(range(n))
    pair_notes = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i):
            shared = len(idsets[i] & idsets[j])
            small = min(len(idsets[i]), len(idsets[j]))
            # same person: they share many exact review ids, or (deck exports of one collection, disjoint
            # review ids) they review in the same 10-minute windows most of the time
            same = small > 0 and shared >= 0.1 * small
            if not same and small > 0:
                bi = {x // 600_000 for x in idsets[i]}
                bj = {x // 600_000 for x in idsets[j]}
                same = len(bi & bj) >= 0.5 * min(len(bi), len(bj))
            pair_notes.append((i, j, shared, small))
            if same:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for gid, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        for i in members:
            rows[i]["donor_group"] = gid
    # unique reviews per donor group = union of review ids
    group_reviews = {}
    for root, members in groups.items():
        union = set().union(*(idsets[i] for i in members))
        group_reviews[root] = len(union)

    # files from one archive were sent by one donor
    for i in range(n):
        for j in range(i):
            if "::" in rows[i]["label"] and rows[i]["label"].split("::")[0] == rows[j]["label"].split("::")[0]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for gid, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        for i in members:
            rows[i]["donor_group"] = gid
    group_reviews = {root: len(set().union(*(idsets[i] for i in members))) for root, members in groups.items()}
    for i, j, shared, small in pair_notes:
        if shared:
            rows[i].setdefault("shared_review_ids", []).append({"with": rows[j]["label"], "shared": shared})
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    unique_reviews = sum(group_reviews.values())
    ok = [r for r in rows if "error" not in r]
    agg = {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "files": len({r["file_sha256"] for r in rows}),
        "collections": n,
        "errors": n - len(ok),
        "exact_duplicates": sum(1 for r in rows if r["duplicate_of"]),
        "mostly_contained": sum(1 for r in rows if r["contained_in"]),
        "donors_estimated": len(groups),
        "reviews_raw_sum": sum(r.get("reviews", 0) for r in ok),
        "reviews_unique": unique_reviews,
        "donors_with_reviews": sum(1 for root, c in group_reviews.items() if c > 0),
        "collections_without_review_history": sum(1 for r in ok if r.get("reviews", 0) == 0),
        "donors_10k_plus_reviews": sum(1 for c in group_reviews.values() if c >= 10_000),
        "first_review": min((r["first_review"] for r in ok if r.get("first_review")), default=None),
        "last_review": max((r["last_review"] for r in ok if r.get("last_review")), default=None),
        # donor sizes as buckets only (no per-donor list), so the aggregate stays non-identifying
        "donors_by_reviews": {
            label: sum(1 for c in group_reviews.values() if lo <= c < hi)
            for label, lo, hi in (("1-9,999", 1, 10_000), ("10k-49,999", 10_000, 50_000),
                                  ("50k-199,999", 50_000, 200_000), ("200k-999,999", 200_000, 1_000_000),
                                  ("1M+", 1_000_000, 10**12))
        },
    }
    Path(args.aggregate).parent.mkdir(parents=True, exist_ok=True)
    Path(args.aggregate).write_text(json.dumps(agg, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
