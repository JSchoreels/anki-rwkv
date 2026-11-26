"""Inventory of donated Anki collections: reviews, cards, date range, duplicates and likely-same-donor groups.

Reads .apkg / .colpkg / .anki2 / .anki21 files (and .zip archives of them) from a donation folder, read-only.
Packages are extracted to a scratch directory outside the repository; zstd-compressed collections
(collection.anki21b) are decompressed with the libzstd.dll that ships with Git for Windows (ctypes, no install).

Outputs:
  --report  local JSON with one row per collection (includes file names = donor labels: keep it local, never commit)
  --aggregate  aggregate-only JSON (counts, no names, no ids) that may be published

Several folders may be given (the campaign first, then partner folders); --source-labels names each one in the
aggregate's reviews_by_source / donors_by_source.

Memory: review ids are kept as sorted unique int64 numpy arrays with a parallel uint8 array of category codes
(about 9 bytes per review), so a few hundred collections stay well under a few GB.

Privacy: never commit the --report output or any extracted collection; only the aggregate may be published.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


def iter_sources(folder, scratch, prefix=""):
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
                path, inner = extract_collection(zf, scratch, f"{prefix}c{n:04d}")
            if path:
                yield entry.name, path, inner, sha
        elif low.endswith(".zip"):
            with zipfile.ZipFile(entry) as outer:
                for info in outer.infolist():
                    if info.filename.lower().endswith((".apkg", ".colpkg")):
                        n += 1
                        nested = scratch / f"{prefix}n{n:04d}.pkg"
                        with outer.open(info) as src, open(nested, "wb") as dst:
                            while block := src.read(1 << 20):
                                dst.write(block)
                        nested_sha = sha256_file(nested)
                        with zipfile.ZipFile(nested) as zf:
                            path, inner = extract_collection(zf, scratch, f"{prefix}c{n:04d}")
                        nested.unlink()
                        if path:
                            yield f"{entry.name}::{info.filename}", path, inner, nested_sha


CATEGORIES = ("Japanese", "Chinese", "Korean", "Medicine", "Other languages", "Other")
CODES = (*CATEGORIES, "Deleted cards")  # uint8 category code = index here
CODE = {c: i for i, c in enumerate(CODES)}
DELETED = CODE["Deleted cards"]
WINDOW_MS = 600_000  # 10-minute review windows
_MED = re.compile(r"anking|zanki|step ?[123]|usmle|comlex|mcat|nclex|nursing|medic|pharm|anatom|physio|patho|"
                  r"microbio|biochem|boards|lightyear|pepper|sketchy|pathoma|b&b|bnb|uworld|amboss|immunol|clinical",
                  re.I)
_JA = re.compile(r"japan|日本|kaishi|jlpt|n[1-5]|kanji|漢字|core ?[26]k|tango|wanikani|kana|文法|漢検|漢檢", re.I)
_ZH = re.compile(r"chinese|mandarin|hsk|cantonese|中文|汉语|漢語", re.I)
_KO = re.compile(r"korean|hangul|한국|topik", re.I)
_LANG = re.compile(r"spanish|french|german|italian|portug|russian|dutch|swedish|polish|arabic|hebrew|greek|latin|"
                   r"turkish|vietnam|thai|hindi|english|vocab|language|langue|idioma|sprache|esperanto|ukrain", re.I)


def _script_shares(text):
    kana = han = hangul = cyr = letters = 0
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            kana += 1
        elif 0x4E00 <= o <= 0x9FFF:
            han += 1
        elif 0xAC00 <= o <= 0xD7AF:
            hangul += 1
        elif 0x0400 <= o <= 0x04FF:
            cyr += 1
        if ch.isalpha():
            letters += 1
    n = max(1, letters)
    return kana / n, han / n, hangul / n, cyr / n


def classify_deck(name, notetypes, sample_text):
    """Broad subject of a deck from its name, its note types and a sample of its note text (never stored)."""
    label = f"{name} {' '.join(notetypes)}"
    kana, han, hangul, cyr = _script_shares(sample_text)
    if _MED.search(label):
        return "Medicine"
    if kana >= 0.02 or _JA.search(label):
        return "Japanese"
    if hangul >= 0.05 or _KO.search(label):
        return "Korean"
    if han >= 0.05 or _ZH.search(label):
        return "Chinese"
    if cyr >= 0.05 or _LANG.search(label):
        return "Other languages"
    return "Other"


def _deck_names(cur):
    try:
        return {did: name.replace("", "::") for did, name in cur.execute("select id, name from decks")}
    except sqlite3.OperationalError:
        decks = json.loads(cur.execute("select decks from col").fetchone()[0])
        return {int(k): v["name"] for k, v in decks.items()}


def _notetype_names(cur):
    try:
        return dict(cur.execute("select id, name from notetypes"))
    except sqlite3.OperationalError:
        models = json.loads(cur.execute("select models from col").fetchone()[0])
        return {int(k): v["name"] for k, v in models.items()}


def reviews_by_category(cur):
    decks, models = _deck_names(cur), _notetype_names(cur)
    per_deck = dict(cur.execute("select c.did, count(*) from revlog r join cards c on c.id = r.cid "
                                "where r.ease between 1 and 4 group by c.did"))
    orphan = cur.execute("select count(*) from revlog r left join cards c on c.id = r.cid "
                         "where r.ease between 1 and 4 and c.id is null").fetchone()[0]
    out = dict.fromkeys(CATEGORIES, 0)
    out["Deleted cards"] = orphan
    deck_cat = {}
    for did, n in per_deck.items():
        rows = cur.execute("select n.mid, n.flds from cards c join notes n on n.id = c.nid where c.did = ? "
                           "limit 300", (did,)).fetchall()
        mids = {models.get(m, "") for m, _ in rows}
        text = " ".join(f[:400] for _, f in rows)
        deck_cat[did] = classify_deck(decks.get(did, ""), mids, text)
        out[deck_cat[did]] += n
    # category of every answered review id, so a donor who sent both a collection and deck exports of it is
    # counted once per review (the same review id appears in both files): sorted unique int64 ids plus a parallel
    # uint8 array of category codes (numpy, not dicts: ~9 bytes per review instead of ~100)
    pairs = np.fromiter(cur.execute("select r.id, coalesce(c.did, -1) from revlog r left join cards c "
                                    "on c.id = r.cid where r.ease between 1 and 4"),
                        dtype=[("rid", np.int64), ("did", np.int64)])
    order = np.argsort(pairs["rid"], kind="stable")
    rids, dids = pairs["rid"][order], pairs["did"][order]
    del pairs, order
    if rids.size > 1:  # a repeated id keeps its last row, as a dict would
        last = np.append(rids[1:] != rids[:-1], True)
        rids, dids = rids[last], dids[last]
    udid, inv = np.unique(dids, return_inverse=True)
    table = np.array([CODE[deck_cat.get(int(d), "Deleted cards")] for d in udid], dtype=np.uint8)
    cats = table[inv.reshape(-1)] if rids.size else np.zeros(0, np.uint8)
    return out, rids, cats


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
    # ids: the answered review ids (ease 1-4), sorted and unique; cats: their category codes
    row["reviews_by_category"], ids, cats = reviews_by_category(cur)
    con.close()
    return row, ids, cats


def count_shared(a, b):
    """Number of values in both sorted unique int64 arrays (exact); disjoint id ranges are skipped at once."""
    if not a.size or not b.size or a[-1] < b[0] or b[-1] < a[0]:
        return 0
    lo, hi = max(a[0], b[0]), min(a[-1], b[-1])
    a = a[np.searchsorted(a, lo):np.searchsorted(a, hi, side="right")]
    b = b[np.searchsorted(b, lo):np.searchsorted(b, hi, side="right")]
    if a.size > b.size:
        a, b = b, a
    if not a.size:
        return 0
    # same count as np.intersect1d(a, b, assume_unique=True).size, without sorting the concatenation
    idx = np.minimum(np.searchsorted(b, a), b.size - 1)
    return int(np.count_nonzero(b[idx] == a))


def merge_group(idlist, catlist):
    """Unique reviews of one donor and its reviews per category, merging the donor's files in order.

    Same rule as a dict filled file by file: a review takes the category of the first file that knows its card, else
    "Deleted cards". The categories come back in the order a review of each first appears (file order, then id), so
    the output and the tie-break of max() match the per-review dict this replaces."""
    if len(idlist) == 1:
        uid, ucat, first = idlist[0], catlist[0], None
    else:
        allid = np.concatenate(idlist)
        allcat = np.concatenate(catlist)
        rank = np.concatenate([np.full(a.size, r, np.int32) for r, a in enumerate(idlist)])
        order = np.lexsort((rank, allcat == DELETED, allid))  # by id, known subject first, then file order
        sid = allid[order]
        start = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]) if sid.size else np.zeros(0, np.int64)
        uid, ucat = sid[start], allcat[order][start]
        first = np.minimum.reduceat(rank[order], start) if start.size else np.zeros(0, np.int32)
        del allid, allcat, rank, order, sid
    if not uid.size:
        return 0, {}
    counts = np.bincount(ucat, minlength=len(CODES))
    seq = ucat if first is None else ucat[np.lexsort((uid, first))]  # dict insertion order: first file, then id
    codes, pos = np.unique(seq, return_index=True)
    per = {CODES[c]: int(counts[c]) for c in codes[np.argsort(pos)]}
    return int(uid.size), per


def peak_memory_gb():
    try:
        import psutil  # optional: only for the log line
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="+", help="donation folders; the first is the campaign, later ones are partners")
    ap.add_argument("--scratch", required=True, help="directory OUTSIDE the repository for extracted collections")
    ap.add_argument("--report", required=True, help="local per-collection JSON (has donor labels: never commit)")
    ap.add_argument("--aggregate", required=True, help="aggregate-only JSON (publishable)")
    ap.add_argument("--keep", action="store_true", help="keep extracted collections")
    ap.add_argument("--source-labels", help="comma-separated public label per folder, e.g. campaign,LMSherlock")
    args = ap.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows, idsets, catsets = [], [], []
    labels = (args.source_labels.split(",") if args.source_labels else ["campaign", "partner"])
    # lazy: one collection is extracted, read and deleted at a time
    sources = ((si, x) for si, f in enumerate(args.folder) for x in iter_sources(f, scratch, prefix=f"s{si}"))
    for si, (label, path, kind, sha) in sources:
        try:
            row, ids, cats = read_collection(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the inventory
            row, ids, cats = {"error": str(exc)[:200]}, np.zeros(0, np.int64), np.zeros(0, np.uint8)
        row.update(label=label, kind=kind, file_sha256=sha, source=labels[min(si, len(labels) - 1)])
        rows.append(row)
        idsets.append(ids)
        catsets.append(cats)
        if not args.keep and Path(path).parent == scratch:
            Path(path).unlink()
        print(f"[inventory] {len(rows):3d} {row.get('reviews', 'ERR')!s:>10} reviews  {kind}", file=sys.stderr)

    n = len(rows)
    size = [int(a.size) for a in idsets]
    windows = [np.unique(a // WINDOW_MS) for a in idsets]  # 10-minute windows with at least one review
    # shared review ids of every pair (computed once; used by the duplicate, containment and same-donor rules)
    shared = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i):
            shared[i, j] = shared[j, i] = count_shared(idsets[i], idsets[j])
    # exact duplicates: same file, or same set of review ids
    for i in range(n):
        rows[i]["duplicate_of"] = None
        rows[i]["contained_in"] = None
        for j in range(i):
            if rows[i]["file_sha256"] == rows[j]["file_sha256"] or (
                    size[i] and size[i] == size[j] and shared[i, j] == size[i]):
                rows[i]["duplicate_of"] = rows[j]["label"]
                break
    # partial overlap: share of this collection's review ids found in a bigger collection
    for i in range(n):
        if rows[i]["duplicate_of"] or not size[i]:
            continue
        best = (0.0, None)
        for j in range(n):
            if i == j or not size[j] or size[j] <= size[i]:
                continue
            share = int(shared[i, j]) / size[i]
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
            sh = int(shared[i, j])
            small = min(size[i], size[j])
            # same person: they share many exact review ids, or (deck exports of one collection, disjoint
            # review ids) they review in the same 10-minute windows most of the time
            same = small > 0 and sh >= 0.1 * small
            if not same and small > 0:
                same = count_shared(windows[i], windows[j]) >= 0.5 * min(windows[i].size, windows[j].size)
            pair_notes.append((i, j, sh, small))
            if same:
                parent[find(i)] = find(j)
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
    for i, j, sh, small in pair_notes:
        if sh:
            rows[i].setdefault("shared_review_ids", []).append({"with": rows[j]["label"], "shared": sh})

    # per donor: union of its review ids (collection + deck exports + backups count each review once) = its unique
    # reviews; subjects: a review keeps a known subject over "Deleted cards" when one of the files still has its
    # card; a donor's main subject is the category with most reviews (deleted cards excluded, since their deck is
    # unknown)
    group_reviews, cat_reviews, cat_donors = {}, {}, {}
    for root, members in groups.items():
        group_reviews[root], per = merge_group([idsets[i] for i in members], [catsets[i] for i in members])
        for k, v in per.items():
            cat_reviews[k] = cat_reviews.get(k, 0) + v
        known = {k: v for k, v in per.items() if k != "Deleted cards" and v > 0}
        if known:
            main_cat = max(known, key=known.get)
            cat_donors[main_cat] = cat_donors.get(main_cat, 0) + 1
            for i in members:
                rows[i]["donor_main_subject"] = main_cat
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    unique_reviews = sum(group_reviews.values())
    # per source: each donor counted once, under the source holding most of its reviews
    src_reviews, src_donors = {}, {}
    for root, members in groups.items():
        if not group_reviews[root]:
            continue
        weight = {}
        for i in members:
            weight[rows[i]["source"]] = weight.get(rows[i]["source"], 0) + size[i]
        src = max(weight, key=weight.get)
        src_reviews[src] = src_reviews.get(src, 0) + group_reviews[root]
        src_donors[src] = src_donors.get(src, 0) + 1
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
        "reviews_by_source": src_reviews,
        "donors_by_source": src_donors,
        "reviews_by_category": cat_reviews,
        "donors_by_category": cat_donors,
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
    print(f"[inventory] {n} collections in {time.time() - t0:.0f} s, peak memory {peak_memory_gb()} GB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
