"""Convert donated Anki collections into the anki-revlogs-10k layout (one user per donor), CPU only.

Pipeline (docs/DONATIONS_DATASET.md has the full rules):
  pass 1  every collection of the donation folders is extracted, read and deleted one at a time with
          scripts/donor_inventory.py's own functions (iter_sources, read_collection); its answered reviews, cards,
          decks and day-boundary config are saved as numpy arrays in a scratch folder OUTSIDE the repository;
  group   files are grouped into donors with donor_inventory.group_donors (shared review ids, shared 10-minute
          windows, same archive), exactly as the inventory does;
  pass 2  donor by donor: union of the review ids (each review once), cards/decks from the file that holds them,
          then the anki-revlogs-dataset-builder rules (commit e299465, --raw-ids --show-time --emit-review-time)
          give revlogs/cards/decks parquet files with the schema of data/anki-revlogs-10k.

Outputs (under --out, default data/donations-YYYYMMDD, gitignored):
  revlogs|cards|decks/user_id=N/data.parquet  N = 1,000,001, 1,000,002, ... (cannot collide with the 10k dataset)
  sources.parquet   user_id -> source label (campaign / LMSherlock / Expertium), files merged, counts, day boundary
  summary.json      aggregate counts and the validation results (no names, no file names)
--answer-time-out DIR also writes the answer-time layout of data/anki-revlogs-10k-original-20261011 (no review_time
column; day_offset and intervals from the answer time), which the training run and the held-out scorer read.

Privacy: donation folders are only read; extracted collections and the per-file arrays live in the scratch folder
and are deleted at the end. Nothing written here holds a donor name or file name.

  python scripts/donations_to_dataset.py [--out data/donations-20261126] [--answer-time-out DIR] [--validate 5]
"""

import argparse
import ctypes
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import donor_inventory as inv  # noqa: E402

FOLDERS = ("D:/Dropbox/Anki Data", "D:/Dropbox/Jarrett User Collections", "D:/Dropbox/Expertium Survey Collections")
SOURCES = ("campaign", "LMSherlock", "Expertium")
USER_BASE = 1_000_000          # donor users are 1,000,001+; the 10k dataset uses 1..10,000
DEFAULT_ROLLOVER = 4           # Anki's own correction when a v2 collection has no rollover set
REV_DT = np.dtype([("id", np.int64), ("cid", np.int64), ("ease", np.int8), ("time", np.int64), ("type", np.int8),
                   ("factor", np.int64)])
CARD_DT = np.dtype([("id", np.int64), ("nid", np.int64), ("did", np.int64)])
DECK_DT = np.dtype([("id", np.int64), ("parent", np.int64), ("preset", np.int64)])
REVLOG_COLUMNS = ["card_id", "day_offset", "rating", "state", "duration", "elapsed_days", "elapsed_seconds",
                  "review_time"]


# ---------------------------------------------------------------------------------------------------------------
# pass 1: one collection -> numpy arrays

def _varint(buf, pos):
    shift = value = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, pos


def _proto_fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos:pos + size], pos + size
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"wire type {wire}")
        yield number, wire, value


def _deck_config_id(kind_blob):
    """decks.kind (Deck.kind oneof): field 1 = Normal {config_id = 1}, field 2 = Filtered (no preset)."""
    for number, wire, value in _proto_fields(bytes(kind_blob or b"")):
        if number == 1 and wire == 2:
            for n2, w2, v2 in _proto_fields(value):
                if n2 == 1 and w2 == 0:
                    return v2 - (1 << 64) if v2 >= 1 << 63 else v2
            return 0  # proto3 default
    return None


def read_decks(cur):
    """Normal decks as the research export writes them (rslib export_dataset): id, immediate parent's id (0 if
    none) and preset (deck options) id; filtered decks have no preset and are left out."""
    try:
        raw = [(did, name, _deck_config_id(kind), "\x1f") for did, name, kind in
               cur.execute("select id, name, kind from decks")]
    except sqlite3.OperationalError:  # legacy schema: decks as JSON in col.decks
        decks = json.loads(cur.execute("select decks from col").fetchone()[0] or "{}")
        raw = [(int(k), v.get("name", ""), None if v.get("dyn") else int(v.get("conf", 1)), "::")
               for k, v in decks.items()]
    by_name = {name: did for did, name, _, _ in raw}
    out = []
    for did, name, preset, sep in raw:
        if preset is None:
            continue
        parent = by_name.get(name.rsplit(sep, 1)[0], 0) if sep in name else 0
        out.append((did, parent, preset))
    return np.array(out, dtype=DECK_DT)


def _config_json(value):
    if value is None:
        return None
    try:
        return json.loads(bytes(value).decode("utf-8") if isinstance(value, (bytes, memoryview)) else value)
    except (ValueError, UnicodeDecodeError):
        return None


def read_day_config(cur):
    """Day-boundary settings: rollover hour, localOffset / creationOffset (minutes west of UTC), schedVer, crt."""
    keys = ("rollover", "localOffset", "creationOffset", "schedVer")
    conf = {}
    try:
        for key, val in cur.execute(f"select key, val from config where key in {keys}"):
            conf[key] = _config_json(val)
    except sqlite3.OperationalError:  # legacy schema: config as JSON in col.conf
        legacy = json.loads(cur.execute("select conf from col").fetchone()[0] or "{}")
        conf = {k: legacy.get(k) for k in keys}
    conf["crt"] = cur.execute("select crt from col").fetchone()[0]
    return {k: (int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None) for k, v in conf.items()}


def extract_arrays(path):
    """Answered revlog rows (ease 1..4, as the research export's button_chosen >= 1), cards and decks of one file.
    Values are read as rslib's row_to_revlog_entry decodes them: a taken time that is not an integer in u32 range
    becomes 0, an unknown review kind becomes Learning (0)."""
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro&immutable=1", uri=True)
    cur = con.cursor()
    rev = np.fromiter(cur.execute(
        "select id, cid, ease, "
        "case when typeof(time) = 'integer' and time between 0 and 4294967295 then time else 0 end, "
        "case when typeof(type) = 'integer' and type between 0 and 5 then type else 0 end, "
        "cast(coalesce(factor, 0) as integer) from revlog where ease between 1 and 4"), dtype=REV_DT)
    cards = np.fromiter(cur.execute("select id, nid, case when odid = 0 then did else odid end from cards"),
                        dtype=CARD_DT)
    decks = read_decks(cur)
    day = read_day_config(cur)
    con.close()
    return rev, cards, decks, day


# ---------------------------------------------------------------------------------------------------------------
# day boundary

def next_day_at(day, max_review_ms):
    """Timestamp (s) of the first rollover after the donor's last review, as Anki's timing_today computes
    next_day_at at export time: v2 = the rollover hour in a fixed UTC offset (localOffset, else creationOffset,
    else UTC); v1 (schedVer 1) = crt + k days. Only its phase matters: day_offset is shifted to start at 0."""
    now = int(max_review_ms // 1000)
    if day.get("schedVer") == 1 and day.get("crt") is not None:
        return now + (int(day["crt"]) - now - 1) % 86_400 + 1, {"rule": "v1-crt", "offset_source": "crt"}
    rollover = day.get("rollover")
    rollover_source = "config"
    if rollover is None:
        rollover, rollover_source = DEFAULT_ROLLOVER, "default-4"
    rollover %= 24
    for key in ("localOffset", "creationOffset"):
        if day.get(key) is not None:
            west, offset_source = max(-23 * 60, min(23 * 60, int(day[key]))), key
            break
    else:
        west, offset_source = 0, "fallback-utc"
    phase = rollover * 3600 + west * 60  # rollover in UTC seconds of the day (local = UTC - minutes west)
    nda = now + (phase - now - 1) % 86_400 + 1  # strictly after the last review, like rollover_passed
    return nda, {"rule": "v2", "rollover_hour": rollover, "rollover_source": rollover_source,
                 "utc_offset_minutes_west": west, "offset_source": offset_source}


# ---------------------------------------------------------------------------------------------------------------
# pass 2: one donor -> dataset frames

def merge_donor(parts):
    """parts: [(rev, cards, decks)] in priority order. Union of review ids (each review once: the row of the first
    file that holds the card, else of the first file), cards and decks from the first file that holds them."""
    revs = np.concatenate([p[0] for p in parts])
    rank = np.concatenate([np.full(p[0].size, r, np.int32) for r, p in enumerate(parts)])
    known = np.concatenate([np.isin(p[0]["cid"], p[1]["id"]) for p in parts])
    order = np.lexsort((rank, ~known, revs["id"]))
    sid = revs["id"][order]
    first = np.r_[True, sid[1:] != sid[:-1]] if sid.size else np.zeros(0, bool)
    rev = revs[order[first]]
    del revs, rank, known, order, sid, first
    cards = np.concatenate([p[1] for p in parts])
    cards = cards[np.sort(np.unique(cards["id"], return_index=True)[1])]
    decks = np.concatenate([p[2] for p in parts])
    decks = decks[np.sort(np.unique(decks["id"], return_index=True)[1])]
    return rev, cards, decks


def build_revlogs(rev, nda, show_time=True):
    """The dataset builder's process_revlogs on one user's answered rows (numpy port):
    drop filtered-deck rows without rescheduling (type 3, factor 0); per card in (card id, review id) order keep
    the rows from its last learning start (a type-0 row whose previous row of the card is not type 0) and drop
    cards without one; state = type + 1, 0 on that learning start; review_time = id - taken ms (show time) or id;
    day_offset = trunc((review_time / 1000 - next_day_at) / 86400) minus its minimum; elapsed_days = day_offset
    diff, elapsed_seconds = trunc(review_time diff / 1000) clamped at 0, both -1 on state 0; rows sorted by
    review_time (stable, ties in card/review order)."""
    rev = rev[~((rev["type"] == 3) & (rev["factor"] == 0))]
    rev = rev[np.lexsort((rev["id"], rev["cid"]))]
    n = rev.size
    if not n:
        return None
    cid, kind = rev["cid"], rev["type"].astype(np.int64)
    card_start = np.r_[True, cid[1:] != cid[:-1]]
    learn = (kind == 0) & (card_start | np.r_[True, kind[:-1] != 0])
    seg = np.flatnonzero(card_start)
    card_index = np.cumsum(card_start) - 1
    last_learn = np.maximum.reduceat(np.where(learn, np.arange(n), -1), seg)
    keep = (last_learn[card_index] >= 0) & (np.arange(n) >= last_learn[card_index])
    rev, learn_row = rev[keep], (np.arange(n) == last_learn[card_index])[keep]
    if not rev.size:
        return None
    state = np.where(learn_row, 0, rev["type"].astype(np.int64) + 1)
    review_time = rev["id"] - rev["time"] if show_time else rev["id"].copy()
    day = np.trunc((review_time / 1000 - nda) / 86400).astype(np.int64)
    day -= day.min()
    elapsed_days = np.r_[0, np.diff(day)]
    elapsed_seconds = np.r_[0, (np.diff(review_time) / 1000).astype(np.int64)]
    elapsed_seconds = np.maximum(elapsed_seconds, 0)
    elapsed_days[state == 0] = -1
    elapsed_seconds[state == 0] = -1
    frame = pd.DataFrame({"card_id": rev["cid"], "day_offset": day, "rating": rev["ease"].astype(np.int64),
                          "state": state, "duration": rev["time"], "elapsed_days": elapsed_days,
                          "elapsed_seconds": elapsed_seconds, "review_time": review_time})
    frame = frame.iloc[np.argsort(review_time, kind="stable")]
    return frame if show_time else frame.drop(columns="review_time")


def write_user(root, user, revlogs, cards, decks):
    for table, frame, index in (("revlogs", revlogs, True), ("cards", cards, False), ("decks", decks, False)):
        folder = root / table / f"user_id={user}"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / "data.parquet", index=index)


# ---------------------------------------------------------------------------------------------------------------

def peak_memory_gb():
    return inv.peak_memory_gb()


def validate_loader(root, users):
    """Build the training features of `users` with the real loader (rwkv.data_processing.get_rwkv_data)."""
    sys.path.insert(0, str(REPO))
    from rwkv.data_processing import get_rwkv_data
    out = {}
    for user in users:
        t0 = time.time()
        try:
            frame = get_rwkv_data(root, user)
            numeric = frame.select_dtypes("number")
            # label_review_th is NaN on each card's last row in the 10k dataset too (no next review)
            nonfinite = [c for c in numeric.columns if not np.isfinite(numeric[c].to_numpy(np.float64)).all()]
            out[str(user)] = {"ok": nonfinite in ([], ["label_review_th"]), "rows": int(len(frame)),
                              "columns": int(frame.shape[1]), "nonfinite_columns": nonfinite,
                              "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            out[str(user)] = {"ok": False, "error": repr(exc)[:300]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", action="append", help="donation folder (repeatable; default: the three folders)")
    ap.add_argument("--source-labels", default=",".join(SOURCES))
    ap.add_argument("--out", default=str(REPO / "data" / f"donations-{date.today():%Y%m%d}"))
    ap.add_argument("--answer-time-out", default="", help="also write the answer-time layout here")
    ap.add_argument("--scratch", default="", help="scratch folder OUTSIDE the repository (default: a new temp dir)")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--validate", type=int, default=5, help="users loaded with get_rwkv_data afterwards (0 = none)")
    args = ap.parse_args()
    if os.name == "nt":  # below normal priority: a GPU job may share the machine
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    folders = args.folder or list(FOLDERS)
    labels = args.source_labels.split(",")
    out = Path(args.out).resolve()
    answer_out = Path(args.answer_time_out).resolve() if args.answer_time_out else None
    for target in (out, answer_out):
        if target is not None and REPO in target.parents and (REPO / "data") not in [target, *target.parents]:
            sys.exit(f"{target}: outputs inside the repository go under data/ (gitignored)")
        if target is not None and target.exists() and any(target.iterdir()):
            sys.exit(f"{target} exists and is not empty")
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="anki-rwkv-donations-"))
    if REPO in scratch.resolve().parents or scratch.resolve() == REPO:
        sys.exit(f"scratch {scratch} is inside the repository")
    extract_dir, arrays_dir = scratch / "extract", scratch / "arrays"
    extract_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        # pass 1 ---------------------------------------------------------------------------------------------
        rows, idsets, catsets, days, answered = [], [], [], [], []
        sources = ((si, x) for si, f in enumerate(folders) for x in inv.iter_sources(f, extract_dir, prefix=f"s{si}"))
        for si, (label, path, kind, sha) in sources:
            i = len(rows)
            try:
                row, ids, cats = inv.read_collection(path)
                rev, cards, decks, day = extract_arrays(path)
                np.save(arrays_dir / f"{i}-rev.npy", rev)
                np.save(arrays_dir / f"{i}-cards.npy", cards)
                np.save(arrays_dir / f"{i}-decks.npy", decks)
                n_answered = int(rev.size)
                del rev, cards, decks
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the conversion (as the inventory)
                row, ids, cats = {"error": str(exc)[:200]}, np.zeros(0, np.int64), np.zeros(0, np.uint8)
                day, n_answered = {}, 0
                for suffix, dt in (("rev", REV_DT), ("cards", CARD_DT), ("decks", DECK_DT)):
                    np.save(arrays_dir / f"{i}-{suffix}.npy", np.zeros(0, dt))
            row.update(label=label, kind=kind, file_sha256=sha, source=labels[min(si, len(labels) - 1)])
            rows.append(row)
            idsets.append(ids)
            catsets.append(cats)
            days.append(day)
            answered.append(n_answered)
            if Path(path).parent == extract_dir:
                Path(path).unlink()
            print(f"[donations] file {i + 1:3d}: {row.get('reviews', 'ERR')!s:>9} reviews "
                  f"({time.time() - t0:.0f} s, peak {peak_memory_gb()} GB)", file=sys.stderr, flush=True)

        # group ----------------------------------------------------------------------------------------------
        n, size, groups = inv.group_donors(rows, idsets)
        donors = sorted(groups.values(), key=lambda members: members[0])  # inventory's donor_group order

        # pass 2 ---------------------------------------------------------------------------------------------
        meta, mismatches = [], []
        totals = {"donors": len(donors), "users": 0, "skipped_no_reviews": 0, "skipped_no_learning_start": 0}
        next_user = USER_BASE + 1
        for gid, members in enumerate(donors):
            inv_unique, _ = inv.merge_group([idsets[i] for i in members], [catsets[i] for i in members])
            if inv_unique == 0:
                totals["skipped_no_reviews"] += 1
                continue
            weight = {}
            for i in members:
                weight[rows[i]["source"]] = weight.get(rows[i]["source"], 0) + size[i]
            source = max(weight, key=weight.get)
            prio = sorted(members, key=lambda i: (-answered[i], i))  # biggest file first, then file order
            parts = [tuple(np.load(arrays_dir / f"{i}-{s}.npy") for s in ("rev", "cards", "decks")) for i in prio]
            rev, cards, decks = merge_donor(parts)
            del parts
            union = int(rev.size)
            if union != inv_unique:
                mismatches.append({"donor_group": gid, "converter": union, "inventory": inv_unique})
            day_cfg = next((days[i] for i in prio if days[i]), {})
            nda, day_info = next_day_at(day_cfg, int(rev["id"].max()))
            show = build_revlogs(rev, nda, show_time=True)
            if show is None:
                totals["skipped_no_learning_start"] += 1
                continue
            user = next_user
            next_user += 1
            cards_df = pd.DataFrame({"card_id": cards["id"], "note_id": cards["nid"], "deck_id": cards["did"]})
            decks_df = pd.DataFrame({"deck_id": decks["id"], "parent_id": decks["parent"],
                                     "preset_id": decks["preset"]})
            write_user(out, user, show, cards_df, decks_df)
            if answer_out is not None:
                write_user(answer_out, user, build_revlogs(rev, nda, show_time=False), cards_df, decks_df)
            kept_cards = show["card_id"].unique()
            meta.append({"user_id": user, "source": source, "files": len(members),
                         "reviews_unique": inv_unique, "reviews": int(len(show)),
                         "cards_reviewed": int(kept_cards.size),
                         "reviews_deleted_cards": int((~show["card_id"].isin(cards["id"])).sum()),
                         "cards": int(cards.size), "decks": int(decks.size),
                         "day_rule": day_info["rule"], "rollover_hour": day_info.get("rollover_hour"),
                         "rollover_source": day_info.get("rollover_source"),
                         "utc_offset_minutes_west": day_info.get("utc_offset_minutes_west"),
                         "offset_source": day_info["offset_source"]})
            totals["users"] += 1
            print(f"[donations] user {user}: {len(show):>9} reviews of {inv_unique} unique ({source}, "
                  f"{len(members)} files; {time.time() - t0:.0f} s, peak {peak_memory_gb()} GB)",
                  file=sys.stderr, flush=True)
            del rev, cards, decks, show, cards_df, decks_df

        sources_df = pd.DataFrame(meta)
        sources_df.to_parquet(out / "sources.parquet", index=False)
        if answer_out is not None:
            sources_df.to_parquet(answer_out / "sources.parquet", index=False)
        by_source = {s: {"users": int((sources_df["source"] == s).sum()),
                         "reviews": int(sources_df.loc[sources_df["source"] == s, "reviews"].sum()),
                         "reviews_unique": int(sources_df.loc[sources_df["source"] == s, "reviews_unique"].sum())}
                     for s in labels}
        summary = {
            "generated": time.strftime("%Y-%m-%dT%H:%M"), "collections": n,
            "collection_errors": sum(1 for r in rows if "error" in r), **totals,
            "reviews": int(sources_df["reviews"].sum()), "reviews_unique": int(sources_df["reviews_unique"].sum()),
            "reviews_unique_all_donors": int(sum(inv.merge_group([idsets[i] for i in m], [catsets[i] for i in m])[0]
                                                 for m in donors)),
            "by_source": by_source,
            "count_check": {"donors_compared": len(meta) + totals["skipped_no_learning_start"],
                            "mismatches": mismatches, "ok": not mismatches},
            "day_boundary": {k: {str(v): int(c) for v, c in sources_df[k].value_counts(dropna=False).items()}
                             for k in ("day_rule", "rollover_source", "offset_source")},
            "user_ids": [int(sources_df["user_id"].min()), int(sources_df["user_id"].max())] if len(meta) else [],
            "answer_time_out": bool(answer_out),
            "seconds": round(time.time() - t0), "peak_memory_gb": peak_memory_gb(),
        }
        if args.validate and len(meta):
            picks = sources_df.sort_values("reviews")["user_id"].tolist()
            step = max(1, len(picks) // args.validate)
            summary["loader_check"] = validate_loader(out, picks[::step][:args.validate])
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary, indent=1))
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
