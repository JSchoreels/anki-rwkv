"""Publish the data-campaign page data (pages/data/donations.json) from the donated-collections folders.

usage: publish_donations_page.py [--folder DIR ...] [--force] [--check]

The folders (DEFAULT_FOLDERS: the campaign, LMSherlock's shared collections, Expertium's survey collections) are fingerprinted from the list of
(folder name, file name, size, mtime) of their files, hashed with SHA-256; only the
hash is stored (as `source_fingerprint` in the json), never the list. When the hash is unchanged the inventory is not
rerun and only the goals block is recomputed. When it changed, scripts/donor_inventory.py runs with its scratch, its
per-collection report and its aggregate under %TEMP%/anki-rwkv-donations (outside the repository); everything there is
deleted afterwards, so the per-collection report (which has donor names) never survives the run.

Privacy: the json keeps only whitelisted aggregate fields (counts, date range, donor sizes as buckets) plus the goals
block. Before writing, the output is checked to contain no file name of the folder.

--check  writes nothing; exits 1 when the json is missing, the folder changed since the json was written, or the json
         fails the privacy check, else 0.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pages" / "data" / "donations.json"
HISTORY = REPO / "pages" / "data" / "donations-history.json"  # aggregate totals over time (no names)


def append_history(doc):
    """Add one aggregate point (time, donors, unique reviews, collections) when the totals changed."""
    d = doc["donations"]
    point = {"t": doc.get("inventory_run") or doc.get("generated"), "donors": d["donors_with_reviews"],
             "reviews": d["reviews_unique"], "collections": d["collections"]}
    hist = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else []
    if hist and all(hist[-1].get(k) == point[k] for k in ("donors", "reviews", "collections")):
        return
    hist.append(point)
    HISTORY.write_text(json.dumps(hist, indent=1) + "\n", encoding="utf-8")


# donation folders: the campaign's Dropbox file request first, then collections shared with us by partners
DEFAULT_FOLDERS = ("D:/Dropbox/Anki Data", "D:/Dropbox/Jarrett User Collections",
                   "D:/Dropbox/Expertium Survey Collections")
DEFAULT_FOLDER = DEFAULT_FOLDERS  # old name, kept for importers
SCRATCH = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "anki-rwkv-donations"

# anki-revlogs-10k, from the parquet footers of data/anki-revlogs-10k/revlogs (row counts per user)
DATASET = {
    "name": "anki-revlogs-10k",
    "users": 10_000,
    "reviews": 727_738_650,
    "reviews_per_user_median": 30_990,
    "reviews_per_user_mean": 72_774,
    "reviews_per_user_p10": 7_535,
    "reviews_per_user_p90": 180_672,
}
EVAL_SET = {"users": 732, "reviews": 30_500_000}  # our standard 732-user evaluation set, about 30.5M reviews

# aggregate fields of donor_inventory.py that may be published (no names, no ids, donor sizes as buckets only)
AGG_KEYS = (
    "files", "collections", "errors", "exact_duplicates", "mostly_contained", "donors_estimated",
    "reviews_raw_sum", "reviews_unique", "donors_with_reviews", "collections_without_review_history",
    "donors_10k_plus_reviews", "first_review", "last_review", "donors_by_reviews",
    "reviews_by_category", "donors_by_category", "reviews_by_source", "donors_by_source",
)
SOURCES = ("campaign", "LMSherlock", "Expertium")  # public label per DEFAULT_FOLDERS entry (credited)
CATEGORIES = ("Japanese", "Chinese", "Korean", "Medicine", "Other languages", "Other", "Deleted cards")
GENERIC = {"anki", "collection", "collections", "export", "backup", "colpkg", "apkg", "anki2", "anki21",
           "review", "reviews", "history", "deck", "decks", "data", "dataset",
           "current"}  # "current": common in file names, and part of a fixed milestone name
BUCKETS = ("1-9,999", "10k-49,999", "50k-199,999", "200k-999,999", "1M+")


def _folders(folders):
    return [Path(folders)] if isinstance(folders, (str, Path)) else [Path(f) for f in folders]


def folder_files(folders):
    return [e for f in _folders(folders) for e in sorted((e for e in f.iterdir() if e.is_file()), key=lambda e: e.name)]


def fingerprint(folders):
    items = [(e.parent.name, e.name, (st := e.stat()).st_size, st.st_mtime_ns) for e in folder_files(folders)]
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


def run_inventory(folder):
    if SCRATCH.resolve().is_relative_to(REPO):
        sys.exit(f"scratch {SCRATCH} is inside the repository")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    agg_path = SCRATCH / "aggregate.json"
    try:
        cmd = [sys.executable, str(REPO / "scripts" / "donor_inventory.py"), *map(str, _folders(folder)),
               "--source-labels", ",".join(SOURCES), "--scratch", str(SCRATCH / "extract"), "--report", str(SCRATCH / "report-local.json"),
               "--aggregate", str(agg_path)]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            sys.exit(f"donor_inventory.py failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        return json.loads(agg_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)  # extracted collections and the per-collection report


def goals_block(agg):
    unique = int(agg["reviews_unique"])
    donors = int(agg["donors_with_reviews"])
    avg = unique / donors if donors else 0
    mean10k = DATASET["reviews_per_user_mean"]
    # Named goals (user 11-26). Reviews = users x the 10k dataset's mean per user. Accuracy estimates come from the
    # data-scaling test (SM174: from scratch, no teacher, on 10k-dataset users) and stay None until it reports.
    specs = (
        ("m1", "Milestone 1 (Minimum Viable)", 1_500, 500,
         "First MIT-licensed model trained only on donations", True, None),
        ("m2", "Milestone 2 (srs-benchmark same training dataset)", 5_000, 2_000,
         "As many training users as the srs-benchmark models (5,000)", False, None),
        ("m3", "Milestone 3 (Current Anki Fork user trained)", 8_000, 2_000,
         "As many users as the models in today's Anki fork", False, None),
    )
    out = []
    for gid, name, train, test, unlocks, provisional, accuracy in specs:
        users = train + test
        target = EVAL_SET["reviews"] if gid == "test-bench" else users * mean10k
        need = max(0, target - unique)
        users_pct = 100 * min(donors, users) / users
        reviews_pct = 100 * min(unique, target) / target
        out.append({
            "id": gid, "name": name, "label": name, "unlocks": unlocks, "provisional": provisional,
            "users_target": users, "train_users": train, "test_users": test, "reviews_target": target,
            "users_pct": round(users_pct, 3), "reviews_pct": round(reviews_pct, 3),
            # the bar uses the lower of the two: a model needs many different people, not only many reviews
            "progress_pct": round(min(users_pct, reviews_pct), 3),
            "reviews_needed": need, "users_needed": max(0, users - donors),
            "donors_needed_at_current_average": math.ceil(need / avg) if avg else None,
            "donors_needed_at_dataset_mean": math.ceil(need / mean10k),
            "accuracy_estimate_pct": accuracy,
        })
    return {
        "current_average_reviews_per_donor": round(avg),
        "dataset_mean_reviews_per_user": mean10k,
        "pct_of_dataset_reviews": round(100 * unique / DATASET["reviews"], 3),
        "pct_of_dataset_users": round(100 * donors / DATASET["users"], 3),
        "goals": out,
    }


def build(agg, fp, inventory_run):
    pub = {k: agg[k] for k in AGG_KEYS if k in agg}
    pub["donors_by_reviews"] = {b: int(agg.get("donors_by_reviews", {}).get(b, 0)) for b in BUCKETS}
    return {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "inventory_run": inventory_run,
        "source_fingerprint": fp,
        "donations": pub,
        "dataset": DATASET,
        "goals": goals_block(pub),
    }


def privacy_problems(doc, folder):
    probs = []
    if set(doc) - {"generated", "inventory_run", "source_fingerprint", "donations", "dataset", "goals"}:
        probs.append("unexpected top-level keys")
    if set(doc.get("donations", {})) - set(AGG_KEYS):
        probs.append("unexpected aggregate keys")
    for key in ("reviews_by_source", "donors_by_source"):
        if set(doc.get("donations", {}).get(key, {})) - set(SOURCES):
            probs.append(f"unexpected {key} labels")
    for key in ("reviews_by_category", "donors_by_category"):
        if set(doc.get("donations", {}).get(key, {})) - set(CATEGORIES):
            probs.append(f"unexpected {key} labels")
    if set(doc.get("donations", {}).get("donors_by_reviews", {})) - set(BUCKETS):
        probs.append("unexpected donor buckets")
    # string values only (keys are fixed field names); generic words that any file name may contain are skipped
    strings = []

    def walk(v):
        if isinstance(v, str):
            strings.append(v.lower())
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(doc)
    text = "\n".join(strings)
    if folder and all(f.is_dir() for f in _folders(folder)):
        for e in folder_files(folder):
            for token in {e.name, e.stem, *re.split(r"[\s_\-.()\[\]]+", e.stem)}:
                token = token.strip().lower()
                if len(token) >= 5 and not token.isdigit() and token not in GENERIC and token in text:
                    probs.append(f"a file-name token appears in the output (length {len(token)})")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", action="append", help="donation folder (repeatable; default: DEFAULT_FOLDERS)")
    ap.add_argument("--force", action="store_true", help="rerun the inventory even if the folder is unchanged")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if the json is stale or unsafe")
    args = ap.parse_args()
    folder = [Path(f) for f in (args.folder or DEFAULT_FOLDERS)]
    missing = [str(f) for f in folder if not f.is_dir()]
    if missing:
        sys.exit(f"folder not found: {missing}")
    fp = fingerprint(folder)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None

    if args.check:
        probs = [] if old else ["pages/data/donations.json is missing"]
        if old and old.get("source_fingerprint") != fp:
            probs.append("the donation folder changed since the json was written")
        if old:
            probs += privacy_problems(old, folder)
        print("\n".join(probs) if probs else "donations.json is up to date")
        return 1 if probs else 0

    if old and old.get("source_fingerprint") == fp and not args.force:
        agg, run = old["donations"], old.get("inventory_run")
        print("folder unchanged: inventory not rerun")
    else:
        agg = run_inventory(folder)
        run = agg.get("generated")
    doc = build(agg, fp, run)
    probs = privacy_problems(doc, folder)
    if probs:
        sys.exit("refusing to write: " + "; ".join(probs))
    strip = lambda d: {k: v for k, v in (d or {}).items() if k != "generated"}  # noqa: E731
    if old and strip(old) == strip(doc):
        print("donations.json unchanged")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    append_history(doc)
    d, g = doc["donations"], doc["goals"]
    print(f"wrote {OUT.relative_to(REPO).as_posix()}: {d['donors_with_reviews']} donors with reviews, "
          f"{d['reviews_unique']:,} unique reviews ({g['pct_of_dataset_reviews']}% of the 10k dataset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
