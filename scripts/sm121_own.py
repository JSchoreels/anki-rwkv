#!/usr/bin/env python
"""Timed recall on your own collection (SM121): the answer-time heads of the time heads for the family (SM117) and of
the time heads for the new candidates (SM119), scored on the user's own Anki collection (CPU only, aggregate output),
against recent-history and card-history baselines, with discrimination read-outs and a separate reproduction control
of the Mac's replay (artifacts/small-model/sm121/design.md, pre-registered).

Answer time = front + back: question shown -> grade pressed, capped at the preset's maximum answer seconds. "Within
S s" always means answered within S s, reading the back included.

Sources (read only; nothing is ever opened under AppData):
  main   the collection copy of 2026-10-14 and its conversion (anki-rwkv artifacts/collection-eval/2026-10-14), the
         inputs of every own-collection score so far (173,690 benchmark test reviews, label DB)
  repro  a plain copy of the profile's collection.anki2 (last written 2026-11-03), converted here with the same
         convert_collection.py (artifacts/small-model/sm121/work/repro, gitignored)
Per-review arrays stay in artifacts/small-model/sm121/work/ (gitignored, never committed); result files hold
aggregates only (no review / card ids, no timestamps, no deck names).

  convert                                       the repro copy -> work/repro/collection-dataset (CPU, ~1 min)
  trunk   --model KEY [--source S] [--state fp32|s4c8n] [--cap N] --out F.npz
                                                one frozen model on CPU: P(grade) and its heads' P(t | grade) on all
                                                time-labelled non-first reviews (SM117 rule)
  check   --out F                               the tiny CPU check (first --cap reviews) + proofs
  report                                        main analysis -> sm121-result.{md,json}
  repro                                         reproduction control -> sm121-repro.{md,json}
"""
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sm117_head as h117  # noqa: E402
from scripts import sm120_fast as f120  # noqa: E402
from scripts.sm117_head import MODELS, verified_master, build_model, Frozen, load_head, show_time_batch, cdf_at  # noqa: E402
from scripts.sm117_timelib import (BINS, TIME_BIN_EDGES_MS, CAP_MS, time_labels, time_user_cap, explicit_table,  # noqa: E402
                                   revlog, atomic_json, median_log_s)

COLL = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14")   # read only
OUT = ROOT / "artifacts/small-model/sm121"
WORK = OUT / "work"                     # gitignored per-review intermediates: never git add -f
REPRO = WORK / "repro"
SOURCES = {"main": {"dir": COLL, "data": COLL / "collection-dataset", "copy": COLL / "collection-copy" / "collection.anki2",
                    "th_map": COLL / "review-th-to-revlog-id.parquet", "ldb": COLL / "collection-label-db"},
           "repro": {"dir": REPRO, "data": REPRO / "collection-dataset", "copy": REPRO / "collection-copy" / "collection.anki2",
                     "th_map": REPRO / "review-th-to-revlog-id.parquet", "ldb": None}}
USER = 1
PRIORS = ROOT / "artifacts/small-model/sm120/priors.json"   # SM120's pooled priors (benchmark training users)
S_GRID = f120.S_GRID                    # 2, 3, 4, 5, 6, 8, 10, 15, 20, 30 s
SETS = f120.SETS                        # Rec (Hard, Good, Easy), GE (Good, Easy), E (Easy); 0-based grades
SET_NAMES = f120.SET_NAMES
SPEED = f120.SPEED_GRADES
LN_VAL = f120.LN_VAL
M = 10.0                                # pseudo-count of the cumulative recent-history baselines (SM117 / SM120)
ROLL = 2000                             # rolling window (reviews)
CARD_LAST = 3                           # card history: the card's last 3 earlier positive-duration answers (any day, any grade)
CARD_C = 3.0                            # card-history strength (main); sensitivity 1 and 10 (Mac baselines)
CARD_SENS = (1.0, 10.0)
MAC_GRADE_C = 10.0                      # Mac's per-grade recent shrink strength toward the pooled success fraction
MAC_DATE = datetime(2026, 4, 28)        # the Mac's slice start (sensitivity slice)
BOOT = 2000
ECE_BINS = 20
AUC_BINS = 200
YOMITAN = "yomitan"                     # deck-name substring (case-insensitive) that defines "the Yomitan decks"
FORK = Path("C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference")   # read only (fork main checkout)
FORK_SHA = {"sm92/time_head.f32.bin": "814e9b891987204a9e802b714e12c72282cc267c69511c27520131aaf0d2f09d",
            "sm92/weights.f32.bin": "4fbc5f963b6ef18d23dbc0f11358fed8a9c7c97c271d6eda95df3b8ed6fbeb6e",
            "sm98/weights.f32.bin": "a006dcc039d19142dc95d257d9c238fb93c7efa12b78e18ec6ae881ac9009a24"}
MAC_BACKUP_SHA = "2515ef66b956ed9b28e065aa895a92db539fc2b126fabd5660217711cf36f66b"
MAC_COUNTS = {"canonical_reviews": 226374, "eligible_prefix": 143125, "scored": 35782, "scored_yomitan": 27431}
REPRO_T = (2, 4, 6, 8, 10, 15, 20, 30)
MAC_REF = {   # the Mac's LogLoss: time head / R x recent / R x card / Grade x recent / Grade x card
    ("yomitan", 2): (0.100498, 0.113126, 0.115532, 0.111090, 0.112567),
    ("yomitan", 4): (0.490168, 0.539743, 0.527310, 0.530051, 0.522184),
    ("yomitan", 30): (0.583622, 0.583512, 0.587449, 0.583465, 0.586303),
    ("full", 4): (0.503622, 0.571159, 0.543742, 0.560409, 0.540229)}
MAC_ORDER = ("head", "mac_succ", "mac_succ_card", "mac_grade", "mac_grade_card")

SM117 = Path("C:/Users/jonat/workspace/wt/sm117/artifacts/small-model")   # read only
SM119 = Path("C:/Users/jonat/workspace/wt/sm119/artifacts/small-model")   # read only
TRUNKS = {   # trunk key -> head dir (its master.pth is the frozen trunk) and {head name: (file, state-dict prefix)}
    "sm92": {"dir": SM117 / "sm117-sm92-head", "heads": {"sm117": ("head.pth", ""),
             "sm83": (str(MODELS["sm92"]["shipped_head"]), "time_mlps.0.")}},
    "sm98": {"dir": SM117 / "sm117-sm98-head", "heads": {"sm117": ("head.pth", "")}},
    "sm106": {"dir": SM117 / "sm117-sm106-head", "heads": {"sm117": ("head.pth", "")}},
    "sm91s2": {"dir": SM117 / "sm117-sm91s2-head", "heads": {"sm117": ("head.pth", "")}},
    "sm99": {"dir": SM117 / "sm117-sm99-head", "heads": {"sm117": ("head.pth", "")}},
    "sm113": {"dir": SM119 / "sm119-sm113-head", "heads": {"sm119": ("head.pth", "")}},
    "sm115": {"dir": SM119 / "sm119-sm115-head", "heads": {"sm119": ("head.pth", "")}},
}
COMBOS = {   # key -> (grade-probability trunk, (time-head trunk, head name), plain name)
    "mac_pair": ("sm98", ("sm92", "sm83"), "the Mac's Small pair: grades of the Small Instant-Only model (SM98), time from the "
                 "answer-time head release (SM83) on the show-time retrain (SM92)"),
    "small_pair_sm98": ("sm98", ("sm92", "sm117"), "Small pair: grades of the Small Instant-Only model (SM98), time from the "
                        "show-time retrain (SM92)'s SM117 head"),
    "small_pair_sm113": ("sm113", ("sm92", "sm117"), "Small pair as the fork runs it today: grades of the longer Small "
                         "Instant-Only model (SM113), time from the show-time retrain (SM92)'s SM117 head"),
    "sm92": ("sm92", ("sm92", "sm117"), "the show-time retrain (SM92) with its own SM117 head"),
    "sm98": ("sm98", ("sm98", "sm117"), "the Small Instant-Only model (SM98) with its own SM117 head"),
    "sm113": ("sm113", ("sm113", "sm119"), "the longer Small Instant-Only model (SM113) with its own SM119 head"),
    "tiny_pair_sm115": ("sm115", ("sm106", "sm117"), "Tiny pair as the fork runs it today: grades of the longer cool-down for "
                        "Tiny (SM115), time from the Tiny two-head model (SM106)'s SM117 head"),
    "sm106": ("sm106", ("sm106", "sm117"), "the Tiny two-head model (SM106) with its own SM117 head"),
    "sm115": ("sm115", ("sm115", "sm119"), "the longer cool-down for Tiny (SM115) with its own SM119 head"),
    "sm91s2": ("sm91s2", ("sm91s2", "sm117"), "the show-time big model (SM91s2) with its own SM117 head"),
    "sm99": ("sm99", ("sm99", "sm117"), "the show-time instant-only big model (SM99) with its own SM117 head (the Big pair "
             "as the fork runs it)"),
}
FAST_BAR = ("raw", "succ_hist", "grade_hist", "rate", "roll", "succ_card", "grade_card",
            "mac_succ", "mac_grade", "mac_succ_card", "mac_grade_card")
FAST_SENS = ("mac_succ_card_c1", "mac_succ_card_c10", "mac_grade_card_c1", "mac_grade_card_c10")
FAST_NAMES = {"head": "time head", "raw": "(1) raw P(set), time ignored",
              "succ_hist": "(2) P(set) x recent share of the set within S",
              "grade_hist": "(3) sum_g P(g) x recent share of grade g within S",
              "rate": "(b) running rate of the event", "roll": "(b2) rolling 2,000-review rate of the event",
              "succ_card": "(4) as (2), card-adjusted", "grade_card": "(5) as (3), card-adjusted",
              "mac_succ": "Mac R x recent", "mac_grade": "Mac grade x recent",
              "mac_succ_card": "Mac R x card", "mac_grade_card": "Mac grade x card",
              "mac_succ_card_c1": "Mac R x card, strength 1", "mac_succ_card_c10": "Mac R x card, strength 10",
              "mac_grade_card_c1": "Mac grade x card, strength 1", "mac_grade_card_c10": "Mac grade x card, strength 10"}
SPEED_BAR = ("hist", "card", "mac_hist", "mac_card")
GRADE_BAR = ("rate", "roll")
DISC_MASKS = ("bench/full", "bench/yomitan", "mac/full", "mac/yomitan")


def below_normal():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thr(d, s, strict):
    return d < 1000 * s if strict else d <= 1000 * s


# ------------------------------------------------------------------ repro conversion (CPU)
def convert(args):
    """The repro copy through the 2026-10-14 convert_collection.py (imported read only, its HERE / OUT redirected)."""
    import importlib.util
    src = SOURCES["repro"]
    if not src["copy"].exists():
        raise SystemExit(f"missing {src['copy']}")
    spec = importlib.util.spec_from_file_location("sm121_convert", COLL / "convert_collection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE, mod.OUT = src["dir"], src["data"]
    mod.main()   # writes collection-dataset/, review-th-to-revlog-id.parquet, collection-dataset-summary.json (all in work/)
    s = json.loads((src["dir"] / "collection-dataset-summary.json").read_text())
    atomic_json(src["dir"] / "copy-info.json", {"sha256": sha256(src["copy"]), "bytes": src["copy"].stat().st_size,
                                                "last_review_month": s["last_review"][:7], "kept_reviews": s["kept_reviews"]})
    print(f"[sm121-convert] {s['kept_reviews']:,} canonical reviews -> {src['data']}")
    return 0


# ------------------------------------------------------------------ trunk forward (CPU)
def trunk(args):
    """P(grade) of the trunk and P(t | grade) of each of its heads on every time-labelled non-first review (SM117 rule:
    duration > 0, elapsed_days >= 0), show-time inputs, leak-free (neighbour; the conversion has no deleted cards),
    fp32 weights, state fp32 (main) or int4 card / int8 note (s4c8n, as the fork runs it), whole history as one
    sequence (seed 2025 + user, as every collection score)."""
    from scripts import small_model_heldout as smh  # first: RWKV_NO_JIT + patches
    import torch
    below_normal()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    assert not torch.cuda.is_available(), "CUDA must stay hidden (CPU only)"
    started = time.perf_counter()
    key, src = args.model, SOURCES[args.source]
    t_spec = TRUNKS[key]
    info = verified_master(key)
    hd = t_spec["dir"]
    trunk_sd = torch.load(hd / "master.pth", map_location="cpu", weights_only=True)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    if set(trunk_sd) != set(released) or not all(torch.equal(trunk_sd[k].float(), released[k].float()) for k in released):
        raise SystemExit(f"{hd / 'master.pth'} is not the released model of {key}")
    model = build_model(key, trunk_sd, "cpu")
    if args.state == "s4c8n":
        if MODELS[key]["kind"] == "fork":
            raise SystemExit("s4c8n is only wired for the small engine here")
        smh.set_expertium_state_bits(model, (4, 8))
    frozen = Frozen(model, MODELS[key]["kind"])
    heads = {}
    for name, (fname, prefix) in t_spec["heads"].items():
        path = Path(fname) if Path(fname).is_absolute() else hd / fname
        heads[name] = load_head(path, frozen.inputs, prefix=prefix)
    rev = revlog(src["data"], USER, args.cap or None)
    lower, _, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy(), same_day=True)
    ths = np.flatnonzero(lower >= 0) + 1
    table = explicit_table(rev)
    t0 = time.perf_counter()
    batch, _ = show_time_batch(src["data"], USER, cap=args.cap or None)
    prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.inference_mode():
        qi, q_th, feats, o = frozen.forward(batch)
        p_logits = o[3][0].float()
        order = torch.argsort(q_th)
        q_th, qi, feats = q_th[order], qi[order], feats[order]
        want = torch.as_tensor(ths)
        pos = torch.searchsorted(q_th, want).clamp(max=len(q_th) - 1)
        found = q_th[pos] == want
        if not bool(found.all()):
            raise SystemExit(f"{int((~found).sum())} time-labelled reviews without an instant row")
        x = torch.cat([feats[pos], torch.as_tensor(table)[want - 1]], -1)
        pg = torch.softmax(p_logits[qi[pos]], -1).double()
        pt = {n: torch.softmax(h(x).float().view(-1, 4, BINS), -1).double() for n, h in heads.items()}
        nonfinite = int((~torch.isfinite(x)).any(-1).sum())
    del o, batch
    fwd = time.perf_counter() - t1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, th=ths.astype(np.int64), pg=pg.numpy().astype(np.float32),
                        **{f"pt_{n}": p.numpy().astype(np.float32) for n, p in pt.items()})
    meta = {"model": key, "name": MODELS[key]["name"], "source": args.source, "state": args.state, "heads": list(heads),
            "reviews_in_history": int(len(rev)), "time_labelled_rows": int(len(ths)), "cap": args.cap,
            "nonfinite_inputs": nonfinite, "p_grade_sum_maxdev": float((pg.sum(1) - 1).abs().max()),
            "p_time_sum_maxdev": max(float((p.sum(2) - 1).abs().max()) for p in pt.values()),
            "prepare_seconds": round(prep, 1), "forward_seconds": round(fwd, 1),
            "total_seconds": round(time.perf_counter() - started, 1), "peak_gb": peak_gb(),
            "threads": torch.get_num_threads(), "inputs": h117.INPUTS, "deleted_cards": "neighbour", "precision": "fp32 weights"}
    atomic_json(out.with_suffix(".json"), meta)
    print(f"[sm121-trunk] {key} ({args.source}, {args.state}): {len(ths):,} rows, prepare {prep:.0f} s, forward {fwd:.0f} s, "
          f"peak {meta['peak_gb']} GB -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ history, decks, slices
def label_ths(src):
    import lmdb
    from scripts.compress_metrics import read_labels
    env = lmdb.open(str(src["ldb"]), readonly=True, lock=False)
    with env.begin() as txn:
        lab = read_labels(txn, USER)
    env.close()
    return np.unique(np.asarray(lab[0], np.int64))


def yomitan_deck_ids(src):
    """Small deck ids (convert_collection.py numbering: sorted used original deck ids) whose full deck name contains
    'yomitan' (any case). Names are only used for the privacy scan."""
    import sqlite3
    con = sqlite3.connect(f"file:{src['copy'].as_posix()}?mode=ro&immutable=1", uri=True)
    used = [d for (d,) in con.execute("""
        select distinct case when c.odid != 0 then c.odid else c.did end from revlog r join cards c on c.id = r.cid
        where r.ease between 1 and 4 and r.type in (0, 1, 2, 3, 4, 5) and not (r.type = 3 and r.factor = 0)""")]
    names = dict(con.execute("select id, name from decks").fetchall())
    con.close()
    used = sorted(used)
    return {i for i, d in enumerate(used) if YOMITAN in str(names.get(d, "")).lower()}, len(used), [str(names.get(d, "")) for d in used]


def history(source):
    """Whole-history arrays (revlog order = chronological) of the collection user, plus the card history: per row the
    card's last CARD_LAST earlier answers with a positive duration (any day, any grade)."""
    import pandas as pd
    src = SOURCES[source]
    rev = pd.read_parquet(src["data"] / "revlogs" / f"user_id={USER}",
                          columns=["card_id", "day_offset", "rating", "duration", "elapsed_days"])
    cards = pd.read_parquet(src["data"] / "cards" / f"user_id={USER}")
    n_dataset_decks = len(pd.read_parquet(src["data"] / "decks" / f"user_id={USER}"))
    ymap, n_decks, _ = yomitan_deck_ids(src)
    deck = rev["card_id"].map(dict(zip(cards["card_id"], cards["deck_id"]))).to_numpy()
    when = pd.read_parquet(src["th_map"]).sort_values("review_th")["revlog_id"].to_numpy()
    if len(when) != len(rev):
        raise SystemExit("review_th map and revlog differ in length")
    card, d, g = rev["card_id"].to_numpy(np.int64), rev["duration"].to_numpy(np.int64), rev["rating"].to_numpy(np.int64) - 1
    n = len(d)
    cd = np.full((n, CARD_LAST), -1, np.int64)
    cg = np.full((n, CARD_LAST), -1, np.int64)
    last = {}
    for i in range(n):
        q = last.get(card[i])
        if q:
            k = len(q)
            cd[i, :k] = [x[0] for x in q]
            cg[i, :k] = [x[1] for x in q]
        if d[i] > 0:
            if q is None:
                q = last[card[i]] = deque(maxlen=CARD_LAST)
            q.append((int(d[i]), int(g[i])))
    return {"src": src, "rev": rev, "card": card, "day": rev["day_offset"].to_numpy(np.int64), "grade": g, "d": d,
            "ed": rev["elapsed_days"].to_numpy(np.int64), "yomitan": np.isin(deck, list(ymap)), "when_ms": when.astype(np.int64),
            "n_decks": n_decks, "n_dataset_decks": n_dataset_decks, "yomitan_decks": len(ymap), "cd": cd, "cg": cg}


def slices(H, ths):
    """Row masks over the scored rows ths: bench = label-DB benchmark test reviews (the 732-set row rule); mac = later-day
    (elapsed_days >= 1), positive time, latest 20% chronologically; mac_date = the same filter from the Mac's start date."""
    t = ths - 1
    out = {}
    if H["src"]["ldb"] is not None:
        out["bench"] = np.isin(ths, label_ths(H["src"]))
    later = (H["ed"][t] >= 1) & (H["d"][t] > 0)
    idx = np.flatnonzero(later)
    mac = np.zeros(len(ths), bool)
    mac[idx[len(idx) - int(math.ceil(0.2 * len(idx))):]] = True
    out["mac"] = mac
    out["mac_date"] = later & (H["when_ms"][t] >= int(MAC_DATE.timestamp() * 1000))
    return out


# ------------------------------------------------------------------ baselines (causal)
def recent_baselines(H, ths, pri):
    """Cumulative recent history (all earlier time-labelled reviews: duration > 0, not a first review, same-day included),
    pseudo-count M at the pooled priors (SM120's running: F, E, R, Bg), the set share (2) and the rolling 2,000 rates."""
    rev, t = H["rev"], ths - 1
    F, E, R, Bg, _, _ = f120.running(rev, t, pri)
    d, g = H["d"], H["grade"]
    lower, _, _ = time_labels(d, H["ed"], same_day=True)
    lab = lower >= 0
    ex = lambda a: (np.cumsum(a, 0) - a)[t].astype(np.float64)   # noqa: E731
    succ = {}
    for sn, gs in SETS.items():
        inset = lab & np.isin(g, gs)
        n_set = ex(inset.astype(np.int64))
        p0 = np.asarray(pri["E0_event"][sn]) / pri["R0_set"][sn]
        under = np.stack([ex((inset & (d <= 1000 * s)).astype(np.int64)) for s in S_GRID], 1)
        succ[sn] = (under + M * p0[None]) / (n_set[:, None] + M)
    labpos = np.flatnonzero(lab)
    j = np.searchsorted(labpos, t)
    w = np.minimum(j, ROLL)
    roll_fast, roll_grade = {}, {}
    for sn, gs in SETS.items():
        inset_l = np.isin(g[labpos], gs)
        c = np.r_[0, np.cumsum(inset_l)]
        roll_grade[sn] = (c[j] - c[j - w] + M * pri["R0_set"][sn]) / (w + M)
        cols = []
        for k, s in enumerate(S_GRID):
            ev = np.r_[0, np.cumsum(inset_l & (d[labpos] <= 1000 * s))]
            cols.append((ev[j] - ev[j - w] + M * pri["E0_event"][sn][k]) / (w + M))
        roll_fast[sn] = np.stack(cols, 1)
    return {"F": F, "E": E, "R": R, "Bg": Bg, "succ": succ, "roll_fast": roll_fast, "roll_grade": roll_grade}


def card_counts(H, ths, Svals, strict):
    """Per row, among the card's last 3 earlier positive-duration answers: n and n within S of each grade set and grade."""
    t = ths - 1
    cd, cg = H["cd"][t], H["cg"][t]
    has = cg >= 0
    out = {"set_n": {}, "set_fast": {}, "g_n": {}, "g_fast": {}}
    fast = np.stack([has & thr(cd, s, strict) for s in Svals], 1)   # [n, S, 3]
    for sn, gs in SETS.items():
        m = has & np.isin(cg, gs)
        out["set_n"][sn] = m.sum(1).astype(np.float64)
        out["set_fast"][sn] = (fast & m[:, None]).sum(2).astype(np.float64)
    for g in (1, 2, 3):
        m = cg == g
        out["g_n"][g] = m.sum(1).astype(np.float64)
        out["g_fast"][g] = (fast & m[:, None]).sum(2).astype(np.float64)
    return out


def mac_window(H, ths, scope, Svals, strict):
    """The Mac's recent history: the preceding ROLL eligible reviews (seen before, >= 1 day elapsed, positive
    duration) inside the scope, updated after each prediction. rec_frac[set] = (fast + 0.5) / (n + 1);
    grade_frac[g] = (fast_g + 10 x rec_frac[Rec]) / (n_g + 10)."""
    t = ths - 1
    elig = (H["ed"] >= 1) & (H["d"] > 0) & scope
    pos = np.flatnonzero(elig)
    j = np.searchsorted(pos, t)
    w = np.minimum(j, ROLL)
    gp, dp = H["grade"][pos], H["d"][pos]

    def wc(ind):
        c = np.r_[0, np.cumsum(ind)]
        return (c[j] - c[j - w]).astype(np.float64)
    n_g = {g: wc(gp == g) for g in (1, 2, 3)}
    f_g = {g: np.stack([wc((gp == g) & thr(dp, s, strict)) for s in Svals], 1) for g in (1, 2, 3)}
    rec_frac = {sn: (sum(f_g[g] for g in gs if g >= 1) + 0.5) / (sum(n_g[g] for g in gs if g >= 1)[:, None] + 1)
                for sn, gs in SETS.items()}
    grade_frac = {g: (f_g[g] + MAC_GRADE_C * rec_frac["Rec"]) / (n_g[g][:, None] + MAC_GRADE_C) for g in (1, 2, 3)}
    return {"rec_frac": rec_frac, "grade_frac": grade_frac}


def mac_preds(pg, sn, k, W, C, c=CARD_C):
    """The Mac's four baselines for set sn at threshold index k (W = mac_window, C = card_counts)."""
    gs = list(SETS[sn])
    raw = pg[:, gs].sum(1)
    rf = W["rec_frac"][sn][:, k]
    return {"mac_succ": raw * rf,
            "mac_grade": sum(pg[:, g] * W["grade_frac"][g][:, k] for g in gs),
            "mac_succ_card": raw * (C["set_fast"][sn][:, k] + c * rf) / (C["set_n"][sn] + c),
            "mac_grade_card": sum(pg[:, g] * (C["g_fast"][g][:, k] + c * W["grade_frac"][g][:, k]) / (C["g_n"][g] + c) for g in gs)}


# ------------------------------------------------------------------ predictions per group (main analysis)
def groups(pg, pt, H, ths, B, Cc, Wm, cap):
    """Yields ((family, set or grade, S), {predictor: p}, y, valid) for every event of one combination."""
    t = ths - 1
    gr, d = H["grade"][t], H["d"][t]
    for sn, gs in SETS.items():
        gs = list(gs)
        raw = pg[:, gs].sum(1)
        yield ("grade", sn, None), {"model": raw, "rate": B["R"][sn], "roll": B["roll_grade"][sn]}, \
            np.isin(gr, gs).astype(np.float64), np.ones(len(t), bool)
    for sn, gs in SETS.items():
        gs = list(gs)
        raw = pg[:, gs].sum(1)
        for k, s in enumerate(S_GRID):
            cdf = cdf_at(pt[:, gs], s)
            P = {"head": (pg[:, gs] * cdf).sum(1), "raw": raw, "succ_hist": raw * B["succ"][sn][:, k],
                 "grade_hist": (pg[:, gs] * B["F"][:, gs, k]).sum(1), "rate": B["E"][sn][:, k], "roll": B["roll_fast"][sn][:, k],
                 "succ_card": raw * (Cc["set_fast"][sn][:, k] + CARD_C * B["succ"][sn][:, k]) / (Cc["set_n"][sn] + CARD_C),
                 "grade_card": sum(pg[:, g] * (Cc["g_fast"][g][:, k] + CARD_C * B["F"][:, g, k]) / (Cc["g_n"][g] + CARD_C) for g in gs)}
            P.update(mac_preds(pg, sn, k, Wm, Cc))
            for c in CARD_SENS:
                mp = mac_preds(pg, sn, k, Wm, Cc, c)
                P[f"mac_succ_card_c{int(c)}"], P[f"mac_grade_card_c{int(c)}"] = mp["mac_succ_card"], mp["mac_grade_card"]
            yield ("fast", sn, s), P, (np.isin(gr, gs) & (d <= 1000 * s)).astype(np.float64), np.full(len(t), cap > 1000 * s)
    for gn, g in SPEED.items():
        m = gr == g
        for k, s in enumerate(S_GRID):
            P = {"head": cdf_at(pt[:, g], s), "hist": B["F"][:, g, k],
                 "card": (Cc["g_fast"][g][:, k] + CARD_C * B["F"][:, g, k]) / (Cc["g_n"][g] + CARD_C),
                 "mac_hist": Wm["grade_frac"][g][:, k],
                 "mac_card": (Cc["g_fast"][g][:, k] + CARD_C * Wm["grade_frac"][g][:, k]) / (Cc["g_n"][g] + CARD_C)}
            yield ("speed", gn, s), P, (d <= 1000 * s).astype(np.float64), m & (cap > 1000 * s)


# ------------------------------------------------------------------ metrics
def ll_rows(p, y, clip=1e-6):
    p = np.clip(p, clip, 1 - clip)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def auc_exact(p, y):
    from scipy.stats import rankdata
    pos, n = y.sum(), len(y)
    return float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (n - pos))) if 0 < pos < n else float("nan")


def calib(p, y):
    """ECE over 20 equal-mass bins of the sorted predictions (exact, per row), the reliability table [n, mean p,
    observed], AUC and a fixed-width 10-bin table [n, mean p, observed]."""
    n = len(p)
    o = np.argsort(p, kind="stable")
    ps, ys = p[o], y[o]
    edges = np.linspace(0, n, ECE_BINS + 1).astype(np.int64)
    table, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            table.append([int(b - a), round(float(ps[a:b].mean()), 5), round(float(ys[a:b].mean()), 5)])
            ece += abs(ps[a:b].sum() - ys[a:b].sum())
    fb = np.minimum((p * 10).astype(np.int64), 9)
    fixed = [[int((fb == k).sum()), round(float(p[fb == k].mean()), 5) if (fb == k).any() else None,
              round(float(y[fb == k].mean()), 5) if (fb == k).any() else None] for k in range(10)]
    return {"ece": round(float(ece / n), 5), "auc": round(auc_exact(p, y), 5),
            "max_bin_gap": round(max(abs(r[1] - r[2]) for r in table), 5),
            "reliability_equal_mass": table, "reliability_fixed_10": fixed}


def boot_weights(blocks, seed=0):
    """[BOOT, n_blocks] resampling counts (blocks drawn with replacement, seed 0)."""
    nb = int(blocks.max()) + 1
    idx = np.random.default_rng(seed).integers(0, nb, size=(BOOT, nb))
    W = np.zeros((BOOT, nb), np.float32)
    for r in range(BOOT):
        W[r] = np.bincount(idx[r], minlength=nb)
    return W


def block_sums(blocks, cols, nb):
    return np.stack([np.bincount(blocks, weights=c, minlength=nb) for c in cols], 1)


def ci(boot):
    return [round(float(np.nanpercentile(boot, 2.5)), 6), round(float(np.nanpercentile(boot, 97.5)), 6)]


def evaluate(gen, mask, bd, bc, Wd, Wc, clip=1e-6, bar=None, calibrate=True):
    """Metrics of every group of `gen` on the rows `mask`: per predictor LogLoss / Brier / mean p (+ calibration),
    head-minus-baseline LogLoss deltas with day-block (primary) and card-block CIs, reliable flags."""
    res, cols_d, cols_c, ncols_d, ncols_c, jobs = {}, [], [], [], [], []
    nbd, nbc = Wd.shape[1], Wc.shape[1]
    for gk, P, y, valid in gen:
        top = "model" if gk[0] == "grade" else "head"
        v = valid & mask
        n = int(v.sum())
        g = {"rows": n, "predictors": {}, "vs": {}}
        res[gk] = g
        if n == 0:
            continue
        yv = y[v]
        g["event_rate"] = round(float(yv.mean()), 5)
        ll = {pn: ll_rows(p[v], yv, clip) for pn, p in P.items()}
        for pn, p in P.items():
            pv = np.clip(p[v], clip, 1 - clip)
            g["predictors"][pn] = {"logloss": round(float(ll[pn].mean()), 6), "brier": round(float(((pv - yv) ** 2).mean()), 6),
                                   "mean_p": round(float(pv.mean()), 5), **(calib(pv, yv) if calibrate or pn == top else {})}
        for pn in P:
            if pn == top:
                continue
            diff = ll[top] - ll[pn]
            g["vs"][pn] = {"delta": round(float(diff.mean()), 6), "relative_pct": round(100 * float(diff.mean()) / float(ll[pn].mean()), 2)}
            jobs.append((gk, pn))
            cols_d.append(np.bincount(bd[v], weights=diff, minlength=nbd)); ncols_d.append(np.bincount(bd[v], minlength=nbd))
            cols_c.append(np.bincount(bc[v], weights=diff, minlength=nbc)); ncols_c.append(np.bincount(bc[v], minlength=nbc))
    for cols, ncols, W, tag in ((cols_d, ncols_d, Wd, "ci_day"), (cols_c, ncols_c, Wc, "ci_card")):
        if not jobs:
            break
        D, N = np.stack(cols, 1).astype(np.float32), np.stack(ncols, 1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (W @ D) / (W @ N)
        lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
        for i, (gk, pn) in enumerate(jobs):
            res[gk]["vs"][pn][tag] = [round(float(lo[i]), 6), round(float(hi[i]), 6)]
    for gk, g in res.items():
        top = "model" if gk[0] == "grade" else "head"
        if not g["vs"]:
            g["reliable"] = False
            continue
        names = bar[gk[0]] if bar else list(g["vs"])
        names = [pn for pn in names if pn in g["vs"]]
        g["bar_baselines"] = names
        g["strongest_baseline"] = min(names, key=lambda pn: g["predictors"][pn]["logloss"])
        ece_ok = g["predictors"][top].get("ece", 1) < 0.01
        g["reliable"] = bool(ece_ok and all(g["vs"][pn]["ci_day"][1] < 0 for pn in names))
        g["reliable_card_block"] = bool(ece_ok and all(g["vs"][pn]["ci_card"][1] < 0 for pn in names))
    return {f"{f}/{sn}/{s if s is not None else '-'}": g for (f, sn, s), g in res.items()}


# ------------------------------------------------------------------ discrimination (spread, lift, AUC, Murphy, R^2, within-card)
def card_bins(H, ths, g, Bg):
    """Card-history bin distribution of grade g: (the card's last-3 answers of grade g per bin + 3 x recent Bg[g]) / (n_g + 3)."""
    t = ths - 1
    cd, cg = H["cd"][t], H["cg"][t]
    cnt = np.zeros((len(t), BINS))
    b = np.searchsorted(np.asarray(TIME_BIN_EDGES_MS), cd.astype(np.float64), side="right")
    for j in range(CARD_LAST):
        m = cg[:, j] == g
        cnt[np.flatnonzero(m), b[m, j]] += 1
    return (cnt + CARD_C * Bg[:, g]) / (cnt.sum(1, keepdims=True) + CARD_C)


def spread(x):
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"sd": round(float(x.std()), 5), "iqr": round(float(q[3] - q[1]), 5), "p10": round(float(q[0]), 5),
            "p50": round(float(q[2]), 5), "p90": round(float(q[4]), 5), "mean": round(float(x.mean()), 5)}


def deciles(key, d, rows):
    """Rows binned by the predicted ln median (10 equal-count bins): n, mean predicted median s, observed median s,
    observed rate within 4 s and 8 s."""
    o = np.argsort(key[rows], kind="stable")
    idx = np.flatnonzero(rows)[o]
    out = []
    for part in np.array_split(idx, 10):
        if len(part) == 0:
            continue
        out.append({"n": int(len(part)), "pred_median_s": round(float(np.exp(key[part]).mean()), 3),
                    "obs_median_s": round(float(np.median(d[part]) / 1000), 3),
                    "obs_within_4s": round(float((d[part] <= 4000).mean()), 4), "obs_within_8s": round(float((d[part] <= 8000).mean()), 4)})
    return out


def binned_auc_boot(scores, y, bd, W):
    """Day-block bootstrap of AUC from 200 quantile bins per score: [BOOT] per score (ties in a bin count half)."""
    out = []
    nb = W.shape[1]
    for s in scores:
        edges = np.unique(np.quantile(s, np.linspace(0, 1, AUC_BINS + 1)[1:-1]))
        b = np.searchsorted(edges, s, side="right")
        k = len(edges) + 1
        pos = np.zeros((nb, k)); neg = np.zeros((nb, k))
        np.add.at(pos, (bd, b), y); np.add.at(neg, (bd, b), 1 - y)
        P, N = W @ pos, W @ neg
        below = np.cumsum(N, 1) - N
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1)))
    return out


def murphy_sums(p, y, nbins=ECE_BINS):
    o = np.argsort(p, kind="stable")
    b = np.empty(len(p), np.int64)
    b[o] = (np.arange(len(p)) * nbins) // len(p)
    return b


def murphy_from(nk, spk, syk):
    """REL, RES, UNC (Murphy) from per-bin n, sum p, sum y; arrays may carry a leading bootstrap axis."""
    N = nk.sum(-1)
    obar = np.asarray(syk.sum(-1) / N)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.nansum(np.where(nk > 0, (spk - syk) ** 2 / nk, 0), -1) / N
        res = np.nansum(np.where(nk > 0, nk * (syk / nk - obar[..., None]) ** 2, 0), -1) / N
    return rel, res, obar * (1 - obar)


def discrimination(pg, pt, H, ths, B, Cc, Wm, mask, bd, Wd, cap):
    t = ths - 1
    gr, d = H["grade"][t], H["d"][t]
    out = {}
    GOOD = 2
    lnmed_head = {g: median_log_s(pt[:, g]) for g in range(4)}
    cb = {g: card_bins(H, ths, g, B["Bg"]) for g in range(4)}
    lnmed_card = {g: median_log_s(cb[g]) for g in range(4)}
    k4, k8 = S_GRID.index(4), S_GRID.index(8)
    ge = [2, 3]
    ge4 = {"head": (pg[:, ge] * cdf_at(pt[:, ge], 4)).sum(1),
           "card": sum(pg[:, g] * (Cc["g_fast"][g][:, k4] + CARD_C * B["F"][:, g, k4]) / (Cc["g_n"][g] + CARD_C) for g in ge)}
    p4good = {"head": cdf_at(pt[:, GOOD], 4), "card": (Cc["g_fast"][GOOD][:, k4] + CARD_C * B["F"][:, GOOD, k4]) / (Cc["g_n"][GOOD] + CARD_C)}
    out["spread"] = {"rows": int(mask.sum()),
                     "pred_median_s_given_good": {"head": spread(np.exp(lnmed_head[GOOD][mask])), "card": spread(np.exp(lnmed_card[GOOD][mask]))},
                     "p_within_4s_given_good": {k: spread(v[mask]) for k, v in p4good.items()},
                     "p_good_or_easy_within_4s": {k: spread(v[mask]) for k, v in ge4.items()}}
    good = mask & (gr == GOOD)
    key_head_all = lnmed_head[0].copy(); key_card_all = lnmed_card[0].copy()
    for g in range(1, 4):
        key_head_all[gr == g] = lnmed_head[g][gr == g]; key_card_all[gr == g] = lnmed_card[g][gr == g]
    out["decile_lift"] = {"good": {"head": deciles(lnmed_head[GOOD], d, good), "card": deciles(lnmed_card[GOOD], d, good)},
                          "all": {"head": deciles(key_head_all, d, mask), "card": deciles(key_card_all, d, mask)}}
    # AUC given Good at 4 s and 8 s
    out["auc_given_good"] = {}
    for s, k in ((4, k4), (8, k8)):
        v = good & (cap > 1000 * s)
        if v.sum() < 50:
            continue
        y = (d[v] <= 1000 * s).astype(np.float64)
        sc = {"head": cdf_at(pt[v, GOOD], s), "card": (Cc["g_fast"][GOOD][v, k] + CARD_C * B["F"][v, GOOD, k]) / (Cc["g_n"][GOOD][v] + CARD_C),
              "recent": B["F"][v, GOOD, k]}
        bh, bcard = binned_auc_boot([sc["head"], sc["card"]], y, bd[v], Wd)
        out["auc_given_good"][f"{s}"] = {"rows": int(v.sum()), "rate": round(float(y.mean()), 4),
                                         **{n: round(auc_exact(x, y), 5) for n, x in sc.items()},
                                         "head_minus_card": round(auc_exact(sc["head"], y) - auc_exact(sc["card"], y), 5),
                                         "head_minus_card_ci_day": ci(bh - bcard)}
    # Murphy decomposition, Good or Easy within 4 s / 8 s
    out["murphy_good_or_easy"] = {}
    for s, k in ((4, k4), (8, k8)):
        v = mask & (cap > 1000 * s)
        y = (np.isin(gr, ge) & (d <= 1000 * s)).astype(np.float64)[v]
        P = {"head": (pg[:, ge] * cdf_at(pt[:, ge], s)).sum(1),
             "grade_card": sum(pg[:, g] * (Cc["g_fast"][g][:, k] + CARD_C * B["F"][:, g, k]) / (Cc["g_n"][g] + CARD_C) for g in ge),
             "grade_hist": (pg[:, ge] * B["F"][:, ge, k]).sum(1)}
        P.update({n: x for n, x in mac_preds(pg, "GE", k, Wm, Cc).items() if n == "mac_grade_card"})
        r, boots = {}, {}
        for n, p in P.items():
            pv = np.clip(p[v], 1e-6, 1 - 1e-6)
            b = murphy_sums(pv, y)
            nk = np.bincount(b, minlength=ECE_BINS).astype(float); spk = np.bincount(b, pv, ECE_BINS); syk = np.bincount(b, y, ECE_BINS)
            rel, res_, unc = murphy_from(nk, spk, syk)
            r[n] = {"brier": round(float(((pv - y) ** 2).mean()), 6), "reliability": round(float(rel), 6),
                    "resolution": round(float(res_), 6), "uncertainty": round(float(unc), 6)}
            nb = Wd.shape[1]
            cube = np.zeros((nb, ECE_BINS, 3))
            np.add.at(cube, (bd[v], b, 0), 1); np.add.at(cube, (bd[v], b, 1), pv); np.add.at(cube, (bd[v], b, 2), y)
            R_ = np.einsum("rb,bkc->rkc", Wd, cube)
            boots[n] = murphy_from(R_[..., 0], R_[..., 1], R_[..., 2])
        r["head_minus_grade_card"] = {"resolution_ci_day": ci(boots["head"][1] - boots["grade_card"][1]),
                                      "reliability_ci_day": ci(boots["head"][0] - boots["grade_card"][0]),
                                      "resolution": round(r["head"]["resolution"] - r["grade_card"]["resolution"], 6)}
        out["murphy_good_or_easy"][f"{s}"] = {"rows": int(v.sum()), **r}
    # R^2 of predicted mean ln time on uncensored Good reviews, and the within-card version
    unc = good & (d > 0) & (d < min(cap, CAP_MS))
    yl = np.log(d[unc] / 1000.0)
    preds = {"head": pt[unc, GOOD] @ LN_VAL, "card": cb[GOOD][unc] @ LN_VAL, "recent": B["Bg"][unc, GOOD] @ LN_VAL}
    nb = Wd.shape[1]
    S_ = block_sums(bd[unc], [np.ones(len(yl)), yl, yl ** 2] + [(yl - p) ** 2 for p in preds.values()], nb)
    R_ = Wd @ S_
    sst = R_[:, 2] - R_[:, 1] ** 2 / R_[:, 0]
    r2b = {n: 1 - R_[:, 3 + i] / sst for i, n in enumerate(preds)}
    sst0 = ((yl - yl.mean()) ** 2).sum()
    out["r2_mean_ln_time_good"] = {"rows": int(unc.sum()), **{n: round(float(1 - ((yl - p) ** 2).sum() / sst0), 5) for n, p in preds.items()},
                                   "head_minus_card_ci_day": ci(r2b["head"] - r2b["card"]),
                                   "head_minus_recent_ci_day": ci(r2b["head"] - r2b["recent"])}
    cards = H["card"][t][unc]
    _, cid, cnt = np.unique(cards, return_inverse=True, return_counts=True)
    keep = cnt[cid] >= 2
    wc = {"rows": int(keep.sum()), "cards": int((cnt >= 2).sum())}
    if keep.sum() > 50:
        def dev(x):
            m = np.bincount(cid, x, len(cnt)) / cnt
            return (x - m[cid])[keep]
        dy = dev(yl)
        dx = {"head": dev(lnmed_head[GOOD][unc]), "card": dev(lnmed_card[GOOD][unc])}
        bk = bd[unc][keep]
        S_ = block_sums(bk, [dy ** 2] + sum([[dx[n] * dy, dx[n] ** 2] for n in dx], []), nb)
        R_ = Wd @ S_
        cb_ = {n: R_[:, 1 + 2 * i] / np.sqrt(R_[:, 2 + 2 * i] * R_[:, 0]) for i, n in enumerate(dx)}
        wc.update({"sd_actual_within_card_ln": round(float(dy.std()), 5),
                   **{f"sd_pred_within_card_ln_{n}": round(float(x.std()), 5) for n, x in dx.items()},
                   **{f"corr_{n}": round(float((x * dy).sum() / math.sqrt((x ** 2).sum() * (dy ** 2).sum())), 5) if (x ** 2).sum() > 0 else None
                      for n, x in dx.items()},
                   "head_minus_card_corr_ci_day": ci(cb_["head"] - cb_["card"])})
    out["within_card_good"] = wc
    return out


# ------------------------------------------------------------------ main analysis report
def load_trunk(key, prefix="trunk"):
    p = WORK / f"{prefix}-{key}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"th": z["th"], "pg": z["pg"].astype(np.float64), **{k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("pt_")}}


def combos_available(prefix="trunk", only=None):
    tr = {k: load_trunk(k, prefix) for k in TRUNKS}
    data, missing = {}, []
    for ck, (gk, (tk, hn), _) in COMBOS.items():
        if only and ck not in only:
            continue
        if tr.get(gk) is None or tr.get(tk) is None or hn not in tr[tk]:
            missing.append(ck)
            continue
        if not np.array_equal(tr[gk]["th"], tr[tk]["th"]):
            raise SystemExit(f"{ck}: trunks {gk} and {tk} scored different rows")
        data[ck] = (tr[gk]["pg"], tr[tk][hn])
    ths = next((tr[k]["th"] for k in tr if tr[k] is not None), None)
    return data, ths, missing


def block_codes(H, ths, m):
    t = ths - 1
    bd = np.zeros(len(t), np.int64); bc = np.zeros(len(t), np.int64)
    bd[m] = np.unique(H["day"][t][m], return_inverse=True)[1]
    bc[m] = np.unique(H["card"][t][m], return_inverse=True)[1]
    return bd, bc


def report(args):
    H = history("main")
    data, ths, missing = combos_available()
    if not data:
        raise SystemExit("no trunk files in work/")
    pri = json.loads(PRIORS.read_text())
    cap = time_user_cap(H["d"])
    t = ths - 1
    B = recent_baselines(H, ths, pri)
    Cc = card_counts(H, ths, S_GRID, strict=False)
    scopes = {"full": np.ones(len(H["d"]), bool), "yomitan": H["yomitan"]}
    Wmac = {sc: mac_window(H, ths, m, S_GRID, strict=False) for sc, m in scopes.items()}
    Sl = slices(H, ths)
    res = {"rows": {}, "user_cap_s": cap / 1000, "combos": {}, "discrimination": {}, "blocks": {}, "missing_combos": missing,
           "slice_start_month": {sl: datetime.fromtimestamp(H["when_ms"][t][m].min() / 1000).strftime("%Y-%m") for sl, m in Sl.items() if m.any()},
           "yomitan_decks": H["yomitan_decks"]}
    bar = {"fast": FAST_BAR, "speed": SPEED_BAR, "grade": GRADE_BAR}
    for sl, m0 in Sl.items():
        for sub in ("full", "yomitan"):
            m = m0 & (H["yomitan"][t] if sub == "yomitan" else True)
            key = f"{sl}/{sub}"
            res["rows"][key] = int(m.sum())
            if not m.any():
                continue
            bd, bc = block_codes(H, ths, m)
            Wd, Wc = boot_weights(bd[m]), boot_weights(bc[m])
            res["blocks"][key] = {"days": int(Wd.shape[1]), "cards": int(Wc.shape[1])}
            for ck, (pg, pt) in data.items():
                st = time.time()
                res["combos"].setdefault(ck, {})[key] = evaluate(groups(pg, pt, H, ths, B, Cc, Wmac[sub], cap), m, bd, bc, Wd, Wc, bar=bar,
                                                                  calibrate=(sl != "mac_date"))
                if key in DISC_MASKS:
                    res["discrimination"].setdefault(ck, {})[key] = discrimination(pg, pt, H, ths, B, Cc, Wmac[sub], m, bd, Wd, cap)
                print(f"[sm121-report] {ck} {key}: {int(m.sum()):,} rows, {time.time() - st:.0f} s", flush=True)
    res["trunks"] = {}
    for k in TRUNKS:
        p = WORK / f"trunk-{k}.json"
        if p.exists():
            meta = json.loads(p.read_text()); meta.pop("cap", None)
            res["trunks"][k] = meta
    res["design"] = {"S": list(S_GRID), "outcome": "set and duration <= S (SM117 rule)", "pseudo_count_recent": M, "rolling_window": ROLL,
                     "card_last": CARD_LAST, "card_strength": CARD_C, "card_strength_sensitivity": list(CARD_SENS),
                     "mac_grade_strength": MAC_GRADE_C, "bootstrap": BOOT, "seed": 0, "ece_bins": ECE_BINS,
                     "data": "own collection copy of 2026-10-14 (reviews up to 2026-10-04), full history, show-time inputs, fp32 state"}
    atomic_json(OUT / "sm121-result.json", res)
    write_md(res)
    privacy_scan("main")
    rel = {ck: r.get("bench/full", {}).get("fast/Rec/4", {}).get("reliable") for ck, r in res["combos"].items()}
    print(f"[sm121-report] {len(res['combos'])} combinations (missing {missing}); recalled within 4 s reliable on the benchmark rows: {rel}")
    return 0


def fmt_d(v, key="ci_day"):
    if key not in v:
        return f"{v['delta']:+.4f} ({v['relative_pct']:+.1f}%)"
    return f"{v['delta']:+.4f} [{v[key][0]:+.4f}, {v[key][1]:+.4f}] ({v['relative_pct']:+.1f}%)"


TOP = ("**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the "
       "question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The \"Show answer\" "
       "moment is not stored, so front-only and back-only times cannot be computed. Every \"within S s\" below means answered within "
       "S s, reading the back included.")


def write_md(res):
    C = res["combos"]
    names = {ck: COMBOS[ck][2] for ck in C}
    L = ["# Timed recall on your own collection (SM121): result", "", TOP, "",
         "Scoring only, CPU, on your own collection (the copy of 2026-10-14, reviews up to 2026-10-04; aggregate numbers only). "
         "Pre-registered in `design.md`. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of the "
         "bar (recent history and card history, listed below) with the 95% CI below zero (2,000 day-block bootstraps, seed 0; "
         "card-block bootstraps as a sensitivity). Deltas are head minus baseline. LogLoss is compared only within the same "
         "outcome and threshold; the untimed grade events have their own table.", "",
         "Rows: " + ", ".join(f"{k} {v:,}" for k, v in res["rows"].items()) + ". `bench` = the benchmark test reviews with a "
         "recorded time (the 732-set row rule: same-day reviews included, first reviews excluded); `mac` = the Mac's slice "
         f"(later-day reviews with a prior review and a positive time, latest 20% chronologically, from {res['slice_start_month'].get('mac')}); "
         f"`mac_date` = the same filter from 28 April 2026 on. `yomitan` = the Yomitan decks ({res['yomitan_decks']} decks).", "",
         "Baselines (which earlier reviews feed each one): (1) raw P(set), time ignored; (2) P(set) x the share of earlier "
         "answers of the set within S s and (3) sum_g P(g) x the share of earlier answers of grade g within S s, both over every "
         "earlier time-labelled review (positive time, not a first review, same-day included) with pseudo-count 10 at the "
         "pooled priors; (b) the running rate of the event over the same reviews; (b2) its rate over the previous 2,000 of them; "
         "(4) and (5) = (2) and (3) adjusted by the card's last three earlier positive-time answers (any day, any grade; "
         "(k + 3 x share) / (n + 3)); the Mac's four (window = the preceding 2,000 eligible reviews, i.e. later-day with a "
         "positive time, in the scope: the whole collection, or only the Yomitan decks for the Yomitan rows): R x recent, "
         "grade x recent, R x card, grade x card (card strength 3; 1 and 10 as a sensitivity outside the bar).", ""]
    L += ["## Where \"recalled within S s\" is reliable (every baseline beaten)", "",
          "| combination | bench, full | bench, Yomitan | mac, full | mac, Yomitan |", "|---|---|---|---|---|"]
    for ck, r in C.items():
        cells = []
        for sl in DISC_MASKS:
            x = r.get(sl, {})
            ss = [s for s in S_GRID if x.get(f"fast/Rec/{s}", {}).get("reliable")]
            cells.append(", ".join(f"{s} s" for s in ss) if ss else "none")
        L.append(f"| {names[ck]} | " + " | ".join(cells) + " |")
    for sn in SETS:
        L += ["", f"## {SET_NAMES[sn]} within S s: head vs the baselines", ""]
        for sl in ("bench/full", "mac/full", "mac/yomitan"):
            L += [f"### {sl}", "", "| combination | S | rows | event rate | head LogLoss | strongest baseline (LogLoss) | head vs strongest | "
                  "vs Mac grade x card | vs (5) | vs (b2) rolling | ECE | AUC | reliable |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for ck, r in C.items():
                for s in (2, 4, 6, 8, 15, 30):
                    x = r.get(sl, {}).get(f"fast/{sn}/{s}")
                    if not x or not x.get("vs"):
                        continue
                    st = x["strongest_baseline"]; p = x["predictors"]
                    L.append(f"| {names[ck]} | {s} s | {x['rows']:,} | {x['event_rate']:.3f} | {p['head']['logloss']:.4f} | {FAST_NAMES[st]} "
                             f"({p[st]['logloss']:.4f}) | {fmt_d(x['vs'][st])} | {fmt_d(x['vs']['mac_grade_card'])} | {fmt_d(x['vs']['grade_card'])} | "
                             f"{fmt_d(x['vs']['roll'])} | {p['head']['ece']:.4f} | {p['head']['auc']:.3f} | {'yes' if x['reliable'] else 'no'} |")
            L.append("")
    L += ["## Untimed grade events: P(recalled), P(Good or Easy), P(Easy) (no time; not comparable with the timed tables)", "",
          "| combination | slice | event | rows | LogLoss model / running rate / rolling 2,000 | model vs running rate | model vs rolling | ECE | AUC | reliable |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sl in ("bench/full", "mac/full", "mac/yomitan"):
        for ck, r in C.items():
            for sn in SETS:
                x = r.get(sl, {}).get(f"grade/{sn}/-")
                if not x or not x.get("vs"):
                    continue
                p = x["predictors"]
                L.append(f"| {names[ck]} | {sl} | {SET_NAMES[sn]} | {x['rows']:,} | {p['model']['logloss']:.4f} / {p['rate']['logloss']:.4f} / "
                         f"{p['roll']['logloss']:.4f} | {fmt_d(x['vs']['rate'])} | {fmt_d(x['vs']['roll'])} | {p['model']['ece']:.4f} | "
                         f"{p['model']['auc']:.3f} | {'yes' if x['reliable'] else 'no'} |")
    L += ["", "## Diagnostic: speed given the observed grade, P(answered within S s | grade)", "",
          "| combination | slice | grade | S | rows | LogLoss head / recent / card / Mac recent / Mac card | head vs strongest | ECE | reliable |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sl in ("bench/full", "mac/yomitan"):
        for ck, r in C.items():
            for gn in SPEED:
                for s in (4, 8):
                    x = r.get(sl, {}).get(f"speed/{gn}/{s}")
                    if not x or not x.get("vs"):
                        continue
                    p = x["predictors"]; st = x["strongest_baseline"]
                    L.append(f"| {names[ck]} | {sl} | {gn} | {s} s | {x['rows']:,} | " + " / ".join(f"{p[n]['logloss']:.4f}" for n in ("head",) + SPEED_BAR)
                             + f" | {fmt_d(x['vs'][st])} ({st}) | {p['head']['ece']:.4f} | {'yes' if x['reliable'] else 'no'} |")
    L += ["", "## Does it discriminate between reviews, or is it a big average?", "",
          "Head vs the card-history baseline (card-adjusted recent shares / bin distributions). Spread = across reviews; AUC given Good "
          "= on reviews graded Good, P(t <= S | Good) ranks the fast ones from the slow ones (timing only, the grade is fixed); "
          "resolution (Murphy) is the part of the Brier score that is more than an average; R^2 of the predicted mean ln time on "
          "uncensored Good reviews; within-card = each review's predicted ln median minus that card's mean prediction, correlated "
          "with the same for the actual ln time (does it follow a card getting faster or slower). Day-block CIs on head minus card.", "",
          "| combination | slice | SD / IQR of predicted median given Good, head vs card (s) | P(within 4 s given Good) p10-p90, head vs card | "
          "AUC given Good at 4 s head / card / recent (diff CI) | resolution GE within 4 s head / card (diff CI) | R^2 head / card / recent | within-card corr head / card (diff CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for ck, dd in res["discrimination"].items():
        for sl, x in dd.items():
            sp = x["spread"]; a = x["auc_given_good"].get("4", {}); mu = x["murphy_good_or_easy"].get("4", {}); r2 = x["r2_mean_ln_time_good"]; wc = x["within_card_good"]
            ph, pc = sp["pred_median_s_given_good"]["head"], sp["pred_median_s_given_good"]["card"]
            qh, qc = sp["p_within_4s_given_good"]["head"], sp["p_within_4s_given_good"]["card"]
            L.append(f"| {names[ck]} | {sl} | {ph['sd']:.2f} / {ph['iqr']:.2f} vs {pc['sd']:.2f} / {pc['iqr']:.2f} | {qh['p10']:.2f}-{qh['p90']:.2f} vs "
                     f"{qc['p10']:.2f}-{qc['p90']:.2f} | " + (f"{a['head']:.3f} / {a['card']:.3f} / {a['recent']:.3f} ({a['head_minus_card']:+.3f} "
                     f"[{a['head_minus_card_ci_day'][0]:+.3f}, {a['head_minus_card_ci_day'][1]:+.3f}])" if a else "")
                     + " | " + (f"{mu['head']['resolution']:.4f} / {mu['grade_card']['resolution']:.4f} ({mu['head_minus_grade_card']['resolution']:+.4f} "
                     f"[{mu['head_minus_grade_card']['resolution_ci_day'][0]:+.4f}, {mu['head_minus_grade_card']['resolution_ci_day'][1]:+.4f}])" if mu else "")
                     + f" | {r2['head']:.3f} / {r2['card']:.3f} / {r2['recent']:.3f} | "
                     + (f"{wc['corr_head']:.3f} / {wc['corr_card']:.3f} ([{wc['head_minus_card_corr_ci_day'][0]:+.3f}, {wc['head_minus_card_corr_ci_day'][1]:+.3f}])"
                        if "corr_head" in wc and wc["corr_head"] is not None and wc["corr_card"] is not None else "") + " |")
    L += ["", "Decile lift tables (observed median time and fast rate per decile of the predicted median, Good and all grades, head "
          "and card), every S, every slice, Brier, card-block CIs, reliability tables with counts, fixed 10-bin calibration: "
          "`sm121-result.json`. The reproduction control of the Mac's numbers is separate: `sm121-repro.md`."]
    if res.get("missing_combos"):
        L += ["", f"Not scored (trunk or head missing): {', '.join(res['missing_combos'])}."]
    (OUT / "sm121-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ reproduction control
def repro(args):
    """The Mac's replay, from its spec: the Mac's Small pair (SM98 grades, SM83 head on SM92), int4 card / int8 note
    state, eligible = seen before, >= 1 day elapsed, positive duration; the last 20% of the eligible reviews scored;
    outcome Hard/Good/Easy and duration < T (strict); LogLoss clipped at 1e-7; the Mac's four baselines (windows in
    the scope)."""
    src = SOURCES["repro"]
    H = history("repro")
    data, ths, missing = combos_available("repro-trunk", only=("mac_pair", "small_pair_sm98"))
    if "mac_pair" not in data:
        raise SystemExit(f"repro trunks missing: {missing}")
    t = ths - 1
    gr, d = H["grade"][t], H["d"][t]
    elig = (H["ed"][t] >= 1) & (d > 0)
    idx = np.flatnonzero(elig)
    n_tail = int(math.ceil(0.2 * len(idx)))
    tail = np.zeros(len(ths), bool); tail[idx[len(idx) - n_tail:]] = True
    date_tail = elig & (H["when_ms"][t] >= int(MAC_DATE.timestamp() * 1000))
    info = json.loads((src["dir"] / "copy-info.json").read_text())
    res = {"copy": {"sha256": info["sha256"], "matches_mac_backup": info["sha256"] == MAC_BACKUP_SHA, "last_review_month": info["last_review_month"],
                    "canonical_reviews": int(len(H["d"])), "mac": MAC_COUNTS},
           "eligible": int(len(idx)), "prefix": int(len(idx) - n_tail), "scored": int(n_tail), "scored_yomitan": int((tail & H["yomitan"][t]).sum()),
           "tail_start_month": datetime.fromtimestamp(H["when_ms"][t][tail].min() / 1000).strftime("%Y-%m"),
           "date_tail_rows": int(date_tail.sum()), "trunks": {}, "tables": {}}
    for k in ("sm92", "sm98"):
        p = WORK / f"repro-trunk-{k}.json"
        if p.exists():
            meta = json.loads(p.read_text()); meta.pop("cap", None); res["trunks"][k] = meta
    scopes = {"full": np.ones(len(H["d"]), bool), "yomitan": H["yomitan"]}
    Cc = card_counts(H, ths, REPRO_T, strict=True)
    for sc, smask in scopes.items():
        Wm = mac_window(H, ths, smask, REPRO_T, strict=True)
        for tname, tm in (("last20", tail), ("from_mac_date", date_tail)):
            m = tm & (H["yomitan"][t] if sc == "yomitan" else True)
            if not m.any():
                continue
            bd, bc = block_codes(H, ths, m)
            Wd, Wc = boot_weights(bd[m]), boot_weights(bc[m])
            for ck, (pg, pt) in data.items():
                def gen():
                    for k, T in enumerate(REPRO_T):
                        P = {"head": (pg[:, 1:] * cdf_at(pt[:, 1:], T)).sum(1)}
                        P.update(mac_preds(pg, "Rec", k, Wm, Cc))
                        for c in CARD_SENS:
                            mp = mac_preds(pg, "Rec", k, Wm, Cc, c)
                            P[f"mac_succ_card_c{int(c)}"], P[f"mac_grade_card_c{int(c)}"] = mp["mac_succ_card"], mp["mac_grade_card"]
                        yield ("fast", "Rec", T), P, ((gr >= 1) & (d < 1000 * T)).astype(np.float64), np.ones(len(ths), bool)
                ev = evaluate(gen(), m, bd, bc, Wd, Wc, clip=1e-7, bar={"fast": MAC_ORDER[1:]}, calibrate=False)
                if tname == "last20":
                    for T in REPRO_T:
                        ref = MAC_REF.get((sc, T))
                        x = ev[f"fast/Rec/{T}"]
                        if ref:
                            x["mac_reference"] = dict(zip(MAC_ORDER, ref))
                            x["ours_minus_mac"] = {n: round(x["predictors"][n]["logloss"] - r, 6) for n, r in zip(MAC_ORDER, ref)}
                res["tables"].setdefault(ck, {})[f"{tname}/{sc}"] = ev
    atomic_json(OUT / "sm121-repro.json", res)
    write_repro_md(res)
    privacy_scan("repro", [OUT / "sm121-repro.md", OUT / "sm121-repro.json"])
    x = res["tables"]["mac_pair"]["last20/yomitan"]["fast/Rec/4"]
    print(f"[sm121-repro] scored {res['scored']:,} (Mac {MAC_COUNTS['scored']:,}); Yomitan 4 s head {x['predictors']['head']['logloss']:.6f} "
          f"(Mac {MAC_REF[('yomitan', 4)][0]}) -> {OUT / 'sm121-repro.md'}")
    return 0


def write_repro_md(res):
    c = res["copy"]
    L = ["# Timed recall on your own collection (SM121): reproduction control of the Mac's replay", "", TOP, "",
         "Kept separate from the main analysis. Spec from the Mac (11-16): the Mac's Small pair (button probabilities of the Small "
         "Instant-Only model (SM98), time from the answer-time head release (SM83) shipped with the show-time retrain (SM92)), f32 "
         "weights, int4 card / int8 note state; eligible = card seen before, at least 1 day elapsed, positive recorded time; the "
         "latest 20% of the eligible reviews scored, all earlier reviews (same-day included) feed the model state and the card "
         "history; outcome = Hard/Good/Easy and recorded time < T (strict); LogLoss clipped at 1e-7; baselines with windows = the "
         "preceding 2,000 eligible reviews in the scope (whole collection, or only the Yomitan decks).", "",
         f"**Input.** The Mac's backup (sha256 `{MAC_BACKUP_SHA[:16]}...`, {c['mac']['canonical_reviews']:,} canonical reviews) is not on "
         f"this computer. Used instead: the closest copy here, the profile's collection file (sha256 `{c['sha256'][:16]}...`, last review "
         f"{c['last_review_month']}, {c['canonical_reviews']:,} canonical reviews). Matches the Mac's backup: {'yes' if c['matches_mac_backup'] else 'no'}. "
         f"Eligible {res['eligible']:,} (prefix {res['prefix']:,}, scored {res['scored']:,}, of which Yomitan {res['scored_yomitan']:,}; "
         f"the Mac: prefix {c['mac']['eligible_prefix']:,}, scored {c['mac']['scored']:,}, Yomitan {c['mac']['scored_yomitan']:,}); tail from "
         f"{res['tail_start_month']}. Numbers cannot match to 2e-6 on a different copy; the comparison checks the method.", "",
         "| combination | scope | T | rows | time head | R x recent | R x card | grade x recent | grade x card | Mac's values (same order) | ours minus Mac (head) | head vs grade x card (day CI) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ck, tabs in res["tables"].items():
        for sc in ("full", "yomitan"):
            ev = tabs.get(f"last20/{sc}", {})
            for T in REPRO_T:
                x = ev.get(f"fast/Rec/{T}")
                if not x or not x.get("vs"):
                    continue
                p = x["predictors"]
                ref = x.get("mac_reference")
                L.append(f"| {COMBOS[ck][2]} | {sc} | {T} s | {x['rows']:,} | " + " | ".join(f"{p[n]['logloss']:.6f}" for n in MAC_ORDER)
                         + " | " + (" / ".join(f"{ref[n]:.6f}" for n in MAC_ORDER) if ref else "") + " | "
                         + (f"{x['ours_minus_mac']['head']:+.6f}" if ref else "") + f" | {fmt_d(x['vs']['mac_grade_card'])} |")
    L += ["", "Card strength sensitivity (1 and 10), card-block CIs, the tail from 28 April 2026 instead of the latest 20%: `sm121-repro.json`."]
    (OUT / "sm121-repro.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def privacy_scan(source, paths=None):
    """The committed outputs must hold no deck name (other than the label) and no per-review fields."""
    _, _, names = yomitan_deck_ids(SOURCES[source])
    paths = paths or [OUT / "sm121-result.md", OUT / "sm121-result.json"]
    bad = []
    for p in paths:
        if not Path(p).exists():
            continue
        text = Path(p).read_text(encoding="utf-8")
        low = text.lower()
        for n in names:
            for part in n.split("\x1f"):
                if len(part) >= 4 and part.lower() not in ("default", "yomitan") and part.lower() in low:
                    bad.append((str(p), "a deck name"))
        for k in ('"review_th"', '"revlog_id"', '"card_id"', '"th"'):
            if k in text:
                bad.append((str(p), k))
    if bad:
        raise SystemExit(f"privacy scan failed: {sorted(set(bad))[:5]}")
    print(f"[sm121-privacy] clean: {[Path(p).name for p in paths]}")
    return True


# ------------------------------------------------------------------ CPU check
def check(args):
    """Tiny CPU check: the first --cap reviews through the Small pair's trunks (fp32 main, s4c8n repro), then proofs on
    whole-history arrays; also the fork files the Mac used against our head / trunks."""
    import subprocess
    import torch
    below_normal()
    started = time.time()
    checks = {}
    WORK.mkdir(parents=True, exist_ok=True)
    tr = {}
    runs = [("sm92", "main", "fp32"), ("sm98", "main", "fp32"), ("sm98", "main", "s4c8n"), ("sm98", "repro", "s4c8n")]
    for key, source, state in runs:
        f = WORK / f"check-trunk-{key}-{source}-{state}.npz"
        r = subprocess.run([sys.executable, "-X", "utf8", __file__, "trunk", "--model", key, "--source", source, "--state", state,
                            "--cap", str(args.cap), "--out", str(f)])
        if r.returncode:
            raise SystemExit(f"check trunk {key} {source} {state} failed")
        z = np.load(f)
        tr[(key, source, state)] = {"th": z["th"], "pg": z["pg"].astype(np.float64),
                                    **{k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("pt_")},
                                    "meta": json.loads(f.with_suffix(".json").read_text())}
    a, b, c, e = tr[("sm92", "main", "fp32")], tr[("sm98", "main", "fp32")], tr[("sm98", "main", "s4c8n")], tr[("sm98", "repro", "s4c8n")]
    checks["K1_trunk_outputs"] = {"pass": bool(np.array_equal(a["th"], b["th"]) and all(x["meta"]["nonfinite_inputs"] == 0 for x in (a, b, c))
                                              and max(x["meta"]["p_grade_sum_maxdev"] for x in (a, b, c)) < 1e-5
                                              and max(x["meta"]["p_time_sum_maxdev"] for x in (a, b, c)) < 1e-5
                                              and set(a["meta"]["heads"]) == {"sm117", "sm83"}),
                                  "rows": a["meta"]["time_labelled_rows"],
                                  "seconds": {f"{k[0]}-{k[1]}-{k[2]}": x["meta"]["total_seconds"] for k, x in tr.items()},
                                  "peak_gb": {f"{k[0]}-{k[1]}-{k[2]}": x["meta"]["peak_gb"] for k, x in tr.items()}}
    # int4 card / int8 note state on the same rows and codes: P(grade) moves a little (the rounding is active), not a lot
    same_rows = bool(np.array_equal(b["th"], c["th"]))
    dev = float(np.abs(b["pg"] - c["pg"]).max()) if same_rows else None
    checks["K1b_state_s4c8n_vs_fp32"] = {"pass": bool(same_rows and dev is not None and 1e-7 < dev < 0.2), "same_rows": same_rows,
                                         "max_abs_p_grade": dev, "mean_abs_p_grade": float(np.abs(b["pg"] - c["pg"]).mean()) if same_rows else None}
    n = min(len(b["th"]), len(e["th"]))
    checks["K1c_repro_copy_runs"] = {"pass": bool(e["meta"]["time_labelled_rows"] > 0 and e["meta"]["nonfinite_inputs"] == 0),
                                     "rows": e["meta"]["time_labelled_rows"], "same_first_rows_as_main": bool(np.array_equal(b["th"][:n], e["th"][:n]))}
    H = history("main")
    lower, _, _ = time_labels(H["d"], H["ed"], same_day=True)
    all_ths = np.flatnonzero(lower >= 0) + 1
    pri = json.loads(PRIORS.read_text())
    B = recent_baselines(H, all_ths, pri)
    F117, E117 = h117.running_baselines(H["rev"], json.loads((ROOT / "artifacts/small-model/sm117/priors.json").read_text()))
    ks = [S_GRID.index(s) for s in h117.S_GRID]
    d1 = float(np.abs(B["F"][:, :, ks] - F117[all_ths - 1]).max())
    d2 = float(np.abs(B["E"]["Rec"][:, ks] - E117[all_ths - 1]).max())
    checks["K2_recent_baselines_match_sm117"] = {"pass": d1 < 1e-6 and d2 < 1e-6, "max_abs_F": d1, "max_abs_E": d2}
    # K3: naive loops on the first 800 history rows (windows shortened to 100 for the test): (2), rolling, card history,
    # the Mac's window (full scope, strict)
    global ROLL
    roll0, ROLL = ROLL, 100
    nmax = 800
    sel = all_ths[all_ths <= nmax]
    Cs = card_counts(H, sel, S_GRID, strict=True)
    Ws = mac_window(H, sel, np.ones(len(H["d"]), bool), S_GRID, strict=True)
    Bs = recent_baselines(H, sel, pri)
    g, d, card, ed = H["grade"], H["d"], H["card"], H["ed"]
    lab = lower >= 0
    worst = 0.0
    for row, th in enumerate(sel):
        i = th - 1
        prev = [j for j in range(i) if lab[j]]
        elig = [j for j in range(i) if ed[j] >= 1 and d[j] > 0][-ROLL:]
        q = [j for j in range(i) if card[j] == card[i] and d[j] > 0][-CARD_LAST:]
        for k, s in enumerate(S_GRID):
            for sn, gs in SETS.items():
                ins = [j for j in prev if g[j] in gs]
                p0 = pri["E0_event"][sn][k] / pri["R0_set"][sn]
                worst = max(worst, abs((sum(d[j] <= 1000 * s for j in ins) + M * p0) / (len(ins) + M) - Bs["succ"][sn][row, k]))
                win = prev[-ROLL:]
                worst = max(worst, abs((sum((g[j] in gs) and d[j] <= 1000 * s for j in win) + M * pri["E0_event"][sn][k]) / (len(win) + M)
                                       - Bs["roll_fast"][sn][row, k]))
                cq = [j for j in q if g[j] in gs]
                worst = max(worst, abs(len(cq) - Cs["set_n"][sn][row]), abs(sum(d[j] < 1000 * s for j in cq) - Cs["set_fast"][sn][row, k]))
            es = [j for j in elig if g[j] >= 1]
            rf = (sum(d[j] < 1000 * s for j in es) + 0.5) / (len(es) + 1)
            worst = max(worst, abs(rf - Ws["rec_frac"]["Rec"][row, k]))
            for gg in (1, 2, 3):
                eg = [j for j in elig if g[j] == gg]
                gf = (sum(d[j] < 1000 * s for j in eg) + MAC_GRADE_C * rf) / (len(eg) + MAC_GRADE_C)
                worst = max(worst, abs(gf - Ws["grade_frac"][gg][row, k]))
    ROLL = roll0
    checks["K3_baselines_naive_equal"] = {"pass": worst < 1e-9, "max_abs": worst, "rows": int(len(sel)), "test_window": 100}
    # K4: labels / probabilities on the check rows (Small pair: SM98 grades, SM92 SM117 head) through the real generator
    ths = a["th"]
    cap = time_user_cap(H["d"])
    Bc = recent_baselines(H, ths, pri)
    Cc = card_counts(H, ths, S_GRID, strict=False)
    Wm = mac_window(H, ths, np.ones(len(H["d"]), bool), S_GRID, strict=False)
    ok, det = True, {}
    G = {gk: (P, y, v) for gk, P, y, v in groups(b["pg"], a["sm117"], H, ths, Bc, Cc, Wm, cap)}
    for s in S_GRID:
        r = {sn: G[("fast", sn, s)] for sn in SETS}
        if not r["Rec"][2].any():
            continue
        yr = {sn: r[sn][1].mean() for sn in SETS}
        mp = {sn: r[sn][0]["head"].mean() for sn in SETS}
        ok &= bool(yr["E"] <= yr["GE"] <= yr["Rec"] and mp["E"] <= mp["GE"] + 1e-12 <= mp["Rec"] + 2e-12)
        for sn in SETS:
            ok &= bool((r[sn][0]["head"] <= r[sn][0]["raw"] + 1e-9).all())
            for pn, p in r[sn][0].items():
                ok &= bool(np.isfinite(p).all() and (p > 0).all() and (p < 1).all())
        det[str(s)] = {"event_rate": {k: round(float(v), 4) for k, v in yr.items()}, "head_mean_p": {k: round(float(v), 4) for k, v in mp.items()}}
    mono = all(det[str(x)]["event_rate"]["Rec"] <= det[str(y)]["event_rate"]["Rec"] for x, y in zip(S_GRID, S_GRID[1:]) if str(x) in det and str(y) in det)
    checks["K4_labels_and_probabilities"] = {"pass": bool(ok and mono), "user_cap_s": cap / 1000, "per_S": det}
    # K5: bootstrap code
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.1, 5000)
    days = np.repeat(np.arange(50), 100)
    one = np.ones(5000, bool)
    ev_rows = evaluate(iter([(("fast", "Rec", 4), {"head": np.full(5000, 0.5), "b": np.full(5000, 0.5)}, (x > 0).astype(float), one)]),
                       one, np.arange(5000), days, boot_weights(np.arange(5000)), boot_weights(days), calibrate=False)
    v = ev_rows["fast/Rec/4"]["vs"]["b"]
    checks["K5_bootstrap"] = {"pass": bool(v["delta"] == 0 and v["ci_day"] == [0, 0]), "identical_predictors_give_zero": v}
    # K6: deck map
    checks["K6_yomitan_decks"] = {"pass": bool(H["n_decks"] == H["n_dataset_decks"] and H["yomitan_decks"] >= 1 and 0 < H["yomitan"].mean() < 1),
                                  "decks_used": H["n_decks"], "yomitan_decks": H["yomitan_decks"],
                                  "yomitan_share_of_history": round(float(H["yomitan"].mean()), 4)}
    S_all = slices(H, all_ths)
    checks["K7_slices"] = {"pass": bool(S_all["bench"].sum() > 100000 and S_all["mac"].sum() > 10000),
                           "bench_rows": int(S_all["bench"].sum()), "mac_rows": int(S_all["mac"].sum()),
                           "mac_yomitan_rows": int((S_all["mac"] & H["yomitan"][all_ths - 1]).sum()),
                           "mac_date_rows": int(S_all["mac_date"].sum()), "time_labelled_rows": int(len(all_ths)), "history_rows": int(len(H["d"]))}
    # K8: evaluation + discrimination end to end on the check rows
    m = np.ones(len(ths), bool)
    bd, bc = block_codes(H, ths, m)
    Wd, Wc = boot_weights(bd), boot_weights(bc)
    ev = evaluate(groups(b["pg"], a["sm117"], H, ths, Bc, Cc, Wm, cap), m, bd, bc, Wd, Wc,
                  bar={"fast": FAST_BAR, "speed": SPEED_BAR, "grade": GRADE_BAR})
    disc = discrimination(b["pg"], a["sm117"], H, ths, Bc, Cc, Wm, m, bd, Wd, cap)
    x4 = ev["fast/Rec/4"]
    checks["K8_evaluation_runs"] = {"pass": bool(x4["rows"] > 0 and set(FAST_BAR) <= set(x4["vs"]) and "spread" in disc),
                                    "rec_4s_logloss": {pn: v["logloss"] for pn, v in x4["predictors"].items()}, "rows": x4["rows"],
                                    "spread_rows": disc["spread"]["rows"], "r2": disc["r2_mean_ln_time_good"]}
    # K9: the fork's shipped files the Mac used (read only): sha256 and our head / trunk tensors inside them
    fk = {}
    for rel, want in FORK_SHA.items():
        p = FORK / rel
        fk[rel] = {"exists": p.exists(), "sha256_ok": p.exists() and sha256(p) == want}
    blob = (FORK / "sm92/time_head.f32.bin").read_bytes() if fk["sm92/time_head.f32.bin"]["exists"] else b""
    sd = torch.load(MODELS["sm92"]["shipped_head"], map_location="cpu", weights_only=True)
    hk = [k for k in sd if k.startswith("time_mlps.0.")]
    fk["sm83_head_tensors_in_fork_file"] = all(sd[k].float().numpy().tobytes() in blob for k in hk) if hk and blob else False
    for key, rel in (("sm92", "sm92/weights.f32.bin"), ("sm98", "sm98/weights.f32.bin")):
        wb = (FORK / rel).read_bytes() if fk[rel]["exists"] else b""
        st = torch.load(verified_master(key)["master"], map_location="cpu", weights_only=True)
        big = sorted(st, key=lambda k: -st[k].numel())[:3]
        fk[f"{key}_largest_tensors_in_fork_file"] = all(st[k].float().numpy().tobytes() in wb for k in big) if wb else False
    checks["K9_fork_files"] = {"pass": all(v["sha256_ok"] for k, v in fk.items() if isinstance(v, dict)), **fk,
                               "note": "pass = the three files have the Mac's sha256; the tensor-in-file flags are informational "
                               "(the container may store tensors in another layout)"}
    # K10: CPU time projection for the full history (linear in reviews)
    scale = len(H["d"]) / max(args.cap, 1)
    checks["K10_time_projection_minutes"] = {"pass": True, "cap": args.cap,
                                             "projected_full_minutes": {f"{k[0]}-{k[2]}": round((x["meta"]["forward_seconds"] + x["meta"]["prepare_seconds"]) * scale / 60, 1)
                                                                        for k, x in tr.items()}}
    res = {"device": "cpu", "cap_reviews": args.cap, "pass": all(v["pass"] for v in checks.values()), "checks": checks,
           "seconds": round(time.time() - started, 1)}
    out = Path(args.out)
    atomic_json(out, res)
    privacy_scan("main", [out])
    print(f"[sm121-check] {'PASS' if res['pass'] else 'FAIL'}: " + ", ".join(f"{k} {v['pass']}" for k, v in checks.items()))
    return 0 if res["pass"] else 1


def main():
    p0 = ArgumentParser(description=__doc__.split("\n")[0])
    sub = p0.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    p = sub.add_parser("trunk")
    p.add_argument("--model", required=True, choices=list(TRUNKS))
    p.add_argument("--source", default="main", choices=list(SOURCES))
    p.add_argument("--state", default="fp32", choices=("fp32", "s4c8n"))
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--cap", type=int, default=800)
    p.add_argument("--out", default=str(OUT / "check-cpu.json"))
    sub.add_parser("report")
    sub.add_parser("repro")
    a = p0.parse_args()
    return {"convert": convert, "trunk": trunk, "check": check, "report": report, "repro": repro}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
