#!/usr/bin/env python
"""Int3 young tier on the early switch (SM145): per-review and per-card worst-case read-out of the predicted retention,
tiered labels vs int8 cards and notes (REF, w8+s8hx8), on the long-card users (CPU). Report only, no verdict except the
"worth its bytes" criterion of design.md (which reads the per-card share above 0.10).

Input: the scorer's --dump DIR --dump-scope all files (one per user, model and label: per head the review_th of every
review the scorer predicts, i.e. the "all" scope of --prior-buckets, long cards included, and the head's probability as
float32). The dumps hold per-review values and stay outside git (work/); this script writes aggregates only: no user
ids, no card ids, no review rows.

  dR = p(label) - p(REF) on the same review (the same review_th in both dumps; the sets must be equal), per head (curve
       = the "ahead" head of the two-head models, instant).
  reviews: |dR| over every predicted review, pooled over the users, by the card's answers before the review (buckets 0,
       1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+, all): n, mean |dR|, mean signed dR, P50 / P90 / P99 / P99.9 / max of
       |dR|, share and count of reviews with |dR| > 0.05 and > 0.10.
  cards: per card the worst review (max |dR| over the card's predicted reviews), by the card's answers in the whole
       history (1-2, ..., 128+, all): cards, P50 / P90 / P99 / max, share and count of cards above 0.05 and 0.10.
  users: how many users have at least one review above 0.05 / 0.10, the largest per-user share above 0.05.

  python sm145_dr.py aggregate --dump DIR --users users-longcard.json --out sm145-dr.json [--min-users 48]
  python sm145_dr.py selftest
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "AN16": "w8+t4.16.8c4.16.8nhx8", "Y2N": "w8+t3.3.4.16.8c3.3.4.16.8nhx8",
          "Y4C": "w8+t3.5.4.16.8c4.16.8nhx8", "Y4N": "w8+t3.5.4.16.8c3.5.4.16.8nhx8"}
DR_LABELS = ("AN16", "Y2N", "Y4C", "Y4N")
TWO, IO = ("curve", "instant"), ("instant",)
KEY = {"curve": "ahead", "instant": "instant"}
MODELS = {   # key: (tag, plain name, heads, size); the same six models as sm145_report.py
    "sm91s2": ("sm91s2-big-showtime", "the show-time big model (SM91s2)", TWO, "Big"),
    "sm99": ("sm99-big-inst-e2", "the show-time instant-only big model (SM99)", IO, "Big"),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "the show-time two-head cool-down for Small (SM118)", TWO, "Small"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "the longer Small Instant-Only model (SM113)", IO, "Small"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "the six-layer Tiny two-head model (SM130)", TWO, "Tiny"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "the six-layer Tiny Instant-Only model (SM130)", IO, "Tiny"),
}
REV_EDGES = [0, 1, 3, 5, 9, 16, 32, 128]          # answers of the card before the review
REV_NAMES = ["0", "1-2", "3-4", "5-8", "9-15", "16-31", "32-127", "128+"]
CARD_EDGES = [1, 3, 5, 9, 16, 32, 128]            # answers of the card in the whole history
CARD_NAMES = ["1-2", "3-4", "5-8", "9-15", "16-31", "32-127", "128+"]
LIMITS = (0.05, 0.10)


def data_path():
    from scripts import compress_common as cc
    return cc.data_path(cc.run_info())


def card_info(data, user):
    """Per revlog row (review_th - 1, the scorer's frame order): the card's earlier answers, the card's code (0..cards-1)
    and, per card code, the card's answers in the whole history."""
    import pandas as pd
    cid = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id"])["card_id"]
    prior = cid.groupby(cid).cumcount().to_numpy()
    codes, _ = pd.factorize(cid)
    return prior, codes, np.bincount(codes)


def dump_name(user, tag, variant):
    """The scorer's dump file name (dump_all_scope: f"{user}_{label.replace('/', '_')}.npz", label tag/variant@e2s-query)."""
    return f"{user}_{tag}_{variant}{E2S}.npz"


def stats(x, signed=None):
    """Distribution of |dR| values x (float64 array)."""
    n = int(len(x))
    if not n:
        return {"n": 0}
    q = np.percentile(x, [50, 90, 99, 99.9])
    out = {"n": n, "mean_abs": float(x.mean()), "P50": float(q[0]), "P90": float(q[1]), "P99": float(q[2]),
           "P99.9": float(q[3]), "max": float(x.max())}
    if signed is not None:
        out["mean_signed"] = float(signed.mean())
    for lim in LIMITS:
        c = int((x > lim).sum())
        out[f"count_gt_{lim:.2f}"] = c
        out[f"share_gt_{lim:.2f}"] = c / n
    return out


def per_user(prior, codes, ncard, th, p_new, p_ref):
    """(review buckets, |dR|, signed dR, per-card worst |dR|, per-card bucket) of one user, model, head and label."""
    d = p_new.astype(np.float64) - p_ref.astype(np.float64)
    a = np.abs(d)
    rb = np.searchsorted(np.asarray(REV_EDGES[1:]), prior[th - 1], side="right")
    cc = codes[th - 1]
    worst = np.full(len(ncard), -1.0)
    np.maximum.at(worst, cc, a)
    seen = worst >= 0
    cb = np.searchsorted(np.asarray(CARD_EDGES[1:]), ncard[seen], side="right")
    return rb, a, d, worst[seen], cb


def aggregate(dump, users, min_users, out_path, models=MODELS, labels=DR_LABELS, data=None, infos=None):
    dump = Path(dump)
    data = data if data is not None else (data_path() if infos is None else None)
    infos = infos if infos is not None else {}
    res = {"note": "aggregates only (no user or card ids, no review rows); |dR| = |p(label) - p(int8 REF)| per review, "
                   "the scorer's all scope (long cards included); CPU long-card users",
           "limits": list(LIMITS), "review_buckets": REV_NAMES, "card_buckets": CARD_NAMES, "models": {}}
    finite, complete = True, True
    for key, (tag, name, heads, size) in models.items():
        m = {"name": name, "size": size, "heads": list(heads), "labels": {}}
        acc = {(lab, h): {"rb": [], "a": [], "d": [], "w": [], "cb": [], "users": 0, "u05": 0, "u10": 0, "umax05": 0.0}
               for lab in labels for h in heads}
        missing = {lab: 0 for lab in labels}
        for u in users:
            fr = dump / dump_name(u, tag, LABELS["REF"])
            if not fr.exists():
                for lab in labels:
                    missing[lab] += 1
                continue
            if u not in infos:
                infos[u] = card_info(data, u)
            prior, codes, ncard = infos[u]
            ref = np.load(fr)
            for lab in labels:
                f = dump / dump_name(u, tag, LABELS[lab])
                if not f.exists():
                    missing[lab] += 1
                    continue
                z = np.load(f)
                for h in heads:
                    k = KEY[h]
                    th, th0 = z[f"{k}_th"], ref[f"{k}_th"]
                    if not np.array_equal(th, th0):
                        raise SystemExit(f"[sm145-dr] review sets differ: model {key} label {lab} head {h} (a user)")
                    pn, pr = z[f"{k}_p"], ref[f"{k}_p"]
                    finite &= bool(np.isfinite(pn).all() and np.isfinite(pr).all())
                    rb, a, d, w, cb = per_user(prior, codes, ncard, th, pn, pr)
                    x = acc[(lab, h)]
                    x["rb"].append(rb.astype(np.int8)); x["a"].append(a.astype(np.float32)); x["d"].append(d.astype(np.float32))
                    x["w"].append(w.astype(np.float32)); x["cb"].append(cb.astype(np.int8))
                    x["users"] += 1
                    x["u05"] += int((a > 0.05).any()); x["u10"] += int((a > 0.10).any())
                    x["umax05"] = max(x["umax05"], float((a > 0.05).mean()) if len(a) else 0.0)
        for lab in labels:
            m["labels"][lab] = {"missing_users": missing[lab], "heads": {}}
            for h in heads:
                x = acc[(lab, h)]
                if x["users"] < min_users:
                    complete = False
                if not x["users"]:
                    m["labels"][lab]["heads"][h] = {"users": 0}
                    continue
                rb, a, d, w, cb = (np.concatenate(x[k]) for k in ("rb", "a", "d", "w", "cb"))
                a, d, w = a.astype(np.float64), d.astype(np.float64), w.astype(np.float64)
                rev = {"all": stats(a, d)}
                for i, nm in enumerate(REV_NAMES):
                    rev[nm] = stats(a[rb == i], d[rb == i])
                cards = {"all": stats(w)}
                for i, nm in enumerate(CARD_NAMES):
                    cards[nm] = stats(w[cb == i])
                m["labels"][lab]["heads"][h] = {
                    "users": x["users"], "reviews": rev, "cards": cards,
                    "users_with_a_review_gt_0.05": x["u05"], "users_with_a_review_gt_0.10": x["u10"],
                    "largest_user_share_gt_0.05": x["umax05"]}
        res["models"][key] = m
    res["finite"], res["complete"], res["min_users"] = bool(finite), bool(complete), min_users
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    worst = max((hd["reviews"]["all"].get("max", 0.0) for mm in res["models"].values() for lb in mm["labels"].values()
                 for hd in lb["heads"].values() if "reviews" in hd), default=0.0)
    print(f"[sm145-dr] {'finite' if finite else 'NON-FINITE'}, {'complete' if complete else 'INCOMPLETE'}; "
          f"largest |dR| {worst:.4f}; -> {out_path}")
    return 0 if finite else 1


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        infos = {}
        models = {"m": ("tagm", "a model (M)", TWO, "Big")}
        for u in (1, 2):
            cid = rng.integers(0, 30, size=400)
            prior = np.array([np.sum(cid[:i] == c) for i, c in enumerate(cid)])
            codes = np.unique(cid, return_inverse=True)[1]
            infos[u] = (prior, codes, np.bincount(codes))
            th = np.arange(1, 401, dtype=np.int64)
            p = rng.random(400).astype(np.float32)
            for lab, shift in (("REF", 0.0), ("AN16", 0.0), ("Y2N", 0.2), ("Y4C", 0.07), ("Y4N", 0.0)):
                q = p.copy()
                if lab == "Y4C":
                    q[codes == codes[0]] += shift        # one card, every review +0.07
                elif lab == "Y2N":
                    q[0] += shift                        # one review +0.2
                np.savez(tmp / dump_name(u, "tagm", LABELS[lab]), ahead_th=th, ahead_p=q, instant_th=th, instant_p=q)
        out = tmp / "dr.json"
        aggregate(tmp, [1, 2], 2, out, models=models, infos=infos)
        d = json.loads(out.read_text())["models"]["m"]["labels"]
        ok &= d["AN16"]["heads"]["curve"]["reviews"]["all"]["max"] == 0.0
        ok &= d["Y2N"]["heads"]["curve"]["reviews"]["all"]["count_gt_0.10"] == 2              # one review per user
        ok &= d["Y2N"]["heads"]["curve"]["cards"]["all"]["count_gt_0.10"] == 2
        ok &= d["Y4C"]["heads"]["instant"]["reviews"]["all"]["count_gt_0.10"] == 0
        n0 = sum(int((infos[u][1] == infos[u][1][0]).sum()) for u in (1, 2))
        ok &= d["Y4C"]["heads"]["instant"]["reviews"]["all"]["count_gt_0.05"] == n0
        ok &= d["Y4C"]["heads"]["instant"]["cards"]["all"]["count_gt_0.05"] == 2
        ok &= d["Y2N"]["heads"]["curve"]["reviews"]["0"]["count_gt_0.10"] == 2             # review 1 = first answer
        ok &= abs(d["Y2N"]["heads"]["curve"]["reviews"]["all"]["mean_signed"] - 0.4 / 800) < 1e-6
        ok &= sum(d["Y4N"]["heads"]["curve"]["reviews"][nm]["n"] for nm in REV_NAMES) == 800
        ok &= sum(d["Y4N"]["heads"]["curve"]["cards"][nm].get("n", 0) for nm in CARD_NAMES) == \
            d["Y4N"]["heads"]["curve"]["cards"]["all"]["n"]
    print(f"[sm145-dr] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--dump", required=True)
    a.add_argument("--users", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--min-users", type=int, default=48)
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    users = json.loads(Path(args.users).read_text())["users"]
    return aggregate(args.dump, users, args.min_users, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
