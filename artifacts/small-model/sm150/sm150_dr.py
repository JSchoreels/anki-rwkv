#!/usr/bin/env python
"""Low-rank int8 card and note states, post-training (SM150): per-review and per-card worst-case read-out of the predicted
retention vs int8 (REF) on the long-card users, from the CPU dumps (--dump-scope all). The read-out code is the int3 young
tier on the early switch (SM145)'s sm145_dr.py (this tree's copy, sha256 pinned) with the SM150 labels; aggregates only
(no user or card ids, no review rows).

  python sm150_dr.py aggregate --dump DIR --users users-longcard.json --out sm150-dr.json [--min-users 48]
  python sm150_dr.py selftest      sm145_dr.py's own selftest, then the SM150 labels on synthetic dumps
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "sm145" / "sm145_dr.py"
LIB_SHA = {"967e988a25e761b87118a4d9859ae5191c3e3134a75084dddac7f4ed16edbe3b",   # CRLF working copy
           "b53e765b2920659dc0bf7c6d530b6da837a2a773eb996c0d26481965a6f71936"}   # LF (the sm145 worktree)
LABELS = {"REF": "w8+s8hx8", "R1": "w8+lr1q8x8", "R12": "w8+lr1.16.2q8x8"}
DR_LABELS = ("R12", "R1")


def lib():
    if hashlib.sha256(LIB.read_bytes()).hexdigest() not in LIB_SHA:
        raise SystemExit("[sm150-dr] sm145_dr.py changed: refusing")
    sys.path.insert(0, str(LIB.parent))
    import sm145_dr as dr
    if not hasattr(dr, "LABELS_SM145"):
        dr.LABELS_SM145 = dict(dr.LABELS)
    dr.LABELS = dict(LABELS)
    return dr


def aggregate(dump, users, min_users, out, sizes=None):
    """sizes: e.g. {"Small", "Tiny"} (Big long-card users are not scored in SM150)."""
    dr = lib()
    models = {k: v for k, v in dr.MODELS.items() if sizes is None or v[3] in sizes}
    return dr.aggregate(dump, users, min_users, out, models=models, labels=DR_LABELS)


def selftest():
    dr = lib()
    import contextlib
    import io
    dr.LABELS = dict(dr.LABELS_SM145)
    with contextlib.redirect_stdout(io.StringIO()):
        ok = dr.selftest() == 0                      # the library's own checks (its own labels)
    dr.LABELS = dict(LABELS)
    rng = np.random.default_rng(0)
    users = [11, 12]
    models = {"m": ("tagm", "a test model", ("curve", "instant"), "Big")}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        infos = {}
        for u in users:
            cid = np.repeat(np.arange(40), 10)                       # 40 cards x 10 answers, 400 reviews
            prior = np.tile(np.arange(10), 40)
            infos[u] = (prior, cid, np.bincount(cid))
            th = np.arange(1, 401)
            p = rng.uniform(0.5, 0.99, 400).astype(np.float32)
            for lab, shift in (("REF", 0.0), ("R1", 0.2), ("R12", 0.06)):
                q = p.copy()
                if lab == "R1":
                    q[5] = np.float32(p[5] - shift)                  # one review of card 0 off by 0.2
                elif lab == "R12":
                    q[:20] = p[:20] + np.float32(shift)              # cards 0-1: every review off by 0.06
                np.savez(tmp / dr.dump_name(u, "tagm", LABELS[lab]), ahead_th=th, ahead_p=q, instant_th=th, instant_p=q)
        out = tmp / "dr.json"
        dr.aggregate(tmp, users, 2, out, models=models, labels=DR_LABELS, data="unused", infos=infos)
        d = json.loads(out.read_text())["models"]["m"]["labels"]
        ok &= d["R1"]["heads"]["curve"]["reviews"]["all"]["max"] > 0.19
        ok &= d["R1"]["heads"]["curve"]["cards"]["all"]["count_gt_0.10"] == 2               # one card per user
        ok &= abs(d["R1"]["heads"]["curve"]["cards"]["all"]["share_gt_0.10"] - 2 / 80) < 1e-12
        ok &= d["R12"]["heads"]["instant"]["cards"]["all"]["count_gt_0.05"] == 4
        ok &= d["R12"]["heads"]["instant"]["cards"]["all"]["count_gt_0.10"] == 0
    print(f"[sm150-dr] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--dump", required=True)
    a.add_argument("--users", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--min-users", type=int, default=48)
    a.add_argument("--sizes", default="")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    return aggregate(args.dump, json.loads(Path(args.users).read_text())["users"], args.min_users, args.out,
                     set(args.sizes.split(",")) if args.sizes else None)


if __name__ == "__main__":
    raise SystemExit(main())
