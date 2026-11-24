#!/usr/bin/env python
"""Output-aware low-rank truncation (SM157): CPU proofs (design.md "Proofs"), run by cpu-proof.sh.

  python sm157_check.py cpu [--out check-cpu-sm157.json]

K1 synthetic (scripts/state_lowrank selftest, selftest_sm148, selftest_sm157, CPU): the SM147 / SM148 / SM152 identities
   unchanged; the weighted factored fit equals the weighted optimum; young states exact with weights; M = I gives the
   plain fit; the Gramian solves M = R + D M D; the label grammar (plain labels carry no weight).
K2 scorer rows, the 3 smoke users x the 6 models (CPU): REF, Q2 and T24 == SM148's committed CPU smoke rows, every metric
   field identical (bit for bit: the JSON values equal, not within a tolerance); descriptors unchanged (no weight key).
K3 pipeline rows (user 2388 x the 6 models x Q2R, Q2RK, T24G, test statistics of 2 training users): a finite row for
   every (model, label), the descriptor names the weight kind, the benchmark LogLoss differs from the plain label's.
Report only: the per-write cost of the codec with and without weights (microbenchmark, CPU, one thread).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "Q2": "w8+lr2q8x8", "T24": "w8+lr2.16.4q8x8",
          "Q2R": "w8+lr2q8x8wr", "Q2RK": "w8+lr2q8x8wrk", "T24G": "w8+lr2.16.4q8x8wg"}
KEPT = ("REF", "Q2", "T24")
PIPE = {"Q2R": ("Q2", "r"), "Q2RK": ("Q2", "rk"), "T24G": ("T24", "g")}
SKIP_FIELDS = {"label", "variant", "model_seconds", "prepare_seconds", "provenance_commit", "provenance_dirty", "scorer"}
SM148_ROWS = Path("C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/cpu/smoke-cpu-rows.jsonl")


def read(path):
    out = {}
    for raw in open(path, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "benchmark" in r or "nonfinite" in r:
            out[(r["user_id"], r["tag"] if "tag" in r else r["label"].split("/")[0], r["variant"])] = r
    return out


def finite(r):
    return "nonfinite" not in r and all(math.isfinite(r["benchmark"][h]["LogLoss"]) for h in ("ahead", "instant")
                                        if isinstance(r["benchmark"].get(h), dict))


def k1():
    from scripts import state_lowrank
    try:
        state_lowrank.selftest()
        info = {"sm148": state_lowrank.selftest_sm148("cpu"), "sm157": state_lowrank.selftest_sm157("cpu")}
        ok = True
    except AssertionError as e:
        ok, info = False, {"error": repr(e)}
    return {"K1 state_lowrank selftests (SM147, SM148 + SM152, SM157; CPU)": ok}, info


def k2(rows):
    old = read(SM148_ROWS)
    users = sorted({u for u, _, _ in rows})
    tags = sorted({t for _, t, _ in rows})
    bad, compared, fields = [], 0, set()
    for u in users:
        for t in tags:
            for n in KEPT:
                a, b = rows.get((u, t, LABELS[n])), old.get((u, t, LABELS[n]))
                if a is None or b is None:
                    bad.append([u, t, n, "missing", a is None, b is None])
                    continue
                keys = (set(a) | set(b)) - SKIP_FIELDS
                diff = [k for k in sorted(keys) if a.get(k) != b.get(k)]
                fields |= keys
                compared += 1
                if diff:
                    bad.append([u, t, n, diff[:6]])
    res = {"K2 REF, Q2 and T24 rows == SM148's committed CPU smoke rows (every metric and descriptor field identical)":
           not bad and compared == 3 * len(KEPT) * 6}
    return res, {"compared_rows": compared, "users": users, "models": len(tags), "fields_compared": len(fields),
                 "mismatches": bad[:10]}


def k3(pipe, smoke):
    bad = []
    n = 0
    for (u, t, v), r in sorted(pipe.items()):
        name = next((k for k, lab in LABELS.items() if lab + E2S == v or lab == v), None)
        if name not in PIPE:
            continue
        n += 1
        base, kind = PIPE[name]
        plain = smoke.get((u, t, LABELS[base]))
        d = r.get("state_lowrank") or {}
        if not finite(r):
            bad.append([u, t, name, "non-finite"])
        elif d.get("weight") != kind or not d.get("weight_stats_sha256"):
            bad.append([u, t, name, "descriptor", d])
        elif plain is None or plain["benchmark"] == r["benchmark"]:
            bad.append([u, t, name, "equal to plain or plain missing"])
    return ({"K3 pipeline rows: 6 models x 3 weighted labels, finite, weight descriptor, differ from plain": not bad and n == 18},
            {"rows": n, "bad": bad[:10]})


def bench():
    """us per write of LowRankCodec.encode (rank 2, int8 factors), plain vs weighted, one thread, CPU."""
    from scripts import state_lowrank as s
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    out = {}
    for K in (16, 32):
        N, H = 256, 4
        nxt = torch.randn(N, H, K, K, generator=g)
        lp = torch.randn(N, H, K, 2, generator=g)
        v = torch.randn(N, H, K, generator=g)
        n = torch.full((N,), 20, dtype=torch.int64)
        X = torch.randn(H, 4 * K, K, generator=g)
        wts = s.sqrt_pair((X.mT @ X).double() / (4 * K) + 1e-3 * torch.eye(K, dtype=torch.float64))
        codec = s.parse_sm148("lr2q8x8")
        res = {}
        for name, w in (("plain", None), ("weighted", wts)):
            codec.encode(nxt, lp, v, n, None, w)
            best = math.inf
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(10):
                    codec.encode(nxt, lp, v, n, None, w)
                best = min(best, (time.perf_counter() - t0) / 10)
            res[name] = best / N * 1e6
        res["extra_pct"] = 100 * (res["weighted"] / res["plain"] - 1)
        # multiply-adds per head and write: S M^(1/2) as done here (K^3) or in the factored form Q^T S M^(1/2) (m K^2,
        # m = r + 1 = 3), and the decode rf M^(-1/2) (r K^2)
        res["madds_per_head_SxMh_dense"] = K ** 3
        res["madds_per_head_SxMh_factored"] = 3 * K * K
        res["madds_per_head_decode"] = 2 * K * K
        out[f"K{K}_us_per_write_4_heads"] = res
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("cpu",))
    p.add_argument("--out", default=str(HERE / "check-cpu-sm157.json"))
    a = p.parse_args()
    res, info = {}, {}
    r, i = k1()
    res.update(r); info["K1"] = i
    smoke = read(HERE / "cpu" / "smoke-cpu-rows.jsonl")
    r, i = k2(smoke)
    res.update(r); info["K2"] = i
    r, i = k3(read(HERE / "cpu" / "pipe-cpu-rows.jsonl"), smoke)
    res.update(r); info["K3"] = i
    info["bench"] = bench()
    doc = {"all_ok": all(res.values()), "checks": res, "info": info}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(("OK  " if v else "FAIL") + " " + k)
    print("ALL OK" if doc["all_ok"] else "SOME CHECKS FAILED")
    sys.exit(0 if doc["all_ok"] else 1)


if __name__ == "__main__":
    main()
