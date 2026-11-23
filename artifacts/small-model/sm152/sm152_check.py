#!/usr/bin/env python
"""Smaller shift vectors and factors on T24 states (SM152): CPU proofs (before any GPU ticket) and the GPU self-test
(first step of the smoke ticket). design.md "Proofs".

  python sm152_check.py cpu [--out check-cpu-sm152.json]      needs cpu/smoke-cpu-rows.jsonl and the smoke dumps (cpu-proof.sh)
  python sm152_check.py gpu-selftest --out gpu/gpu-selftest.json

K1 bytes and bit packing (sm152_bytes.py: REF / T24 / Q2 bytes = SM148's table; byte order F6S4 < F6 < S4 < T24 < REF;
   pack / unpack round trip of int8 / int6 / int4 codes; the packed codes decode to the codec's own floats).
K2 synthetic (scripts/state_lowrank.selftest / selftest_sm148, CPU): the SM148 identities plus the int6 round trip and the
   SM152 label grammar (q6, x4; q8 / x8 unchanged).
K3 scorer rows, the 3 smoke users x the 6 models (CPU): REF, T24 and Q2 == SM148's committed CPU smoke rows (the codec
   change leaves the existing labels unchanged: benchmark LogLoss within 1e-6 per user and head, counts equal); S4, F6 and
   F6S4 differ from T24 on every (user, model); F6S4 differs from S4 and from F6 on at least one user per model; the rows'
   codec descriptors (factor format, shift bits 4 | float16 scale for x4); every row finite.
K5 the dump: a file per (user, model, label); dumped reviews per bucket == the rows' all-scope counts; same reviews in
   every label; |dR| S4 vs REF > 0; sm152_dr.py aggregates the smoke dumps.
Report only: CPU review-evals/s per label (scorer rows) and the per-write cost of the codec (microbenchmark).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
E2S = "@e2s-query"
LABELS = {"REF": "w8+s8hx8", "T24": "w8+lr2.16.4q8x8", "Q2": "w8+lr2q8x8", "S4": "w8+lr2.16.4q8x4",
          "F6": "w8+lr2.16.4q6x8", "F6S4": "w8+lr2.16.4q6x4"}
SCORED = ("S4", "F6", "F6S4")
KEPT = ("REF", "T24", "Q2")      # existing labels: must equal SM148's rows
DUMPED = ("REF", "S4", "F6", "F6S4")
WANT_DESC = {"S4": ("q8", 260), "F6": ("q6", 264), "F6S4": ("q6", 260), "T24": ("q8", 264)}   # (quant, shift bits)
SKIP_FIELDS = {"label", "variant", "state_bits", "state_tiers", "state_entity_bits", "model_seconds", "prepare_seconds",
               "provenance_commit", "provenance_dirty", "scorer", "state_vq_path", "state_lowrank", "state_scale_bits",
               "state_shift_bits"}   # descriptors, not metrics
CPU_ROWS = HERE / "cpu" / "smoke-cpu-rows.jsonl"
SM148_ROWS = HERE.parent / "sm148" / "cpu" / "smoke-cpu-rows.jsonl"
SMOKE_DUMP = ROOT / "work" / "sm152-dump-smoke"
TOL_REF = 1e-6   # REF, T24, Q2 rescored vs SM148's rows


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


def rows_equal(a, b):
    keys = (set(a) | set(b)) - SKIP_FIELDS
    return [k for k in sorted(keys) if a.get(k) != b.get(k)]


def k2():
    from scripts import state_lowrank
    res = {}
    try:
        state_lowrank.selftest()
        info = state_lowrank.selftest_sm148("cpu")
        res["K2 state_lowrank selftests (SM147, SM148 + SM152 int6 / grammar, CPU)"] = True
    except AssertionError as e:
        res["K2 state_lowrank selftests (SM147, SM148 + SM152 int6 / grammar, CPU)"] = False
        info = {"error": str(e)}
    return res, info


def k3(rows):
    res, info = {}, {"rows": len(rows)}
    users = sorted({u for u, _, _ in rows})
    tags = sorted({t for _, t, _ in rows})
    old = read(SM148_ROWS)
    bad = {k: [] for k in ("sm148", "differ_t24", "finite", "missing", "descriptor")}
    kept_max = 0.0
    f6s4 = {t: {"S4": False, "F6": False} for t in tags}
    for u in users:
        for t in tags:
            get = lambda n: rows.get((u, t, LABELS[n]))   # noqa: E731
            if any(get(n) is None for n in LABELS):
                bad["missing"].append([u, t, [n for n in LABELS if get(n) is None]])
                continue
            for n in LABELS:
                r = get(n)
                if "nonfinite" in r or not all(math.isfinite(r["benchmark"][h]["LogLoss"]) for h in ("ahead", "instant")
                                               if isinstance(r["benchmark"].get(h), dict)):
                    bad["finite"].append([u, t, n])
            for n in KEPT:
                o = old.get((u, t, LABELS[n]))
                if o is None:
                    bad["sm148"].append([u, t, n, "missing"])
                    continue
                for h in ("ahead", "instant"):
                    x, y = get(n)["benchmark"].get(h), o["benchmark"].get(h)
                    if isinstance(x, dict) and isinstance(y, dict):
                        d = abs(x["LogLoss"] - y["LogLoss"])
                        kept_max = max(kept_max, d)
                        if x["size"] != y["size"] or d > TOL_REF:
                            bad["sm148"].append([u, t, n, h, x["LogLoss"], y["LogLoss"]])
                    elif (x is None) != (y is None):
                        bad["sm148"].append([u, t, n, h, "head missing"])
            for n in SCORED:
                if get(n)["benchmark"] == get("T24")["benchmark"]:
                    bad["differ_t24"].append([u, t, n])
            for n in ("S4", "F6"):
                if get("F6S4")["benchmark"] != get(n)["benchmark"]:
                    f6s4[t][n] = True
            for n, (quant, shift) in WANT_DESC.items():
                d = get(n).get("state_lowrank") or {}
                if d.get("quant") != quant or d.get("shift_bits") != shift or d.get("spec") != "2.16.4":
                    bad["descriptor"].append([u, t, n, d])
    res["K3 a row for every (user, model, label)"] = not bad["missing"] and bool(users)
    res["K3 every row finite"] = not bad["finite"]
    res["K3 REF, T24 and Q2 rescored == SM148's committed CPU smoke rows (benchmark LogLoss within 1e-6, same counts)"] = not bad["sm148"]
    res["K3 S4, F6 and F6S4 differ from T24 on every (user, model)"] = not bad["differ_t24"]
    res["K3 F6S4 differs from S4 and from F6 on at least one user per model"] = all(all(v.values()) for v in f6s4.values())
    res["K3 codec descriptors (spec 2.16.4; q8 / q6 factors; shift bits 264 int8 / 260 int4, float16 scale)"] = not bad["descriptor"]
    info.update({k: v[:6] for k, v in bad.items()})
    info["kept_labels_vs_sm148_max_abs_logloss"] = kept_max
    # report only: CPU scorer throughput per label (review evaluations per second, model time only)
    speed = {}
    for t in tags:
        for n in LABELS:
            rr = [rows[(u, t, LABELS[n])] for u in users if (u, t, LABELS[n]) in rows]
            sec = sum(r.get("model_seconds", 0) for r in rr)
            rev = sum(r.get("reviews", 0) for r in rr)
            speed[f"{t} {n}"] = round(rev / sec) if sec else None
    info["cpu_review_evals_per_second"] = speed
    return res, info


def k5(rows):
    import numpy as np
    sys.path.insert(0, str(HERE.parent / "sm145"))
    import sm145_dr as dr
    res, info = {}, {"files": 0}
    users = sorted({u for u, _, _ in rows})
    tags = sorted({t for _, t, _ in rows})
    data = dr.data_path()
    missing, count_bad, keys_bad = [], [], []
    pos_q4 = False
    for u in users:
        prior, _cid, _ncard = dr.card_info(data, u)
        for t in tags:
            ref = None
            for name in DUMPED:
                v = LABELS[name]
                f = SMOKE_DUMP / dr.dump_name(u, t, v)
                r = rows.get((u, t, v))
                if not f.exists() or r is None:
                    missing.append([u, t, name])
                    continue
                info["files"] += 1
                z = np.load(f)
                pb = r["prior_buckets"]
                for h in ("ahead", "instant"):
                    th = z[f"{h}_th"]
                    want = pb["all"].get(h, {"n": [0] * len(pb["edges"])})["n"]
                    got = (np.bincount(np.searchsorted(np.asarray(pb["edges"][1:]), prior[th - 1], side="right"),
                                       minlength=len(pb["edges"])).tolist() if len(th) else [0] * len(pb["edges"]))
                    if got != want:
                        count_bad.append([u, t, name, h])
                if name == "REF":
                    ref = z
                elif ref is not None:
                    for h in ("ahead", "instant"):
                        if not np.array_equal(z[f"{h}_th"], ref[f"{h}_th"]):
                            keys_bad.append([u, t, name, h])
                            continue
                        d = np.abs(z[f"{h}_p"].astype(np.float64) - ref[f"{h}_p"].astype(np.float64))
                        if name == "S4" and len(d) and d.max() > 0:
                            pos_q4 = True
    res["K5 a dump for every (user, model, dumped label)"] = not missing
    res["K5 dumped reviews per bucket == the rows' all-scope counts"] = info["files"] > 0 and not count_bad
    res["K5 same reviews in every label's dump"] = not keys_bad
    res["K5 |dR| S4 vs REF > 0 somewhere"] = pos_q4
    out = HERE / "cpu" / "smoke-dr.json"
    a = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "sm152_dr.py"), "aggregate", "--dump", str(SMOKE_DUMP),
                        "--users", str(HERE / "users-smoke.json"), "--out", str(out), "--min-users", "3"],
                       capture_output=True, text=True)
    ok = a.returncode == 0 and out.exists()
    if ok:
        d = json.loads(out.read_text())
        ok = d.get("finite") is True and d.get("complete") is True
    res["K5 sm152_dr.py aggregates the smoke dumps (finite, complete)"] = ok
    info.update({"missing": missing[:10], "count_bad": count_bad[:10], "keys_bad": keys_bad[:10],
                 "dr_stdout": a.stdout[-400:], "dr_stderr": a.stderr[-400:]})
    return res, info


def per_write_bench():
    """Report only: CPU (1 thread) cost of one state write per head, codec part only (the step itself excluded):
    REF's int8 rows vs the rank-r int8 factor codec, batched (N entities x 4 heads per call, so the per-call overhead of
    torch is spread) and for a single entity (one write, as the fork pays it; dominated by torch's per-op overhead)."""
    from scripts import state_lowrank as sl
    from scripts.state_vq import int_row_quant
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    q8 = int_row_quant(264)
    out = {}
    for K in (16, 32):
        for N in (256, 1):
            S = torch.randn(N, 4, K, K, generator=g)
            L = torch.randn(N, 4, K, 4, generator=g)
            v = torch.randn(N, 4, K, generator=g)
            n = torch.full((N,), 20, dtype=torch.int64)

            def tm(f, it):
                f()
                t0 = time.perf_counter()
                for _ in range(it):
                    f()
                return (time.perf_counter() - t0) / it / (N * 4) * 1e6   # microseconds per head write
            it = 50 if N > 1 else 400
            row = {"REF int8 rows": tm(lambda: q8(S), it)}
            for name, spec, qf in (("rank 4 int8 factors", "4", "q8"), ("rank 4 int6 factors", "4", "q6"),
                                   ("rank 2 int8 factors", "2", "q8"), ("rank 2 int6 factors", "2", "q6")):
                c = sl.LowRankCodec(spec, qf)
                Lp = L[..., :c.rmax].contiguous()
                row[name] = tm(lambda: c.encode(S, Lp, v, n, None), it)
            out[f"K{K} N{N}"] = {k: round(x, 2) for k, x in row.items()}
    torch.set_num_threads(4)
    return out


def cpu(out):
    t0 = time.time()
    b = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "sm152_bytes.py")], capture_output=True, text=True)
    res = {"K1 bytes and bit-packing round trip (sm152_bytes.py)": b.returncode == 0}
    r2, i2 = k2()
    res.update(r2)
    rows = read(CPU_ROWS)
    r3, info = k3(rows)
    res.update(r3)
    r5, i5 = k5(rows)
    res.update(r5)
    info["k2"], info["k5"] = i2, i5
    info["k1_stdout"] = b.stdout[-600:] + b.stderr[-600:]
    info["per_write_microseconds_cpu_1_thread (report only)"] = per_write_bench()
    ok = all(res.values())
    Path(out).write_text(json.dumps({"ok": ok, "checks": res, "info": info, "seconds": round(time.time() - t0, 1)},
                                    indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"[sm152-check] CPU proofs {'ALL OK' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


def gpu_selftest(out):
    """The CUDA pieces: the Jacobi eigen kernel vs torch.linalg.eigh; selftest_sm148 on CUDA (fused encode kernel:
    truncation, young exactness); the fused kernel vs the torch path on the device and vs the CPU loop (float factors
    within 1e-5 relative; int8 factors report only, their rounding flips are discrete)."""
    from scripts import state_lowrank as sl
    from scripts import state_lowrank_cuda as slc
    from expertium.model.rwkv_ops import single_timestep
    res, rep = {}, {}
    try:
        rep["jacobi"] = slc.selftest()
        res["G1 Jacobi eigen kernel == torch.linalg.eigh (eigenvalues, reconstruction, orthonormality)"] = True
    except AssertionError as e:
        res["G1 Jacobi eigen kernel == torch.linalg.eigh (eigenvalues, reconstruction, orthonormality)"] = False
        rep["jacobi"] = str(e)
    try:
        rep["selftest_sm148_cuda"] = sl.selftest_sm148("cuda")
        res["G2 selftest_sm148 on CUDA (fused encode: truncation, young states exact, int8 / int6 round trip)"] = True
    except AssertionError as e:
        res["G2 selftest_sm148 on CUDA (fused encode: truncation, young states exact, int8 / int6 round trip)"] = False
        rep["selftest_sm148_cuda"] = str(e)
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for K in (16, 32):
        for spec, q in (("2.16.4", "f"), ("2.16.4", "q8"), ("2.16.4", "q6"), ("2", "q6")):
            B, T, H = 256, 80, 4
            r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
            kd = torch.nn.functional.normalize(kd, dim=-1)
            a = torch.sigmoid(a)
            w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
            skip = torch.rand(B, T, generator=g) < 0.3
            args = (r, k * a, v, w, a, kd, skip)
            c = sl.LowRankCodec(spec, q)
            o_cpu = sl.lowrank_loop(c, *args, single_timestep, None)
            ga = tuple(x.cuda() for x in args)
            sl.FUSED_CUDA = True
            o_fused = sl.lowrank_loop(c, *ga, single_timestep, None).cpu()
            sl.FUSED_CUDA = False
            o_torch = sl.lowrank_loop(c, *ga, single_timestep, None).cpu()
            sl.FUSED_CUDA = True
            rel = lambda x, y: float((x - y).norm() / y.norm())   # noqa: E731
            rep[f"K{K} {spec} {q}"] = {"fused_vs_cpu": rel(o_fused, o_cpu), "fused_vs_torch_cuda": rel(o_fused, o_torch),
                                       "finite": bool(torch.isfinite(o_fused).all())}
            res[f"G3 fused kernel finite K{K} {spec} {q}"] = rep[f"K{K} {spec} {q}"]["finite"]
            if q == "f":
                worst = max(worst, rep[f"K{K} {spec} {q}"]["fused_vs_cpu"])
    res["G3 fused kernel == CPU loop, float factors (relative <= 1e-5)"] = worst <= 1e-5
    # SM152: on the same inputs, the fused kernel's int6 output differs from its int8 output and is closer to the CPU
    # loop's int6 output than to the CPU loop's int8 output (the qmax argument reaches the kernel)
    g4 = {}
    for K in (16, 32):
        B, T, H = 256, 80, 4
        r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
        kd = torch.nn.functional.normalize(kd, dim=-1)
        a = torch.sigmoid(a)
        w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
        skip = torch.rand(B, T, generator=g) < 0.3
        args = (r, k * a, v, w, a, kd, skip)
        ga = tuple(x.cuda() for x in args)
        o = {}
        for q in ("q8", "q6"):
            c = sl.LowRankCodec("2.16.4", q)
            o[("cpu", q)] = sl.lowrank_loop(c, *args, single_timestep, None)
            o[("cuda", q)] = sl.lowrank_loop(c, *ga, single_timestep, None).cpu()
        rel = lambda x, y: float((x - y).norm() / y.norm())   # noqa: E731
        g4[f"K{K}"] = {"fused_q6_vs_cpu_q6": rel(o[("cuda", "q6")], o[("cpu", "q6")]),
                       "fused_q6_vs_cpu_q8": rel(o[("cuda", "q6")], o[("cpu", "q8")]),
                       "fused_q8_vs_cpu_q8": rel(o[("cuda", "q8")], o[("cpu", "q8")]),
                       "fused_q6_differs_from_q8": not torch.equal(o[("cuda", "q6")], o[("cuda", "q8")])}
    rep["G4"] = g4
    res["G4 fused int6 differs from fused int8 and is closer to the CPU int6 loop than to the CPU int8 loop"] = all(
        x["fused_q6_differs_from_q8"] and x["fused_q6_vs_cpu_q6"] < x["fused_q6_vs_cpu_q8"] for x in g4.values())
    ok = all(res.values())
    Path(out).write_text(json.dumps({"ok": ok, "checks": res, "report_only": rep}, indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"[sm152-check] GPU self-test {'OK' if ok else 'FAILED'}; float factors fused vs CPU max {worst:.1e}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "gpu-selftest"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "cpu":
        return cpu(a.out or str(HERE / "check-cpu-sm152.json"))
    return gpu_selftest(a.out or str(HERE / "gpu" / "gpu-selftest.json"))


if __name__ == "__main__":
    raise SystemExit(main())
