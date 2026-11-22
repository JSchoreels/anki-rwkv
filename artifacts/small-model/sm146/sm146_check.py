#!/usr/bin/env python
"""Aggressive state tiers for Tiny (SM146): CPU proofs of the tier code and the per-review dump (before any GPU
ticket) and the GPU kernel self-test (inside the first GPU ticket).

  python sm146_check.py cpu [--out check-cpu-sm146.json]   K1 bytes, K2 synthetic quantizer proofs, K3 / K4 scorer rows,
                                                           K5 the per-review dump (needs cpu/smoke-cpu-rows.jsonl,
                                                           cpu/emulate-rows.jsonl and the smoke dumps of cpu-proof.sh)
  python sm146_check.py gpu-selftest [--out gpu/gpu-selftest.json]   the tier kernel vs the stock kernel and the CPU
                                                           reference on random inputs (CUDA)

K2 (synthetic, seeded; the tier code is unchanged since SM141): with every tier at one width the tier code equals the stock
code bit for bit; with real tiers (int3 / int4 / int8, and the ternary int2 specs of this experiment, 3-part and 5-part)
each row gets the width of its own update count (manual per-step loop), on the state quantizer, the recurrence and the fork's
patched reference.
K3 (scorer rows, 3 smoke users x the 2 Tiny models, CPU): U8 == w8+s8hx8 and P4 (the 5-part spec 4.5.4.16.8) == AN16 in every
metric field; each aggressive label differs from AN16 and from REF on every user, and the pairs (I3a, I3b), (I2a, I2b), (I2a, I3a),
(I2b, I3b) differ; every row finite; the GPU code path's CPU twin (--emulate-gpu-path) gives U8 == REF.
K4: REF and AN16 CPU rows == the committed CPU smoke rows of SM145, every field (same users, same --prior-buckets edges)
except timing and provenance.
K5 (the --dump-scope all dump): a dump per (user, model, label); per head the dumped reviews per answer-count bucket == the
row's prior_buckets "all" counts; review sets equal across labels; |dR| of U8 vs REF == 0 exactly and of AN16 vs REF > 0;
sm146_dr.py aggregates the smoke dumps (finite, complete).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

LABELS = {"REF": "w8+s8hx8", "AN16": "w8+t4.16.8c4.16.8nhx8", "I3a": "w8+t3.5.4.16.8c3.5.4.16.8nhx8",
          "I3b": "w8+t3.9.4.16.8c3.9.4.16.8nhx8", "I2a": "w8+t2.5.4.16.8c2.5.4.16.8nhx8",
          "I2b": "w8+t2.5.3.16.8c2.5.3.16.8nhx8"}
NEW = ("I3a", "I3b", "I2a", "I2b")
PAIRS = (("I3a", "I3b"), ("I2a", "I2b"), ("I2a", "I3a"), ("I2b", "I3b"))   # each pair of labels differs on every user
PROOF = {"U8": "w8+t8.5.8.128.8c8.5.8.128.8nhx8", "P4": "w8+t4.5.4.16.8c4.5.4.16.8nhx8"}
E2S = "@e2s-query"
SKIP_FIELDS = {"label", "variant", "state_bits", "state_tiers", "state_entity_bits", "model_seconds", "prepare_seconds",
               "provenance_commit", "provenance_dirty", "scorer", "state_vq_path"}   # descriptors, not metrics
CPU_ROWS, EMU_ROWS = HERE / "cpu" / "smoke-cpu-rows.jsonl", HERE / "cpu" / "emulate-rows.jsonl"
SM144_ROWS = HERE.parent / "sm145" / "cpu" / "smoke-cpu-rows.jsonl"
SMOKE_DUMP = HERE.parents[2] / "work" / "sm146-dump-smoke"   # cpu-proof.sh --dump (never committed: work/)


SPECS_A = [(3, 4, 4, 128, 8), (3, 8, 4, 128, 8), (2, 4, 4, 128, 8), (2, 4, 3, 128, 8), (3, 0, 3, 16, 8)]   # (young, M, mid, N, old)
SPECS_C = [(3, 4, 4, 40, 8), (3, 8, 4, 40, 8), (2, 4, 4, 40, 8), (2, 4, 3, 40, 8), (3, 0, 3, 40, 8)]


def states(gen, n=64, K=16, H=4):
    """Random [n, H, K, K] states with zero, tiny, huge and normal rows (f16 clamp and zero-scale rule hit)."""
    x = torch.randn(n, H, K, K, generator=gen)
    x[0] = 0
    x[1, 0] = 0
    x[2] *= 1e-9
    x[3] *= 1e6
    x[4, 1] *= 1e-30
    return x


def rows_equal(a, b):
    keys = (set(a) | set(b)) - SKIP_FIELDS
    return [k for k in sorted(keys) if a.get(k) != b.get(k)]


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


def k2():
    from scripts import compress_quant as cq
    from scripts.small_model_wkv import F16_SCALE, TIER, TIER_NEVER, QMAX, reference, round_scale_f16, tier_bits
    from scripts.state_vq import TierQuant, int_row_quant
    g = torch.Generator().manual_seed(0)
    res = {}
    # a. state_vq TierQuant
    for K in (16, 32):
        x = states(g, K=K)
        n = torch.randint(1, 300, (x.shape[0],), generator=g)
        for w in (8, 4):
            got = TierQuant(tier_bits(w, 5, w, 128, w))(x, None, None, 0, n)
            res[f"TierQuant uniform int{w} == int_row_quant K{K}"] = bool(torch.equal(got, int_row_quant(w | F16_SCALE)(x)))
        for (y_, M_, m_, N_, o_) in SPECS_A:
            spec = tier_bits(y_, M_, m_, N_, o_)
            got = TierQuant(spec)(x, None, None, 0, n)
            ok = True
            for i in range(x.shape[0]):
                w = y_ if n[i] <= M_ else o_ if n[i] >= N_ else m_
                q = QMAX[w]
                scale = round_scale_f16(x[i].abs().amax(-1, keepdim=True) / q)
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                ok &= torch.equal(got[i], torch.clamp(torch.round(x[i] / scale), -q, q) * scale)
                if w in (8, 4):
                    ok &= torch.equal(got[i], int_row_quant(w | F16_SCALE)(x[i:i + 1])[0])
            res[f"TierQuant {y_}/{m_}/{o_} (M {M_}, N {N_}) picks the width of each row's count K{K}"] = bool(ok)
    # b. the patched fork module (TorchScript helpers, called eagerly here)
    mod = cq.install_state_quant_model()
    x = states(g, K=32)
    n = torch.randint(1, 300, (x.shape[0],), generator=g)
    for w in (8, 4):
        res[f"fork _quantize_tier uniform int{w} == _quantize_rows"] = bool(torch.equal(
            mod._quantize_tier(x, tier_bits(w, 5, w, 128, w), n), mod._quantize_rows(x, w | F16_SCALE)))
    res["fork _quantize_tier == state_vq TierQuant (3/4/8)"] = bool(torch.equal(
        mod._quantize_tier(x, tier_bits(3, 4, 4, 128, 8), n), TierQuant(tier_bits(3, 4, 4, 128, 8))(x, None, None, 0, n)))
    # c. the recurrence (small_model_wkv.reference = the GPU kernel's CPU twin) and the fork's patched reference
    for K in (16, 32):
        B, T, H = 3, 300, 4
        r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
        kd = torch.nn.functional.normalize(kd, dim=-1)
        a = torch.sigmoid(a)
        w_ = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
        skip = torch.rand(B, T, generator=g) < 0.3
        skip[:, 0] = False
        args_ = (r, k * a, v, w_, a, kd, skip)
        for w in (8, 4):
            res[f"reference uniform int{w} TIER == plain K{K}"] = bool(torch.equal(
                reference(*args_, tier_bits(w, 5, w, 128, w)), reference(*args_, w | F16_SCALE)))
        from rwkv.model.rwkv_ops import single_timestep
        for ci, (y_, M_, m_, N_, o_) in enumerate(SPECS_C):
            spec = tier_bits(y_, M_, m_, N_, o_)
            got = reference(*args_, spec)
            # manual loop: per row the width of its update count
            state = torch.zeros(B, H, K, K)
            cnt = torch.zeros(B, dtype=torch.long)
            man = torch.empty(B, T, H, K)
            for t in range(T):
                o, nxt = single_timestep(*(z[:, t] for z in args_[:6]), state)
                man[:, t] = o
                cnt += (~skip[:, t]).long()
                q = torch.tensor([QMAX[y_ if c <= M_ else o_ if c >= N_ else m_] for c in cnt.tolist()]).view(B, 1, 1, 1)
                scale = round_scale_f16(nxt.abs().amax(-1, keepdim=True) / q)
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                nxt = torch.maximum(torch.minimum(torch.round(nxt / scale), q), -q) * scale
                state = torch.where(skip[:, t, None, None, None], state, nxt)
            res[f"reference {y_}/{m_}/{o_} (M {M_}, N {N_}) == manual per-count loop K{K}"] = bool(torch.equal(got, man))
            if K == 32 and ci == 0:
                fork = mod.reference_rwkv7_q(*args_, spec)
                res["fork reference_rwkv7_q TIER == small_model_wkv.reference (K32)"] = bool(torch.equal(fork, got))
                res["fork reference_rwkv7_q uniform int8 TIER == plain"] = bool(torch.equal(
                    mod.reference_rwkv7_q(*args_, tier_bits(8, 5, 8, 128, 8)), mod.reference_rwkv7_q(*args_, 8 | F16_SCALE)))
    assert TIER == 1024 and TIER_NEVER == 65535
    return res


def k3k4():
    res, info = {}, {}
    rows, emu = read(CPU_ROWS), read(EMU_ROWS)
    users = sorted({u for u, _, _ in rows})
    tags = sorted({t for _, t, _ in rows})
    info["users"], info["tags"], info["rows"], info["emulate_rows"] = users, tags, len(rows), len(emu)
    nonfin = [k for k, r in rows.items() if "nonfinite" in r]
    res["every CPU smoke row finite"] = not nonfin
    want = {(u, t, v) for u in users for t in tags for v in [*LABELS.values(), *PROOF.values()]}
    res["every (user, model, label) CPU smoke row present"] = want <= set(rows)
    every = {**LABELS, **PROOF}
    bad = {}
    for u in users:
        for t in tags:
            def get(name):
                return rows.get((u, t, every[name]))
            for x, y in (("U8", "REF"), ("P4", "AN16")):
                if not (get(x) and get(y)) or rows_equal(get(x), get(y)):
                    bad.setdefault(f"{x} == {y}", []).append([u, t, rows_equal(get(x) or {}, get(y) or {})[:4]])
            for lab in NEW:   # the aggressive tiers take effect on every user
                if not (get(lab) and get("AN16")) or not rows_equal(get(lab), get("AN16")):
                    bad.setdefault(f"{lab} differs from AN16 (every user)", []).append([u, t])
            for lab in NEW:
                if not (get(lab) and get("REF")) or not rows_equal(get(lab), get("REF")):
                    bad.setdefault(f"{lab} differs from REF (every user)", []).append([u, t])
            for x, y in PAIRS:
                if not (get(x) and get(y)) or not rows_equal(get(x), get(y)):
                    bad.setdefault(f"{x} differs from {y} (every user)", []).append([u, t])
    for name in ("U8 == REF", "P4 == AN16", *(f"{lab} differs from AN16 (every user)" for lab in NEW),
                 *(f"{lab} differs from REF (every user)" for lab in NEW), *(f"{x} differs from {y} (every user)" for x, y in PAIRS)):
        res[name] = name not in bad
    info["failures"] = bad
    # K4: the int3 young tier on the early switch (SM145)'s committed CPU smoke rows: REF and AN16 are the same labels, users and edges
    old = read(SM144_ROWS)
    n4, d4 = 0, []
    for (u, t, v), r in old.items():
        if v in (LABELS["REF"], LABELS["AN16"]) and (u, t, v) in rows:
            n4 += 1
            d = rows_equal(rows[(u, t, v)], r)
            if d:
                d4.append([u, t, v, d[:4]])
    res["K4 REF and AN16 CPU rows == SM145's committed CPU smoke rows (every field, prior_buckets included)"] = (
        n4 == 2 * len(users) * len(tags) and not d4)
    info["k4_compared"], info["k4_diffs"] = n4, d4
    # GPU code path's CPU twin (--emulate-gpu-path): uniform == REF there too; vs the Hooks path (report only)
    emu_bad, emu_diff = [], {}
    for (u, t, v), r in emu.items():
        if v == PROOF["U8"] and (u, t, LABELS["REF"]) in emu and rows_equal(r, emu[(u, t, LABELS["REF"])]):
            emu_bad.append([u, t])
        if (u, t, v) in rows:
            for h in ("ahead", "instant"):
                if h in r.get("benchmark", {}):
                    d = abs(r["benchmark"][h]["LogLoss"] - rows[(u, t, v)]["benchmark"][h]["LogLoss"])
                    emu_diff[v] = max(emu_diff.get(v, 0.0), d)
    res["emulate-gpu-path U8 == REF"] = bool(emu) and not emu_bad
    info["emulate_vs_cpu_max_abs_benchmark_logloss"] = emu_diff   # report only
    return res, info, rows


def k5(rows):
    """The per-review dump (--dump-scope all) of the smoke users: counts per bucket == the rows' all-scope counts."""
    import numpy as np
    import subprocess
    sys.path.insert(0, str(HERE))
    import sm146_dr as dr
    res, info = {}, {"files": 0}
    every = {**LABELS, **PROOF}
    users = sorted({u for u, _, _ in rows})
    tags = sorted({t for _, t, _ in rows})
    data = dr.data_path()
    missing, count_bad, keys_bad = [], [], []
    zero_u8, pos_an16 = True, False
    for u in users:
        prior, _cid, _ncard = dr.card_info(data, u)
        for t in tags:
            ref = None
            for name in ("REF", *[k for k in every if k != "REF"]):
                v = every[name]
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
                        if name == "U8" and len(d) and d.max() != 0:
                            zero_u8 = False
                        if name == "AN16" and len(d) and d.max() > 0:
                            pos_an16 = True
    res["K5 a dump for every (user, model, label)"] = not missing
    res["K5 dumped reviews per bucket == the rows' all-scope counts"] = info["files"] > 0 and not count_bad
    res["K5 same reviews in every label's dump"] = not keys_bad
    res["K5 |dR| U8 vs REF == 0 exactly"] = zero_u8 and info["files"] > 0
    res["K5 |dR| AN16 vs REF > 0 somewhere"] = pos_an16
    out = HERE / "cpu" / "smoke-dr.json"
    a = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "sm146_dr.py"), "aggregate", "--dump", str(SMOKE_DUMP),
                        "--users", str(HERE / "users-smoke.json"), "--out", str(out), "--min-users", "3"],
                       capture_output=True, text=True)
    ok = a.returncode == 0 and out.exists()
    if ok:
        d = json.loads(out.read_text())
        ok = d.get("finite") is True and d.get("complete") is True
    res["K5 sm146_dr.py aggregates the smoke dumps (finite, complete)"] = ok
    info.update({"missing": missing[:10], "count_bad": count_bad[:10], "keys_bad": keys_bad[:10],
                 "dr_stdout": a.stdout[-400:], "dr_stderr": a.stderr[-400:]})
    return res, info


def cpu(out):
    t0 = time.time()
    import subprocess
    b = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "sm146_bytes.py")], capture_output=True, text=True)
    res = {"K1 bytes (sm146_bytes.py)": b.returncode == 0}
    res.update(k2())
    r3, info, rows = k3k4()
    res.update(r3)
    r5, info5 = k5(rows)
    res.update(r5)
    info["k5"] = info5
    ok = all(res.values())
    Path(out).write_text(json.dumps({"ok": ok, "checks": res, "info": info, "seconds": round(time.time() - t0, 1)},
                                    indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"[sm146-check] CPU proofs {'ALL OK' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


def gpu_selftest(out):
    from scripts.small_model_wkv import F16_SCALE, reference, tier_bits, wkv_long
    g = torch.Generator().manual_seed(1)
    res, rep = {}, {}
    for K in (16, 32):
        B, T, H = 3, 500, 4
        r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
        kd = torch.nn.functional.normalize(kd, dim=-1)
        a = torch.sigmoid(a)
        w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
        skip = torch.rand(B, T, generator=g) < 0.3
        skip[:, 0] = False
        cpu_args = (r, k * a, v, w, a, kd, skip)
        gpu_args = tuple(x.cuda() for x in cpu_args)
        for wd in (8, 4):
            u = wkv_long(*gpu_args, tier_bits(wd, 5, wd, 128, wd)).cpu()
            p = wkv_long(*gpu_args, wd | F16_SCALE).cpu()
            res[f"GPU tier kernel uniform int{wd} == stock kernel K{K} (bitwise)"] = bool(torch.equal(u, p))
        for name, spec in (("3/4/8 M4 N40", tier_bits(3, 4, 4, 40, 8)), ("3/4/8 M8 N40", tier_bits(3, 8, 4, 40, 8)),
                           ("2/4/8 M4 N40", tier_bits(2, 4, 4, 40, 8)), ("2/3/8 M4 N40", tier_bits(2, 4, 3, 40, 8)),
                           ("3/8 N40", tier_bits(3, 0, 3, 40, 8)), ("4/8 N40", tier_bits(4, 0, 4, 40, 8))):
            got = wkv_long(*gpu_args, spec).cpu()
            ref = reference(*cpu_args, spec)
            err = (got - ref).abs()
            res[f"GPU tier kernel {name} finite K{K}"] = bool(torch.isfinite(got).all())
            rep[f"{name} K{K}"] = {"max_abs": float(err.max()), "frac_over_1e-4": float((err > 1e-4).float().mean())}
    ok = all(res.values())
    Path(out).write_text(json.dumps({"ok": ok, "checks": res, "vs_cpu_reference (report only)": rep}, indent=1) + "\n",
                         encoding="utf-8")
    for k, v in res.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"[sm146-check] GPU self-test {'OK' if ok else 'FAILED'}; vs CPU reference: "
          + "; ".join(f"{k} max {v['max_abs']:.1e}" for k, v in rep.items()))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "gpu-selftest"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "cpu":
        return cpu(a.out or HERE / "check-cpu-sm146.json")
    return gpu_selftest(a.out or HERE / "gpu" / "gpu-selftest.json")


if __name__ == "__main__":
    raise SystemExit(main())
