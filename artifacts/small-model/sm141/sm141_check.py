#!/usr/bin/env python
"""Tiered card state by answer count (SM141): CPU proofs of the tier code (before any GPU ticket) and the GPU kernel
self-test (inside the first GPU ticket).

  python sm141_check.py cpu [--out check-cpu-sm141.json]   K1 bytes, K2 synthetic quantizer proofs, K3 / K4 scorer rows
                                                           (needs cpu/smoke-cpu-rows.jsonl + cpu/emulate-rows.jsonl of
                                                           cpu-proof.sh)
  python sm141_check.py gpu-selftest [--out gpu/gpu-selftest.json]   the tier kernel vs the stock kernel and the CPU
                                                           reference on random inputs (CUDA)

K2 (synthetic, seeded): with every tier at one width the tier code equals the stock code bit for bit (state_vq
TierQuant vs int_row_quant, the patched fork _quantize_tier vs _quantize_rows, small_model_wkv.reference with TIER bits
vs the plain flag bits, the fork reference_rwkv7_q with TIER bits vs small_model_wkv.reference); with real tiers each
row gets the width of its update count (manual per-step loop).
K3 (scorer rows, 3 smoke users x 6 models, CPU): uniform tier labels U8 / U4 == w8+s8hx8 / w8+s4c8nhx8 in every metric
field; A == w8+s4c8nhx8 for a user without a card of 128 answers and != for the long-card user; the young tiers change
the rows; the GPU code path's CPU twin (--emulate-gpu-path, expertium models) gives U8 == w8+s8hx8 bit for bit.
K4: this tree's w8+s8hx8 CPU rows == the committed CPU rows of the release-candidate checks (SM139) for the common
users (every common field except timing and provenance; --prior-buckets only adds its field).
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

LABELS = {"REF": "w8+s8hx8", "A": "w8+t4.128.8c8nhx8", "AN": "w8+t4.128.8c4.128.8nhx8",
          "B4": "w8+t3.5.4.128.8c3.5.4.128.8nhx8", "B8": "w8+t3.9.4.128.8c3.9.4.128.8nhx8",
          "D4": "w8+t2.5.4.128.8c2.5.4.128.8nhx8"}
PROOF = {"U8": "w8+t8.5.8.128.8c8.5.8.128.8nhx8", "U4": "w8+t4.5.4.128.4c8.5.8.128.8nhx8", "C4": "w8+s4c8nhx8"}
E2S = "@e2s-query"
SKIP_FIELDS = {"label", "variant", "state_bits", "state_tiers", "state_entity_bits", "model_seconds", "prepare_seconds",
               "provenance_commit", "provenance_dirty", "scorer", "state_vq_path"}   # descriptors, not metrics
CPU_ROWS, EMU_ROWS = HERE / "cpu" / "smoke-cpu-rows.jsonl", HERE / "cpu" / "emulate-rows.jsonl"
SM139_ROWS = HERE.parent / "sm139" / "cpu" / "smoke-cpu-rows.jsonl"
LONG_USER = 1866   # the smoke user with a card of >= 128 answers (SM136's GPU smoke user)


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
        spec = tier_bits(3, 4, 4, 128, 8)
        got = TierQuant(spec)(x, None, None, 0, n)
        ok = True
        for i in range(x.shape[0]):
            w = 3 if n[i] <= 4 else 8 if n[i] >= 128 else 4
            q = QMAX[w]
            scale = round_scale_f16(x[i].abs().amax(-1, keepdim=True) / q)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            ok &= torch.equal(got[i], torch.clamp(torch.round(x[i] / scale), -q, q) * scale)
            if w != 3:
                ok &= torch.equal(got[i], int_row_quant(w | F16_SCALE)(x[i:i + 1])[0])
        res[f"TierQuant 3/4/8 picks the width of each row's count K{K}"] = bool(ok)
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
        spec = tier_bits(3, 4, 4, 40, 8)
        got = reference(*args_, spec)
        # manual loop: per row the width of its update count
        from rwkv.model.rwkv_ops import single_timestep
        state = torch.zeros(B, H, K, K)
        cnt = torch.zeros(B, dtype=torch.long)
        man = torch.empty(B, T, H, K)
        for t in range(T):
            o, nxt = single_timestep(*(z[:, t] for z in args_[:6]), state)
            man[:, t] = o
            cnt += (~skip[:, t]).long()
            q = torch.tensor([QMAX[3 if c <= 4 else 8 if c >= 40 else 4] for c in cnt.tolist()]).view(B, 1, 1, 1)
            scale = round_scale_f16(nxt.abs().amax(-1, keepdim=True) / q)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            nxt = torch.maximum(torch.minimum(torch.round(nxt / scale), q), -q) * scale
            state = torch.where(skip[:, t, None, None, None], state, nxt)
        res[f"reference 3/4/8 (M 4, N 40) == manual per-count loop K{K}"] = bool(torch.equal(got, man))
        if K == 32:
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
    bad = {}
    for u in users:
        for t in tags:
            def get(name):
                return rows.get((u, t, {**LABELS, **PROOF}[name]))
            pairs = [("U8", "REF"), ("U4", "C4")]
            for x, y in pairs:
                if get(x) and get(y):
                    d = rows_equal(get(x), get(y))
                    if d:
                        bad.setdefault(f"{x} == {y}", []).append([u, t, d[:4]])
            if get("A") and get("C4"):
                long = max(get("A")["prior_buckets"]["all"].get("instant", {"max_prior": 0})["max_prior"], 0)
                same = not rows_equal(get("A"), get("C4"))
                if (long >= 128) == same:   # a review reading a state of >= 128 updates must change A vs C4, else equal
                    bad.setdefault("A vs C4 by the 128-answer threshold", []).append([u, t, long, same])
            for y in ("B4", "B8", "D4"):
                if get(y) and get("AN") and not rows_equal(get(y), get("AN")):
                    bad.setdefault(f"{y} differs from AN", []).append([u, t])
    for name in ("U8 == REF", "U4 == C4", "A vs C4 by the 128-answer threshold", "B4 differs from AN",
                 "B8 differs from AN", "D4 differs from AN"):
        res[name] = name not in bad
    info["failures"] = bad
    # GPU code path's CPU twin (--emulate-gpu-path): uniform == REF there too; vs the Hooks path
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
    # K4: vs SM139's committed CPU rows (w8+s8hx8)
    old = read(SM139_ROWS)
    k4 = []
    n4 = 0
    for (u, t, v), r in old.items():
        if v == LABELS["REF"] and (u, t, v) in rows:
            n4 += 1
            mine = {k: x for k, x in rows[(u, t, v)].items() if k != "prior_buckets"}
            d = rows_equal(mine, r)
            if d:
                k4.append([u, t, d[:4]])
    res["K4 w8+s8hx8 rows == SM139 committed CPU rows"] = n4 > 0 and not k4
    info["k4_compared"], info["k4_diffs"] = n4, k4
    return res, info


def cpu(out):
    t0 = time.time()
    import subprocess
    b = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "sm141_bytes.py")], capture_output=True, text=True)
    res = {"K1 bytes (sm141_bytes.py)": b.returncode == 0}
    res.update(k2())
    r3, info = k3k4()
    res.update(r3)
    ok = all(res.values())
    Path(out).write_text(json.dumps({"ok": ok, "checks": res, "info": info, "seconds": round(time.time() - t0, 1)},
                                    indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"[sm141-check] CPU proofs {'ALL OK' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
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
        for name, spec in (("3/4/8 M4 N40", tier_bits(3, 4, 4, 40, 8)), ("2/4/8 M4 N40", tier_bits(2, 4, 4, 40, 8)),
                           ("4/8 N40", tier_bits(4, 0, 4, 40, 8))):
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
    print(f"[sm141-check] GPU self-test {'OK' if ok else 'FAILED'}; vs CPU reference: "
          + "; ".join(f"{k} max {v['max_abs']:.1e}" for k, v in rep.items()))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "gpu-selftest"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "cpu":
        return cpu(a.out or HERE / "check-cpu-sm141.json")
    return gpu_selftest(a.out or HERE / "gpu" / "gpu-selftest.json")


if __name__ == "__main__":
    raise SystemExit(main())
