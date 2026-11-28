#!/usr/bin/env python
"""AUC ceiling (SM182): CPU proofs before the screen (design.md "Proofs"). About 10 minutes, 4 threads, < 4 GB.

  python sm182_check.py cpu [--out check-cpu-sm182.json]     (C1, C2, C3, C4; C4 fixes lambda of arm M)

C1 option off = stock, bitwise: 3 real updates on each of users 4745 and 2308 (six-layer Tiny two-head (SM130), release
   weights and Muon state, release options with 256-review windows, so that windows are carried) in the stock tree
   (SM171's run tree, cea0f931) and in the SM182 run tree without "pair_rank": every update's loss, model input and
   gradients equal bit for bit.
C2 the pair term on synthetic batches: every sampled pair is (passed row, failed row) inside the mask; each passed /
   failed row is used its fair share of pairs (max - min count below 25% of the share: the Kronecker sequence has bounded discrepancy); the sampled mean is within 1% of the exact all-pairs
   mean; adding a constant to every score leaves it unchanged; an empty class gives exactly 0 with zero gradients;
   gradients are <= 0 on passed rows, >= 0 on failed rows, 0 outside the mask; the instant score equals
   logit(1 - P(Again)) of the float32 softmax (1e-3); deterministic (two calls bitwise equal).
C3 the real trainer path with the option on (arms R and M, 3 updates with optimizer steps on each user): everything
   finite; at the first update the model input equals stock, R's loss equals curve + instant pair terms and M's loss
   equals the stock loss + lambda x (curve + instant) (1e-6 relative).
C4 the scale of the two terms at the start weights (no step): the pair term P = curve + instant on the first
   2,048-review window of each of the first 28 users of the screen's training subset (CPU, without the teacher: the
   term reads hard labels only; 8,192-review windows went over 6 GB on the CPU, 2,048 stay far under), review-weighted;
   the stock loss L0 =
   0.7582, SM171's GPU probe on the same 28 users from the same weights; lambda_M = L0 / P0 rounded to two significant
   digits (so that both terms start at the same value).
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
STOCK = W / "sm171-run"          # cea0f931, never edited
RUN = W / "sm182-run"            # branch sm182-run
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
CTL = W / "sm171-run/artifacts/small-model/sm171-tiny-ctl"
N_SCALE, SCALE_CHUNK = 28, 2048
G = HERE / "gpu" / "cpu-check"


def env():
    e = {k: v for k, v in os.environ.items() if k not in ("SM179_INPUTS", "RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START",
                                                          "RWKV_NO_JIT")}
    e.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", RWKV_DELETED_CARDS="neighbour")
    return e


def updates(tree, name, pair_rank=None, users="4745,2308", chunk=256, steps=3, no_step=False, no_kd=False):
    out = G / f"updates-{name}.json"
    cmd = [PY, "-X", "utf8", str(HERE / "sm182_updates.py"), "--tree", str(tree), "--users", users, "--chunk",
           str(chunk), "--steps", str(steps), "--out", str(out)]
    if pair_rank:
        cmd += ["--pair-rank", json.dumps(pair_rank)]
    if no_step:
        cmd += ["--no-step"]
    if no_kd:
        cmd += ["--no-kd"]
    r = subprocess.run(cmd, env=env(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (G / f"updates-{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{name}: exit {r.returncode}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())


def c1():
    s = updates(STOCK, "stock")["steps"]
    o = updates(RUN, "off")["steps"]
    keys = ("loss_hex", "start_sha256", "grad_sha256")
    ok = len(s) == len(o) >= 6 and all(a[k] == b[k] for a, b in zip(s, o) for k in keys)
    return ok, {"updates": len(s), "equal_fields": list(keys), "ok": ok, "stock_losses": [x["loss"] for x in s],
                "first_grad_sha256": s[0]["grad_sha256"]}, s


SYNTH = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from expertium.model import variant_model as vm
out = {}
rng = np.random.default_rng(182)
torch.manual_seed(182)


def exact(s, y, m):
    sp, sf = s[m & (y > 0.5)], s[m & (y <= 0.5)]
    if len(sp) == 0 or len(sf) == 0:
        return 0.0
    return float(np.mean(np.logaddexp(0.0, sf[None, :] - sp[:, None])))


def indices(y, m, pairs):
    passed = np.flatnonzero(m & (y > 0.5)); failed = np.flatnonzero(m & (y <= 0.5))
    i = np.arange(pairs, dtype=np.float64) + 0.5
    a, b = np.mod(i * 0.6180339887498949, 1.0), np.mod(i * 0.41421356237309515, 1.0)
    ip = np.minimum((a * len(passed)).astype(np.int64), max(len(passed) - 1, 0))
    jf = np.minimum((b * len(failed)).astype(np.int64), max(len(failed) - 1, 0))
    return passed[ip], failed[jf], passed, failed

checks = {}
worst_rel, spread, pair_ok, sign_ok, shift_max, det_ok = 0.0, 0.0, True, True, 0.0, True
for case in range(12):
    B, T = int(rng.integers(1, 4)), int(rng.integers(50, 1500))
    s = rng.normal(0, 1.5, (B, T)).astype(np.float32)
    y = (rng.random((B, T)) < rng.uniform(0.6, 0.95)).astype(np.float32)
    s = s + 1.2 * y                        # informative scores
    m = (rng.random((B, T)) < 0.8).astype(np.int32)
    pairs = 65536
    st = torch.tensor(s, requires_grad=True)
    v = vm.pair_logistic(st, torch.tensor(y), torch.tensor(m), pairs)
    v.backward()
    v = v.detach()
    g = st.grad.numpy().reshape(-1)
    sf, yf, mf = s.reshape(-1), y.reshape(-1), m.reshape(-1) > 0
    ex = exact(sf, yf, mf)
    worst_rel = max(worst_rel, abs(float(v) / ex - 1))
    # the reference sampled mean (numpy indices) and the pair validity
    ip, jf, passed, failed = indices(yf, mf, pairs)
    ref = float(np.mean(np.logaddexp(0.0, sf[jf].astype(np.float64) - sf[ip])))
    pair_ok = pair_ok and abs(ref / float(v) - 1) < 1e-5 and bool(np.all(yf[ip] > 0.5) and np.all(yf[jf] <= 0.5)
                                                                      and np.all(mf[ip]) and np.all(mf[jf]))
    cp, cf = np.bincount(ip, minlength=len(sf))[passed], np.bincount(jf, minlength=len(sf))[failed]
    spread = max(spread, (cp.max() - cp.min()) / (pairs / len(passed)), (cf.max() - cf.min()) / (pairs / len(failed)))
    sign_ok = sign_ok and bool(np.all(g[yf > 0.5] <= 0) and np.all(g[(yf <= 0.5)] >= 0) and np.all(g[~mf] == 0))
    v2 = vm.pair_logistic(torch.tensor(s + 5.0), torch.tensor(y), torch.tensor(m), pairs)
    shift_max = max(shift_max, abs(float(v2) - float(v)))
    v3 = vm.pair_logistic(torch.tensor(s), torch.tensor(y), torch.tensor(m), pairs)
    det_ok = det_ok and float(v3).hex() == float(v).hex()
checks["sampled_vs_exact_all_pairs_max_rel"] = worst_rel
checks["pairs_valid_and_equal_numpy_sampler"] = pair_ok
checks["share_spread_over_fair_share_max"] = spread
checks["gradient_signs"] = sign_ok
checks["shift_invariance_max_abs"] = shift_max
checks["deterministic"] = det_ok
# empty classes
s = torch.randn(2, 300, requires_grad=True)
zero = []
for y, m in ((torch.ones(2, 300), torch.ones(2, 300)), (torch.zeros(2, 300), torch.ones(2, 300)),
             (torch.ones(2, 300), torch.zeros(2, 300))):
    if s.grad is not None:
        s.grad = None
    v = vm.pair_logistic(s, y, m, 4096)
    v.backward()
    v = v.detach()
    zero.append(float(v) == 0.0 and bool(torch.all(s.grad == 0)))
checks["empty_class_zero"] = all(zero)
# instant score
l = torch.randn(3, 50, 4) * 3
p = torch.softmax(l, -1)
ref = torch.log((1 - p[..., 0]) / p[..., 0])
checks["instant_score_max_abs"] = float((vm.instant_score(l) - ref).abs().max())
ok = (checks["sampled_vs_exact_all_pairs_max_rel"] < 0.01 and pair_ok and spread < 0.25 and sign_ok
      and checks["shift_invariance_max_abs"] < 1e-5 and det_ok and checks["empty_class_zero"]
      and checks["instant_score_max_abs"] < 1e-3)
checks["ok"] = ok
print(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in checks.items()}))
'''


def c2():
    r = subprocess.run([PY, "-X", "utf8", "-c", SYNTH, str(RUN)], env=env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return info["ok"], info


def c3(stock_steps, lam):
    info, ok = {}, True
    for arm, spec in (("R", {"weight": 1.0, "logloss": 0.0}), ("M", {"weight": lam, "logloss": 1.0})):
        st = updates(RUN, f"arm-{arm}", spec)["steps"]
        s0 = st[0]
        pair = s0["pair_curve"] + s0["pair_instant"]
        want = pair if arm == "R" else stock_steps[0]["loss"] + lam * pair
        rel = abs(s0["loss"] / want - 1)
        a_ok = (len(st) == 6 and rel <= 1e-6 and s0["start_sha256"] == stock_steps[0]["start_sha256"]
                and all(math.isfinite(x["loss"]) and x["grads_finite"] and x["weights_finite"] for x in st))
        ok = ok and a_ok
        info[arm] = {"ok": a_ok, "spec": spec, "first_loss_rel_diff": rel, "losses": [round(x["loss"], 6) for x in st],
                     "pair_curve": [round(x["pair_curve"], 6) for x in st],
                     "pair_instant": [round(x["pair_instant"], 6) for x in st],
                     "grad_norms": [round(x["grad_norm"], 4) for x in st]}
    return ok, info


def c4():
    """P0 on the CPU (no teacher: the pair term reads hard labels only); L0 is the training loss that SM171's GPU probe
    (ticket 001614) measured at the release weights on the first 28 users of this same subset (both arms equal)."""
    probe = json.loads(Path("C:/Users/jonat/workspace/wt/sm171/artifacts/small-model/sm171/gpu/probe-summary.json")
                       .read_text())
    l0 = probe["probe"]["review_weighted_loss"]
    rows = json.loads((CTL / "protocol.json").read_text())["rows"][:N_SCALE]
    users = ",".join(str(r["user_id"]) for r in rows)
    P = updates(RUN, "scale-pair", {"weight": 1.0, "logloss": 0.0}, users, SCALE_CHUNK, 1, True, no_kd=True)["steps"]
    if len(P) != N_SCALE:
        raise RuntimeError(f"scale: {len(P)} windows, expected {N_SCALE}")
    n = sum(x["reviews"] for x in P)
    p0 = sum(x["loss"] * x["reviews"] for x in P) / n
    pc = sum(x["pair_curve"] * x["reviews"] for x in P) / n
    pi = sum(x["pair_instant"] * x["reviews"] for x in P) / n
    lam = float(f"{l0 / p0:.2g}")
    ok = all(math.isfinite(x["loss"]) for x in P) and 0.05 <= lam <= 20
    return ok, {"L0": l0, "L0_source": "SM171 probe ticket 001614 (gpu/probe-summary.json): logged (stock) training loss "
                "over its first 81 updates from the release weights (learning rate warming up), review-weighted, the "
                "first 28 users of the subset (2,134,503 reviews)",
                "users": [x["user"] for x in P], "window": SCALE_CHUNK,
                "reviews": n, "P0": p0, "P0_curve": pc, "P0_instant": pi, "lambda_M": lam,
                "per_user": [{"user": b["user"], "reviews": b["reviews"], "P": round(b["loss"], 5),
                              "curve": round(b["pair_curve"], 5), "instant": round(b["pair_instant"], 5),
                              "grad_norm_P": round(b["grad_norm"], 4)} for b in P], "ok": ok}, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu"])
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm182.json"))
    args = ap.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    heads = {}
    for t in (STOCK, RUN):
        heads[t.name] = {"head": subprocess.run(["git", "-C", str(t), "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip(),
                         "clean": not subprocess.run(["git", "-C", str(t), "status", "--porcelain", "rwkv", "scripts",
                                                      "expertium"], capture_output=True, text=True).stdout.strip()}
    ok1, i1, stock_steps = c1()
    print(f"[sm182-check] C1 {'OK' if ok1 else 'FAIL'}", flush=True)
    ok2, i2 = c2()
    print(f"[sm182-check] C2 {'OK' if ok2 else 'FAIL'}", flush=True)
    ok4, i4, lam = c4()
    print(f"[sm182-check] C4 {'OK' if ok4 else 'FAIL'}: L0 {i4['L0']:.4f}, P0 {i4['P0']:.4f}, lambda_M {lam}", flush=True)
    ok3, i3 = c3(stock_steps, lam)
    print(f"[sm182-check] C3 {'OK' if ok3 else 'FAIL'}", flush=True)
    res = {"trees": heads, "C1_off_equals_stock": i1, "C2_pair_term": i2, "C3_trainer_path": i3, "C4_scale": i4,
           "lambda_M": lam, "all_ok": ok1 and ok2 and ok3 and ok4 and all(h["clean"] for h in heads.values())}
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[sm182-check] ALL {'OK' if res['all_ok'] else 'FAIL'}")
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
