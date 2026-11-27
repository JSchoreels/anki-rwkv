#!/usr/bin/env python
"""Session inputs (SM179): CPU proofs before the screen (design.md "Proofs"). About 10 minutes, 4 threads, < 3 GB.

  python sm179_check.py cpu [--out check-cpu-sm179.json]

C1 flag off = stock, bitwise: 3 real updates on each of users 4745 and 2308 (six-layer Tiny two-head (SM130), release
   weights and Muon state, release options, 256-review windows with carried windows) in the stock tree (SM171's run
   tree, cea0f931) and in the SM179 run tree with SM179_INPUTS unset: every update's loss, model input and gradients
   equal bit for bit.
C2 flag on, step 0 = release: the same updates with SM179_INPUTS=session / state from the zero-padded starts of the
   arms: the first update's loss equals the stock loss (to 1e-6 relative: only the matmul's summation length changes),
   the extra columns are non-zero, their gradients are non-zero, everything finite.
C3 the session values: (a) equal a plain per-review loop (review k reads only reviews j < k); (b) causal: recomputed
   on every one of 40 prefixes of the history they equal the full-history values on that prefix; (c) changing review
   k's rating or duration leaves review k's values unchanged; (d) through the scorer's sample builder (show-time
   inputs): the query row of a review carries exactly its update row's extra columns; flag off the samples equal the
   stock tree's bit for bit.
C4 the scorer: the release model (flag off) and the padded session start (flag on) give the same benchmark LogLoss
   (1e-6) on 2 small users of the 732 set (fp32 and REF w8+s8hx8, show-time inputs, deleted cards neighbour).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
STOCK = W / "sm171-run"          # cea0f931, never edited
RUN = W / "sm179-run"            # branch sm179-run
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
RELEASE = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a"
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
ARMS = {"session": RUN / "artifacts/small-model/sm179-tiny-ses/start", "state": RUN / "artifacts/small-model/sm179-tiny-st/start"}
U732 = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
G = HERE / "gpu" / "cpu-check"


def env(mode):
    e = {k: v for k, v in os.environ.items() if k not in ("SM179_INPUTS", "RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START")}
    e.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", RWKV_DELETED_CARDS="neighbour")
    if mode:
        e["SM179_INPUTS"] = mode
    return e


def updates(tree, mode, master, optimizer, name):
    out = G / f"updates-{name}.json"
    cmd = [PY, "-X", "utf8", str(HERE / "sm179_updates.py"), "--tree", str(tree), "--master", str(master),
           "--optimizer", str(optimizer), "--out", str(out)]
    r = subprocess.run(cmd, env=env(mode), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (G / f"updates-{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{name}: exit {r.returncode}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())


def c1_c2():
    stock = updates(STOCK, "", RELEASE / "master.pth", RELEASE / "optimizer.pth", "stock")
    off = updates(RUN, "", RELEASE / "master.pth", RELEASE / "optimizer.pth", "off")
    keys = ("loss_hex", "start_sha256", "grad_sha256", "start_width")
    s, o = stock["steps"], off["steps"]
    c1 = len(s) == len(o) >= 6 and all(a[k] == b[k] for a, b in zip(s, o) for k in keys)
    c1_info = {"updates": len(s), "equal_fields": list(keys), "ok": c1,
               "first": {k: s[0][k] for k in ("loss", "start_width", "grad_sha256")}}
    c2_info, c2 = {}, True
    for mode, start in ARMS.items():
        on = updates(RUN, mode, start / "master.pth", start / "optimizer.pth", mode)["steps"]
        rel = abs(on[0]["loss"] / s[0]["loss"] - 1)
        ok = (len(on) == len(s) and rel <= 1e-6 and on[0]["start_width"] == 92 + {"session": 6, "state": 4}[mode]
              and all(st["extra_columns_abs_max"] > 0 for st in on) and all(st["extra_grad_abs_sum"] > 0 for st in on)
              and all(st["weights_finite"] for st in on)
              and all(abs(st["loss"]) < 10 for st in on))
        c2 = c2 and ok
        c2_info[mode] = {"ok": ok, "first_loss_rel_diff_vs_stock": rel, "width": on[0]["start_width"],
                         "losses": [round(st["loss"], 6) for st in on], "stock_losses": [round(st["loss"], 6) for st in s],
                         "extra_grad_abs_sum": [st["extra_grad_abs_sum"] for st in on]}
    return c1, c1_info, c2, c2_info


C3 = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from rwkv import extra_inputs as ex
out = {}
users = [int(u) for u in sys.argv[2].split(",")]
for u in users:
    show, dur = ex._time_frame(sys.argv[3], u)
    import pandas as pd
    rating = pd.read_parquet(f"{sys.argv[3]}/revlogs/user_id={u}", columns=["rating"]).rating.to_numpy()
    full = ex.session_values(show, dur, rating)
    n = len(show)
    lim = min(n, 3000)
    # (a) plain loop
    ans = show + dur
    ref = {k: [] for k in ex.SESSION}
    start = 0
    for k in range(lim):
        gap = 0.0 if k == 0 else max(show[k] - ans[k - 1], 0) / 1000
        if k == 0 or gap >= ex.SESSION_GAP_SECONDS:
            start = k
        n10 = sum(1 for j in range(k) if ans[j] >= show[k] - 600000)
        n60 = sum(1 for j in range(k) if ans[j] >= show[k] - 3600000)
        prev = [rating[j] == 1 for j in range(max(start, k - ex.AGAIN_WINDOW), k)]
        vals = {"sm179_gap": np.log1p(gap), "sm179_n10": np.log1p(n10), "sm179_n60": np.log1p(n60),
                "sm179_sess_min": np.log1p(max(show[k] - show[start], 0) / 60000), "sm179_sess_pos": np.log1p(k - start),
                "sm179_sess_again": (sum(prev) / len(prev)) if prev else 0.0}
        for name in ex.SESSION:
            ref[name].append(vals[name])
    loop = max(float(np.max(np.abs(np.array(ref[k]) - full[k][:lim]))) for k in ex.SESSION)
    # (b) prefixes
    rng = np.random.default_rng(u)
    cuts = sorted(set(rng.integers(1, n + 1, size=40).tolist()))
    prefix = max(float(np.max(np.abs(ex.session_values(show[:c], dur[:c], rating[:c])[k] - full[k][:c])))
                 for c in cuts for k in ex.SESSION)
    # (c) this review's own answer
    own = 0.0
    for k in rng.integers(1, n, size=40).tolist():
        r2, d2 = rating.copy(), dur.copy()
        r2[k] = 4 if r2[k] == 1 else 1
        d2[k] = d2[k] + 37000
        alt = ex.session_values(show, d2, r2)
        own = max(own, max(abs(float(alt[name][k] - full[name][k])) for name in ex.SESSION))
    out[u] = {"reviews": n, "loop_reviews": lim, "loop_max_abs_diff": loop, "prefixes": len(cuts),
              "prefix_max_abs_diff": prefix, "own_answer_max_abs_diff": own,
              "ok": loop <= 1e-12 and prefix == 0.0 and own == 0.0}
print(json.dumps(out))
'''

SAMPLES = r'''
import hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
import numpy as np, torch
from scripts import sm31_inputs
out = {}
for u in [int(x) for x in sys.argv[2].split(",")]:
    from pathlib import Path
    s = sm31_inputs.samples(Path(sys.argv[3]), u, "e2s-query", "neighbour")[0]
    x = s.card_features.numpy()
    info = {"width": int(x.shape[1]), "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16],
            "rows": int(x.shape[0])}
    try:
        from rwkv import extra_inputs as ex
        n = ex.DIMS
    except ImportError:
        n = 0
    if n:
        th, q = s.review_ths.numpy(), s.skips.numpy()
        upd = {int(t): i for i, t in enumerate(th) if not q[i]}
        diff = max(float(np.abs(x[i, 24:24 + n] - x[upd[int(th[i])], 24:24 + n]).max()) for i in np.flatnonzero(q))
        base = np.concatenate([x[:, :24], x[:, 24 + n:]], 1)
        info.update(query_vs_update_max_abs_diff=diff, base_sha256=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()[:16],
                    extra_abs_max=float(np.abs(x[:, 24:24 + n]).max()))
    out[u] = info
print(json.dumps(out))
'''


def run_py(code, args, mode, cwd):
    e = env(mode)
    e.pop("RWKV_DELETED_CARDS")   # the scorer's samples apply the deleted-card transform themselves
    r = subprocess.run([PY, "-X", "utf8", "-c", code, *args], env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def c3():
    users = "4745,2308,1269"
    vals = run_py(C3, [str(RUN), users, DATA], "session", str(RUN))
    stock = run_py(SAMPLES, [str(STOCK), "4745,2308", DATA], "", str(STOCK))
    off = run_py(SAMPLES, [str(RUN), "4745,2308", DATA], "", str(RUN))
    ses = run_py(SAMPLES, [str(RUN), "4745,2308", DATA], "session", str(RUN))
    st = run_py(SAMPLES, [str(RUN), "4745,2308", DATA], "state", str(RUN))
    ok = (all(v["ok"] for v in vals.values()) and stock == off
          and all(ses[u]["query_vs_update_max_abs_diff"] == 0 and st[u]["query_vs_update_max_abs_diff"] == 0
                  and ses[u]["base_sha256"] == stock[u]["sha256"] and st[u]["base_sha256"] == stock[u]["sha256"]
                  and ses[u]["width"] == 30 and st[u]["width"] == 28 for u in ses))
    return ok, {"values": vals, "samples_stock": stock, "samples_off": off, "samples_session": ses, "samples_state": st}


def c4():
    users = json.loads(U732.read_text())["users"]
    import pandas as pd
    sizes = sorted((len(pd.read_parquet(f"{DATA}/revlogs/user_id={u}", columns=["rating"])), u) for u in users[:200])
    pick = [u for n, u in sizes if n >= 400][:2]
    uf = G / "c4-users.json"
    uf.write_text(json.dumps({"users": pick}))
    rows = {}
    for name, mode, master in (("off", "", RELEASE / "master.pth"), ("ses", "session", ARMS["session"] / "master.pth")):
        out = G / f"c4-{name}.jsonl"
        if out.exists():
            out.unlink()
        e = env(mode)
        e.pop("RWKV_DELETED_CARDS")
        e["PYTORCH_JIT"] = "0"
        cmd = [PY, "-X", "utf8", str(RUN / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(uf),
               "--model", f"m=expertium:w64-c1-d2p1u1:{master}", "--only", "m/fp32", "--only", "m/w8+s8hx8",
               "--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour", "--device", "cpu",
               "--workers", "1", "--time-limit", "1200", "--out", str(out)]
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(RUN))
        (G / f"c4-{name}.log").write_text(r.stdout[-5000:] + r.stderr[-5000:], encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"c4 {name}: {r.stderr[-1500:]}")
        rows[name] = {(x["user_id"], x["label"]): x["benchmark"] for x in map(json.loads, open(out, encoding="utf-8"))
                      if "benchmark" in x}
    keys = sorted(rows["off"])
    diffs = [abs(rows["off"][k][h]["LogLoss"] - rows["ses"][k][h]["LogLoss"]) for k in keys for h in ("ahead", "instant")]
    ok = len(keys) == 4 and set(keys) == set(rows["ses"]) and max(diffs) <= 1e-6
    return ok, {"users": pick, "labels": len(keys), "max_abs_logloss_diff": max(diffs) if diffs else None}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm179.json"))
    args = p.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    head = lambda tree: subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = subprocess.run(["git", "-C", str(RUN), "status", "--porcelain", "rwkv", "scripts", "expertium"],
                           capture_output=True, text=True).stdout.strip()
    report = {"stock_tree": {"path": str(STOCK), "head": head(STOCK)}, "run_tree": {"path": str(RUN), "head": head(RUN),
              "dirty": bool(dirty)}, "checks": {}}
    ok1, i1, ok2, i2 = c1_c2()
    report["checks"]["C1_flag_off_equals_stock_bitwise"] = i1
    report["checks"]["C2_flag_on_step0_equals_release"] = i2
    ok3, i3 = c3()
    report["checks"]["C3_session_values_and_samples"] = dict(i3, ok=ok3)
    ok4, i4 = c4()
    report["checks"]["C4_scorer_padded_equals_release"] = dict(i4, ok=ok4)
    report["all_ok"] = bool(ok1 and ok2 and ok3 and ok4 and not dirty)
    report["seconds"] = round(time.time() - t0)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"C1": ok1, "C2": ok2, "C3": ok3, "C4": ok4, "dirty": bool(dirty), "all_ok": report["all_ok"],
                      "seconds": report["seconds"]}))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
