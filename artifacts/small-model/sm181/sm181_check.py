#!/usr/bin/env python
"""Session inputs confirmation (SM181): CPU proofs before the GPU tickets (design.md "Proofs"). 4 threads, < 4 GB.

  python sm181_check.py cpu [--out check-cpu-sm181.json]

K1 the show-time data of every user this run reads with the session flag on: the 794 training users of the Small arm
   and the 765 decisive users (users-decisive.json). For each, rwkv/extra_inputs.py's own alignment (shared unique
   parquet row index, rating and duration equal row by row) succeeds and the 6 session values are finite; the number
   of answer-time inversions is reported (the code is exact with them; SM179 found 0 on its 1,667 users).
K2 Small two-head (SM118), the real trainer path on CPU (release weights and Muon state, release options, 256-review
   windows, users 1088 and 2104 (training users of the arm), 3 updates each): (a) flag off in the run tree (cd220759) = the stock tree of SM149's
   control (sm149-run, cea0f931) bit for bit (loss, model input, every gradient); (b) flag on (session) from the padded
   start of sm181-small-ses: first-update loss equal to the stock loss within 1e-6 relative, extra columns and their
   gradients non-zero, everything finite.
K3 the scorer: (a) Small: the release model (flag off) and the padded Small start (flag on) give the same benchmark
   LogLoss (1e-6) on 2 small users of the 732 set (fp32 and REF w8+s8hx8, show-time inputs, deleted cards neighbour);
   (b) six-layer Tiny two-head (SM130) with --pool 1500 on the 2 smallest decisive users (SM169's smoke users 5625,
   9094): the release model (flag off) and SM179's padded session start (flag on) give the same LogLoss (1e-6).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
STOCK = W / "sm149-run"          # cea0f931, the tree of SM149's control, never edited
RUN = W / "sm179-run"            # branch sm179-run, cd220759
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
SMALL = W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2"
SMALL_CASE = SMALL / "expertium/E89-F65e-F77a"
SMALL_START = RUN / "artifacts/small-model/sm181-small-ses/start"
SMALL_CTL = W / "sm149-run/artifacts/small-model/sm149-sm118-ctl"
TINY_CASE = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a"
TINY_START = RUN / "artifacts/small-model/sm179-tiny-ses/start"
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
DECISIVE = HERE / "users-decisive.json"
G = HERE / "gpu" / "cpu-check"


def env(mode):
    e = {k: v for k, v in os.environ.items() if k not in ("SM179_INPUTS", "RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START",
                                                          "SM11_INPUTS")}
    e.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", RWKV_DELETED_CARDS="neighbour")
    if mode:
        e["SM179_INPUTS"] = mode
    return e


SCAN = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from rwkv import extra_inputs as ex
import pandas as pd
users = json.loads(open(sys.argv[2]).read())
out = {"users": 0, "rows": 0, "inversions": 0, "users_with_inversions": [], "failed": {}, "nonfinite": []}
for u in users:
    try:
        show, dur = ex._time_frame(sys.argv[3], u)
        rating = pd.read_parquet(f"{sys.argv[3]}/revlogs/user_id={u}", columns=["rating"]).rating.to_numpy()
        v = ex.session_values(show, dur, rating)
    except Exception as e:  # noqa: BLE001
        out["failed"][str(u)] = repr(e)[:300]
        continue
    ans = show + dur
    inv = int((np.diff(ans) < 0).sum())
    out["users"] += 1
    out["rows"] += int(len(show))
    out["inversions"] += inv
    if inv:
        out["users_with_inversions"].append(u)
    if not all(np.isfinite(v[k]).all() for k in ex.SESSION):
        out["nonfinite"].append(u)
print(json.dumps(out))
'''


def k1():
    train = [r["user_id"] for r in json.loads((SMALL_CTL / "protocol.json").read_text())["rows"]]
    decisive = json.loads(DECISIVE.read_text())["users"]
    info = {}
    for name, users in (("small_training", train), ("decisive", decisive)):
        uf = G / f"k1-{name}.json"
        uf.write_text(json.dumps(users))
        e = env("session")
        e.pop("RWKV_DELETED_CARDS")
        r = subprocess.run([PY, "-X", "utf8", "-c", SCAN, str(RUN), str(uf), DATA], env=e, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", cwd=str(RUN))
        if r.returncode:
            raise RuntimeError(f"k1 {name}: {r.stderr[-2000:]}")
        res = json.loads(r.stdout.strip().splitlines()[-1])
        res["expected_users"] = len(users)
        res["ok"] = res["users"] == len(users) and not res["failed"] and not res["nonfinite"]
        info[name] = res
    return all(v["ok"] for v in info.values()), info


def updates(tree, mode, master, optimizer, name):
    out = G / f"updates-{name}.json"
    cmd = [PY, "-X", "utf8", str(HERE / "sm181_updates.py"), "--tree", str(tree), "--master", str(master),
           "--optimizer", str(optimizer), "--release", str(SMALL), "--arch", "w64", "--users", "1088,2104",
           "--out", str(out)]
    r = subprocess.run(cmd, env=env(mode), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (G / f"updates-{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{name}: exit {r.returncode}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())


def k2():
    stock = updates(STOCK, "", SMALL_CASE / "master.pth", SMALL_CASE / "optimizer.pth", "stock")["steps"]
    off = updates(RUN, "", SMALL_CASE / "master.pth", SMALL_CASE / "optimizer.pth", "off")["steps"]
    keys = ("loss_hex", "start_sha256", "grad_sha256", "start_width")
    a = len(stock) == len(off) >= 6 and all(x[k] == y[k] for x, y in zip(stock, off) for k in keys)
    on = updates(RUN, "session", SMALL_START / "master.pth", SMALL_START / "optimizer.pth", "session")["steps"]
    rel = abs(on[0]["loss"] / stock[0]["loss"] - 1)
    b = (len(on) == len(stock) and rel <= 1e-6 and on[0]["start_width"] == 98
         and all(st["extra_columns_abs_max"] > 0 for st in on) and all(st["extra_grad_abs_sum"] > 0 for st in on)
         and all(st["weights_finite"] for st in on) and all(abs(st["loss"]) < 10 for st in on))
    return a and b, {"a_flag_off_equals_stock": {"ok": a, "updates": len(stock), "equal_fields": list(keys)},
                     "b_flag_on_step0": {"ok": b, "first_loss_rel_diff_vs_stock": rel, "width": on[0]["start_width"],
                                         "losses": [round(s["loss"], 6) for s in on],
                                         "stock_losses": [round(s["loss"], 6) for s in stock],
                                         "extra_grad_abs_sum": [s["extra_grad_abs_sum"] for s in on]}}


def score_pair(name, pool, users, arch, release, start, labels):
    uf = G / f"k3-{name}-users.json"
    uf.write_text(json.dumps({"users": users}))
    rows = {}
    for side, mode, master in (("off", "", release), ("ses", "session", start)):
        out = G / f"k3-{name}-{side}.jsonl"
        if out.exists():
            out.unlink()
        e = env(mode)
        e.pop("RWKV_DELETED_CARDS")
        e["PYTORCH_JIT"] = "0"
        cmd = [PY, "-X", "utf8", str(RUN / "scripts/small_model_heldout.py"), "--pool", str(pool), "--users-file", str(uf),
               "--model", f"m=expertium:{arch}:{master}"]
        for lab in labels:
            cmd += ["--only", f"m/{lab}"]
        cmd += ["--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour", "--device", "cpu",
                "--workers", "1", "--time-limit", "1800", "--out", str(out)]
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(RUN))
        (G / f"k3-{name}-{side}.log").write_text(r.stdout[-5000:] + r.stderr[-5000:], encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"k3 {name} {side}: {r.stderr[-1500:]}")
        rows[side] = {(x["user_id"], x["label"]): x["benchmark"] for x in map(json.loads, open(out, encoding="utf-8"))
                      if "benchmark" in x}
    keys = sorted(rows["off"])
    diffs = [abs(rows["off"][k][h]["LogLoss"] - rows["ses"][k][h]["LogLoss"]) for k in keys for h in ("ahead", "instant")]
    ok = len(keys) == len(users) * len(labels) and set(keys) == set(rows["ses"]) and bool(diffs) and max(diffs) <= 1e-6
    return ok, {"users": users, "pool": pool, "labels": len(keys), "max_abs_logloss_diff": max(diffs) if diffs else None}


def k3():
    ok_a, a = score_pair("small", 732, [516, 260], "w64", SMALL_CASE / "master.pth", SMALL_START / "master.pth",
                         ["fp32", "w8+s8hx8"])
    ok_b, b = score_pair("tiny1500", 1500, [5625, 9094], "w64-c1-d2p1u1", TINY_CASE / "master.pth",
                         TINY_START / "master.pth", ["fp32", "w8+s8hx8"])
    return ok_a and ok_b, {"a_small_732": dict(a, ok=ok_a), "b_tiny_pool1500": dict(b, ok=ok_b)}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm181.json"))
    c.add_argument("--only", default="k1,k2,k3")
    args = p.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    head = lambda tree: subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = subprocess.run(["git", "-C", str(RUN), "status", "--porcelain", "rwkv", "scripts", "expertium"],
                           capture_output=True, text=True).stdout.strip()
    report = {"stock_tree": {"path": str(STOCK), "head": head(STOCK)}, "run_tree": {"path": str(RUN), "head": head(RUN),
              "dirty": bool(dirty)}, "checks": {}}
    oks = {}
    for name, fn in (("k1", k1), ("k2", k2), ("k3", k3)):
        if name in args.only.split(","):
            t1 = time.time()
            oks[name], info = fn()
            report["checks"][name] = dict(info, ok=oks[name], seconds=round(time.time() - t1))
            print(json.dumps({name: oks[name], "seconds": round(time.time() - t1)}), flush=True)
    report["all_ok"] = bool(len(oks) == 3 and all(oks.values()) and not dirty)
    report["seconds"] = round(time.time() - t0)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({**oks, "dirty": bool(dirty), "all_ok": report["all_ok"], "seconds": report["seconds"]}))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
