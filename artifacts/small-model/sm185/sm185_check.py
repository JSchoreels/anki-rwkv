#!/usr/bin/env python
"""Small cool-down with session inputs (SM185): the proofs (design.md "Proofs"); SM183's sm183_check.py with the
sources changed to Small two-head (SM118) (chain th) and Small Instant-Only (SM113) (chain io).

  python sm185_check.py cpu [--out check-cpu-sm185.json]                  CPU, 4 threads, < 4 GB, a few minutes
  python sm185_check.py ctl --chain th [--out check-ctl-sm185.json]       after the GPU control reproduction ticket
  python sm185_check.py ctl --chain io [--out check-ctl-io-sm185.json]    (CPU, seconds)

K1 the show-time data of every user this run reads with the session flag on: the 8,000 training users (the same rows
   for both chains) and the 732 set. For each, rwkv/extra_inputs.py's own alignment (shared unique parquet row index,
   rating and duration equal row by row) succeeds and the 6 session values are finite; answer-time inversions are
   reported. (The 765 decisive users passed the same scan in SM181's K1.)
K2 the real trainer path on CPU (each source's pass-1 options: strided-lane carry, show-time inputs, replay KD targets,
   fresh Muon; io: curve loss weight 0; 256-review windows, users 4745 and 2308, 3 updates each):
   (a) per chain, the source's pass-1 start, flag off: the run tree (cd220759) = the tree that trained the source bit
       for bit (loss, model input, every gradient): th = wt/sm118 (SM118 trained in its own worktree), io = wt/sm113;
   (b) per chain, the padded start of sm185-<chain>-ses-p1 with the flag on: first-update loss equal to the run tree's
       flag-off loss from the source's start within 1e-6 relative, input width 98, extra columns and their gradients
       non-zero, everything finite.
G0 control reproduction (GPU, `ctl --chain c`): sm185-<c>-ctlcheck (the source's pass 1, flag unset, run tree) trained
   to its first periodic snapshot (400 users) must equal the source's own pass 1 at that snapshot bit for bit: every
   master.pth tensor, every optimizer.pth state tensor, the snapshot's counters and done users, and the per-update log
   (user, offset, reviews, seed, loss, gradient norm, labels, learning rate) of all updates up to it. The source's
   snapshot: SM118's pass 1 (never resumed); SM113's pass 1 attempt 1 (its first segment, before the break stop).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
W = Path("C:/Users/jonat/workspace/wt")
STOCK = {"th": W / "sm118", "io": W / "sm113"}   # the trees that trained the sources, clean scripts / rwkv / expertium
RUN = W / "sm179-run"               # branch sm179-run, cd220759
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
SRC = {"th": W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p1",
       "io": W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p1"}
RS = RUN / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
ARCH = "w64"
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
USERS732 = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
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


def users_732():
    d = json.loads(USERS732.read_text())
    return [int(u) for u in (d["users"] if isinstance(d, dict) else d)]


def k1():
    train = [r["user_id"] for r in json.loads((SRC["th"] / "protocol.json").read_text())["rows"]]
    io = [r["user_id"] for r in json.loads((SRC["io"] / "protocol.prepared.json").read_text())["rows"]]
    if sorted(io) != sorted(train):
        raise RuntimeError("the two chains train different users")
    info = {}
    for name, users in (("training_8000", train), ("set_732", users_732())):
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
        res["users_with_inversions"] = res["users_with_inversions"][:50]
        res["ok"] = res["users"] == len(users) and not res["failed"] and not res["nonfinite"]
        info[name] = res
    return all(v["ok"] for v in info.values()), info


def updates(tree, mode, master, folder, name):
    out = G / f"updates-{name}.json"
    cmd = [PY, "-X", "utf8", str(HERE / "sm185_updates.py"), "--tree", str(tree), "--master", str(master),
           "--optimizer", "none", "--release", str(folder), "--arch", ARCH, "--users", "4745,2308", "--out", str(out)]
    r = subprocess.run(cmd, env=env(mode), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (G / f"updates-{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{name}: exit {r.returncode}: {r.stderr[-2000:]}")
    return json.loads(out.read_text())["steps"]


def k2():
    keys = ("loss_hex", "start_sha256", "grad_sha256", "start_width")
    res, oks = {}, []
    for chain in ("th", "io"):
        src = SRC[chain]
        q = src / "protocol.prepared.json"
        start = json.loads((q if q.exists() else src / "protocol.json").read_text())["full_run"]["init_master"]
        off = updates(RUN, "", start, src, f"{chain}-off")
        stock = updates(STOCK[chain], "", start, src, f"{chain}-stock")
        a = len(stock) == len(off) >= 6 and all(x[k] == y[k] for x, y in zip(stock, off) for k in keys)
        res[f"a_{chain}_flag_off_equals_source_tree"] = {"ok": a, "tree": str(STOCK[chain]), "updates": len(stock),
                                                        "equal_fields": list(keys),
                                                        "losses": [round(s["loss"], 6) for s in stock]}
        oks.append(a)
        on = updates(RUN, "session", RS / f"sm185-{chain}-ses-p1/start/master.pth", src, f"{chain}-session")
        rel = abs(on[0]["loss"] / off[0]["loss"] - 1)
        b = (len(on) == len(off) >= 6 and rel <= 1e-6 and on[0]["start_width"] == 98
             and all(st["extra_columns_abs_max"] > 0 for st in on) and all(st["extra_grad_abs_sum"] > 0 for st in on)
             and all(st["weights_finite"] for st in on) and all(abs(st["loss"]) < 10 for st in on))
        res[f"b_{chain}_flag_on_step0"] = {"ok": b, "first_loss_rel_diff_vs_flag_off": rel, "width": on[0]["start_width"],
                                           "losses": [round(s["loss"], 6) for s in on],
                                           "flag_off_losses": [round(s["loss"], 6) for s in off],
                                           "extra_grad_abs_sum": [s["extra_grad_abs_sum"] for s in on]}
        oks.append(b)
    return all(oks), res


def cpu(args):
    G.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    head = lambda tree: subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = {str(t): subprocess.run(["git", "-C", str(t), "status", "--porcelain", "rwkv", "scripts", "expertium"],
                                    capture_output=True, text=True).stdout.strip() for t in (RUN, *STOCK.values())}
    report = {"source_trees": {c: {"path": str(t), "head": head(t), "dirty": bool(dirty[str(t)])} for c, t in STOCK.items()},
              "run_tree": {"path": str(RUN), "head": head(RUN), "dirty": bool(dirty[str(RUN)])}, "checks": {}}
    oks = {}
    for name, fn in (("k1", k1), ("k2", k2)):
        if name in args.only.split(","):
            t1 = time.time()
            oks[name], info = fn()
            report["checks"][name] = dict(info, ok=oks[name], seconds=round(time.time() - t1))
            print(json.dumps({name: oks[name], "seconds": round(time.time() - t1)}), flush=True)
    report["all_ok"] = bool(len(oks) == 2 and all(oks.values()) and not any(dirty.values()))
    report["seconds"] = round(time.time() - t0)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({**oks, "dirty": any(dirty.values()), "all_ok": report["all_ok"], "seconds": report["seconds"]}))
    return 0 if report["all_ok"] else 1


UPDATE_KEYS = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")


def ctl(args):
    """G0: the control reproduction check."""
    import torch
    snap = "snapshots/users-00400"
    def first_with_snapshot(f):
        places = [f / "expertium" / TRACK] + sorted(x / "expertium" / TRACK for x in (f / "attempts").glob("attempt-*"))
        return next((x for x in places if (x / snap / "snapshot.json").exists()), places[0])
    a = first_with_snapshot(RS / f"sm185-{args.chain}-ctlcheck")   # a restarted check: its attempt
    b = first_with_snapshot(SRC[args.chain])                       # SM113's pass 1: attempt 1
    if not (b / snap / "snapshot.json").exists():
        raise SystemExit(f"the source has no {snap}")
    problems, info = [], {}
    if not (a / snap / "snapshot.json").exists():
        problems.append(f"{a / snap} missing (the check ticket did not reach 400 users)")
    else:
        sa, sb = (json.loads((x / snap / "snapshot.json").read_text()) for x in (a, b))
        for k in ("progress_users", "updates", "update_offset_end", "review_presentations", "done_users"):
            if sa.get(k) != sb.get(k):
                problems.append(f"snapshot {k} differs")
        info["snapshot"] = {k: sb.get(k) for k in ("progress_users", "updates", "update_offset_end", "review_presentations")}
        ma, mb = (torch.load(x / snap / "master.pth", map_location="cpu", weights_only=True) for x in (a, b))
        bad = [k for k in mb if k not in ma or not torch.equal(ma[k], mb[k])] + [k for k in ma if k not in mb]
        info["master_tensors"] = len(mb)
        if bad:
            problems.append(f"master.pth differs in {len(bad)} tensors (first {bad[:3]})")
        oa, ob = (torch.load(x / snap / "optimizer.pth", map_location="cpu", weights_only=False) for x in (a, b))
        n, badopt = 0, []

        def walk(x, y, path):
            nonlocal n
            if torch.is_tensor(y):
                n += 1
                if not (torch.is_tensor(x) and x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)):
                    badopt.append(path)
            elif isinstance(y, dict):
                if not isinstance(x, dict) or set(x) != set(y):
                    badopt.append(path + " (keys)")
                    return
                for k in y:
                    walk(x[k], y[k], f"{path}/{k}")
            elif isinstance(y, (list, tuple)):
                if not isinstance(x, (list, tuple)) or len(x) != len(y):
                    badopt.append(path + " (length)")
                    return
                for i, (p, q) in enumerate(zip(x, y)):
                    walk(p, q, f"{path}[{i}]")
            elif x != y:
                badopt.append(path)
        walk(oa, ob, "optimizer")
        info["optimizer_tensors"] = n
        if badopt:
            problems.append(f"optimizer.pth differs at {len(badopt)} places (first {badopt[:3]})")
        k = int(sb["updates"])
        ua, ub = ([json.loads(l) for _, l in zip(range(k), open(x / "updates.jsonl", encoding="utf-8"))] for x in (a, b))
        diff = [i for i, (x, y) in enumerate(zip(ua, ub)) if any(x.get(f) != y.get(f) for f in UPDATE_KEYS)]
        info["updates_compared"] = min(len(ua), len(ub))
        if len(ua) != k or len(ub) != k:
            problems.append(f"update logs hold {len(ua)} / {len(ub)} of {k} updates")
        if diff:
            problems.append(f"{len(diff)} updates differ (first step {ua[diff[0]].get('step')})")
    info["compared_folder"] = str(a)
    info["source_folder"] = str(b)
    out = {"what": f"G0: sm185-{args.chain}-ctlcheck (run tree cd220759, flag unset) vs {SRC[args.chain].name} at its 400-user snapshot",
           "ok": not problems, "problems": problems, **info, "update_fields": list(UPDATE_KEYS)}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"g0_ok": out["ok"], "problems": problems[:3]}))
    return 0 if out["ok"] else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm185.json"))
    c.add_argument("--only", default="k1,k2")
    c = sub.add_parser("ctl")
    c.add_argument("--chain", required=True, choices=["th", "io"])
    c.add_argument("--out")
    args = p.parse_args()
    if args.cmd == "ctl" and not args.out:
        args.out = str(HERE / ("check-ctl-sm185.json" if args.chain == "th" else "check-ctl-io-sm185.json"))
    return cpu(args) if args.cmd == "cpu" else ctl(args)


if __name__ == "__main__":
    raise SystemExit(main())
