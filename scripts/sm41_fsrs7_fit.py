#!/usr/bin/env python
"""The small model in the review simulator (SM41): pre-split FSRS-7 refits for the FSRS-7 baseline policy.

The baseline's first version used each user's srs-benchmark fit (result/FSRS-7-short-secs.jsonl). The benchmark
saves the LAST fold's parameters of its 5-fold TimeSeriesSplit, fitted on the first ~5/6 of the user's reviews, so
for most simulated windows the fit had seen the simulated days. Here FSRS-7 is refitted per (user, split) on the
user's reviews BEFORE the split day only, with srs-benchmark's own code and flags of that file
(`--algo FSRS-7 --short --secs`: its feature pipeline, Trainer, seed 42, 1 torch thread), on the dataset the simulator
uses. Run with the srs-benchmark venv (it needs the benchmark's packages; the benchmark checkout is only read).

  fit    --users U.json --out F.jsonl          one line per (user, split): parameters + fit facts (resumable)
  proof  --users 3876:225,... --out P.json     (a) the same code on the benchmark's own last fold reproduces the
         saved parameters; (b) the numpy port (scripts/sm41_fsrs7.py) == the benchmark's torch predictions on the
         pre-split reviews with the refit parameters; (c) the simulator's warm-up update rule (every review, the
         first one initialises, then elapsed seconds / 86400) == the benchmark's predictions; (d) LogLoss of the
         refit, the benchmark fit and the defaults on the pre-split reviews and on the 60-day window
"""
from argparse import ArgumentParser
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time

BENCH = Path("C:/Users/jonat/workspace/srs-benchmark")
DATA = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011")
FITS = BENCH / "result" / "FSRS-7-short-secs.jsonl"
HERE = Path(__file__).resolve().parents[1]
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH))
_argv = sys.argv
sys.argv = ["script.py", "--algo", "FSRS-7", "--short", "--secs", "--data", str(DATA)]
import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402
import torch   # noqa: E402
import script as SB   # noqa: E402  (parses the benchmark flags above at import; writes nothing)
from data_loader import UserDataLoader   # noqa: E402
from models.model_factory import create_model   # noqa: E402
from sklearn.model_selection import TimeSeriesSplit   # noqa: E402
from utils import Collection, result_parameters   # noqa: E402
from scripts.sm41_fsrs7 import FSRS7, FSRS7_DEFAULT   # noqa: E402
sys.argv = _argv
LEAD = dt.timedelta(minutes=10)


def hold_reason(now=None):
    """scripts/sm41_drift.py's rule (copied: that module needs the anki-rwkv venv): a quiet.json window of scope all /
    cpu, re-read at every check, running or starting within 10 min; quiet.json unreadable -> break 19:50-01:00."""
    now = now or dt.datetime.now()
    try:
        windows = json.loads(QUIET.read_text())
    except Exception:  # noqa: BLE001
        hm = now.hour * 60 + now.minute
        return "quiet.json unreadable: default break 19:50-01:00" if (hm >= 19 * 60 + 50 or hm < 60) else None
    for w in windows:
        if w.get("scope", "all") not in ("all", "cpu"):
            continue
        a, b = dt.datetime.fromisoformat(w["start"]), dt.datetime.fromisoformat(w["end"])
        if a - LEAD <= now < b:
            return f"quiet window {w['start']}..{w['end']}"
    return None


def wait_clear(tag):
    said = False
    while True:
        r = hold_reason()
        if r is None:
            return
        if not said:
            print(f"[sm41-fsrs7-fit] {tag}: holding ({r}) at {dt.datetime.now():%m-%d %H:%M}", flush=True)
            said = True
        time.sleep(300)


def load(user):
    return UserDataLoader(SB.config).load_user_data(user)


def fit(train):
    """srs-benchmark's per-fold fit (script._fit_trainable_weights, no recency weighting for this file), seeded."""
    torch.manual_seed(SB.config.seed)
    w = SB._fit_trainable_weights(SB._apply_recency_weighting(train))
    return result_parameters({0: w})["0"]


def bench_params(user):
    with FITS.open() as fh:
        for line in fh:
            if f'"user": {user},' in line:
                r = json.loads(line)
                if int(r["user"]) == int(user):
                    return r["parameters"]["0"]
    return None


def bench_metrics(user):
    with FITS.open() as fh:
        for line in fh:
            if f'"user": {user},' in line:
                r = json.loads(line)
                if int(r["user"]) == int(user):
                    return r["metrics"]
    return {}


def torch_predict(w, df):
    p, _, _ = Collection(create_model(SB.config, list(map(float, w))), SB.config).batch_predict(df.copy())
    return np.asarray(p, np.float64)


def port_predict(w, df):
    """The numpy port on the benchmark's own history tensors (history rows = [interval days, rating])."""
    f = FSRS7(w)
    out = np.empty(len(df))
    for j, (h, t) in enumerate(zip(df["tensor"], df["delta_t"])):
        h = h.numpy().astype(np.float64)
        st = np.zeros(3)
        for dt_, g in h:
            st = f.step(np.array(dt_), np.array(g), st)
        out[j] = f.recall(np.array(float(t)), st[0], st[1], st[2])
    return out


def stream_predict(w, user, review_th):
    """The simulator's warm-up rule (scripts/sm41_engine.py Warm.advance) on the raw revlog: every review of the card
    in order, the first one initialises, later ones step with max(0, elapsed_seconds) / 86400; P(recall) read just
    before the reviews whose review_th is asked."""
    raw = pd.read_parquet(DATA / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "elapsed_seconds"])
    f = FSRS7(w)
    want = {int(x): k for k, x in enumerate(review_th)}
    out = np.full(len(review_th), np.nan)
    fs = {}
    for i, (c, g, es) in enumerate(zip(raw["card_id"].to_numpy(), raw["rating"].to_numpy(),
                                       raw["elapsed_seconds"].to_numpy())):
        st = fs.get(c, np.zeros(3))
        dt_ = max(0.0, es / 86400.0)
        k = want.get(i + 1)
        if k is not None:
            out[k] = f.recall(np.array(dt_), st[0], st[1], st[2])
        fs[c] = f.step(np.array(dt_), np.array(float(g)), st)
    return out


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_one(ds, user, D):
    t = time.perf_counter()
    train = ds[ds["day_offset"] < D]
    w = fit(train)
    return {"user": int(user), "split_day": int(D), "parameters": w, "train_rows": int(len(train)),
            "train_max_day": int(train["day_offset"].max()), "dataset_rows": int(len(ds)),
            "fit_seconds": round(time.perf_counter() - t, 1),
            "code": "srs-benchmark script._fit_trainable_weights, --algo FSRS-7 --short --secs, seed 42, 1 thread",
            "benchmark_last_fold_train_max_day": last_fold_max_day(ds)}


def last_fold_max_day(ds):
    tr, _ = list(TimeSeriesSplit(n_splits=SB.config.n_splits).split(ds))[-1]
    return int(ds.iloc[tr]["day_offset"].max())


def done_keys(path):
    p = Path(path)
    if not p.exists():
        return set()
    return {(json.loads(l)["user"], json.loads(l)["split_day"]) for l in p.read_text().splitlines() if l.strip()}


def cmd_fit(args):
    users = json.loads(Path(args.users).read_text())["users"]
    done = done_keys(args.out)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        users = users[k::n]
    if args.only:
        users = [u for u in users if int(u["user"]) == args.only]
    for u in users:
        todo = [s["split_day"] for s in u["splits"] if (u["user"], s["split_day"]) not in done]
        if not todo:
            continue
        wait_clear(f"user {u['user']}")
        ds = load(u["user"])
        for D in todo:
            r = fit_one(ds, u["user"], D)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"[sm41-fsrs7-fit] user {u['user']} split {D}: {r['train_rows']} pre-split rows (benchmark fit "
                  f"saw up to day {r['benchmark_last_fold_train_max_day']}), {r['fit_seconds']} s", flush=True)


def cmd_proof(args):
    out = {"created": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "flags": "--algo FSRS-7 --short --secs",
           "dataset": str(DATA), "users": []}
    ok = True
    for u, D in args.users:
        ds = load(u)
        rec = {"user": u, "split_day": D}
        # (a) the benchmark's own last fold: same code, same data -> the saved parameters
        tr, _ = list(TimeSeriesSplit(n_splits=SB.config.n_splits).split(ds))[-1]
        wb = bench_params(u)
        wa = fit(ds.iloc[tr])
        rec["a_last_fold_max_abs_param_gap"] = float(np.max(np.abs(np.array(wa) - np.array(wb))))
        rec["a_benchmark_fit_saw_up_to_day"] = int(ds.iloc[tr]["day_offset"].max())
        # the benchmark's whole 5-fold protocol with this code: LogLoss against the file's
        ps_, ys_ = [], []
        for k, (tr_k, te_k) in enumerate(TimeSeriesSplit(n_splits=SB.config.n_splits).split(ds)):
            w_k = wa if k == SB.config.n_splits - 1 else fit(ds.iloc[tr_k])
            ps_.append(torch_predict(w_k, ds.iloc[te_k])); ys_.append(ds.iloc[te_k]["y"].to_numpy(np.float64))
        rec["a_5fold_logloss"] = round(logloss(np.concatenate(ps_), np.concatenate(ys_)), 6)
        rec["a_5fold_logloss_file"] = bench_metrics(u).get("LogLoss")
        rec["a_pass"] = bool(rec["a_last_fold_max_abs_param_gap"] <= 1e-4
                             or abs(rec["a_5fold_logloss"] - rec["a_5fold_logloss_file"]) <= 1e-5)
        # pre-split refit
        pre = ds[ds["day_offset"] < D]
        win = ds[(ds["day_offset"] >= D) & (ds["day_offset"] < D + 60)]
        wr = fit(pre)
        rec["pre_rows"], rec["window_rows"] = int(len(pre)), int(len(win))
        # (b) port == torch on the pre-split rows (refit parameters)
        pt = torch_predict(wr, pre)
        pp = port_predict(wr, pre)
        rec["b_port_vs_torch_max_abs"] = float(np.max(np.abs(pt - pp)))
        # (c) the simulator's update stream == torch (refit parameters), pre-split rows
        ps = stream_predict(wr, u, pre["review_th"].to_numpy())
        rec["c_stream_vs_torch_max_abs"] = float(np.nanmax(np.abs(pt - ps)))
        rec["c_stream_vs_torch_mean_abs"] = float(np.nanmean(np.abs(pt - ps)))
        rec["c_stream_vs_torch_share_over_1e-4"] = float(np.mean(np.abs(pt - ps) > 1e-4))
        # (d) LogLoss: refit / benchmark fit / defaults, pre-split (in-sample) and on the 60-day window
        y_pre, y_win = pre["y"].to_numpy(np.float64), win["y"].to_numpy(np.float64)
        for name, w in (("refit", wr), ("benchmark_fit", wb), ("default", FSRS7_DEFAULT)):
            rec[f"d_logloss_pre_{name}"] = round(logloss(torch_predict(w, pre) if name != "refit" else pt, y_pre), 5)
            rec[f"d_logloss_window_{name}"] = round(logloss(torch_predict(w, win), y_win), 5) if len(win) else None
        good = rec["a_pass"] and rec["b_port_vs_torch_max_abs"] <= 1e-4 \
            and rec["c_stream_vs_torch_max_abs"] <= 1e-4
        rec["pass"] = bool(good)
        ok &= good
        out["users"].append(rec)
        print(f"[sm41-fsrs7-proof] {json.dumps(rec)}", flush=True)
    out["pass"] = bool(ok)
    out["tolerance"] = ("(a) last-fold parameters within 1e-4 (the file rounds to 4 decimals) or, failing that, the "
                        "5-fold LogLoss within 1e-5 of the file's (srs-benchmark notes eager vs --compile LogLoss "
                        "differ by up to 7.4e-6 per user); (b), (c) predictions within 1e-4")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[sm41-fsrs7-proof] {'PASS' if ok else 'FAIL'} -> {args.out}", flush=True)


def main():
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--users", required=True); f.add_argument("--out", required=True)
    f.add_argument("--shard", default=""); f.add_argument("--only", type=int, default=0, help="this user only")
    p = sub.add_parser("proof")
    p.add_argument("--users", type=lambda x: [tuple(map(int, y.split(":"))) for y in x.split(",")], required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "proof": cmd_proof}[a.cmd](a)


if __name__ == "__main__":
    main()
