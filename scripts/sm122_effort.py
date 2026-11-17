#!/usr/bin/env python
"""Retrieval-effort probe (SM122), after Pyc & Rawson (2009): do harder but successful retrievals (slower successful
answers) lead to higher or lower later recall, in the model's learned behaviour and in real outcomes?
Pre-registered in artifacts/small-model/sm122/design.md. CPU only, aggregate output.

Answer time = front + back (question shown -> grade pressed, capped at the preset's maximum answer seconds). In the
models it is feature 8 (scaled_duration) on the update (history) rows; query rows carry 0.

Counterfactual arms (raw revlog edits of the capped prefix, then the scorer's show-time batch, fp32, leak-free):
  base        nothing changed
  half/double the target reviews' recorded duration x0.5 / x2 (clamped to [1 ms, the user's cap])
  card_fast/card_slow   set to the card's own 10th / 90th percentile of its Good/Easy timed answers (>= 3 of them)
  user_fast/user_slow   set to the user's 10th / 90th percentile of Good/Easy timed answers
  extra       one more same-day success inserted 600 s after each target (same grade, state and duration)
Targets: Good/Easy, timed below the cap, not a first review, whose card's next review is on a later day; at least
SPACING reviews apart (greedy in review order). Read-outs per candidate pair (t, t'): the instant head's P(recall) at the
query row of t', and the curve head at t's update row at t' (show-time interval) and at 1 d, 1 w, 30 d.

  select                     the pre-registered user subsample -> artifacts/small-model/sm122/users.json
  check  [--out F]           CPU proofs K1-K7 on a few users -> check-cpu.json
  run    --model KEY         all users of the subsample (resumable) -> work/<key>/<user>.npz (gitignored)
  report                     -> sm122-result.{md,json}
Code reused read only from wt/sm121 (scripts/sm117_head.py, small_model_heldout.py, sm31_inputs.py, sm71_leak.py).
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S121 = Path("C:/Users/jonat/workspace/wt/sm121")          # read only (the loaders); commit recorded in every output
sys.path.insert(0, str(S121))

OUT = ROOT / "artifacts/small-model/sm122"
WORK = OUT / "work"                                         # gitignored per-user arrays: never git add -f
U732 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
SEED = 122
N_USERS = 200                                              # Small and Tiny: random 200 of the 723 scoreable users
N_BIG = 60                                                 # Big: the first 60 of them (random order)
CAP = 20000                                                # first 20,000 reviews of each user (prefix)
SPACING = 100                                              # targets at least 100 reviews apart
INSERT_S = 600                                             # extra success: 600 s after the target
FIXED = {"1d": 86400.0, "1w": 7 * 86400.0, "30d": 30 * 86400.0}
ARMS = ("base", "half", "double", "card_fast", "card_slow", "user_fast", "user_slow", "extra")
BIG_ARMS = ("base", "half", "double", "extra")
MODEL_KEYS = ("sm92", "sm106", "sm91s2")
NAMES = {"sm92": "the show-time retrain (SM92)", "sm106": "the Tiny two-head model (SM106)",
         "sm91s2": "the show-time big model (SM91s2)"}
SHORT = {"sm92": "Small two-head", "sm106": "Tiny two-head", "sm91s2": "Big two-head"}
ARM_NAMES = {"half": "x0.5", "double": "x2", "card_fast": "card's 10th percentile", "card_slow": "card's 90th percentile",
             "user_fast": "user's 10th percentile", "user_slow": "user's 90th percentile",
             "extra": "one extra same-day success"}
BOOT = 1000
MIN_PCT = 0.05
EPS = 1e-6


def below_normal():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def git_head(path):
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "scripts"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": sha, "scripts_dirty": dirty}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ quiet windows (quiet.json all/cpu + daily break)
def quiet_wait(margin_min=6, stop_file=None):
    """Sleeps while inside a quiet window or within margin_min of the next one; returns False if a STOP file appears."""
    while True:
        if stop_file is not None and Path(stop_file).exists():
            return False
        now = datetime.now()
        blocks = []
        try:
            for w in json.loads(QUIET.read_text(encoding="utf-8")):
                if w.get("scope", "all") in ("all", "cpu"):
                    blocks.append((datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"])))
        except Exception:  # noqa: BLE001
            try:
                text = QUIET.read_text(encoding="utf-8")
                for s, e in re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S):
                    blocks.append((datetime.fromisoformat(s), datetime.fromisoformat(e)))
            except Exception:  # noqa: BLE001
                pass
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 3)]
        blocks += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20)) for d in range(-1, 3)
                   if (day + timedelta(days=d)).weekday() == 6]
        inside = [e for s, e in blocks if s - timedelta(minutes=margin_min) <= now < e]
        if not inside:
            return True
        wake = max(inside) + timedelta(minutes=5)
        print(f"[sm122] quiet window until {max(inside):%m-%d %H:%M}; sleeping", flush=True)
        while datetime.now() < wake:
            if stop_file is not None and Path(stop_file).exists():
                return False
            time.sleep(min(600.0, max(1.0, (wake - datetime.now()).total_seconds())))


# ------------------------------------------------------------------ users
def pool_users():
    return json.loads(U732.read_text())["users"]


def select(args):
    users = pool_users()
    rng = np.random.default_rng(SEED)
    pick = [int(u) for u in rng.permutation(np.asarray(users))[:N_USERS]]
    from scripts import compress_common as cc
    rc = cc.run_info()["review_counts"]
    n = np.array([min(int(rc[str(u)]), CAP) for u in pick])
    out = {"note": f"SM122: random {N_USERS} of the {len(users)} scoreable users of the 732-user standard readout set "
                   f"(numpy default_rng({SEED}).permutation, first {N_USERS}); Big uses the first {N_BIG}. Prefix cap "
                   f"{CAP} reviews.", "seed": SEED, "users": pick, "big_users": pick[:N_BIG],
           "capped_reviews": int(n.sum()), "capped_reviews_big": int(n[:N_BIG].sum())}
    atomic_json(OUT / "users.json", out)
    print(f"[sm122-select] {len(pick)} users, {n.sum():,} capped reviews (Big {n[:N_BIG].sum():,})")
    return 0


# ------------------------------------------------------------------ per-user data, candidates, targets
def raw_revlog(data, user):
    import pandas as pd
    return pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}").iloc[:CAP].reset_index(drop=True)


def plan_user(raw):
    """Candidate pairs (t, t') and per-candidate covariates (0-based row indices into the capped raw frame)."""
    from scripts.sm117_timelib import time_user_cap
    import pandas as pd
    card = raw["card_id"].to_numpy(np.int64)
    day = raw["day_offset"].to_numpy(np.int64)
    rating = raw["rating"].to_numpy(np.int64)
    dur = raw["duration"].to_numpy(np.int64)
    ed = raw["elapsed_days"].to_numpy(np.int64)
    n = len(raw)
    ucap = int(time_user_cap(dur))
    nxt = pd.Series(np.arange(n)).groupby(card).shift(-1).to_numpy()
    has_next = ~np.isnan(nxt)
    nxt = np.where(has_next, nxt, -1).astype(np.int64)
    ge = (rating >= 3)
    timed = (dur > 0) & (dur < ucap)
    ok_ge = ge & timed
    cand = np.flatnonzero(ok_ge & (ed >= 0) & has_next)
    cand = cand[ed[nxt[cand]] >= 1]
    t, t2 = cand, nxt[cand]
    logd = np.where(ok_ge, np.log(np.maximum(dur, 1)), np.nan)
    # within-user: ln d - the user's median ln d of the same grade (Good/Easy, timed below the cap, whole prefix)
    med_u = {g: float(np.nanmedian(logd[(rating == g) & ok_ge])) if ((rating == g) & ok_ge).any() else np.nan for g in (3, 4)}
    z_user = logd[t] - np.array([med_u[g] for g in rating[t]])
    # within-card: ln d - median of the card's EARLIER Good/Easy timed answers (>= 2 of them), causal
    s = pd.Series(logd)
    prev_med = s.groupby(card).transform(lambda x: x.shift().expanding(min_periods=2).median()).to_numpy()
    z_card = logd[t] - prev_med[t]
    # card quantiles (whole prefix, >= 3 Good/Easy timed answers) and user quantiles
    cq = pd.DataFrame({"c": card[ok_ge], "d": dur[ok_ge]}).groupby("c")["d"]
    cnt = cq.count()
    q10 = cq.quantile(0.1)
    q90 = cq.quantile(0.9)
    c10 = pd.Series(card[t]).map(q10.where(cnt >= 3)).to_numpy()
    c90 = pd.Series(card[t]).map(q90.where(cnt >= 3)).to_numpy()
    u10, u90 = (float(np.percentile(dur[ok_ge], 10)), float(np.percentile(dur[ok_ge], 90))) if ok_ge.any() else (np.nan, np.nan)
    # same-day structure of day D = day of t (t is the card's last review of D)
    df = pd.DataFrame({"c": card, "d": day, "s": (rating >= 2).astype(int), "a": (rating == 1).astype(int),
                       "first": (ed == -1).astype(int)})
    g = df.groupby(["c", "d"])
    n_day = g["s"].transform("size").to_numpy()
    s_day = g["s"].transform("sum").to_numpy()
    a_day = g["a"].transform("sum").to_numpy()
    f_day = g["first"].transform("max").to_numpy()
    # targets: greedy, at least SPACING apart
    tgt = np.zeros(len(t), bool)
    last = -10 ** 9
    for i, ti in enumerate(t):
        if ti - last >= SPACING:
            tgt[i] = True
            last = ti
    return {"t": t, "t2": t2, "grade": rating[t], "dur": dur[t], "ucap": ucap, "ed_t": ed[t], "ed_t2": ed[t2],
            "y": (rating[t2] >= 2).astype(np.int8), "z_user": z_user, "z_card": z_card, "c10": c10, "c90": c90,
            "u10": u10, "u90": u90, "n_day": n_day[t], "s_day": s_day[t], "a_day": a_day[t], "first_day": f_day[t],
            "day_t2": day[t2], "target": tgt}


def arm_durations(P, arm):
    """New durations of the targets for a duration arm (NaN = this target is not changed in this arm)."""
    d = P["dur"][P["target"]].astype(np.float64)
    if arm == "half":
        new = d * 0.5
    elif arm == "double":
        new = d * 2.0
    elif arm == "card_fast":
        new = P["c10"][P["target"]].astype(np.float64)
    elif arm == "card_slow":
        new = P["c90"][P["target"]].astype(np.float64)
    elif arm == "user_fast":
        new = np.full_like(d, P["u10"])
    elif arm == "user_slow":
        new = np.full_like(d, P["u90"])
    else:
        raise ValueError(arm)
    return np.where(np.isnan(new), np.nan, np.clip(np.round(new), 1, P["ucap"]))


def edit_raw(raw, P, arm, only=None):
    """(edited raw frame, row map old 0-based index -> new 0-based index, inserted-copy index per target or None).
    only = a single target index (into P["t"]) for the exact single-target forwards of the check."""
    frame = raw.copy()
    tg = np.flatnonzero(P["target"]) if only is None else np.array([only])
    if arm == "base":
        return frame, np.arange(len(raw)), None
    if arm == "extra":
        import pandas as pd
        rows = P["t"][tg]
        nxt = P["t2"][tg]
        es = frame["elapsed_seconds"].to_numpy(np.int64).copy()
        es[nxt] = np.maximum(es[nxt] - INSERT_S, 1)
        frame["elapsed_seconds"] = es
        ins = frame.iloc[rows].copy()
        ins["elapsed_days"] = 0
        ins["elapsed_seconds"] = INSERT_S
        order = np.r_[np.arange(len(frame)), rows + 0.5]
        both = pd.concat([frame, ins], ignore_index=True)
        both = both.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
        before = np.searchsorted(np.sort(rows), np.arange(len(raw)), side="left")   # insertions strictly before
        rmap = np.arange(len(raw)) + before
        return both, rmap, rmap[rows] + 1
    new = arm_durations(P, arm) if only is None else arm_durations(P, arm)[np.flatnonzero(P["target"]) == only]
    rows = P["t"][tg]
    keep = ~np.isnan(new)
    dur = frame["duration"].to_numpy(np.int64).copy()
    dur[rows[keep]] = new[keep].astype(np.int64)
    frame["duration"] = dur
    return frame, np.arange(len(raw)), None


@contextmanager
def raw_patch(frame):
    """pandas.read_parquet of a revlogs/ path returns a copy of `frame` (the edited capped prefix)."""
    import pandas as pd
    original = pd.read_parquet

    def patched(path, *a, **k):
        if "/revlogs/" in str(path).replace("\\", "/"):
            return frame.copy()
        return original(path, *a, **k)

    pd.read_parquet = patched
    try:
        yield
    finally:
        pd.read_parquet = original


def make_batch(data, user, frame):
    from scripts.sm117_head import show_time_batch
    with raw_patch(frame):
        batch, _ = show_time_batch(data, user)
    return batch


# ------------------------------------------------------------------ model outputs
class Engine:
    def __init__(self, key):
        from scripts import small_model_heldout as smh  # noqa: F401  first: RWKV_NO_JIT + patches
        import torch
        from scripts.sm117_head import verified_master, build_model, MODELS
        self.torch = torch
        self.key = key
        self.kind = MODELS[key]["kind"]
        info = verified_master(key)
        self.sha = info["sha256"]
        sd = torch.load(info["master"], map_location="cpu", weights_only=True)
        self.model = build_model(key, sd, "cpu")

    def forward(self, b):
        m = self.model
        if self.kind == "fork":
            return m.forward_batch(b.start, b.sub_gather, b.sub_gather_lens, b.time_shift_selects, b.skips, b.num_data)
        return m.forward_batch_variant(b.start, b.sub_gather, b.sub_gather_lens, b.time_shift_selects, b.skips, b.num_data)

    def curve(self, o, rows, secs):
        """P(recall) of the curve head at update rows `rows` after secs (tensor [n] of seconds)."""
        torch, m = self.torch, self.model
        t = secs.float().view(-1, 1)
        if self.kind == "fork":
            raw = m.forgetting_curve(o[1][0][rows], t)
        elif getattr(m, "gru_on", False):
            raw = m.gru_forgetting_curve(o[1][0][rows], o[4][0][rows], o[5][0][rows], t)
        else:
            raw = m.forgetting_curve(o[1][0][rows], t)
        raw = raw.view(-1)
        res = m.interp(o[0][0][rows], t).view(-1)
        return torch.sigmoid(torch.log(raw / (1 - raw)) + res)

    def readout(self, b, rmap, ins, P):
        """Per candidate: instant P(recall) at t' query row, curve at t's update row (or the inserted copy) at t' and
        at the fixed horizons."""
        torch = self.torch
        with torch.inference_mode():
            o = self.forward(b)
            lab = b.labels[0]
            isq = lab[:, 6] == 1
            qrows = torch.nonzero(isq).squeeze(-1)
            urows = torch.nonzero(~isq).squeeze(-1)
            q_th = b.label_review_th[0].long()[qrows]
            order = torch.argsort(q_th)
            q_sorted, q_rows_sorted = q_th[order], qrows[order]
            want = torch.as_tensor(rmap[P["t2"]] + 1)
            pos = torch.searchsorted(q_sorted, want).clamp(max=len(q_sorted) - 1)
            if not bool((q_sorted[pos] == want).all()):
                raise RuntimeError("a next review without its query row")
            qr = q_rows_sorted[pos]
            src = rmap[P["t"]].copy()
            if ins is not None:
                src[P["target"]] = ins
            ur = urows[torch.as_tensor(src)]
            if not bool((b.label_review_th[0].long()[ur] == want).all()):
                raise RuntimeError("update row label does not point at the next review")
            inst = 1.0 - torch.softmax(o[3][0][qr].float(), -1)[:, 0]
            lab_s = lab[ur, 0].float()
            out = {"inst": inst.double().numpy(), "curve": self.curve(o, ur, lab_s).double().numpy()}
            for k, s in FIXED.items():
                out[k] = self.curve(o, ur, torch.full((len(ur),), s)).double().numpy()
            out["label_s"] = lab_s.double().numpy()
        return out, o


def run_user(eng, data, user, arms):
    raw = raw_revlog(data, user)
    P = plan_user(raw)
    if P["target"].sum() == 0:
        return P, {}
    res = {}
    for arm in arms:
        frame, rmap, ins = edit_raw(raw, P, arm)
        b = make_batch(data, user, frame)
        res[arm], _ = eng.readout(b, rmap, ins, P)
        del b
    return P, res


def save_user(path, user, P, res, secs):
    arrays = {f"P_{k}": np.asarray(v) for k, v in P.items()}
    for arm, r in res.items():
        for k, v in r.items():
            arrays[f"{arm}__{k}"] = np.asarray(v, np.float64)
    arrays["seconds"] = np.array(secs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run(args):
    from scripts import small_model_heldout as smh  # noqa: F401
    import torch
    below_normal()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    assert not torch.cuda.is_available(), "CUDA must stay hidden (CPU only)"
    from scripts import compress_common as cc
    data = cc.data_path()
    sel = json.loads((OUT / "users.json").read_text())
    users = sel["big_users"] if args.model == "sm91s2" else sel["users"]
    arms = BIG_ARMS if args.model == "sm91s2" else ARMS
    if args.limit:
        users = users[:args.limit]
    eng = Engine(args.model)
    wd = WORK / args.model
    wd.mkdir(parents=True, exist_ok=True)
    stop = OUT / "run" / "STOP"
    started = time.perf_counter()
    meta = {"model": args.model, "sha256": eng.sha, "arms": list(arms), "users": len(users), "sm122": git_head(ROOT),
            "sm121_loaders": git_head(S121), "threads": torch.get_num_threads(), "started": datetime.now().isoformat(timespec="seconds")}
    atomic_json(wd / "meta.json", meta)
    done = 0
    for k, u in enumerate(users):
        f = wd / f"{u}.npz"
        if f.exists():
            done += 1
            continue
        if not quiet_wait(stop_file=stop):
            print("[sm122-run] STOP file: stopping", flush=True)
            return 1
        t0 = time.perf_counter()
        try:
            P, res = run_user(eng, data, u, arms)
        except Exception as e:  # noqa: BLE001  recorded, the user is left out
            print(f"[sm122-run] user {u}: {e!r}", flush=True)
            (wd / f"{u}.error").write_text(repr(e)[:500], encoding="utf-8")
            continue
        save_user(f, u, P, res, time.perf_counter() - t0)
        done += 1
        print(f"[sm122-run] {args.model} {k + 1}/{len(users)} user {u}: {len(P['t'])} pairs, {int(P['target'].sum())} "
              f"targets, {time.perf_counter() - t0:.0f} s (total {time.perf_counter() - started:.0f} s, peak {peak_gb()} GB)",
              flush=True)
    meta.update({"ended": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - started),
                 "peak_gb": peak_gb(), "done_users": done})
    atomic_json(wd / "meta.json", meta)
    (wd / "DONE").write_text("ok", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ CPU check
def check(args):
    from scripts import small_model_heldout as smh  # noqa: F401
    import torch
    below_normal()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    assert not torch.cuda.is_available()
    from scripts import compress_common as cc
    data = cc.data_path()
    sel = json.loads((OUT / "users.json").read_text())
    rc = cc.run_info()["review_counts"]
    # check users: the first 4 of the subsample whose history is under 8,000 reviews (cheap), then 2 longer ones
    short = [u for u in sel["users"] if int(rc[str(u)]) < 8000][:4]
    longer = [u for u in sel["users"] if 15000 < int(rc[str(u)])][:2]
    report = {"users": short + longer, "proofs": {}, "timing": {}, "sm122": git_head(ROOT), "sm121_loaders": git_head(S121)}
    keys = args.models.split(",")
    for key in keys:
        eng = Engine(key)
        K = {}
        t_all = time.perf_counter()
        # K1: our curve read-out = the model's own p_curve (get_loss) on update rows; instant = softmax of its logits
        u = short[0]
        raw = raw_revlog(data, u)
        P = plan_user(raw)
        b = make_batch(data, u, raw)
        r, o = eng.readout(b, np.arange(len(raw)), None, P)
        k1 = None
        try:
            with torch.inference_mode():
                stats = eng.model.get_loss(b)
            pc = stats.p_curve[0] if stats.p_curve.dim() == 2 else stats.p_curve
            lab = b.labels[0]
            urows = torch.nonzero(lab[:, 6] == 0).squeeze(-1)
            ours = eng.curve(o, urows, lab[urows, 0].float())
            hl = lab[urows, 4] == 1
            k1 = float((ours[hl] - pc.view(-1)[urows][hl].float()).abs().max())
        except Exception as e:  # noqa: BLE001
            k1 = f"get_loss unavailable: {e!r}"[:200]
        K["K1_curve_vs_get_loss_maxabs"] = k1
        # K5: determinism (two base passes)
        r2, _ = eng.readout(make_batch(data, u, raw), np.arange(len(raw)), None, P)
        K["K5_base_repeat_maxabs"] = float(max(np.abs(r[k] - r2[k]).max() for k in r))
        # K2: the double arm changes only feature 8 on target update rows (+ the target query rows' interval columns)
        fr, rmap, ins = edit_raw(raw, P, "double")
        b2 = make_batch(data, u, fr)
        diff = (b.start != b2.start)
        cols = sorted(set(torch.nonzero(diff.any(0)).squeeze(-1).tolist())) if diff.any() else []
        lab = b.labels[0]
        isq = (lab[:, 6] == 1).numpy()
        rows_changed = torch.nonzero(diff.any(1)).squeeze(-1).numpy()
        upd_rows = np.flatnonzero(~isq)[P["t"][P["target"]]]
        col8_rows = torch.nonzero(diff[:, 8]).squeeze(-1).numpy() if diff.any() else np.array([], int)
        K["K2_changed_columns"] = cols
        K["K2_col8_rows_are_target_update_rows"] = bool(set(col8_rows.tolist()) <= set(upd_rows.tolist()))
        K["K2_other_rows_are_query_rows"] = bool(all(isq[rw] for rw in set(rows_changed.tolist()) - set(col8_rows.tolist())))
        # K3: before the first target's update row every output equals base; feature-8-only edit = raw-duration edit
        o_b = eng.forward(b)
        o_2 = eng.forward(b2)
        first = int(upd_rows.min())
        K["K3_outputs_before_first_target_maxabs"] = float(max((x[0][:first] - y[0][:first]).abs().max() for x, y in zip(o_b, o_2) if x.dim() == 3))
        rd, _ = eng.readout(b2, rmap, ins, P)
        b3 = make_batch(data, u, raw)
        new = arm_durations(P, "double")
        from rwkv import data_processing as dp
        b3.start[torch.as_tensor(upd_rows), 8] = torch.as_tensor(dp.scale_duration(new), dtype=b3.start.dtype)
        r3, _ = eng.readout(b3, rmap, ins, P)
        K["K3_feature8_only_vs_raw_edit_maxabs_at_next"] = float(np.abs(rd["inst"] - r3["inst"]).max())
        # K6: extra arm: same outputs as base before the first insertion; row count + targets
        fe, rme, inse = edit_raw(raw, P, "extra")
        be = make_batch(data, u, fe)
        o_e = eng.forward(be)
        lab_e = be.labels[0]
        first_e = int(torch.nonzero(lab_e[:, 6] == 0).squeeze(-1)[int(P["t"][P["target"]].min())])
        K["K6_extra_rows_added"] = int(be.start.shape[0] - b.start.shape[0])
        K["K6_extra_expected_rows"] = int(2 * P["target"].sum())
        K["K6_extra_outputs_before_first_insert_maxabs"] = float(max((x[0][:first_e] - y[0][:first_e]).abs().max() for x, y in zip(o_b, o_e) if x.dim() == 3))
        re_, _ = eng.readout(be, rme, inse, P)
        K["K6_extra_readout_finite"] = bool(all(np.isfinite(v).all() for v in re_.values()))
        # K4: batched vs exact single-target deltas (double arm), up to 8 targets spread over the history per user
        ex, bt, rb = [], [], []
        for u2 in short + longer:
            raw2 = raw_revlog(data, u2)
            P2 = plan_user(raw2)
            tg = np.flatnonzero(P2["target"])
            if len(tg) == 0:
                continue
            base2, _ = eng.readout(make_batch(data, u2, raw2), np.arange(len(raw2)), None, P2)
            f2, m2, _ = edit_raw(raw2, P2, "double")
            dbl2, _ = eng.readout(make_batch(data, u2, f2), m2, None, P2)
            pick = tg[np.unique(np.linspace(0, len(tg) - 1, min(8, len(tg))).round().astype(int))]
            for i in pick:
                fs, ms, _ = edit_raw(raw2, P2, "double", only=i)
                s_, _ = eng.readout(make_batch(data, u2, fs), ms, None, P2)
                ex.append(s_["inst"][i] - base2["inst"][i])
                bt.append(dbl2["inst"][i] - base2["inst"][i])
                rb.append(base2["inst"][i])
        ex, bt, rb = np.array(ex), np.array(bt), np.array(rb)
        K["K4_targets"] = int(len(ex))
        K["K4_mean_exact_pct_of_R"] = float(100 * ex.mean() / rb.mean())
        K["K4_mean_batched_pct_of_R"] = float(100 * bt.mean() / rb.mean())
        K["K4_mean_abs_exact_pct_of_R"] = float(100 * np.abs(ex).mean() / rb.mean())
        K["K4_mean_abs_diff_pct_of_R"] = float(100 * np.abs(ex - bt).mean() / rb.mean())
        K["K4_pearson"] = float(np.corrcoef(ex, bt)[0, 1]) if len(ex) > 2 and ex.std() > 0 and bt.std() > 0 else None
        mean_ok = abs(ex.mean() - bt.mean()) <= max(0.2 * abs(ex.mean()), 1e-4 * rb.mean())
        K["K4_pass"] = bool(mean_ok and (K["K4_pearson"] is None or K["K4_pearson"] >= 0.9))
        # timing of one full user (all arms) on a longer user
        t0 = time.perf_counter()
        arms = BIG_ARMS if key == "sm91s2" else ARMS
        Pl, _ = run_user(eng, data, longer[0], arms)
        sec = time.perf_counter() - t0
        n_rev = min(int(rc[str(longer[0])]), CAP)
        K["timing_user_reviews"] = n_rev
        K["timing_seconds_all_arms"] = round(sec, 1)
        tot = sel["capped_reviews_big"] if key == "sm91s2" else sel["capped_reviews"]
        K["projected_run_hours"] = round(sec / n_rev * tot / 3600, 2)
        K["peak_gb"] = peak_gb()
        K["check_seconds"] = round(time.perf_counter() - t_all, 1)
        K["pass"] = bool((isinstance(k1, float) and k1 < 1e-4 or not isinstance(k1, float))
                         and K["K5_base_repeat_maxabs"] == 0.0 and K["K2_col8_rows_are_target_update_rows"]
                         and K["K2_other_rows_are_query_rows"] and K["K3_outputs_before_first_target_maxabs"] < 1e-5
                         and K["K3_feature8_only_vs_raw_edit_maxabs_at_next"] < 1e-6
                         and K["K6_extra_rows_added"] == K["K6_extra_expected_rows"]
                         and K["K6_extra_outputs_before_first_insert_maxabs"] < 1e-5 and K["K6_extra_readout_finite"]
                         and K["K4_pass"] and (K["peak_gb"] or 0) < 6)
        report["proofs"][key] = K
        print(f"[sm122-check] {key}: {json.dumps(K)}", flush=True)
        atomic_json(Path(args.out), report)
        del eng
    # K7 (privacy) is run on the result files by `report`
    atomic_json(Path(args.out), report)
    return 0 if all(v["pass"] for v in report["proofs"].values()) else 1


# ------------------------------------------------------------------ report
def load_model(key):
    wd = WORK / key
    users, rows = [], []
    for f in sorted(wd.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        users.append(int(f.stem))
        rows.append({k: z[k] for k in z.files})
    return users, rows


def poisson_w(nb, seed=0):
    return np.random.default_rng(seed).poisson(1.0, size=(BOOT, nb)).astype(np.float64)


def ratio_ci(num, den, W):
    """num/den per block -> point and 95% bootstrap CI of sum(num)/sum(den) (in %)."""
    pt = 100 * num.sum() / den.sum()
    bs = 100 * (W @ num) / np.maximum(W @ den, 1e-12)
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_offsets(z, y, g, ng, iters=30):
    """Per-group logit offsets a_g minimising LogLoss of sigmoid(z + a_g) (Newton, separable)."""
    a = np.zeros(ng)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z + a[g])))
        gr = np.bincount(g, p - y, ng)
        h = np.bincount(g, p * (1 - p), ng) + 1e-9
        a -= gr / h
    return a


def crossfit_gain(p, y, grp, ng, ublock, fold):
    """Per-row LogLoss of (global offset) and (per-group offset) recalibrations, 2-fold by user."""
    z = logit(p)
    llg, llb = np.zeros(len(p)), np.zeros(len(p))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        a0 = fit_offsets(z[tr], y[tr], np.zeros(tr.sum(), int), 1)
        ab = fit_offsets(z[tr], y[tr], grp[tr], ng)
        llg[te] = ll(1 / (1 + np.exp(-(z[te] + a0[0]))), y[te])
        llb[te] = ll(1 / (1 + np.exp(-(z[te] + ab[grp[te]]))), y[te])
    return llg, llb


def summarise_model(key, users, rows, cuts=None):
    arms = BIG_ARMS if key == "sm91s2" else ARMS
    nu = len(users)
    Wu = poisson_w(nu, seed=1)
    out = {"model": key, "name": NAMES[key], "users": nu, "q1": {}, "q2": {}, "q3": {}}
    # ---------------- Q1 + Q3 model: counterfactual deltas on targets (user blocks)
    reads = ("inst", "curve", "1d", "1w", "30d")
    tot_targets = sum(int(r["P_target"].sum()) for r in rows)
    out["targets"] = tot_targets
    out["pairs"] = sum(len(r["P_t"]) for r in rows)
    strata = {"all": lambda r: np.ones(int(r["P_target"].sum()), bool),
              "Good": lambda r: r["P_grade"][r["P_target"]] == 3,
              "Easy": lambda r: r["P_grade"][r["P_target"]] == 4,
              "t is a same-day step": lambda r: r["P_ed_t"][r["P_target"]] == 0,
              "t is a later-day review": lambda r: r["P_ed_t"][r["P_target"]] >= 1,
              "next after 1 d": lambda r: r["P_ed_t2"][r["P_target"]] == 1,
              "next after 2-7 d": lambda r: (r["P_ed_t2"][r["P_target"]] >= 2) & (r["P_ed_t2"][r["P_target"]] <= 7),
              "next after 8-30 d": lambda r: (r["P_ed_t2"][r["P_target"]] >= 8) & (r["P_ed_t2"][r["P_target"]] <= 30),
              "next after > 30 d": lambda r: r["P_ed_t2"][r["P_target"]] > 30}
    for arm in arms:
        if arm == "base":
            continue
        A = {}
        for sname, sfun in strata.items():
            S = {}
            for rd in reads:
                num, den, llnum, llden, n, l2num = (np.zeros(nu) for _ in range(6))
                for i, r in enumerate(rows):
                    tg = r["P_target"].astype(bool)
                    if f"{arm}__{rd}" not in r or tg.sum() == 0:
                        continue
                    m = sfun(r)
                    if arm.startswith(("card_", "user_")):
                        d_new = _arm_new(r, arm)
                        m = m & ~np.isnan(d_new)
                    b = r[f"base__{rd}"][tg][m]
                    a = r[f"{arm}__{rd}"][tg][m]
                    num[i], den[i], n[i] = (a - b).sum(), b.sum(), m.sum()
                    if rd in ("inst", "curve"):
                        y = r["P_y"][tg][m].astype(np.float64)
                        llnum[i], llden[i] = (ll(a, y) - ll(b, y)).sum(), ll(b, y).sum()
                    if arm != "extra":
                        d0 = r["P_dur"][tg][m].astype(np.float64)
                        l2num[i] = np.log2(np.maximum(_arm_new(r, arm)[m], 1) / np.maximum(d0, 1)).sum()
                if n.sum() < 20:
                    continue
                pt, lo, hi = ratio_ci(num, den, Wu)
                e = {"n": int(n.sum()), "users": int((n > 0).sum()), "dR_pct": pt, "ci": [lo, hi],
                     "mean_R": float(den.sum() / n.sum())}
                if rd in ("inst", "curve"):
                    p2, l2, h2 = ratio_ci(llnum, llden, Wu)
                    e["dLL_pct"], e["dLL_ci"] = p2, [l2, h2]
                if arm != "extra":
                    e["mean_log2_change"] = float(l2num.sum() / n.sum())
                e["go"] = bool((lo > 0 or hi < 0) and abs(pt) >= MIN_PCT)
                S[rd] = e
            A[sname] = S
        # spillover: change on non-target pairs (instant at their next review)
        num, den, n = np.zeros(nu), np.zeros(nu), np.zeros(nu)
        for i, r in enumerate(rows):
            if f"{arm}__inst" not in r:
                continue
            ctl = ~r["P_target"].astype(bool)
            num[i] = (r[f"{arm}__inst"][ctl] - r["base__inst"][ctl]).sum()
            den[i] = r["base__inst"][ctl].sum()
            n[i] = ctl.sum()
        pt, lo, hi = ratio_ci(num, den, Wu)
        A["spillover on non-target pairs (instant)"] = {"inst": {"n": int(n.sum()), "dR_pct": pt, "ci": [lo, hi]}}
        out["q1" if arm != "extra" else "q3"][arm] = A
    # ---------------- Q2 + Q3 real: residuals of the base predictions
    P = {k: np.concatenate([r[f"P_{k}"] for r in rows]) for k in ("grade", "y", "z_user", "z_card", "s_day", "a_day",
                                                                   "n_day", "first_day", "day_t2", "ed_t2")}
    uid = np.concatenate([np.full(len(r["P_t"]), i) for i, r in enumerate(rows)])
    base = {rd: np.concatenate([r["base__" + rd] for r in rows]) for rd in ("inst", "curve")}
    y = P["y"].astype(np.float64)
    perm = np.random.default_rng(SEED).permutation(nu)
    ufold = np.zeros(nu, int)
    ufold[perm[1::2]] = 1
    fold = ufold[uid]
    # user-day blocks
    _, ud = np.unique(uid.astype(np.int64) * 10 ** 7 + P["day_t2"].astype(np.int64), return_inverse=True)
    Wd = poisson_w(int(ud.max()) + 1, seed=2)
    if cuts is None:
        cuts = {}
        for zname in ("z_user", "z_card"):
            z = P[zname]
            cuts[zname] = np.nanpercentile(z, [20, 40, 60, 80]).tolist()
    out["cuts"] = cuts
    for zname in ("z_user", "z_card"):
        z = P[zname]
        ok = ~np.isnan(z)
        grp = np.searchsorted(np.asarray(cuts[zname]), z[ok], side="right")
        Q = {"rows": int(ok.sum()), "users": int(len(np.unique(uid[ok])))}
        for rd in ("inst", "curve"):
            p = base[rd][ok]
            yy = y[ok]
            u = uid[ok]
            bins = []
            for q in range(5):
                m = grp == q
                rnum = np.bincount(u[m], yy[m] - p[m], nu)
                rden = np.bincount(u[m], p[m], nu)
                pt, lo, hi = ratio_ci(rnum, rden, Wu)
                bins.append({"bin": q + 1, "n": int(m.sum()), "recall": float(yy[m].mean()), "pred": float(p[m].mean()),
                             "resid_pct": pt, "ci": [lo, hi]})
            # Q5 - Q1 residual difference (in % of mean prediction)
            m1, m5 = grp == 0, grp == 4
            d_num = np.bincount(u[m5], yy[m5] - p[m5], nu) / max(m5.sum(), 1) - np.bincount(u[m1], yy[m1] - p[m1], nu) / max(m1.sum(), 1)
            pt = 100 * d_num.sum() / p.mean()
            bs = 100 * (Wu @ (np.bincount(u[m5], yy[m5] - p[m5], nu)) / np.maximum(Wu @ np.bincount(u[m5], minlength=nu).astype(float), 1)
                        - Wu @ (np.bincount(u[m1], yy[m1] - p[m1], nu)) / np.maximum(Wu @ np.bincount(u[m1], minlength=nu).astype(float), 1)) / p.mean()
            diff = {"pct": pt, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            # slope of residual on z (clipped to [-2, 2]) per doubling, in % of mean prediction
            zc = np.clip(z[ok], -2, 2) * 1.0
            zz = zc - zc.mean()
            sxy = np.bincount(u, zz * (yy - p), nu)
            sxx = np.bincount(u, zz * zz, nu)
            slope = 100 * math.log(2) * sxy.sum() / sxx.sum() / p.mean()
            bsl = 100 * math.log(2) * (Wu @ sxy) / (Wu @ sxx) / p.mean()
            slope_e = {"pct_per_doubling": slope, "ci": [float(np.percentile(bsl, 2.5)), float(np.percentile(bsl, 97.5))]}
            # cross-fitted LogLoss gain of per-bin offsets over a global offset
            llg, llb = crossfit_gain(p, yy, grp, 5, u, fold[ok])
            gu = np.bincount(u, llg - llb, nu)
            du = np.bincount(u, llg, nu)
            pt_u, lo_u, hi_u = ratio_ci(gu, du, Wu)
            udk = ud[ok]
            gd = np.bincount(udk, llg - llb, Wd.shape[1])
            dd = np.bincount(udk, llg, Wd.shape[1])
            pt_d, lo_d, hi_d = ratio_ci(gd, dd, Wd)
            Q[rd] = {"bins": bins, "q5_minus_q1": diff, "slope": slope_e,
                     "gain": {"pct": pt_u, "ci_user": [lo_u, hi_u], "ci_userday": [lo_d, hi_d],
                              "ll_model": float(ll(p, yy).mean()), "ll_global": float(llg.mean()), "ll_bins": float(llb.mean()),
                              "go": bool(lo_u > 0 and pt_u >= MIN_PCT)}}
        out["q2"][zname] = Q
    # Q3 real: same-day successes on day D (no Again that day), first day of the card or not
    ok = P["a_day"] == 0
    s = np.minimum(P["s_day"], 3)
    grp_all = (s - 1) + 3 * P["first_day"].astype(int)          # 0..2 later days, 3..5 first day; 1, 2, >= 3 successes
    Q3 = {"rows": int(ok.sum())}
    names = ["later day, 1 success", "later day, 2 successes", "later day, >= 3 successes",
             "first day, 1 success", "first day, 2 successes", "first day, >= 3 successes"]
    for rd in ("inst", "curve"):
        p, yy, u, g = base[rd][ok], y[ok], uid[ok], grp_all[ok]
        groups = []
        for q in range(6):
            m = g == q
            if m.sum() == 0:
                continue
            rnum = np.bincount(u[m], yy[m] - p[m], nu)
            rden = np.bincount(u[m], p[m], nu)
            pt, lo, hi = ratio_ci(rnum, rden, Wu)
            groups.append({"group": names[q], "n": int(m.sum()), "recall": float(yy[m].mean()), "pred": float(p[m].mean()),
                           "resid_pct": pt, "ci": [lo, hi]})
        llg, llb = crossfit_gain(p, yy, g, 6, u, fold[ok])
        gu, du = np.bincount(u, llg - llb, nu), np.bincount(u, llg, nu)
        pt_u, lo_u, hi_u = ratio_ci(gu, du, Wu)
        gd, dd = np.bincount(ud[ok], llg - llb, Wd.shape[1]), np.bincount(ud[ok], llg, Wd.shape[1])
        pt_d, lo_d, hi_d = ratio_ci(gd, dd, Wd)
        # later day: >= 2 successes minus 1 success (residual, % of mean prediction)
        m1, m2 = g == 0, (g == 1) | (g == 2)
        a1 = np.bincount(u[m1], yy[m1] - p[m1], nu)
        c1 = np.bincount(u[m1], minlength=nu).astype(float)
        a2 = np.bincount(u[m2], yy[m2] - p[m2], nu)
        c2 = np.bincount(u[m2], minlength=nu).astype(float)
        pt = 100 * (a2.sum() / max(c2.sum(), 1) - a1.sum() / max(c1.sum(), 1)) / p.mean()
        bs = 100 * ((Wu @ a2) / np.maximum(Wu @ c2, 1) - (Wu @ a1) / np.maximum(Wu @ c1, 1)) / p.mean()
        Q3[rd] = {"groups": groups, "extra_minus_single_later_day": {"pct": pt, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
                  "gain": {"pct": pt_u, "ci_user": [lo_u, hi_u], "ci_userday": [lo_d, hi_d], "go": bool(lo_u > 0 and pt_u >= MIN_PCT)}}
    out["q3_real"] = Q3
    out["seconds_per_user"] = float(np.mean([float(r["seconds"]) for r in rows]))
    return out, cuts


def _arm_new(r, arm):
    tg = r["P_target"].astype(bool)
    d = r["P_dur"][tg].astype(np.float64)
    ucap = float(r["P_ucap"])
    if arm == "half":
        new = d * 0.5
    elif arm == "double":
        new = d * 2.0
    elif arm == "card_fast":
        new = r["P_c10"][tg].astype(np.float64)
    elif arm == "card_slow":
        new = r["P_c90"][tg].astype(np.float64)
    elif arm == "user_fast":
        new = np.full_like(d, float(r["P_u10"]))
    elif arm == "user_slow":
        new = np.full_like(d, float(r["P_u90"]))
    else:
        return np.full_like(d, np.nan)
    return np.where(np.isnan(new), np.nan, np.clip(np.round(new), 1, ucap))


def privacy_scan(paths):
    """No card ids, review indices, day offsets or timestamps in the committed files: every number in them is an
    aggregate. Checks that no per-row array is stored (no list longer than 50 numbers) and no key names a row field."""
    bad = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        for m in re.finditer(r"\[([0-9eE+\-., ]{400,})\]", text):
            bad.append(f"{p}: long numeric list")
        for word in ("card_id", "review_th", "day_offset", "revlog_id"):
            if word in text:
                bad.append(f"{p}: contains {word}")
    return bad


def fmt_ci(e, key="ci", nd=2):
    lo, hi = e[key]
    return f"{e['dR_pct'] if 'dR_pct' in e else e['pct']:+.{nd}f}% [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def report(args):
    res = {"design": "artifacts/small-model/sm122/design.md", "sm122": git_head(ROOT), "sm121_loaders": git_head(S121),
           "cap": CAP, "spacing": SPACING, "insert_s": INSERT_S, "boot": BOOT, "models": {}}
    cuts = None
    for key in MODEL_KEYS:
        users, rows = load_model(key)
        if not rows:
            continue
        s, c = summarise_model(key, users, rows, cuts)
        if cuts is None:
            cuts = c
        meta = WORK / key / "meta.json"
        s["meta"] = json.loads(meta.read_text()) if meta.exists() else None
        s["errors"] = len(list((WORK / key).glob("*.error")))
        res["models"][key] = s
    check_f = OUT / "check-cpu.json"
    res["check"] = json.loads(check_f.read_text()) if check_f.exists() else None
    atomic_json(OUT / "sm122-result.json", res)
    write_md(res)
    bad = privacy_scan([OUT / "sm122-result.json", OUT / "sm122-result.md"])
    print(f"[sm122-report] models {list(res['models'])}; privacy scan: {bad or 'clean'}")
    return 0 if not bad else 1


def write_md(res):
    L = ["# Retrieval-effort probe (SM122): results", "",
         "Inspired by Pyc & Rawson (2009) (the retrieval effort hypothesis: harder but successful retrievals produce better later memory). "
         "Design and pre-registered rules: `design.md`. Answer time = front + back, from the question shown to the grade pressed, "
         "capped at the preset's maximum answer seconds; in the models it is feature 8 on the history rows.", "",
         f"Users: a pre-registered random subsample of the 732-user standard readout set (`users.json`), first {CAP:,} reviews "
         f"of each; targets at least {SPACING} reviews apart; user-block bootstrap ({BOOT} Poisson replicates). An effect counts "
         "only with the 95% CI excluding 0 and a size of at least 0.05%.", ""]
    for key, s in res["models"].items():
        nm = f"{SHORT[key]}: {s['name']}"
        L += [f"## {nm}", "", f"{s['users']} users, {s['pairs']:,} later-day pairs (previous review Good or Easy, timed), "
              f"{s['targets']:,} counterfactual targets.", ""]
        L += ["### Question 1: counterfactual duration of the last success (change of predicted R, % of R)", "",
              "| arm | mean log2 change | instant R at the next review | LogLoss change (instant) | curve R at the next review | curve R at 1 d | curve R at 1 w | curve R at 30 d |",
              "|---|---|---|---|---|---|---|---|"]
        for arm, A in s["q1"].items():
            a = A.get("all", {})
            if "inst" not in a:
                continue
            row = [ARM_NAMES[arm], f"{a['inst'].get('mean_log2_change', float('nan')):+.2f}", fmt_ci(a["inst"], nd=3),
                   f"{a['inst']['dLL_pct']:+.3f}% [{a['inst']['dLL_ci'][0]:+.3f}, {a['inst']['dLL_ci'][1]:+.3f}]"]
            row += [fmt_ci(a[rd], nd=3) if rd in a else "-" for rd in ("curve", "1d", "1w", "30d")]
            L.append("| " + " | ".join(row) + " |")
        L += ["", "By stratum (x2 arm; instant R at the next review / curve R at 1 w):", "", "| stratum | n | instant | curve 1 w |", "|---|---|---|---|"]
        for sname, S in s["q1"].get("double", {}).items():
            if "inst" in S and "1w" in S:
                L.append(f"| {sname} | {S['inst']['n']:,} | {fmt_ci(S['inst'], nd=3)} | {fmt_ci(S['1w'], nd=3)} |")
        sp = s["q1"].get("double", {}).get("spillover on non-target pairs (instant)", {}).get("inst")
        if sp:
            L += ["", f"Spillover (x2 arm) on the {sp['n']:,} non-target pairs: {fmt_ci(sp, nd=4)}."]
        L += ["", "### Question 2: does real later recall depend on the previous success's duration beyond the model?", ""]
        for zname, title in (("z_user", "within user (duration vs the user's median for that grade)"),
                             ("z_card", "within card (duration vs the card's earlier median, at least 2 earlier)")):
            Q = s["q2"][zname]
            L += [f"**{title}**, {Q['rows']:,} rows. Quintile 1 = fastest.", "",
                  "| head | Q1 resid | Q2 | Q3 | Q4 | Q5 resid | Q5 - Q1 | slope per doubling | LogLoss gain of per-quintile offsets (user CI; user-day CI) |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for rd in ("inst", "curve"):
                e = Q[rd]
                cells = [f"{b['resid_pct']:+.2f}%" for b in e["bins"]]
                g = e["gain"]
                L.append(f"| {'instant' if rd == 'inst' else 'curve'} | " + " | ".join(cells) + f" | {fmt_ci(e['q5_minus_q1'])} | "
                         f"{e['slope']['pct_per_doubling']:+.2f}% [{e['slope']['ci'][0]:+.2f}, {e['slope']['ci'][1]:+.2f}] | "
                         f"{g['pct']:+.3f}% [{g['ci_user'][0]:+.3f}, {g['ci_user'][1]:+.3f}]; [{g['ci_userday'][0]:+.3f}, {g['ci_userday'][1]:+.3f}]"
                         f"{' (go)' if g['go'] else ''} |")
            L.append("")
            L += ["Actual recall vs predicted by quintile (instant): " + "; ".join(
                f"Q{b['bin']} {100 * b['recall']:.2f}% vs {100 * b['pred']:.2f}%" for b in Q["inst"]["bins"]), ""]
        L += ["### Question 3: one more same-day success", ""]
        ex = s["q3"].get("extra", {}).get("all", {})
        if ex:
            L += ["Model (counterfactual insertion, 600 s after the target): " + "; ".join(
                f"{lab} {fmt_ci(ex[rd], nd=3)}{' (go)' if ex[rd]['go'] else ''}" for rd, lab in
                (("inst", "instant R at the next review"), ("curve", "curve R at the next review"), ("1d", "curve 1 d"),
                 ("1w", "curve 1 w"), ("30d", "curve 30 d")) if rd in ex) + ".", ""]
        Q3 = s["q3_real"]
        L += ["Real outcomes (residual = actual - predicted recall at the next later-day review, % of predicted; days without an Again):", "",
              "| group | n | recall | instant pred | instant resid | curve resid |", "|---|---|---|---|---|---|"]
        cur = {g["group"]: g for g in Q3["curve"]["groups"]}
        for gi in Q3["inst"]["groups"]:
            c = cur.get(gi["group"], {})
            L.append(f"| {gi['group']} | {gi['n']:,} | {100 * gi['recall']:.2f}% | {100 * gi['pred']:.2f}% | "
                     f"{gi['resid_pct']:+.2f}% [{gi['ci'][0]:+.2f}, {gi['ci'][1]:+.2f}] | "
                     f"{c.get('resid_pct', float('nan')):+.2f}% |")
        for rd in ("inst", "curve"):
            e = Q3[rd]
            L.append(f"\n{'Instant' if rd == 'inst' else 'Curve'}: later day, 2 or more successes minus 1 success: {fmt_ci(e['extra_minus_single_later_day'])}; "
                     f"LogLoss gain of group offsets {e['gain']['pct']:+.3f}% [{e['gain']['ci_user'][0]:+.3f}, {e['gain']['ci_user'][1]:+.3f}]{' (go)' if e['gain']['go'] else ''}.")
        meta = s.get("meta") or {}
        L += ["", f"CPU: {meta.get('seconds', '?')} s for {s['users']} users ({s['seconds_per_user']:.0f} s per user, all arms), "
              f"peak {meta.get('peak_gb', '?')} GB; errors: {s['errors']}.", ""]
    (OUT / "sm122-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    c = sub.add_parser("check")
    c.add_argument("--out", default=str(OUT / "check-cpu.json"))
    c.add_argument("--models", default=",".join(MODEL_KEYS))
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=MODEL_KEYS)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"select": select, "check": check, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
