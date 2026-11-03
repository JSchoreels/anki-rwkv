#!/usr/bin/env python
"""The small model in the review simulator (SM41), stage 2 part 1: the drift study (CPU).

Free running = the learner (the released leak-free small model + the answer-time head on richer frozen features
(SM82c)) reads its own sampled grades and answer times. Stage 1 found that it drifts. This script measures the drift
on the REAL schedule (same cards, days and review order as the real user, so every lane sees the same reviews) and
tests mitigations, all lanes of one split advancing together in one batch (lock-step replay: B lanes = one batched
model step per review, states stacked along the batch dimension, common random numbers: every lane draws the same 3
uniforms per review whatever it needs, lanes of the same seed share them).

Lanes (ARMS): tf = teacher forcing (real grades and times); fr-* = free running (non-first grades sampled from the
instant head, later-day answer times from the answer-time head; first-review grades/times and same-day times stay
real, as in stage 1's real-fr lane) with one mitigation each. See ARMS and artifacts/small-model/sm41/DESIGN.md.

Interval convention (the duration leak, duration-leak scoring (SM31)): the training data's elapsed_seconds runs from
the card's previous answer to THIS answer, so it contains this review's own answer time, in the review (commit) rows
and in the instant (query) rows alike. The app queries when the card is shown. conv "show" (default): query rows at
show time (the lane's own previous answer to the real show moment, floor, >= 0, cumulative = earlier intervals + it:
exactly the SM31 e2s-query inputs for teacher forcing), commit rows answer-to-answer with the lane's own answer time
(the training convention); conv "answer": every row with the data's elapsed seconds (stage 1; in a free-running lane
the query row then contains the REAL answer time).

  select   pick the users and split days (the 732 set, seed fixed) -> artifacts/small-model/sm41/drift/users.json
  run      --user U --splits D1,D2 --days H   one warm-up pass, every split's lanes; resumable (a split already in
           --out is skipped); holds new splits in quiet.json windows (from 10 min before)
  proof    reproducibility (same seed -> same lanes), batched == unbatched, teacher forcing == stage 1
  report   the drift tables -> drift/drift-tables.md, drift/drift-result.json
"""
from argparse import ArgumentParser
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RWKV_NO_JIT", "1")
from scripts import sm41_sim as S   # noqa: E402
from rwkv import data_processing as DP   # noqa: E402

OUT = S.HERE / "drift"
DAY = S.SECONDS_PER_DAY
BUCKET = 15                      # days per bucket
SHARED = ("deck_id", "preset_id", "user_id")   # the streams shared across cards (re-anchoring target)

# arm -> settings. conv: interval convention; grade_temp / time_temp: sampling temperatures (logits / T);
# time: "sample" | "expected" (exp of the expected ln time of the chosen grade) | "real"; anchor_days k: every k-th
# day (days k-1, 2k-1, ...) teacher forced (real grades and times); anchor_shared k: at the start of every k-th day the
# deck / preset / user states are replaced by the teacher-forced lane's (card and note states stay the lane's own).
ARMS = {
    "tf": dict(teacher=True),
    "tf-answer": dict(teacher=True, conv="answer"),
    "fr": dict(),
    "fr-answer": dict(conv="answer"),
    "fr-treal": dict(time="real"),
    "fr-texp": dict(time="expected"),
    "fr-tt0.5": dict(time_temp=0.5),
    "fr-gt0.8": dict(grade_temp=0.8),
    "fr-anchor7": dict(anchor_days=7),
    "fr-shared1": dict(anchor_shared=1),
    "fr-shared7": dict(anchor_shared=7),
    "fr-texp-shared1": dict(time="expected", anchor_shared=1),
}
DEFAULT_ARMS = ",".join(ARMS)


def arm_cfg(name):
    c = dict(teacher=False, conv="show", grade_temp=1.0, time_temp=1.0, time="sample", anchor_days=0, anchor_shared=0)
    c.update(ARMS[name])
    return c


# ------------------------------------------------------------------ CPU break
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
LEAD = dt.timedelta(minutes=10)   # no new split within 10 min of a window start (a split takes a few minutes)


def hold_reason(now=None):
    """Why new work must wait now (None = clear): a quiet.json window of scope all / cpu (re-read at every check; the
    user's breaks live there, e.g. 2026-11-03 17:30-19:30 and 20:00-01:00 from 11-04 on) that is running or starts
    within LEAD. If quiet.json cannot be read, the default break 19:50-01:00 applies."""
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
            print(f"[sm41-drift] {tag}: holding ({r}) at {dt.datetime.now():%m-%d %H:%M}", flush=True)
            said = True
        time.sleep(300)


# ------------------------------------------------------------------ batched model step
def _expand(st, B):
    if st is None:
        return None
    if isinstance(st, tuple):
        return tuple(_expand(t, B) for t in st)
    return st if st.shape[0] == B else st.expand(B, *st.shape[1:])


def _where(st, mask, B):
    """Rows of `mask` take row 0 (the teacher-forced lane)."""
    if st is None:
        return None
    if isinstance(st, tuple):
        return tuple(_where(t, mask, B) for t in st)
    import torch
    if st.shape[0] != B:
        return st
    m = mask.view(B, *([1] * (st.dim() - 1)))
    return torch.where(m, st[0:1].expand_as(st), st)


def step_batch(model, feat, states, snaps=False):
    """sm41_sim.Model.step for B rows (feat [B, 92]); states[stream] = per-layer dict with batch-1 or batch-B tensors.
    Returns (p [B,4], (w, s, d) or None, next states, frozen [B,448] or None)."""
    torch = model.torch
    m = model.rnn
    B = feat.shape[0]
    model.steps += B
    with torch.inference_mode():
        x = torch.from_numpy(feat)
        if m.zero_features:
            x = x.clone()
            x[:, m.zero_features] = 0
        x = m.features2card(x)
        shots = [x] if snaps else None
        nxt = {s: (dict(states[s]) if states.get(s) is not None else {}) for s in m.streams}
        v0 = [None] * len(m.streams)
        ls = m.layer_steps
        for k, (i, layer) in enumerate(ls):
            s = m.streams[i]
            x, v0[i], nxt[s][layer] = m.rwkv_modules[i].blocks[layer](
                in_BC=x, v0_BC=v0[i] if v0[i] is not None else torch.empty_like(x), state=_expand(nxt[s].get(layer), B))
            if snaps and (k + 1 == len(ls) or ls[k + 1][1] != layer):
                shots.append(x)
        h = m.prehead_norm(x)
        hp = m.head_p(h).float()
        p = torch.softmax(m.p_linear(hp), -1).numpy().astype(np.float64)
        frozen = None
        if snaps:
            frozen = torch.cat([torch.nn.functional.layer_norm(t.float(), (64,)) for t in shots] + [hp], -1)
    return p, nxt, frozen


def time_probs_batch(model, frozen, explicit):
    torch = model.torch
    with torch.inference_mode():
        z = torch.cat([frozen, torch.from_numpy(explicit)], -1)
        return torch.softmax(model.head(z).view(-1, 4, 16), -1).numpy().astype(np.float64)


def es_cols(es, cum):
    """Columns 2..7 (elapsed seconds, its sin/cos, the cumulative interval, its sin/cos) for arrays es, cum."""
    es = np.asarray(es, np.float64); cum = np.asarray(cum, np.float64)
    out = np.empty((len(es), 6), np.float32)
    out[:, 0] = DP.scale_elapsed_seconds(es)
    out[:, 1] = np.sin((es % DAY) * 2 * np.pi / DAY)
    out[:, 2] = np.cos((es % DAY) * 2 * np.pi / DAY)
    out[:, 3] = DP.scale_elapsed_seconds_cumulative(cum)
    out[:, 4] = np.sin((cum % DAY) * 2 * np.pi / DAY)
    out[:, 5] = np.cos((cum % DAY) * 2 * np.pi / DAY)
    return out


def sample_cat(p, u):
    """Inverse-CDF draw per row: p [B, K] (rows sum to 1), u [B] -> index [B]."""
    c = np.cumsum(p, -1)
    c[:, -1] = np.inf
    return (u[:, None] >= c).sum(-1)


def temper(p, T):
    """Rows of p at temperature T [B] (logits / T)."""
    if np.all(T == 1.0):
        return p
    q = np.power(np.maximum(p, 1e-300), 1.0 / T[:, None])
    return q / q.sum(-1, keepdims=True)


# ------------------------------------------------------------------ one split, every lane
class Warm:
    """The warm-up: every real review up to a split, one row at a time (teacher forced, the training convention)."""

    def __init__(self, model, rev, user):
        self.model, self.rev, self.user = model, rev, user
        self.builder = S.Builder(day0=int(rev["day_offset"][0]), seed=[41, user])
        self.explicit = S.Explicit()
        self.states = {s: {} for s in model.streams}
        self.i = 0
        self.ids_of = {}
        for j in range(len(rev["card_id"])):
            c = int(rev["card_id"][j])
            self.ids_of.setdefault(c, (c, int(rev["note_id"][j]), int(rev["deck_id"][j]), int(rev["preset_id"][j])))
        self.seconds = 0.0

    def advance(self, end):
        rev, m = self.rev, self.model
        t = time.perf_counter()
        for i in range(self.i, end):
            c = int(rev["card_id"][i])
            ids = self.ids_of[c]
            _, f = self.builder.row(*ids, int(rev["day_offset"][i]), int(rev["elapsed_days"][i]),
                                    int(rev["elapsed_seconds"][i]), int(rev["rating"][i]), int(rev["duration"][i]))
            keys = dict(zip(m.streams, (ids[0], ids[2], ids[1], ids[3], 0)))
            _, nxt, _ = step_batch(m, f.reshape(1, 92), {s: self.states[s].get(keys[s]) for s in m.streams})
            for s in m.streams:
                self.states[s][keys[s]] = nxt[s]
            self.explicit.push(c, int(rev["rating"][i]), int(rev["duration"][i]))
        self.seconds += time.perf_counter() - t
        self.i = end


def run_split(model, warm, habits, D, N, lanes, seeds_n, base_seed):
    """Every lane of one split over [D, D+N) on the real schedule, lock-step. lanes: list of (arm, seed index or -1).
    Returns the result dict (per-lane counters by bucket)."""
    rev = warm.rev
    day = rev["day_offset"]
    end = int(np.searchsorted(day, D, side="left"))
    stop = int(np.searchsorted(day, D + N, side="left"))
    assert warm.i == end
    B = len(lanes)
    cfg = [arm_cfg(a) for a, _ in lanes]
    import torch
    teacher = np.array([c["teacher"] for c in cfg])
    answer = np.array([c["conv"] == "answer" for c in cfg])
    gT = np.array([c["grade_temp"] for c in cfg], np.float64)
    tT = np.array([c["time_temp"] for c in cfg], np.float64)
    tmode = [c["time"] for c in cfg]
    t_real = np.array([m == "real" for m in tmode]); t_exp = np.array([m == "expected" for m in tmode])
    a_days = np.array([c["anchor_days"] for c in cfg]); a_shared = np.array([c["anchor_shared"] for c in cfg])
    assert not a_shared.any() or (lanes[0][0] == "tf"), "shared-stream anchoring needs the tf lane first"
    seed_of = np.array([max(s, 0) for _, s in lanes])
    rng = np.random.default_rng([41, warm.user, D, N, base_seed])
    m = model
    states = {s: dict(v) for s, v in warm.states.items()}
    builder = copy.deepcopy(warm.builder)
    expl = [copy.deepcopy(warm.explicit) for _ in range(B)]
    old = {int(c) for c in rev["card_id"][:end]}
    mid = (m.lo + m.hi) / 2
    # per card: lane cumulative interval [B], lane minus real previous answer time (s) [B]
    cum, dprev = {}, {}
    nb = (N + BUCKET - 1) // BUCKET
    z = lambda *sh: np.zeros((B, nb) + sh)   # noqa: E731
    acc = {"later_old": z(3), "later_new": z(3), "first_touch": z(3),   # n, pass, sum p(pass)
           "grades_same": z(4), "grades_later": z(4),
           "time_later": z(4),      # n, sum seconds, sum ln seconds, sum model-expected seconds (chosen grade)
           "implied_follow": z(2),  # same-day follow-ups implied by the user's step rules: n reviews, sum P(follow)
           "grade_ll": z(4)}        # tf diagnostics: sum -ln p(real grade) later, n later, same-day sum, n same
    seen = set()
    rules = habits["rules"]
    t0 = time.perf_counter()
    cur_day = None
    for i in range(end, stop):
        c = int(rev["card_id"][i])
        ids = warm.ids_of[c]
        d = int(day[i]) - D
        b = d // BUCKET
        if d != cur_day:
            cur_day = d
            sh = (a_shared > 0) & (d % np.maximum(a_shared, 1) == 0)
            if sh.any():
                mask = torch.from_numpy(sh)
                for s in SHARED:
                    for key, st in states[s].items():
                        states[s][key] = {layer: _where(v, mask, B) for layer, v in st.items()}
            tf_day = teacher | ((a_days > 0) & (d % np.maximum(a_days, 1) == np.maximum(a_days, 1) - 1))
        ed, es = int(rev["elapsed_days"][i]), int(rev["elapsed_seconds"][i])
        g_r, u_r = int(rev["rating"][i]), int(rev["duration"][i])
        kind = 0 if ed == -1 else (1 if ed == 0 else 2)
        st_b = builder.cards.get(c)
        prev_cum = float(st_b[3]) if st_b is not None else 0.0
        q, f = builder.row(c, ids[1], ids[2], ids[3], int(day[i]), ed, es, g_r, u_r)
        U = rng.random((seeds_n, 3))[seed_of]
        keys = dict(zip(m.streams, (ids[0], ids[2], ids[1], ids[3], 0)))
        cur = {s: states[s].get(keys[s]) for s in m.streams}
        F = np.repeat(f[None], B, 0)
        lane_cum = cum.get(c)
        if lane_cum is None:
            lane_cum = np.full(B, prev_cum)
        dp = dprev.get(c, np.zeros(B))
        if kind == 0:
            rating = np.full(B, g_r); dur = np.full(B, u_r)
            c_es = np.full(B, float(es))
        else:
            x = es - u_r / 1000.0 + dp
            q_es = np.where(answer, float(es), np.floor(np.maximum(x, 0.0)))
            Q = np.repeat(q[None], B, 0)
            Q[:, 2:8] = np.where(answer[:, None], Q[:, 2:8], es_cols(q_es, lane_cum + q_es))
            if not np.isfinite(Q).all():
                raise FloatingPointError(f"non-finite query row: user {warm.user} split {D} review {i}")
            p, _, frozen = step_batch(m, Q, cur, snaps=kind == 2)
            p = p / p.sum(-1, keepdims=True)
            g_s = sample_cat(temper(p, gT), U[:, 0]) + 1
            rating = np.where(tf_day, g_r, g_s)
            if kind == 2:
                ex = np.stack([e.features(c, ed) for e in expl])
                tp = time_probs_batch(m, frozen, ex)
                pb = tp[np.arange(B), rating - 1]                       # [B, 16] of the lane's grade
                pbs = temper(pb / pb.sum(-1, keepdims=True), tT)
                bins = sample_cat(pbs, U[:, 1])
                lo, hi = m.lo[bins], m.hi[bins]
                ms_s = np.where(bins == 15, 60000, np.round(1000 * np.exp(lo + U[:, 2] * (hi - lo))))
                ms_e = np.round(1000 * np.exp((pb * mid).sum(-1)))
                dur = np.where(tf_day | t_real, u_r, np.where(t_exp, ms_e, ms_s)).astype(np.int64)
                acc["time_later"][:, b, 3] += (pb * np.exp(mid)).sum(-1)
                ppass = 1.0 - p[:, 0]
                tgt = acc["later_old"] if c in old else acc["later_new"]
                tgt[:, b, 0] += 1; tgt[:, b, 1] += rating > 1; tgt[:, b, 2] += ppass
                if c not in seen and c in old:
                    acc["first_touch"][:, b, 0] += 1; acc["first_touch"][:, b, 1] += rating > 1
                    acc["first_touch"][:, b, 2] += ppass
                acc["grades_later"][np.arange(B), b, rating - 1] += 1
                acc["time_later"][:, b, 0] += 1; acc["time_later"][:, b, 1] += dur / 1000.0
                acc["time_later"][:, b, 2] += np.log(np.maximum(dur, 1) / 1000.0)
                acc["grade_ll"][:, b, 0] += -np.log(np.maximum(p[:, g_r - 1], 1e-7)); acc["grade_ll"][:, b, 1] += 1
            else:
                dur = np.full(B, u_r)
                acc["grades_same"][np.arange(B), b, rating - 1] += 1
                acc["grade_ll"][:, b, 2] += -np.log(np.maximum(p[:, g_r - 1], 1e-7)); acc["grade_ll"][:, b, 3] += 1
            # commit row: answer-to-answer with the lane's own answer time (== the data's for teacher forcing). A lane
            # whose previous answer of this card ran past the real show moment (x < 0: e.g. a long sampled time before
            # a short real same-day step) is shown when that answer ends: commit = max(x, 0) + its own answer time
            c_es = np.round(es + dp + (dur - u_r) / 1000.0)
            differs = (dp != 0) | (dur != u_r)
            c_es = np.where(differs, np.maximum(c_es, np.round(np.maximum(x, 0.0) + dur / 1000.0)), c_es)
            c_es = np.where(answer, float(es), c_es)
        seen.add(c)
        acc["implied_follow"][np.arange(B), b, 0] += 1
        acc["implied_follow"][:, b, 1] += np.array([rules[(kind, int(g))][0] for g in rating])
        new_cum = lane_cum + c_es
        F[:, 2:8] = np.where(answer[:, None], F[:, 2:8], es_cols(c_es, new_cum))
        F[:, 8] = DP.scale_duration(dur.astype(np.float64)).astype(np.float32)
        F[:, 9:13] = 0.0
        F[np.arange(B), 8 + rating] = 1.0
        if not np.isfinite(F).all():
            raise FloatingPointError(f"non-finite review row: user {warm.user} split {D} review {i} lanes "
                                     f"{np.where(~np.isfinite(F).all(1))[0].tolist()}")
        _, nxt, _ = step_batch(m, F, cur)
        for s in m.streams:
            states[s][keys[s]] = nxt[s]
        for e, g, u in zip(expl, rating, dur):
            e.push(c, int(g), int(u))
        cum[c] = np.where(answer, prev_cum + es, new_cum)
        dprev[c] = np.where(answer, 0.0, (u_r - dur) / 1000.0)
    wall = time.perf_counter() - t0
    reals = {"reviews": stop - end, "active_days": int(len(np.unique(day[end:stop]))),
             "first": int((rev["elapsed_days"][end:stop] == -1).sum())}
    return {"user": warm.user, "split_day": D, "days": N, "warmup_reviews": end, "window_reviews": stop - end,
            "lanes": [f"{a}#{s}" if s >= 0 else a for a, s in lanes], "real": reals, "bucket_days": BUCKET,
            "acc": {k: v.round(6).tolist() for k, v in acc.items()}, "wall_seconds": round(wall, 2),
            "lane_reviews_per_second": round(B * (stop - end) / max(wall, 1e-9), 1),
            "ticks_per_second": round((stop - end) / max(wall, 1e-9), 1)}


def lane_list(arms, seeds):
    out = []
    for a in arms.split(","):
        if ARMS[a].get("teacher"):
            out.append((a, -1))
        else:
            out += [(a, s) for s in range(seeds)]
    return out


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["user"], r["split_day"], r["days"]))
    return keys


def run(args):
    S.low_priority()
    user = args.user
    splits = sorted(int(x) for x in args.splits.split(","))
    todo = [D for D in splits if (user, D, args.days) not in done_keys(args.out)]
    if not todo:
        print(f"[sm41-drift] user {user}: all splits done", flush=True)
        return
    wait_clear(f"user {user}")
    rev, _ = S.load_user(user)
    model = S.Model(threads=args.threads)
    warm = Warm(model, rev, user)
    lanes = lane_list(args.arms, args.seeds)
    for D in todo:
        end = int(np.searchsorted(rev["day_offset"], D, side="left"))
        wait_clear(f"user {user} split {D}")
        warm.advance(end)
        habits = S.user_habits(rev, end)
        res = run_split(model, warm, habits, D, args.days, lanes, args.seeds, args.seed)
        res["warmup_seconds"] = round(warm.seconds, 1)
        res["good_only_window"] = None
        res["finished"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[sm41-drift] user {user} split {D} days {args.days}: {res['window_reviews']} reviews x {len(lanes)} "
              f"lanes in {res['wall_seconds']} s ({res['lane_reviews_per_second']} lane reviews/s); warm-up "
              f"{end} reviews {warm.seconds:.0f} s", flush=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


# ------------------------------------------------------------------ user selection
def select(args):
    import pandas as pd
    from scripts import compress_common as cc
    data = cc.data_path()
    pool = json.loads((S.HERE.parent / "eval-users-732.json").read_text())["users"]
    rng = np.random.default_rng(args.seed)
    order = [int(u) for u in rng.permutation(sorted(int(u) for u in pool))]
    H, LH = args.days, args.long_days
    chosen, longs, looked = [], [], 0
    for u in order:
        if len(chosen) >= args.users:
            break
        looked += 1
        df = pd.read_parquet(data / "revlogs" / f"user_id={u}", columns=["day_offset", "elapsed_days", "rating", "duration",
                                                                         "card_id"])
        n = len(df)
        if not 8000 <= n <= 60000 or (df["duration"] > 0).mean() < 0.95:
            continue
        day = df["day_offset"].to_numpy(); ed = df["elapsed_days"].to_numpy(); g = df["rating"].to_numpy()
        card = df["card_id"].to_numpy()
        if ((g[ed >= 1] > 1).mean() if (ed >= 1).any() else 1.0) >= 0.99:
            continue

        def window_ok(D, L, lo_rev, hi_rev, min_days):
            a = int(np.searchsorted(day, D)); b = int(np.searchsorted(day, D + L))
            if a < 3000 or b >= n and day[-1] < D + L - 1:
                return None
            if not lo_rev <= b - a <= hi_rev or len(np.unique(day[a:b])) < min_days:
                return None
            later = ed[a:b] >= 1
            if later.sum() < 300 or (g[a:b][later] > 1).mean() >= 0.995 or (g[a:b][later] == 1).sum() < 30:
                return None
            pre = set(card[:a].tolist()); seen, ft = set(), 0
            for c, e in zip(card[a:b].tolist(), ed[a:b].tolist()):
                if c not in seen:
                    seen.add(c)
                    ft += int(e >= 1 and c in pre)
            if ft < 100:
                return None
            return {"split_day": int(D), "window_reviews": int(b - a), "pre_reviews": a,
                    "active_days": int(len(np.unique(day[a:b]))), "first_touch": ft,
                    "retention_later": round(float((g[a:b][later] > 1).mean()), 4)}
        cands = list(range(int(day[0]) + 30, int(day[-1]) - H + 2, 15))
        rng.shuffle(cands)
        picked = []
        for D in cands:
            if len(picked) >= 3:
                break
            if any(abs(D - p["split_day"]) < H for p in picked):
                continue
            w = window_ok(D, H, 1500, 9000, 40)
            if w:
                picked.append(w)
        if len(picked) < 2:
            continue
        picked.sort(key=lambda w: w["split_day"])
        chosen.append({"user": u, "reviews": n, "splits": picked})
        if len(longs) < args.long_users:
            for D in cands:
                w = window_ok(D, LH, 3000, 15000, 80)
                if w:
                    longs.append({"user": u, "reviews": n, "split": w})
                    break
    out = {"rule": (f"the 732 set in numpy default_rng({args.seed}) order; users with 8k-60k reviews, >=95% timed, "
                    "later-day retention < 0.99 over the history (Good-only excluded); split days on a 15-day grid "
                    f"(shuffled, same rng), each {H}-day window: >=3,000 reviews before, 1,500-9,000 reviews, >=40 active "
                    "days, >=300 later-day reviews with retention < 0.995 and >=30 Again, >=100 first-touch old cards; "
                    f"windows of one user >= {H} days apart; 2-3 splits per user; first {args.users} users that qualify. "
                    f"Long horizon: the first {args.long_users} chosen users with a {LH}-day window (3k-15k reviews, "
                    ">=80 active days, same other rules)"),
           "days": H, "long_days": LH, "looked_at": looked, "users": chosen, "long": longs}
    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    ns = sum(len(u["splits"]) for u in chosen)
    print(f"[sm41-drift] select: {len(chosen)} users, {ns} splits, {sum(w['window_reviews'] for u in chosen for w in u['splits']):,} "
          f"window reviews; long: {len(longs)} users ({sum(x['split']['window_reviews'] for x in longs):,} reviews); "
          f"looked at {looked}", flush=True)


# ------------------------------------------------------------------ proof
def proof(args):
    """(1) the same seed -> the same lanes (two runs of one split); (2) batched == unbatched: every lane alone (B=1;
    shared-stream-anchored lanes with the tf lane, B=2) vs all lanes together, on 2 users; (3) teacher forcing in the
    answer convention reproduces stage 1's real-tf lane (prototype.jsonl) on one stage-1 user, 90 days."""
    S.low_priority()
    res = {"seeds": args.seeds}
    model = S.Model(threads=1)
    lanes = lane_list(DEFAULT_ARMS, args.seeds)

    warms = {}

    def one(user, D, N, lanes_):
        # run_split never writes into the warm-up (it copies the entity maps, the builder and the explicit features),
        # so one warm-up serves every run of a split
        if (user, D) not in warms:
            rev, _ = S.load_user(user)
            warm = Warm(model, rev, user)
            end = int(np.searchsorted(rev["day_offset"], D, side="left"))
            warm.advance(end)
            warms[(user, D)] = (warm, S.user_habits(rev, end))
        warm, habits = warms[(user, D)]
        return run_split(model, warm, habits, D, N, lanes_, args.seeds, 0), warm

    def lane_acc(r, k):
        return {name: np.asarray(v)[k] for name, v in r["acc"].items()}

    # (3) stage 1 reproduction: user 3876 split 428, 90 days, real-tf#0 of prototype.jsonl
    proto = {json.loads(l)["user"]: json.loads(l) for l in (S.HERE / "prototype.jsonl").read_text().splitlines() if l.strip()}
    u3, D3 = 3876, 428
    r3, _ = one(u3, D3, 90, [("tf-answer", -1), ("tf", -1)])
    a = lane_acc(r3, 0)
    lat = a["later_old"].sum(0) + a["later_new"].sum(0)
    ft = a["first_touch"].sum(0)
    tl = a["time_later"].sum(0)
    ref = proto[u3]["lanes"]["real-tf#0"]
    s3 = {"predicted_retention_later": round(lat[2] / lat[0], 4), "ref": ref["predicted_retention_later"],
          "first_touch_n": int(ft[0]), "first_touch_retention": round(ft[1] / ft[0], 4),
          "first_touch_predicted": round(ft[2] / ft[0], 4), "ref_first_touch": ref["first_touch"]["all"],
          "predicted_seconds_per_later_review": round(tl[3] / tl[0], 2), "ref_s": ref["predicted_seconds_per_later_review"],
          "retention_later": round(lat[1] / lat[0], 4), "ref_retention_real": proto[u3]["real"]["retention_later"]}
    b = lane_acc(r3, 1)
    lb = b["later_old"].sum(0) + b["later_new"].sum(0)
    gl_a, gl_b = a["grade_ll"].sum(0), b["grade_ll"].sum(0)
    s3["show_conv_predicted_retention_later"] = round(lb[2] / lb[0], 4)
    s3["grade_logloss_later_answer_vs_show"] = [round(gl_a[0] / gl_a[1], 4), round(gl_b[0] / gl_b[1], 4)]
    s3["grade_logloss_same_day_answer_vs_show"] = [round(gl_a[2] / max(gl_a[3], 1), 4), round(gl_b[2] / max(gl_b[3], 1), 4)]
    s3["ok"] = bool(abs(s3["predicted_retention_later"] - s3["ref"]) <= 1e-4 and abs(s3["predicted_seconds_per_later_review"] - s3["ref_s"]) <= 0.01
                and s3["first_touch_n"] == ref["first_touch"]["n"] and abs(s3["first_touch_predicted"] - ref["first_touch"]["all"]["predicted"]) <= 1e-4)
    res["stage1_teacher_forcing"] = s3
    print(f"[sm41-drift-proof] stage 1 tf: {s3}", flush=True)

    # (1) + (2) on two users
    res["users"] = {}
    ok_all = s3["ok"]
    for u, D in args.users:
        t = time.perf_counter()
        r1, _ = one(u, D, args.days, lanes)
        r2, _ = one(u, D, args.days, lanes)
        same = all(np.array_equal(np.asarray(r1["acc"][k]), np.asarray(r2["acc"][k])) for k in r1["acc"])
        tb = time.perf_counter() - t
        worst, flips = {}, []
        for k, (arm, s) in enumerate(lanes):
            if arm == "tf":
                sub = [(arm, s)]
            elif arm_cfg(arm)["anchor_shared"]:
                sub = [("tf", -1), (arm, s)]
            else:
                sub = [(arm, s)]
            rr, _ = one(u, D, args.days, sub)
            j = len(sub) - 1
            A, Bv = lane_acc(r1, k), lane_acc(rr, j)
            # the draws (integer counters: grades, passes, review counts) must match exactly unless a near-tie flipped
            # one; the float sums (predicted pass, seconds, ln seconds, LogLoss) to a relative 1e-4 (float32 reduction
            # order of a batch-B vs batch-1 matmul, summed over hundreds of reviews; times are rounded to whole ms)
            ints = [A["grades_same"], A["grades_later"], A["later_old"][..., :2], A["later_new"][..., :2],
                    A["first_touch"][..., :2], A["time_later"][..., :1], A["implied_follow"][..., :1]]
            intb = [Bv["grades_same"], Bv["grades_later"], Bv["later_old"][..., :2], Bv["later_new"][..., :2],
                    Bv["first_touch"][..., :2], Bv["time_later"][..., :1], Bv["implied_follow"][..., :1]]
            exact = all(np.array_equal(x, y) for x, y in zip(ints, intb))
            rel = max(float((np.abs(A[n] - Bv[n]) / np.maximum(1.0, np.abs(A[n]))).max()) for n in A)
            gap = max(float(np.abs(A[n] - Bv[n]).max()) for n in A)
            worst[r1["lanes"][k]] = {"draws_identical": bool(exact), "max_rel_gap": rel, "max_abs_gap": gap}
            if not exact:
                flips.append(r1["lanes"][k])
        prob = {"same_seed_identical": same, "batched_vs_unbatched": worst, "lanes_with_a_flipped_draw": flips,
                "batched_wall_seconds": r1["wall_seconds"], "batched_lane_reviews_per_second": r1["lane_reviews_per_second"],
                "two_batched_runs_seconds": round(tb, 1), "window_reviews": r1["window_reviews"]}
        prob["ok"] = bool(same and len(flips) <= 1 and all(v["max_rel_gap"] <= 1e-4 for v in worst.values() if v["draws_identical"]))
        ok_all = ok_all and prob["ok"]
        res["users"][str(u)] = prob
        print(f"[sm41-drift-proof] user {u} split {D}: same seed identical {same}; flips {flips}; "
              f"{r1['lane_reviews_per_second']} lane reviews/s batched", flush=True)
    res["ok"] = bool(ok_all)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[sm41-drift-proof] {'PASS' if ok_all else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ report
ARM_TEXT = {
    "tf": "teacher forcing (real grades and times), show-time query rows",
    "tf-answer": "teacher forcing, stage 1's answer-to-answer query rows",
    "fr": "free running (grades and later-day times sampled), show-time query rows",
    "fr-answer": "free running, answer-to-answer query rows (stage 1)",
    "fr-treal": "free running grades, real times (bound for any time fix)",
    "fr-texp": "expected answer time fed (exp of the expected ln time)",
    "fr-tt0.5": "time head sampled at temperature 0.5",
    "fr-gt0.8": "grade head sampled at temperature 0.8",
    "fr-anchor7": "every 7th day teacher forced (real grades and times)",
    "fr-shared1": "deck / preset / user states re-anchored to teacher forcing every day",
    "fr-shared7": "deck / preset / user states re-anchored every 7 days",
    "fr-texp-shared1": "expected time + daily shared-stream re-anchoring",
}


def lane_metrics(A, k, blocks):
    """Per block (list of bucket ranges): the metrics of lane k from the counters."""
    out = []
    for bl in blocks:
        sl = slice(bl[0], bl[1])

        def s(name):
            return np.asarray(A[name])[k, sl].sum(0)
        ft, lo, ln = s("first_touch"), s("later_old"), s("later_new")
        gs, gl, tl, fo = s("grades_same"), s("grades_later"), s("time_later"), s("implied_follow")
        g = gs + gl
        div = lambda a, b: float(a / b) if b > 0 else float("nan")   # noqa: E731
        out.append({"ft_n": ft[0], "ft_ret": div(ft[1], ft[0]), "ft_pred": div(ft[2], ft[0]),
                    "old_ret": div(lo[1], lo[0]), "old_pred": div(lo[2], lo[0]), "new_ret": div(ln[1], ln[0]),
                    "new_pred": div(ln[2], ln[0]), "again": div(g[0], g.sum()), "hard": div(g[1], g.sum()),
                    "good": div(g[2], g.sum()), "easy": div(g[3], g.sum()), "easy_pass": div(gl[3], gl[1:].sum()),
                    "s": div(tl[1], tl[0]), "s_geo": math.exp(div(tl[2], tl[0])) if tl[0] else float("nan"),
                    "s_exp": div(tl[3], tl[0]), "follow": div(fo[1], fo[0])})
    return out


def report(args):
    import glob
    rows = []
    for pat in args.rows.split(","):
        for path in sorted(glob.glob(pat)):
            rows += [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    print(f"[sm41-drift] report: {len(rows)} rows", flush=True)
    L = ["# The small model in the review simulator (SM41): drift study tables", ""]
    out = {"horizons": {}}
    for H in sorted({r["days"] for r in rows}):
        rs = [r for r in rows if r["days"] == H]
        nbk = (H + BUCKET - 1) // BUCKET
        per = 30 // BUCKET
        blocks = [(i, min(i + per, nbk)) for i in range(0, nbk, per)]
        names = [f"days {b[0] * BUCKET}-{min(b[1] * BUCKET, H) - 1}" for b in blocks]
        # per user-split, per arm (seeds averaged): metric - real (realised from the tf lane), state drift vs tf
        recs = {}     # arm -> user -> list of per-split dicts (block -> metric -> drift)
        users = sorted({r["user"] for r in rs})
        for r in rs:
            A = r["acc"]
            arms = {}
            for k, lane in enumerate(r["lanes"]):
                arms.setdefault(lane.split("#")[0], []).append(k)
            real = lane_metrics(A, arms["tf"][0], blocks)
            for arm, ks in arms.items():
                ms = [lane_metrics(A, k, blocks) for k in ks]
                d = []
                for bi in range(len(blocks)):
                    mean = {key: float(np.nanmean([m[bi][key] for m in ms])) for key in ms[0][bi]}
                    rb = real[bi]
                    d.append({"ft_state": mean["ft_pred"] - rb["ft_pred"], "ft_real": mean["ft_ret"] - rb["ft_ret"],
                              "old": mean["old_ret"] - rb["old_ret"], "old_state": mean["old_pred"] - rb["old_pred"],
                              "new": mean["new_ret"] - rb["new_ret"], "again": mean["again"] - rb["again"],
                              "easy_pass": mean["easy_pass"] - rb["easy_pass"], "s": mean["s"] - rb["s"],
                              "s_exp": mean["s_exp"] - rb["s"], "follow": mean["follow"] - rb["follow"],
                              "tf_pred_vs_real": rb["ft_pred"] - rb["ft_ret"], "real_s": rb["s"], "real_easy": rb["easy_pass"],
                              "real_again": rb["again"]})
                recs.setdefault(arm, {}).setdefault(r["user"], []).append(d)
        arms_order = [a for a in ARMS if a in recs]

        def user_vals(arm, bi, key, absval=False):
            v = []
            for u, splits in recs[arm].items():
                x = [s[bi][key] for s in splits if not math.isnan(s[bi][key])]
                if x:
                    v.append(float(np.mean(np.abs(x) if absval else x)))
            return np.asarray(v)

        def ci(v, scale=100.0, digits=1):
            if len(v) == 0:
                return "—"
            m, h = v.mean() * scale, 1.96 * v.std(ddof=1) / math.sqrt(len(v)) * scale if len(v) > 1 else float("nan")
            return f"{m:+.{digits}f} ± {h:.{digits}f}"
        nsplit = len(rs)
        L += [f"## Horizon {H} days: {len(users)} users, {nsplit} split points", "",
              f"Lanes per split: {len(rs[0]['lanes'])} ({', '.join(rs[0]['lanes'][:4])}, ...). Drift = lane minus the real "
              "user (the teacher-forced lane for predictions) on the same reviews, in points; mean over users (splits of a "
              "user averaged first) ± 95% CI over users. 'first touch' = each old card's first later-day review in the "
              "window (its day was set from real pre-split outcomes, so it does not depend on the lane's samples). 'state' "
              "= the model's own predicted pass rate on those reviews (no outcome noise): the cleanest drift signal.", ""]
        L += ["### First-touch state drift (predicted pass rate, lane minus teacher forcing), points", "",
              "| arm | " + " | ".join(names) + " | mean abs, last block |", "|---|" + "---|" * (len(names) + 1)]
        res = {}
        for arm in arms_order:
            cells = [ci(user_vals(arm, bi, "ft_state")) for bi in range(len(blocks))]
            ab = user_vals(arm, len(blocks) - 1, "ft_state", absval=True)
            L.append(f"| {arm} | " + " | ".join(cells) + f" | {ab.mean() * 100:.2f} |")
            res[arm] = {"ft_state": [float(user_vals(arm, bi, "ft_state").mean()) for bi in range(len(blocks))],
                        "ft_state_abs_last": float(ab.mean())}
        L += ["", "### First-touch realised retention drift (lane minus real), points", "",
              "| arm | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for arm in arms_order:
            L.append(f"| {arm} | " + " | ".join(ci(user_vals(arm, bi, "ft_real")) for bi in range(len(blocks))) + " |")
        L += ["", "### All later-day reviews on the real schedule (old cards / new cards), realised retention drift, points", "",
              "| arm | " + " | ".join(f"old {n}" for n in names) + " | " + " | ".join(f"new {n}" for n in names) + " |",
              "|---|" + "---|" * (2 * len(names))]
        for arm in arms_order:
            L.append(f"| {arm} | " + " | ".join(ci(user_vals(arm, bi, "old")) for bi in range(len(blocks))) + " | "
                     + " | ".join(ci(user_vals(arm, bi, "new")) for bi in range(len(blocks))) + " |")
        lb = len(blocks) - 1
        L += ["", f"### Grades, answer time and implied same-day load ({names[lb]}), lane minus real", "",
              "Again = share of non-first reviews; Easy = share of later-day passes; seconds = mean answer time of later-day "
              "reviews as fed back (fr-texp feeds exp E[ln t], a geometric mean) / the head's arithmetic expectation for the "
              "lane's grade; follow-ups = same-day reviews per review implied by the user's own step rules (on the real "
              "schedule reviews per day are fixed by construction; this is the load a policy lane would add). Real means: "
              f"Again {np.nanmean([s[lb]['real_again'] for u in recs['tf'].values() for s in u]):.3f}, Easy "
              f"{np.nanmean([s[lb]['real_easy'] for u in recs['tf'].values() for s in u]):.3f}, "
              f"{np.nanmean([s[lb]['real_s'] for u in recs['tf'].values() for s in u]):.1f} s.", "",
              "| arm | Again, points | Easy of passes, points | seconds fed | seconds expected | follow-ups per 100 reviews |",
              "|---|---|---|---|---|---|"]
        for arm in arms_order:
            L.append(f"| {arm} | {ci(user_vals(arm, lb, 'again'))} | {ci(user_vals(arm, lb, 'easy_pass'))} | "
                     f"{ci(user_vals(arm, lb, 's'), 1.0)} | {ci(user_vals(arm, lb, 's_exp'), 1.0)} | "
                     f"{ci(user_vals(arm, lb, 'follow'))} |")
            res[arm].update({k: float(np.nanmean(user_vals(arm, lb, k))) for k in ("again", "easy_pass", "s", "s_exp", "follow",
                                                                                     "ft_real", "old", "new")})
        # paired vs fr: |state drift| reduction, last block
        L += ["", f"### Mitigation vs plain free running, paired per user ({names[lb]})", "",
              "Reduction of the mean absolute first-touch state drift and of the absolute Easy and seconds drift; "
              "positive = less drift than fr.", "", "| arm | what it does | state drift reduction, points | Easy | seconds |",
              "|---|---|---|---|---|"]
        for arm in arms_order:
            if arm == "fr" or arm.startswith("tf"):
                continue
            red = {}
            for key, sc in (("ft_state", 100.0), ("easy_pass", 100.0), ("s", 1.0)):
                v = []
                for u in recs[arm]:
                    if u not in recs["fr"]:
                        continue
                    a = [abs(s[lb][key]) for s in recs[arm][u] if not math.isnan(s[lb][key])]
                    f = [abs(s[lb][key]) for s in recs["fr"][u] if not math.isnan(s[lb][key])]
                    if a and f:
                        v.append(np.mean(f) - np.mean(a))
                red[key] = ci(np.asarray(v), sc, 2 if key == "ft_state" else 1)
            L.append(f"| {arm} | {ARM_TEXT.get(arm, '')} | {red['ft_state']} | {red['easy_pass']} | {red['s']} |")
        # the duration leak: tf vs tf-answer
        leak = [r for r in rs if "tf-answer" in r["lanes"]]
        if leak:
            ll = {"tf": np.zeros(4), "tf-answer": np.zeros(4)}
            for r in leak:
                for name in ll:
                    ll[name] += np.asarray(r["acc"]["grade_ll"])[r["lanes"].index(name)].sum(0)
            L += ["", "### The duration leak (duration-leak scoring (SM31)): teacher forcing with show-time vs answer-to-answer query rows", "",
                  "| query rows | grade LogLoss, later-day reviews | grade LogLoss, same-day reviews |", "|---|---|---|"]
            for name, text in (("tf-answer", "answer-to-answer (training data, stage 1)"), ("tf", "show time (the app)")):
                v = ll[name]
                L.append(f"| {text} | {v[0] / v[1]:.5f} | {v[2] / max(v[3], 1):.5f} |")
            out.setdefault("leak", {})[str(H)] = {k: v.tolist() for k, v in ll.items()}
        wall = sum(r["wall_seconds"] for r in rs); lr = sum(r["window_reviews"] * len(r["lanes"]) for r in rs)
        L += ["", f"Speed: {lr:,} lane reviews in {wall / 3600:.2f} process-hours of lanes ({lr / max(wall, 1):,.0f} lane reviews/s "
              f"per process, 1 thread), warm-ups {sum(r['warmup_seconds'] for r in rs) / 3600:.2f} process-hours.", ""]
        out["horizons"][str(H)] = {"users": len(users), "splits": nsplit, "blocks": names, "arms": res}
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print("\n".join(L[:60]))


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--users", type=int, default=50); s.add_argument("--seed", type=int, default=4141)
    s.add_argument("--days", type=int, default=60); s.add_argument("--long-days", type=int, default=120)
    s.add_argument("--long-users", type=int, default=10); s.add_argument("--out", default=str(OUT / "users.json"))
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True); r.add_argument("--splits", required=True)
    r.add_argument("--days", type=int, default=60); r.add_argument("--arms", default=DEFAULT_ARMS)
    r.add_argument("--seeds", type=int, default=3); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--threads", type=int, default=1); r.add_argument("--out", default=str(OUT / "drift.jsonl"))
    q = sub.add_parser("proof"); q.add_argument("--seeds", type=int, default=2); q.add_argument("--days", type=int, default=30)
    q.add_argument("--users", type=lambda x: [tuple(map(int, y.split(":"))) for y in x.split(",")], default="3876:428,2755:406")
    q.add_argument("--out", default=str(OUT / "proof-cpu.json"))
    t = sub.add_parser("report"); t.add_argument("--rows", default=str(OUT / "rows" / "*.jsonl"))
    t.add_argument("--md", default=str(OUT / "drift-tables.md")); t.add_argument("--json", default=str(OUT / "drift-result.json"))
    a = p.parse_args()
    {"select": select, "run": run, "proof": proof, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
