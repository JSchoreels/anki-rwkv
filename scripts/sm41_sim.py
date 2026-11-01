#!/usr/bin/env python
"""The small model in the review simulator (SM41), stage 1: a card-level review simulator driven by the exported small
RWKV model (the released leak-free small model + the answer-time head on richer frozen features (SM82c)), not by
FSRS-6 curves. Design: artifacts/small-model/sm41/DESIGN.md.

A simulated learner = the model conditioned on one real user's history up to a split day (warm-up: every real review
fed through the recurrent model, one row at a time), then rolled forward for N days in "lanes" that share the warm-up
state:
  real-tf   the user's real continuation, real grades and times fed back (teacher forcing: the model's ordinary
            prediction task on these days; its probabilities are recorded)
  real-fr   the real schedule (same cards, days, elapsed seconds), but every output the model covers SAMPLED from it
            and fed back: the grade of every non-first review, the time of every later-day review (free running on the
            real schedule: the drift check); first-review grades and times and same-day times stay real
  real-fr-pool  the same, but first-review grades and times and same-day times drawn from the user's recent pools,
            exactly as in the scheduler lanes (what the pools add to the drift)
  dr-X      a scheduler: every card's next interval = the day its predicted curve reaches desired retention X; grades
            and times sampled and fed back; same-day learning steps follow the user's own empirical step rules; new
            cards = the user's real new cards of each day, in real order; card set = the cards of the real window.

Per review in a lane: (1) the instant (query) row through the model (state not committed) -> P(grade) -> sample the
grade; the answer-time head (SM82c) on the query row's frozen features + the 26 explicit features -> the time bins of
the sampled grade -> sample the time (reviews on a later day; same-day and first reviews draw from the user's own
pre-split times of that kind and grade, as the time head is not trained on them); (2) the review row with that grade
and time -> commit every stream's state; the card's curve (GRU power curves) -> the scheduler.

  check  --user U [--rows N]            CPU proof: feature builder == get_rwkv_data, recurrent == parallel (grade
                                        outputs and the 448 frozen time-head features), explicit features ==
                                        sm82c_head.explicit_features.
  run    --user U --split-day D ...     one user, every lane -> JSONL row (per-day series + summary).
  bench  --batch B ...                  batched-step throughput (lanes stacked along the batch dim).
  report                                the table -> artifacts/small-model/sm41/prototype-result.md / .json.
"""
from argparse import ArgumentParser
import copy
import heapq
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

HERE = ROOT / "artifacts/small-model/sm41"
MASTER = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth")  # read only
HEAD = "time_mlps.0."
SECONDS_PER_DAY = 86400
PERIODS = [3, 7, 30, 100, 365, 3650, 36500]       # rwkv.config.DAY_OFFSET_ENCODE_PERIODS
ID_DIMS = [("card_id", 12), ("note_id", 12), ("deck_id", 8), ("preset_id", 8)]   # RWKV_SUBMODULES order minus user
MAX_SAME_DAY = 8
from rwkv import data_processing as DP   # noqa: E402
from scripts.expertium_options import TIME_BIN_EDGES_MS   # noqa: E402
TIME_EDGES = np.asarray(TIME_BIN_EDGES_MS)


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ feature builder (rwkv.data_processing, incremental)
class Builder:
    """The 92 model inputs of each review row (and its instant / query row), built one review at a time exactly as
    get_rwkv_data + add_queries + prepare_batch.add_encodings build them for a whole history (check C1). The id codes
    and the day-encoding baselines are drawn from this builder's own generator (training draws them at random too)."""

    def __init__(self, day0, seed):
        self.day0 = int(day0)
        self.rng = np.random.default_rng(seed)
        self.base = [int(self.rng.integers(0, p)) for p in PERIODS]
        self.codes = {name: {} for name, _ in ID_DIMS}
        self.th = 0
        self.cum_new = 0
        self.cards = {}            # card -> [last th, cum_new at last, ed cumulative, es cumulative, first day]
        self.day = None
        self.new_today = 0
        self.rev_today = 0
        self.prev_day = None

    def code(self, stream, key, dim):
        table = self.codes[stream]
        c = table.get(key)
        if c is None:
            c = self.rng.integers(0, 4, size=dim).astype(np.float32) - 1.5
            table[key] = c
        return c

    def row(self, card, note, deck, preset, day, ed, es, rating, duration):
        """(query vector or None for a first review, review vector); advances the history by this review."""
        dp = DP
        day = int(day)
        self.th += 1
        first = ed == -1
        if first:
            self.cum_new += 1
        st = self.cards.get(card)
        if st is None:
            st = [0, 0, 0, 0, day]
            self.cards[card] = st
            diff_new, diff_rev = 0.0, 0.0
        else:
            diff_new = float(self.cum_new - st[1])
            diff_rev = float(max(0, self.th - st[0] - 1))
        if day != self.day:
            self.day, self.new_today, self.rev_today = day, 0, 0
        cum_rev_today = self.rev_today
        self.rev_today += 1
        if first:
            self.new_today += 1
        day_diff = 0.0 if self.prev_day is None else float(day - self.prev_day)
        self.prev_day = day
        st[2] += ed
        st[3] += es
        st[0], st[1] = self.th, self.cum_new
        f = np.zeros(92, np.float32)
        f[0] = dp.scale_elapsed_days(np.float64(ed))
        f[1] = dp.scale_elapsed_days_cumulative(np.float64(st[2]))
        f[2] = dp.scale_elapsed_seconds(np.float64(es))
        f[3] = math.sin((es % SECONDS_PER_DAY) * 2 * math.pi / SECONDS_PER_DAY)
        f[4] = math.cos((es % SECONDS_PER_DAY) * 2 * math.pi / SECONDS_PER_DAY)
        f[5] = dp.scale_elapsed_seconds_cumulative(np.float64(st[3]))
        f[6] = math.sin((st[3] % SECONDS_PER_DAY) * 2 * math.pi / SECONDS_PER_DAY)
        f[7] = math.cos((st[3] % SECONDS_PER_DAY) * 2 * math.pi / SECONDS_PER_DAY)
        f[8] = dp.scale_duration(np.float64(duration))
        f[8 + rating] = 1.0                       # rating_1..4 at 9..12
        # 13..15 note/deck/preset *_is_nan: 0 (leak-free inputs, scripts/sm71_leak.py "neighbour")
        f[16] = dp.scale_day_offset_diff(np.float64(day_diff))
        f[17] = (((day - self.day0) % 7) - 3) / 3
        f[18] = dp.scale_diff_new_cards(np.float64(diff_new))
        f[19] = dp.scale_diff_reviews(np.float64(diff_rev))
        f[20] = dp.scale_cum_new_cards_today(np.float64(self.new_today))
        f[21] = dp.scale_cum_reviews_today(np.float64(cum_rev_today))
        f[22] = 0.0      # scaled_state: zeroed by the small model (zero_features 22); the state column is not simulated
        f[23] = 0.0
        k = 24
        for (stream, dim), key in zip(ID_DIMS, (card, note, deck, preset)):
            f[k:k + dim] = self.code(stream, key, dim)
            k += dim
        rel, rel_first = day - self.day0, st[4] - self.day0
        for p, b in zip(PERIODS, self.base):
            fr = 2 * np.pi / p
            f[k] = np.sin(fr * ((b + rel) % p)); f[k + 1] = np.cos(fr * ((b + rel) % p))
            f[k + 2] = np.sin(fr * ((b + rel_first) % p)); f[k + 3] = np.cos(fr * ((b + rel_first) % p))
            k += 4
        if first:
            return None, f
        q = f.copy()
        q[8:13] = 0.0
        q[22] = 0.0
        q[23] = 1.0
        return q, f


# ------------------------------------------------------------------ explicit time features (sm82c_head, incremental)
class Explicit:
    """scripts/sm82c_head.explicit_features one review at a time (check C3)."""

    def __init__(self):
        from scripts.sm82_time import RunningMedian
        self.per = {g: RunningMedian() for g in (1, 2, 3, 4)}
        self.every = RunningMedian()
        self.counts = [0, 0, 0, 0, 0]
        self.cards = {}
        self.prev, self.ema, self.i = None, math.log(8.0), 0

    def features(self, card, elapsed):
        LN8 = math.log(8.0)
        row = np.zeros(26, np.float32)
        st = self.cards.get(card)
        last3, cnt, last_grade = st if st is not None else ([], 0, 0)
        for k in range(3):
            if k < len(last3):
                row[k] = last3[k] - LN8
            else:
                row[3 + k] = 1.0
        row[6] = math.log1p(cnt) / 5.0
        if 1 <= last_grade <= 4:
            row[6 + last_grade] = 1.0
        em = self.every.median()
        fb = em if em is not None else LN8
        for g in (1, 2, 3, 4):
            m = self.per[g].median()
            row[10 + g] = (m if m is not None else fb) - LN8
            row[14 + g] = math.log1p(self.counts[g]) / 10.0
        row[19] = fb - LN8
        if self.prev is None:
            row[21] = 1.0
        else:
            row[20] = self.prev - LN8
        row[22] = self.ema - LN8
        row[23] = math.log1p(max(elapsed, 0)) / 3.0
        row[24] = 1.0 if elapsed < 0 else 0.0
        row[25] = math.log1p(self.i) / 10.0
        return row

    def push(self, card, rating, duration):
        st = self.cards.get(card)
        if st is None:
            st = [[], 0, 0]
            self.cards[card] = st
        st[1] += 1
        st[2] = rating
        if duration > 0:
            x = math.log(max(duration, 1) / 1000.0)
            self.per.get(rating, self.every).push(x)
            self.every.push(x)
            if rating in self.per:
                self.counts[rating] += 1
            st[0].insert(0, x)
            del st[0][3:]
            self.prev = x
            self.ema = 0.9 * self.ema + 0.1 * x
        self.i += 1


# ------------------------------------------------------------------ model
class Model:
    """The recurrent small model (expertium.model.variant_rnn, w64) + the answer-time head (SM82c), CPU fp32."""

    def __init__(self, master=MASTER, threads=1):
        import torch
        from expertium import variants
        from expertium.model.variant_model import TimeMLP
        torch.set_num_threads(threads)
        self.torch = torch
        sd = torch.load(master, map_location="cpu", weights_only=True)
        self.rnn = variants.build("w64", rnn=True).float()
        self.rnn.load_state_dict({k: v.float() for k, v in sd.items() if not k.startswith(HEAD)}, strict=True)
        self.rnn.eval()
        self.head = TimeMLP(474, 128, 64)
        self.head.load_state_dict({k[len(HEAD):]: v.float() for k, v in sd.items() if k.startswith(HEAD)})
        self.head.eval()
        self.streams = self.rnn.streams
        self.steps = 0
        from scripts.sm82_time import LOG_EDGES
        self.lo = np.r_[math.log(0.25), LOG_EDGES]       # ln-second bounds of the 16 bins (bin 15 = the 60 s cap)
        self.hi = np.r_[LOG_EDGES, LOG_EDGES[-1]]

    def step(self, feat, states, snaps=False):
        """VariantSrsRWKVRnn.step (interleaved) for one row, keeping x after features2card and after every layer round
        when `snaps` (the SM82c frozen features). Returns (out, next_states, frozen features [448] or None)."""
        torch = self.torch
        m = self.rnn
        self.steps += 1
        with torch.inference_mode():
            x = torch.from_numpy(feat).view(1, 92)
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
                    in_BC=x, v0_BC=v0[i] if v0[i] is not None else torch.empty_like(x), state=nxt[s].get(layer))
                if snaps and (k + 1 == len(ls) or ls[k + 1][1] != layer):
                    shots.append(x)
            h = m.prehead_norm(x)
            x_w = m.head_w(h).float()
            hp = m.head_p(h).float()
            out = {"w": torch.softmax(torch.nn.functional.linear(x_w, m.gru_w_weight, m.gru_w_bias), -1)[0].numpy(),
                   "s": torch.nn.functional.linear(x_w, m.gru_s_weight, m.gru_s_bias)[0].numpy(),
                   "d": torch.nn.functional.linear(x_w, m.gru_d_weight, m.gru_d_bias)[0].numpy(),
                   "p": torch.softmax(m.p_linear(hp), -1)[0].numpy()}
            frozen = None
            if snaps:
                frozen = torch.cat([torch.nn.functional.layer_norm(t.float(), (64,)) for t in shots] + [hp], -1)[0]
        return out, nxt, frozen

    def time_probs(self, frozen, explicit):
        """[4, 16] bin probabilities per grade."""
        torch = self.torch
        with torch.inference_mode():
            z = torch.cat([frozen, torch.from_numpy(explicit)]).view(1, 474)
            return torch.softmax(self.head(z).view(4, 16), -1).numpy()

    def sample_time_ms(self, probs_g, rng):
        b = int(rng.choice(16, p=probs_g / probs_g.sum()))
        if b == 15:
            return 60000
        return int(round(1000 * math.exp(rng.uniform(self.lo[b], self.hi[b]))))


def curve(w, s_raw, d_raw, t_seconds):
    """Recall at elapsed seconds t (array) from one review row's GRU power curves (VariantSrsRWKVRnn.curve_probability;
    the zero ahead residual adds 1e-5 to the logit)."""
    s = np.exp(np.clip(s_raw, -25, 25)); d = np.exp(np.clip(d_raw, -25, 25))
    t = np.maximum(1.0, np.asarray(t_seconds, np.float64))[..., None]
    raw = 1e-5 + (1 - 2e-5) * np.sum(w * np.exp(-d * np.log1p(t / (1e-7 + s))), -1)
    z = np.log(raw / (1 - raw)) + 1e-5
    return 1 / (1 + np.exp(-z))


def time_at(w, s_raw, d_raw, r):
    """Elapsed seconds where the curve reaches recall r (bisection in ln t over [1 s, 1000 years])."""
    lo, hi = 0.0, math.log(1000 * 365 * SECONDS_PER_DAY)
    if curve(w, s_raw, d_raw, math.exp(hi)) >= r:
        return math.exp(hi)
    if curve(w, s_raw, d_raw, 1.0) < r:
        return 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if curve(w, s_raw, d_raw, math.exp(mid)) >= r:
            lo = mid
        else:
            hi = mid
    return math.exp(lo)


# ------------------------------------------------------------------ user data
def load_user(user):
    from rwkv.data_processing import get_rwkv_data
    from scripts import compress_common as cc
    from scripts import sm71_leak
    data = cc.data_path()
    frame = get_rwkv_data(data, user)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    cols = ["card_id", "note_id", "deck_id", "preset_id", "day_offset", "elapsed_days", "elapsed_seconds", "rating",
            "duration"]
    rev = {c: frame[c].to_numpy().astype(np.int64) for c in cols}
    # get_rwkv_data shifts day_offset to start at 0 (add_segment_features); keep that day numbering
    return rev, frame


def user_habits(rev, end, recent_days=60, min_reviews=2000):
    """The user's own RECENT pre-split habits (the last `recent_days` days before the split, widened to at least the
    last `min_reviews` reviews; users change habits, e.g. user 1190 moved to Good-only): first-rating mix, same-day step
    rules per (kind, rating), time pools for first and same-day reviews per rating, the session-start spread."""
    day_all = rev["day_offset"]
    lo = int(np.searchsorted(day_all, day_all[end - 1] - recent_days + 1, side="left"))
    lo = max(0, min(lo, end - min_reviews))
    sl = slice(lo, end)
    card, day, ed, es = rev["card_id"][sl], day_all[sl], rev["elapsed_days"][sl], rev["elapsed_seconds"][sl]
    rating, dur = rev["rating"][sl], rev["duration"][sl]
    n = len(card)
    nxt = np.full(n, -1)
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(int(card[i]), -1)
        last[int(card[i])] = i
    kind = np.where(ed == -1, 0, np.where(ed == 0, 1, 2))
    follow = np.array([nxt[i] >= 0 and ed[nxt[i]] == 0 for i in range(n)])
    rules = {}
    for k in (0, 1, 2):
        for g in (1, 2, 3, 4):
            m = (kind == k) & (rating == g)
            cnt = int(m.sum())
            p = (int((follow & m).sum()) + (1 if k < 2 and g == 1 else 0)) / (cnt + 2)
            gaps = es[nxt[m & follow]]
            gaps = gaps[gaps > 0]
            rules[(k, g)] = (p, gaps if len(gaps) else np.array([600]))
    first_mix = np.array([((kind == 0) & (rating == g)).sum() + 1 for g in (1, 2, 3, 4)], np.float64)
    pools = {}
    for k in (0, 1):
        for g in (1, 2, 3, 4):
            v = dur[(kind == k) & (rating == g) & (dur > 0)]
            if len(v) < 5:
                v = dur[(kind == k) & (dur > 0)]
            pools[(k, g)] = v if len(v) else np.array([8000])
    later = (ed >= 1)
    delta = (es[later] - SECONDS_PER_DAY * ed[later]).astype(np.float64)
    sigma = float(np.clip(np.std(delta) / math.sqrt(2), 600, 6 * 3600)) if later.sum() > 10 else 3 * 3600.0
    return {"rules": rules, "first_mix": first_mix / first_mix.sum(), "pools": pools, "sigma": sigma}


# ------------------------------------------------------------------ lanes
class Lane:
    def __init__(self, name, model, states, builder, explicit, last_out, last_time, seed):
        self.name, self.model = name, model
        self.states = {s: dict(v) for s, v in states.items()}     # entity maps (tensors are never written in place)
        self.builder = copy.deepcopy(builder)
        self.explicit = copy.deepcopy(explicit)
        self.last_out = dict(last_out)     # card -> (w, s, d) of the card's last review row
        self.last_time = dict(last_time)   # card -> absolute seconds of the card's last answer
        self.rng = np.random.default_rng(seed)
        self.D = 0                         # the split day (absolute day of simulated day 0), set by run
        self.days = {}
        self.records = []                  # (day, kind, rating, ms, p_pass, R_curve_at_review, card)
        self.time_ms = 0
        self.wall = 0.0

    def day_row(self, d):
        r = self.days.get(d)
        if r is None:
            r = {"reviews": 0, "later": 0, "later_pass": 0, "p_pass": 0.0, "first": 0, "same": 0,
                 "grades": [0, 0, 0, 0], "seconds": 0.0, "pred_seconds": 0.0, "time_ll": 0.0, "time_n": 0,
                 "grade_ll": 0.0}
            self.days[d] = r
        return r

    def review(self, ids, d, ed, es, habits, rating=None, duration=None, t_abs=None, real=None):
        """One review of card ids=(card, note, deck, preset) on simulated day d (0 = the split day) with elapsed days / seconds ed, es. rating /
        duration given = teacher forcing; None = sampled. real = (rating, duration) of the real review (real-fr): the outputs
        the model does not cover (first-review grade and time, same-day time) are then taken from it, not from the
        user's pools. Returns (rating, duration)."""
        started = time.perf_counter()
        card, note, deck, preset = ids
        m = self.model
        keys = dict(zip(m.streams, (card, deck, note, preset, 0)))
        cur = {s: self.states[s].get(keys[s]) for s in m.streams}
        row = self.day_row(d)
        day_abs = d + self.D
        kind = 0 if ed == -1 else (1 if ed == 0 else 2)
        p_pass = None
        # both rows at once: only the review row's columns 8..12 (duration, grade) depend on the outcome
        q, f = self.builder.row(card, note, deck, preset, day_abs, ed, es, 1, 0)
        if kind == 0:
            if rating is None and real is not None:
                rating, duration = real
            if rating is None:
                rating = int(self.rng.choice(4, p=habits["first_mix"])) + 1
                duration = int(self.rng.choice(habits["pools"][(0, rating)]))
        else:
            out_q, _, frozen = m.step(q, cur, snaps=kind == 2)
            pg = out_q["p"].astype(np.float64)
            pg = pg / pg.sum()
            p_pass = 1.0 - pg[0]
            tp = None
            if kind == 2:
                tp = m.time_probs(frozen, self.explicit.features(card, ed))
            if rating is None:
                rating = int(self.rng.choice(4, p=pg)) + 1
                if kind == 2:
                    duration = m.sample_time_ms(tp[rating - 1], self.rng)
                else:
                    duration = real[1] if real is not None else int(self.rng.choice(habits["pools"][(1, rating)]))
            row["grade_ll"] += -math.log(max(pg[rating - 1], 1e-7))
            if kind == 2:
                pb = tp[rating - 1]
                row["pred_seconds"] += float(np.sum(pb * np.exp((m.lo + m.hi) / 2)))
                if duration > 0:
                    lb = int(np.searchsorted(TIME_EDGES, float(duration), side="right"))
                    row["time_ll"] += -math.log(max(float(pb[lb]), 1e-7))
                    row["time_n"] += 1
                row["later"] += 1
                row["later_pass"] += int(rating > 1)
                row["p_pass"] += p_pass
        f[8] = DP.scale_duration(np.float64(duration))
        f[9:13] = 0.0
        f[8 + rating] = 1.0
        out, nxt, _ = m.step(f, cur)
        for s in m.streams:
            self.states[s][keys[s]] = nxt[s]
        self.explicit.push(card, rating, duration)
        self.last_out[card] = (out["w"], out["s"], out["d"])
        if t_abs is not None:
            self.last_time[card] = t_abs
        row["reviews"] += 1
        row["first"] += int(kind == 0)
        row["same"] += int(kind == 1)
        row["grades"][rating - 1] += 1
        row["seconds"] += duration / 1000.0
        self.records.append((d, kind, rating, duration, -1.0 if p_pass is None else p_pass, card))
        self.wall += time.perf_counter() - started
        return rating, duration


def memorized(lane, cards, t_end):
    total = 0.0
    for c in cards:
        if c in lane.last_out:
            w, s, dd = lane.last_out[c]
            total += float(curve(w, s, dd, max(1.0, t_end - lane.last_time.get(c, t_end))))
    return total


def run(args):
    low_priority()
    user, D, N = args.user, args.split_day, args.days
    rev, _ = load_user(user)
    n = len(rev["card_id"])
    day = rev["day_offset"]
    end = int(np.searchsorted(day, D, side="left"))
    stop = int(np.searchsorted(day, D + N, side="left"))
    habits = user_habits(rev, end)
    model = Model(threads=args.threads)
    t0 = time.perf_counter()
    builder = Builder(day0=int(day[0]), seed=[41, user])
    explicit = Explicit()
    states = {s: {} for s in model.streams}
    last_out, last_day = {}, {}
    window_cards = set(rev["card_id"][end:stop].tolist())
    ids_of = {}
    for i in range(n):
        c = int(rev["card_id"][i])
        ids_of[c] = (c, int(rev["note_id"][i]), int(rev["deck_id"][i]), int(rev["preset_id"][i]))
    for i in range(end):
        c = int(rev["card_id"][i])
        ids = ids_of[c]
        _, f = builder.row(*ids, int(day[i]), int(rev["elapsed_days"][i]), int(rev["elapsed_seconds"][i]),
                           int(rev["rating"][i]), int(rev["duration"][i]))
        keys = dict(zip(model.streams, (ids[0], ids[2], ids[1], ids[3], 0)))
        out, nxt, _ = model.step(f, {s: states[s].get(keys[s]) for s in model.streams})
        for s in model.streams:
            states[s][keys[s]] = nxt[s]
        explicit.push(c, int(rev["rating"][i]), int(rev["duration"][i]))
        if c in window_cards:
            last_out[c] = (out["w"], out["s"], out["d"])
            last_day[c] = int(day[i])
    # drop card / note states the window never touches (memory)
    warm_s = time.perf_counter() - t0
    mu = 12 * 3600.0
    last_time = {c: d * SECONDS_PER_DAY + mu for c, d in last_day.items()}
    print(f"[sm41] user {user}: warm-up {end} reviews in {warm_s:.1f} s ({end / warm_s:.0f} reviews/s); window "
          f"{stop - end} reviews, {len(window_cards)} cards", flush=True)
    result = {"user": user, "split_day": D, "days": N, "warmup_reviews": end, "warmup_seconds": round(warm_s, 2),
              "window_reviews": stop - end, "window_cards": len(window_cards), "lanes": {}}
    t_end = (D + N) * SECONDS_PER_DAY + mu

    # real continuation
    real = {"reviews": stop - end, "days": {}}
    for i in range(end, stop):
        d = int(day[i]) - D
        r = real["days"].setdefault(d, {"reviews": 0, "later": 0, "later_pass": 0, "grades": [0, 0, 0, 0],
                                        "seconds": 0.0, "first": 0, "same": 0})
        e = int(rev["elapsed_days"][i]); g = int(rev["rating"][i])
        r["reviews"] += 1; r["grades"][g - 1] += 1; r["seconds"] += rev["duration"][i] / 1000.0
        r["first"] += int(e == -1); r["same"] += int(e == 0)
        if e >= 1:
            r["later"] += 1; r["later_pass"] += int(g > 1)
    result["real"] = summarise_days(real["days"], None)
    result["real"]["by_kind"] = by_kind(real_records(rev, end, stop, D))

    lanes = []
    for name in args.lanes.split(","):
        for seed in range(args.seeds if not name == "real-tf" else 1):
            lane = Lane(f"{name}#{seed}", model, states, builder, explicit, last_out, last_time, [41, user, seed, sum(map(ord, name))])
            lane.D = D
            if name.startswith("real"):
                tf = name == "real-tf"
                for i in range(end, stop):
                    c = int(rev["card_id"][i])
                    d = int(day[i])
                    rr = (int(rev["rating"][i]), int(rev["duration"][i]))
                    lane.review(ids_of[c], d - D, int(rev["elapsed_days"][i]), int(rev["elapsed_seconds"][i]), habits,
                                rating=rr[0] if tf else None, duration=rr[1] if tf else None, t_abs=None,
                                real=rr if name == "real-fr" else None)
                    lane.last_time[c] = d * SECONDS_PER_DAY + mu
            else:
                simulate_policy(lane, float(name.split("-")[1]), rev, end, stop, D, N, ids_of, window_cards, habits,
                                last_day, mu)
            mem = memorized(lane, window_cards, t_end)
            summary = summarise_days(lane.days, lane)
            summary["memorized_end"] = round(mem, 1)
            summary["wall_seconds"] = round(lane.wall, 2)
            summary["reviews_per_second"] = round(len(lane.records) / max(lane.wall, 1e-9), 1)
            summary["fluency"] = fluency(lane, rev, end, stop)
            summary["by_kind"] = by_kind(lane.records)
            result["lanes"][lane.name] = summary
            print(f"[sm41] user {user} {lane.name}: {json.dumps({k: v for k, v in summary.items() if k != 'series'})}",
                  flush=True)
            lanes.append(lane)
    result["model_steps"] = model.steps
    result["total_seconds"] = round(time.perf_counter() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(result) + "\n")


def by_kind(records):
    """Grade mix and mean seconds per review kind (first / same-day / later day)."""
    out = {}
    for k, name in ((0, "first"), (1, "same_day"), (2, "later")):
        g = np.array([r[2] for r in records if r[1] == k]); t = np.array([r[3] for r in records if r[1] == k])
        if len(g):
            out[name] = {"n": int(len(g)), "grade_mix": [round(float((g == x).mean()), 4) for x in (1, 2, 3, 4)],
                         "mean_seconds": round(float(t.mean()) / 1000, 2), "median_seconds": round(float(np.median(t)) / 1000, 2)}
    return out


def real_records(rev, end, stop, D):
    return [(int(rev["day_offset"][i]) - D, 0 if rev["elapsed_days"][i] == -1 else (1 if rev["elapsed_days"][i] == 0 else 2),
             int(rev["rating"][i]), int(rev["duration"][i]), -1.0, int(rev["card_id"][i])) for i in range(end, stop)]


def summarise_days(days, lane):
    keys = sorted(days)
    tot = {"reviews": 0, "later": 0, "later_pass": 0, "first": 0, "same": 0, "seconds": 0.0, "grades": [0, 0, 0, 0],
           "p_pass": 0.0, "pred_seconds": 0.0, "time_ll": 0.0, "time_n": 0, "grade_ll": 0.0}
    series = []
    for d in keys:
        r = days[d]
        for k in tot:
            if k == "grades":
                tot[k] = [a + b for a, b in zip(tot[k], r["grades"])]
            elif k in r:
                tot[k] += r[k]
        series.append([d, r["reviews"], round(r["seconds"], 1), r["later"], r["later_pass"]])
    n_days = 90 if not keys else max(keys) + 1
    s = {"reviews": tot["reviews"], "active_days": len(keys), "first_reviews": tot["first"],
         "same_day_reviews": tot["same"], "later_reviews": tot["later"],
         "retention_later": round(tot["later_pass"] / max(tot["later"], 1), 4),
         "grade_mix": [round(g / max(tot["reviews"], 1), 4) for g in tot["grades"]],
         "pass_grade_mix": [round(g / max(sum(tot["grades"][1:]), 1), 4) for g in tot["grades"][1:]],
         "minutes_per_day": round(tot["seconds"] / 60 / n_days, 2), "minutes_total": round(tot["seconds"] / 60, 1),
         "seconds_per_review": round(tot["seconds"] / max(tot["reviews"], 1), 2), "series": series}
    if lane is not None:
        s["predicted_retention_later"] = round(tot["p_pass"] / max(tot["later"], 1), 4)
        s["predicted_seconds_per_later_review"] = round(tot["pred_seconds"] / max(tot["later"], 1), 2)
        s["time_logloss"] = round(tot["time_ll"] / max(tot["time_n"], 1), 4)
        s["grade_logloss_non_first"] = round(tot["grade_ll"] / max(tot["later"] + tot["same"], 1), 4)
    return s


def fluency(lane, rev, end, stop):
    """New cards of the window: share whose predicted 90%-recall interval reached 21 days by the end, and the median
    answer time spent on them until then (seconds)."""
    new_cards = [int(c) for c, e in zip(rev["card_id"][end:stop], rev["elapsed_days"][end:stop]) if e == -1]
    new = set(new_cards)
    spent, done = {}, {}
    for (d, kind, rating, ms, p, card) in lane.records:
        if card in new and card not in done:
            spent[card] = spent.get(card, 0) + ms / 1000.0
    # maturity from the card's final curve (cheap version: one curve per card at the end)
    mature = 0
    for c in new:
        if c in lane.last_out:
            w, s, dd = lane.last_out[c]
            if time_at(w, s, dd, 0.9) >= 21 * SECONDS_PER_DAY:
                mature += 1
    times = [spent[c] for c in new if c in spent]
    return {"new_cards": len(new), "mature_share_end": round(mature / max(len(new), 1), 4),
            "median_seconds_per_new_card": round(float(np.median(times)), 1) if times else None}


def simulate_policy(lane, dr, rev, end, stop, D, N, ids_of, window_cards, habits, last_day, mu):
    """Desired-retention scheduler over N days: due cards (overdue first, then shuffled), then the day's real new cards
    in real order; same-day steps by the user's own rules; interval = the day the card's curve reaches dr."""
    rng = lane.rng
    day = rev["day_offset"]
    new_by_day = {}
    for i in range(end, stop):
        if rev["elapsed_days"][i] == -1:
            new_by_day.setdefault(int(day[i]) - D, []).append(int(rev["card_id"][i]))
    due = {}
    for c in window_cards:
        if c in lane.last_out:
            w, s, dd = lane.last_out[c]
            t = time_at(w, s, dd, dr)
            due[c] = max(0, last_day[c] + max(1, int(round(t / SECONDS_PER_DAY))) - D)
    last_review_day = {c: last_day[c] - D for c in due}
    for d in range(N):
        start = (D + d) * SECONDS_PER_DAY + float(np.clip(12 * 3600 + habits["sigma"] * rng.standard_normal(),
                                                          3600, 20 * 3600))
        now = start
        todays = [c for c, v in due.items() if v <= d]
        rng.shuffle(todays)
        todays.sort(key=lambda c: due[c])
        main = todays + new_by_day.get(d, [])
        main.reverse()
        learn = []
        same_count = {}
        while main or learn:
            if learn and learn[0][0] <= now:
                _, _, c = heapq.heappop(learn)
            elif main:
                c = main.pop()
            else:
                now = learn[0][0]
                continue
            if c in lane.last_time:
                es = int(round(now - lane.last_time[c]))
                ed = d - last_review_day[c]
            else:
                es, ed = -1, -1
            rating, dur = lane.review(ids_of[c], d, ed, max(es, 1) if ed != -1 else -1, habits)
            now += dur / 1000.0
            lane.last_time[c] = now
            last_review_day[c] = d
            kind = 0 if ed == -1 else (1 if ed == 0 else 2)
            p, gaps = habits["rules"][(kind, rating)]
            same_count[c] = same_count.get(c, 0) + 1
            if same_count[c] < MAX_SAME_DAY and rng.random() < p:
                heapq.heappush(learn, (now + float(rng.choice(gaps)), int(rng.integers(1 << 30)), c))
                due.pop(c, None)
            else:
                w, s, dd = lane.last_out[c]
                t = time_at(w, s, dd, dr)
                due[c] = d + max(1, int(round(t / SECONDS_PER_DAY)))


# ------------------------------------------------------------------ CPU proof
def check(args):
    low_priority()
    from scripts import small_model_heldout as smh  # noqa: F401  first (state-quant install, RWKV_NO_JIT, LongWKV)
    import torch
    torch.set_num_threads(2)
    from rwkv.prepare_batch import prepare
    from scripts import compress_common as cc
    from scripts import sm82c_head as h
    user = args.user
    rev, frame = load_user(user)
    rows = args.rows
    res = {"user": user, "rows": rows}

    # C1: the builder's 24 base columns == create_sample's (whole history, query + review rows, section order)
    from rwkv.data_processing import create_sample
    sample = create_sample(user, frame, [], torch.float32, torch.device("cpu"))
    ref = sample.card_features.numpy()
    b = Builder(day0=int(rev["day_offset"][0]), seed=0)
    mine = []
    for i in range(len(rev["card_id"])):
        q, f = b.row(int(rev["card_id"][i]), int(rev["note_id"][i]), int(rev["deck_id"][i]), int(rev["preset_id"][i]),
                     int(rev["day_offset"][i]), int(rev["elapsed_days"][i]), int(rev["elapsed_seconds"][i]),
                     int(rev["rating"][i]), int(rev["duration"][i]))
        if q is not None:
            mine.append(q[:24])
        mine.append(f[:24])
    mine = np.stack(mine)
    ref = ref.copy()
    ref[:, 22] = 0.0    # scaled_state: zeroed by the model (zero_features 22), not simulated
    gap = float(np.abs(mine - ref).max()) if mine.shape == ref.shape else float("inf")
    res["C1_builder_vs_get_rwkv_data_max_abs"] = gap
    res["C1_ok"] = gap < 1e-5
    print(f"[sm41-check] C1 builder == get_rwkv_data base columns ({mine.shape} vs {ref.shape}): max abs {gap:.2e}",
          flush=True)

    # C2: recurrent step (this script's Model.step, interleaved, with snapshots) == the parallel model (grade outputs,
    # curve, and the SM82c 448 frozen features at the instant rows), on the first `rows` reviews
    sub = frame.iloc[:rows]
    batch = prepare(cc.make_samples(sub, user, len(sub)), seed=2025 + user)
    s0 = create_sample(user, sub, [], torch.float32, torch.device("cpu"))
    sd = torch.load(MASTER, map_location="cpu", weights_only=True)
    fz = h.Frozen(h.build_frozen({k: v.float() for k, v in sd.items()}, torch.device("cpu")))
    with torch.inference_mode():
        qi, th, feats, _ = fz.forward(batch)
        par = fz.model.get_loss(batch)
    # C1b: all 92 columns: the builder with the scorer batch's id codes and day baselines reproduces batch.start
    start = batch.start.float().numpy()
    b2 = Builder(day0=int(rev["day_offset"][0]), seed=0)
    rel0 = int(rev["day_offset"][0]) - b2.day0
    k = 24 + sum(dim for _, dim in ID_DIMS)
    for j, P in enumerate(PERIODS):
        ang = math.atan2(start[0, k + 4 * j], start[0, k + 4 * j + 1]) % (2 * math.pi)
        b2.base[j] = int(round(ang / (2 * math.pi / P) - rel0)) % P
    rows_of = {}
    for r in range(start.shape[0]):
        for (stream, dim), off in zip(ID_DIMS, (24, 36, 48, 56)):
            b2.codes[stream].setdefault(int(s0.ids[stream][r]), start[r, off:off + dim].copy())
    mine2 = []
    for i in range(rows):
        qq, ff = b2.row(int(rev["card_id"][i]), int(rev["note_id"][i]), int(rev["deck_id"][i]), int(rev["preset_id"][i]),
                        int(rev["day_offset"][i]), int(rev["elapsed_days"][i]), int(rev["elapsed_seconds"][i]),
                        int(rev["rating"][i]), int(rev["duration"][i]))
        if qq is not None:
            mine2.append(qq)
        mine2.append(ff)
    mine2 = np.stack(mine2)
    ref2 = start.copy()
    ref2[:, 22] = 0.0
    gap1b = float(np.abs(mine2 - ref2).max()) if mine2.shape == ref2.shape else float("inf")
    res["C1b_all_92_columns_max_abs"] = gap1b
    res["C1_ok"] = res["C1_ok"] and gap1b < 1e-5
    print(f"[sm41-check] C1b all 92 columns with the scorer's codes and baselines: max abs {gap1b:.2e}", flush=True)
    model = Model()
    feats_rnn, imm, crv = {}, [], []
    states = {s: {} for s in model.streams}
    qset = set(qi.tolist())
    labels = batch.labels[0]
    for r in range(start.shape[0]):
        keys = {s: int(s0.ids[s][r]) for s in model.streams}
        out, nxt, fr = model.step(start[r].copy(), {s: states[s].get(keys[s]) for s in model.streams}, snaps=r in qset)
        if not bool(s0.skips[r]):
            for s in model.streams:
                states[s][keys[s]] = nxt[s]
        if r in qset:
            feats_rnn[r] = fr.numpy()
        imm.append(1 - out["p"][0])
        crv.append(float(curve(out["w"], out["s"], out["d"], float(labels[r, 0]))))
    lab = (labels[:, 4] > 0).numpy()
    imm = np.asarray(imm); crv = np.asarray(crv)
    q = (labels[:, 6] == 1).numpy() & lab
    c = (labels[:, 6] == 0).numpy() & lab
    gi = float(np.abs(imm[q] - par.p_imm[0].numpy()[q]).max())
    gc = float(np.abs(crv[c] - par.p_curve[0].numpy()[c]).max())
    gf = float(max(np.abs(feats_rnn[int(r)] - feats[k].numpy()).max() for k, r in enumerate(qi.tolist())))
    res.update({"C2_instant_max_abs": gi, "C2_curve_max_abs": gc, "C2_frozen_features_max_abs": gf,
                "C2_ok": gi < 1e-4 and gc < 1e-4 and gf < 1e-3, "C2_instant_rows": int(q.sum()), "C2_curve_rows": int(c.sum())})
    print(f"[sm41-check] C2 recurrent vs parallel on {rows} reviews: instant {gi:.2e}, curve {gc:.2e}, frozen features "
          f"{gf:.2e}", flush=True)

    # C3: incremental explicit features == sm82c_head.explicit_features
    table = h.explicit_features(rev["card_id"], rev["rating"], rev["duration"], rev["elapsed_days"])
    e = Explicit()
    gap3 = 0.0
    for i in range(len(rev["card_id"])):
        row = e.features(int(rev["card_id"][i]), int(rev["elapsed_days"][i]))
        gap3 = max(gap3, float(np.abs(row - table[i]).max()))
        e.push(int(rev["card_id"][i]), int(rev["rating"][i]), int(rev["duration"][i]))
    res["C3_explicit_max_abs"] = gap3
    res["C3_ok"] = gap3 < 1e-6
    print(f"[sm41-check] C3 explicit features incremental vs table: max abs {gap3:.2e}", flush=True)
    res["ok"] = res["C1_ok"] and res["C2_ok"] and res["C3_ok"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[sm41-check] {'PASS' if res['ok'] else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ batched-step throughput
def bench(args):
    """Throughput of one recurrent step over B independent learners (states stacked along the batch dimension): the
    full-run design's inner loop. Random states, CPU or CUDA."""
    low_priority()
    import torch
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    model = Model(threads=args.threads)
    m = model.rnn.to(dev)
    head = model.head.to(dev)
    res = {}
    for B in [int(x) for x in args.batch.split(",")]:
        x0 = torch.randn(B, 92, device=dev)
        st = {}
        with torch.inference_mode():
            # one step from scratch to get state shapes, then reuse them
            for s in m.streams:
                st[s] = None
            def one(states):
                x = m.features2card(x0)
                nxt = {s: dict(states[s]) if states[s] is not None else {} for s in m.streams}
                v0 = [torch.empty_like(x) for _ in m.streams]
                shots = [x]
                for k, (i, layer) in enumerate(m.layer_steps):
                    s = m.streams[i]
                    x, v0[i], nxt[s][layer] = m.rwkv_modules[i].blocks[layer](in_BC=x, v0_BC=v0[i], state=nxt[s].get(layer))
                    shots.append(x)
                hh = m.prehead_norm(x)
                hp = m.head_p(hh)
                z = torch.cat([torch.nn.functional.layer_norm(t, (64,)) for t in shots[:5]] + [hp, torch.zeros(B, 26, device=dev)], -1)
                head(z)
                m.head_w(hh)
                return nxt
            st = one(st)
            reps = args.reps
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                one(st)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            el = time.perf_counter() - t
        # one simulated review = 2 steps (instant row + review row)
        res[B] = {"steps_per_second": round(B * reps / el, 1), "reviews_per_second": round(B * reps / el / 2, 1),
                  "ms_per_step_call": round(1000 * el / reps, 3)}
        print(f"[sm41-bench] {args.device} B={B}: {res[B]}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[f"{args.device}-threads{args.threads}"] = res
    out.write_text(json.dumps(prev, indent=1))


# ------------------------------------------------------------------ report
def block_retention(series, lo, hi):
    later = sum(r[3] for r in series if lo <= r[0] < hi)
    passed = sum(r[4] for r in series if lo <= r[0] < hi)
    return passed / later if later else float("nan")


def report(args):
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["user"])
    bench_path = HERE / "bench.json"
    bench_rows = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    out = {"users": [], "policies": {}, "speed": {}}
    L = ["# The small model in the review simulator (SM41): stage 1 prototype tables", ""]
    L.append(f"{len(rows)} users of the 732 set ([users.json](users.json)), 90 simulated days from a split day inside each "
             "history, CPU (1 thread per process, 4 processes). Learner: the released leak-free small model + the "
             "answer-time head on richer frozen features (SM82c). Design and definitions: [DESIGN.md](DESIGN.md). "
             "Lanes: `real-tf` = the real continuation with real grades and times fed back (teacher forcing); `real-fr` "
             "= the real schedule with sampled grades and times fed back (2 seeds); `dr-r` = desired-retention "
             "scheduler at r (2 seeds). Retention = pass share of later-day reviews (interval >= 1 day).")
    L.append("")
    L.append("## 1. Validity: the real schedule replayed through the model")
    L.append("")
    L.append("| user | split day | warm-up reviews | window reviews | real retention | teacher-forced predicted | "
             "free-running (sampled) | free-running days 0-29 / 60-89 | real pass mix H/G/E | free-running pass mix | "
             "real min/day | free-running min/day | real s per later review | teacher-forced predicted s | "
             "free-running s | grade LogLoss (tf) | time LogLoss (tf) | pools: retention / min/day / s per later review |")
    L.append("|" + "---|" * 18)
    agg = {"real": [], "tf": [], "fr": [], "rmin": [], "frmin": [], "rs": [], "tfs": [], "frs": [], "pr": [],
           "pmin": [], "ps": []}
    for r in rows:
        lanes = r["lanes"]
        tf = lanes["real-tf#0"]
        frs = [v for k, v in lanes.items() if k.split("#")[0] == "real-fr"]
        pools = [v for k, v in lanes.items() if k.split("#")[0] == "real-fr-pool"]
        pool_ret = float(np.mean([v["retention_later"] for v in pools])) if pools else float("nan")
        pool_min = float(np.mean([v["minutes_per_day"] for v in pools])) if pools else float("nan")
        pool_s = float(np.mean([v["by_kind"]["later"]["mean_seconds"] for v in pools])) if pools else float("nan")
        real = r["real"]
        fr_ret = float(np.mean([v["retention_later"] for v in frs]))
        fr_b1 = float(np.nanmean([block_retention(v["series"], 0, 30) for v in frs]))
        fr_b3 = float(np.nanmean([block_retention(v["series"], 60, 90) for v in frs]))
        fr_mix = np.mean([v["pass_grade_mix"] for v in frs], 0)
        fr_min = float(np.mean([v["minutes_per_day"] for v in frs]))
        fr_s = float(np.mean([v["by_kind"]["later"]["mean_seconds"] for v in frs]))
        rs = real["by_kind"]["later"]["mean_seconds"]
        good_only = real["retention_later"] >= 0.995
        flag = " (Good-only)" if good_only else ""
        L.append(f"| {r['user']}{flag} | {r['split_day']} | {r['warmup_reviews']:,} | {r['window_reviews']:,} | "
                 f"{real['retention_later']:.3f} | {tf['predicted_retention_later']:.3f} | {fr_ret:.3f} | "
                 f"{fr_b1:.3f} / {fr_b3:.3f} | {'/'.join(f'{x:.2f}' for x in real['pass_grade_mix'])} | "
                 f"{'/'.join(f'{x:.2f}' for x in fr_mix)} | {real['minutes_per_day']:.1f} | {fr_min:.1f} | {rs:.1f} | "
                 f"{tf['predicted_seconds_per_later_review']:.1f} | {fr_s:.1f} | {tf['grade_logloss_non_first']:.3f} | "
                 f"{tf['time_logloss']:.3f} | {pool_ret:.3f} / {pool_min:.1f} / {pool_s:.1f} |")
        u = {"user": r["user"], "good_only": good_only, "real_retention": real["retention_later"],
             "tf_predicted_retention": tf["predicted_retention_later"], "fr_retention": fr_ret,
             "fr_retention_days_0_29": fr_b1, "fr_retention_days_60_89": fr_b3,
             "real_min_per_day": real["minutes_per_day"], "fr_min_per_day": fr_min, "real_s_later": rs,
             "tf_pred_s_later": tf["predicted_seconds_per_later_review"], "fr_s_later": fr_s,
             "real_pass_mix": real["pass_grade_mix"], "fr_pass_mix": fr_mix.tolist(),
             "tf_grade_logloss": tf["grade_logloss_non_first"], "tf_time_logloss": tf["time_logloss"],
             "fr_pool_retention": pool_ret, "fr_pool_min_per_day": pool_min, "fr_pool_s_later": pool_s}
        out["users"].append(u)
        if not good_only:
            for k, v in (("real", real["retention_later"]), ("tf", tf["predicted_retention_later"]), ("fr", fr_ret),
                         ("rmin", real["minutes_per_day"]), ("frmin", fr_min), ("rs", rs),
                         ("tfs", tf["predicted_seconds_per_later_review"]), ("frs", fr_s), ("pr", pool_ret),
                         ("pmin", pool_min), ("ps", pool_s)):
                agg[k].append(v)
    if agg["real"]:
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        out["validity_mean_excluding_good_only"] = m
        L.append(f"| mean, {len(agg['real'])} users without Good-only | | | | {m['real']:.3f} | {m['tf']:.3f} | "
                 f"{m['fr']:.3f} | | | | {m['rmin']:.1f} | {m['frmin']:.1f} | {m['rs']:.1f} | {m['tfs']:.1f} | "
                 f"{m['frs']:.1f} | | | {m['pr']:.3f} / {m['pmin']:.1f} / {m['ps']:.1f} |")
    L.append("")
    L.append("## 2. Validity and outcomes: the desired-retention scheduler")
    L.append("")
    L.append("Same card set as the real window (old cards reviewed in the window + the window's real new cards, in real "
             "order and on their real day). Memorised = sum of the cards' predicted recall at the end of day 90; the "
             "real row uses the teacher-forced lane's curves. Day-0 reviews = the split day's reviews (the backlog of "
             "old cards already due under the policy). Mean of 2 seeds.")
    L.append("")
    L.append("| user | schedule | later-day retention | instant-head predicted | reviews/day | min/day | "
             "memorised at day 90 | min per memorised card | day-0 reviews | new cards mature (90% interval >= 21 d) |")
    L.append("|" + "---|" * 10)
    pol_agg = {}
    for r in rows:
        lanes = r["lanes"]
        real, tf = r["real"], lanes["real-tf#0"]
        n_days = r["days"]
        good_only = real["retention_later"] >= 0.995
        flag = " (Good-only)" if good_only else ""
        day0_real = sum(x[1] for x in real["series"] if x[0] == 0)
        L.append(f"| {r['user']}{flag} | real | {real['retention_later']:.3f} | {tf['predicted_retention_later']:.3f} | "
                 f"{real['reviews'] / n_days:.1f} | {real['minutes_per_day']:.1f} | {tf['memorized_end']:.0f} | "
                 f"{real['minutes_total'] / max(tf['memorized_end'], 1):.2f} | {day0_real} | "
                 f"{tf['fluency']['mature_share_end']:.2f} |")
        names = sorted({k.split("#")[0] for k in lanes if k.startswith("dr-")})
        for name in names:
            vs = [v for k, v in lanes.items() if k.split("#")[0] == name]

            def g(key):
                return float(np.mean([v[key] for v in vs]))
            day0 = float(np.mean([sum(x[1] for x in v["series"] if x[0] == 0) for v in vs]))
            mature = float(np.mean([v["fluency"]["mature_share_end"] for v in vs]))
            rec = {"retention": g("retention_later"), "predicted": g("predicted_retention_later"),
                   "reviews_per_day": g("reviews") / n_days, "min_per_day": g("minutes_per_day"),
                   "memorized": g("memorized_end"), "min_per_memorized": g("minutes_total") / max(g("memorized_end"), 1),
                   "day0_reviews": day0, "mature": mature}
            out["policies"].setdefault(str(r["user"]), {})[name] = rec
            L.append(f"| {r['user']}{flag} | {name} | {rec['retention']:.3f} | {rec['predicted']:.3f} | "
                     f"{rec['reviews_per_day']:.1f} | {rec['min_per_day']:.1f} | {rec['memorized']:.0f} | "
                     f"{rec['min_per_memorized']:.2f} | {day0:.0f} | {mature:.2f} |")
            if not good_only:
                pol_agg.setdefault(name, []).append(rec)
    if pol_agg:
        L.append("")
        L.append("Mean over the users without Good-only:")
        L.append("")
        L.append("| schedule | target | later-day retention | instant-head predicted | min/day | memorised at day 90 | "
                 "min per memorised card |")
        L.append("|---|---|---|---|---|---|---|")
        for name, recs in sorted(pol_agg.items()):
            mm = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
            out.setdefault("policy_means", {})[name] = mm
            L.append(f"| {name} | {name.split('-')[1]} | {mm['retention']:.3f} | {mm['predicted']:.3f} | "
                     f"{mm['min_per_day']:.1f} | {mm['memorized']:.0f} | {mm['min_per_memorized']:.2f} |")
    L.append("")
    L.append("## 3. Speed")
    L.append("")
    wr = [r["warmup_reviews"] / r["warmup_seconds"] for r in rows]
    lr = [v["reviews_per_second"] for r in rows for v in r["lanes"].values()]
    tot_rev = sum(v["reviews"] for r in rows for v in r["lanes"].values())
    tot_s = sum(r["total_seconds"] for r in rows)
    out["speed"] = {"warmup_reviews_per_second_mean": float(np.mean(wr)),
                    "lane_reviews_per_second_mean": float(np.mean(lr)), "simulated_reviews": int(tot_rev),
                    "process_seconds": tot_s}
    L.append(f"- Warm-up (one model step per real review): mean {np.mean(wr):.0f} reviews/s per process "
             f"(range {min(wr):.0f}-{max(wr):.0f}).")
    L.append(f"- Simulated reviews (2 model steps each: instant row + review row, plus the time head): mean "
             f"{np.mean(lr):.0f} reviews/s per process (range {min(lr):.0f}-{max(lr):.0f}); {tot_rev:,} lane reviews, "
             f"{tot_s / 60:.1f} process-minutes in all incl. warm-up (4 processes at below-normal priority, other CPU "
             "work running).")
    for key, res in bench_rows.items():
        L.append(f"- Batched step (`bench`, {key}): " + ", ".join(
            f"B={b}: {v['reviews_per_second']:,.0f} reviews/s ({v['ms_per_step_call']} ms per step)" for b, v in res.items()))
    L.append("")
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print("\n".join(L))


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--user", type=int, default=722); c.add_argument("--rows", type=int, default=3000)
    c.add_argument("--out", default=str(HERE / "check-cpu.json"))
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True); r.add_argument("--split-day", type=int, required=True)
    r.add_argument("--days", type=int, default=90); r.add_argument("--lanes", default="real-tf,real-fr,real-fr-pool,dr-0.80,dr-0.90,dr-0.95")
    r.add_argument("--seeds", type=int, default=1); r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", default=str(HERE / "prototype.jsonl"))
    b = sub.add_parser("bench"); b.add_argument("--batch", default="1,16,256,2048"); b.add_argument("--device", default="cpu")
    b.add_argument("--threads", type=int, default=1); b.add_argument("--reps", type=int, default=20)
    b.add_argument("--out", default=str(HERE / "bench.json"))
    q = sub.add_parser("report"); q.add_argument("--rows", default=str(HERE / "prototype.jsonl"))
    q.add_argument("--md", default=str(HERE / "prototype-tables.md"))
    q.add_argument("--json", default=str(HERE / "prototype-result.json"))
    a = p.parse_args()
    {"check": check, "run": run, "bench": bench, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
