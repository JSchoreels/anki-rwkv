#!/usr/bin/env python
"""The small model in the review simulator (SM41), stage 2 part 2: the general vectorised lane engine (CPU).

Part 1 (scripts/sm41_drift.py) advanced every lane of a split in lock-step on the REAL schedule: one card per tick for
all lanes. Here lanes run their own schedules (a policy picks each card's next day), so at a tick every lane reviews a
different card. Design: artifacts/small-model/sm41/DESIGN.md section 9.

- State arenas: per stream, per layer, per state tensor one tensor [entities, lanes, ...] (the card set of the window,
  their notes / decks / presets, the user), filled from the shared warm-up; a tick gathers (entity of lane b, lane b)
  for the active lanes, runs one batched query step and one batched commit step, and scatters the commit back. A
  "fresh" mask per (entity, lane) marks entities the lane has not seen yet: in the batched step their rows behave
  exactly like the stock first step (state None: the token shift reads the row's own layer-normed input, zero wkv
  state), via forward hooks on the mixers' layer norms (no copy of the model code).
- Features: the 92 inputs of `sm41_sim.Builder` vectorised over lanes (per-lane counters, per lane x card history),
  the entity id codes drawn once in the real window order (common to all lanes, = part 1's codes).
- Interval convention (DESIGN.md 8.1, conv "show"): query rows from the lane's previous answer of the card to the show
  moment, commit rows answer to answer with the lane's own answer time.
- Re-anchoring (daily shared-stream re-anchoring (fr-shared1), part 1's best arm): lane 0 is teacher forcing on the
  real schedule; at the start of every day the deck / preset / user states of every anchored lane are replaced by
  lane 0's. Card and note states stay the lane's own.
- Lanes: replay lanes (`tf`, `fr`, `fr-shared1`, ...) replay the real schedule exactly as part 1 (same draws: proof
  P3); policy lanes (`ihr-R`, `fsrs7-R`, `curve-R`, suffix `-free` = no re-anchoring) run their own sessions:
  due cards (overdue first), then the day's real new cards in real order, same-day steps by the user's own rules
  (sm41_sim.user_habits), every card's next day from the policy:
    ihr-R    instant-head desired retention: K candidate query rows per card (the card's post-commit state, elapsed =
             each candidate interval), one batched step, the first interval where P(pass) falls to R (log-linear
             interpolation on the grid);
    fsrs7-R  FSRS-7 (34 parameters, srs-benchmark's finished dual-stability model, ported to numpy) at desired
             retention R, the user's own srs-benchmark fit (FSRS-7 short secs), updated with the lane's own grades and
             answer-to-answer intervals: the baseline scheduler;
    curve-R  stage 1's curve-head desired retention (the commit row's GRU power curves).
- Draws (policy lanes): counter-based uniforms keyed (user, split, seed, card, the card's review number in the lane,
  slot), so lanes of the same seed share draws card by card whatever the policy (common random numbers), and a lane
  run alone draws exactly what it draws in a batch. Replay lanes use part 1's stream.

  proof    P1 batched == each lane alone; P2 engine == a scalar reference built on stage 1's Builder and the stock
           recurrent step; P3 replay lanes reproduce part 1's fr-shared1 rows
  run      --user U --split D --days N --lanes ...   one user, one split -> JSONL row (resumable, quiet.json holds)
  report   the validation tables -> artifacts/small-model/sm41/engine/engine-tables.md / .json
"""
from argparse import ArgumentParser
import copy
import datetime as dt
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
from scripts import sm41_sim as S   # noqa: E402
from scripts import sm41_drift as DR   # noqa: E402
from rwkv import data_processing as DP   # noqa: E402

OUT = S.HERE / "engine"
DAY = S.SECONDS_PER_DAY
MU = 12 * 3600.0                       # warm-up cards: their last answer at 12:00 of their last day (stage 1)
BUCKET = DR.BUCKET
SHARED = DR.SHARED
T_GRID = np.unique(np.round(np.geomspace(1, 1000, 16)))     # candidate intervals, days (16 values)
FSRS7_FITS = Path("C:/Users/jonat/workspace/srs-benchmark/result/FSRS-7-short-secs.jsonl")   # read only


# ------------------------------------------------------------------ lane configuration
def lane_cfg(name):
    """'tf' | replay arm of part 1 ('fr', 'fr-shared1', 'fr-shared7') + '#seed' | policy 'ihr-0.90', 'fsrs7-0.90',
    'curve-0.90' (+ '-free': no shared-stream re-anchoring) + '#seed'."""
    base, _, seed = name.partition("#")
    seed = int(seed) if seed else 0
    if base == "tf" or base.startswith("fr"):
        a = DR.arm_cfg(base)
        assert a["conv"] == "show" and a["time"] == "sample" and a["grade_temp"] == 1.0 and a["time_temp"] == 1.0 \
            and not a["anchor_days"], f"replay arm {base}: only teacher forcing / plain / shared-stream arms"
        return {"name": name, "base": base, "kind": "replay", "teacher": a["teacher"], "anchor": a["anchor_shared"],
                "seed": seed}
    parts = base.split("-")
    free = parts[-1] == "free"
    return {"name": name, "base": base, "kind": "policy", "teacher": False, "anchor": 0 if free else 1,
            "policy": parts[0], "r": float(parts[1]), "seed": seed}


# ------------------------------------------------------------------ counter-based draws
_G = [np.uint64(x) for x in (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB)]


def _mix(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * _G[1]
        z = (z ^ (z >> np.uint64(27))) * _G[2]
    return z ^ (z >> np.uint64(31))


def uniforms(key, *parts):
    """Uniform [0, 1) per element, a pure function of (key, parts...) (arrays broadcast): splitmix64 chain."""
    with np.errstate(over="ignore"):
        z = _mix(np.asarray(key, np.uint64) * _G[0])
        for p in parts:
            z = _mix(z + np.asarray(p, np.int64).astype(np.uint64) * _G[0])
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


# ------------------------------------------------------------------ FSRS-7 (numpy port of srs-benchmark models/fsrs_v7.py)
FSRS7_DEFAULT = [0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0, 1.3826, 0.7024, 0.5999, 0.8146,
                 0.6398, 1.0, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567, 0.0801, 0.2421,
                 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048]


class FSRS7:
    """Finished dual-stability FSRS-7 (34 parameters); state [..., 3] = long-term S, short-term S, D (zeros = new)."""
    S_MIN, S_MAX, D_MIN, D_MAX = 0.0001, 36500.0, 1.0, 10.0      # --secs: s_min 1e-4 (srs-benchmark config.py)

    def __init__(self, w):
        self.w = np.asarray(w, np.float64)

    def init_d(self, rating):
        return self.w[4] - np.exp(self.w[5] * (rating - 1)) + 1

    def r1(self, t, ss):
        w = self.w
        t = np.maximum(t, 0.0)
        decay1 = -np.clip(w[23] * ss ** (w[33] - 0.3), 0.01, 0.95)
        factor1 = np.exp(np.minimum(np.log(w[25]) / decay1, 60.0)) - 1.0
        return (t / ss * factor1 + 1.0) ** decay1

    def recall(self, t, s, ss, d):
        w = self.w
        t = np.maximum(t, 0.0)
        r1 = self.r1(t, ss)
        decay2 = -np.clip(w[24], 0.01, 0.95)
        factor2 = w[26] ** (1.0 / decay2) - 1.0
        r2 = (t / s * factor2 * np.exp((d - 5.0) * (w[32] - 0.3)) + 1.0) ** decay2
        w1 = w[27] * ss ** (-w[29])
        w2 = w[28] * s ** w[30] * np.exp((d - 5.0) * (w[31] - 0.5))
        return ((w1 * r1 + w2 * r2) / (w1 + w2)) * (1.0 - 2e-5) + 1e-5

    def _stab(self, last_s, last_d, r, rating, k):
        w = self.w
        hard = np.where(rating == 2, w[k + 6], 1.0)
        easy = np.where(rating == 4, w[k + 7], 1.0)
        fail = w[k + 3] * ((last_s + 1.0) ** w[k + 4] - 1.0) * np.exp((1.0 - r) * w[k + 5])
        pls = np.minimum(last_s, fail)
        sinc = np.exp(w[k] - 1.5) * (11.0 - last_d) * last_s ** (-w[k + 1]) * (np.exp((1.0 - r) * w[k + 2]) - 1.0) \
            * hard * easy + 1.0
        return np.where(rating > 1, np.maximum(pls, last_s * sinc), pls)

    def step(self, dt_days, rating, state):
        w = self.w
        rating = np.asarray(rating, np.float64)
        first = (state == 0).all(-1)
        idx = np.clip(rating.astype(np.int64), 1, 4) - 1
        i_s = w[idx]
        i_d = np.clip(self.init_d(rating), self.D_MIN, self.D_MAX)
        s = np.clip(state[..., 0], self.S_MIN, self.S_MAX)
        ss = np.clip(state[..., 1], self.S_MIN, self.S_MAX)
        d = np.clip(state[..., 2], self.D_MIN, self.D_MAX)
        r = self.recall(dt_days, s, ss, d)
        u_s = self._stab(s, d, r, rating, 7)
        u_ss = self._stab(ss, d, self.r1(dt_days, ss), rating, 15)
        u_ss = np.where(rating == 1, np.minimum(u_ss, 0.8 * u_s), u_ss)
        delta = -w[6] * (rating - 3)
        delta = np.where(rating == 1, delta * (r + 0.1), delta)
        u_d = d + delta * (10 - d) / 9
        u_d = np.clip(0.01 * self.init_d(4.0) + 0.99 * u_d, self.D_MIN, self.D_MAX)
        out = np.stack([np.where(first, i_s, u_s), np.where(first, 0.8 * i_s, u_ss), np.where(first, i_d, u_d)], -1)
        out[..., :2] = np.clip(out[..., :2], self.S_MIN, self.S_MAX)
        out[..., 2] = np.clip(out[..., 2], self.D_MIN, self.D_MAX)
        return out

    def interval(self, state, r):
        """Days t where recall(t) = r (bisection in ln t over [1e-4, 1e6] days), per row."""
        s, ss, d = state[..., 0], state[..., 1], state[..., 2]
        r = np.broadcast_to(np.asarray(r, np.float64), s.shape)
        lo = np.full(s.shape, math.log(1e-4)); hi = np.full(s.shape, math.log(1e6))
        for _ in range(40):
            mid = (lo + hi) / 2
            ok = self.recall(np.exp(mid), s, ss, d) >= r
            lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
        return np.exp(lo)


def fsrs7_params(user):
    if FSRS7_FITS.exists():
        with FSRS7_FITS.open() as fh:
            for line in fh:
                if f'"user": {user},' not in line:
                    continue
                r = json.loads(line)
                if int(r["user"]) == int(user) and r.get("parameters"):
                    ps = r["parameters"]
                    return ps[sorted(ps, key=int)[-1]], "user FSRS-7 fit (srs-benchmark FSRS-7-short-secs)"
    return FSRS7_DEFAULT, "FSRS-7 default parameters"


# ------------------------------------------------------------------ curve-head interval (vectorised sm41_sim.time_at)
def curve_np(w, s_raw, d_raw, t):
    return S.curve(w, s_raw, d_raw, t)


def time_at_vec(w, s_raw, d_raw, r):
    """sm41_sim.time_at per row (w, s, d [n, 3], r [n]): the same bounds, 40 bisection steps in ln t."""
    n = w.shape[0]
    top = math.log(1000 * 365 * DAY)
    lo = np.zeros(n); hi = np.full(n, top)
    at_top = curve_np(w, s_raw, d_raw, np.full(n, math.exp(top))) >= r
    at_one = curve_np(w, s_raw, d_raw, np.ones(n)) < r
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = curve_np(w, s_raw, d_raw, np.exp(mid)) >= r
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return np.where(at_top, math.exp(top), np.where(at_one, 1.0, np.exp(lo)))


# ------------------------------------------------------------------ instant-head interval from the candidate grid
def choose_interval(pg, r):
    """pg [n, K] P(pass) at T_GRID, r [n]: (t days float, rounded days >= 1, P(pass) at the rounded days by log-linear
    interpolation). t = the first crossing of r (log-linear between grid points); 1 if P(pass at 1 day) < r already;
    the grid end if never."""
    lg = np.log(T_GRID)
    below = pg < r[:, None]
    j = np.argmax(below, 1)
    none = ~below.any(1)
    jm = np.maximum(j - 1, 0)
    rows = np.arange(len(r))
    p0, p1 = pg[rows, jm], pg[rows, j]
    frac = np.where(p0 > p1, (p0 - r) / np.where(p0 > p1, p0 - p1, 1.0), 0.0)
    t = np.exp(lg[jm] + np.clip(frac, 0, 1) * (lg[j] - lg[jm]))
    t = np.where(j == 0, T_GRID[0], t)
    t = np.where(none, T_GRID[-1], t)
    ti = np.maximum(1, np.round(t)).astype(np.int64)
    aim = np.array([np.interp(math.log(x), lg, pg[i]) for i, x in enumerate(ti)])
    return t, ti, aim


# ------------------------------------------------------------------ fresh-row hooks (state None emulated in a batch)
def install_fresh_hooks(model):
    if getattr(model, "_sm41_hooks", False):
        return

    def hook(mod, inp, out):
        f = mod.__dict__.get("_sm41_fill")
        if f is not None:
            buf, idx = f
            buf[idx] = out[idx]
    for mod in model.rnn.rwkv_modules:
        for blk in mod.blocks:
            blk.time_mixer.layer_norm.register_forward_hook(hook)
            if not getattr(blk, "cmix_stripped", False):
                blk.channel_mixer.layer_norm.register_forward_hook(hook)
    model._sm41_hooks = True


def _set_fill(mod, val):
    mod.__dict__["_sm41_fill"] = val


def vstep(model, X, st, fresh, snaps=False, curve=False):
    """One batched step: X [n, 92]; st[stream][layer] = nested state with [n, ...] leaves (gathered copies); fresh[stream]
    = bool [n] (rows whose entity has no state yet). Returns p [n, 4] (float64), next states, frozen [n, 448] | None,
    (w, s, d) numpy | None."""
    torch = model.torch
    m = model.rnn
    model.steps += X.shape[0]
    with torch.inference_mode():
        x = torch.from_numpy(X)
        if m.zero_features:
            x = x.clone()
            x[:, m.zero_features] = 0
        x = m.features2card(x)
        shots = [x] if snaps else None
        nxt = {s: {} for s in m.streams}
        v0 = [None] * len(m.streams)
        ls = m.layer_steps
        for k, (i, layer) in enumerate(ls):
            s = m.streams[i]
            blk = m.rwkv_modules[i].blocks[layer]
            state = st[s][layer]
            fr = fresh[s]
            filled = False
            if fr is not None and fr.any():
                idx = torch.from_numpy(np.nonzero(fr)[0])
                (shift, wkv), cstate = state
                wkv[idx] = 0
                _set_fill(blk.time_mixer.layer_norm, (shift, idx))
                if cstate is not None:
                    _set_fill(blk.channel_mixer.layer_norm, (cstate, idx))
                filled = True
            x, v0[i], nxt[s][layer] = blk(in_BC=x, v0_BC=v0[i] if v0[i] is not None else torch.empty_like(x),
                                          state=state)
            if filled:
                _set_fill(blk.time_mixer.layer_norm, None)
                _set_fill(blk.channel_mixer.layer_norm, None)
            if snaps and (k + 1 == len(ls) or ls[k + 1][1] != layer):
                shots.append(x)
        h = m.prehead_norm(x)
        hp = m.head_p(h).float()
        p = torch.softmax(m.p_linear(hp), -1).numpy().astype(np.float64)
        frozen = None
        if snaps:
            frozen = torch.cat([torch.nn.functional.layer_norm(t.float(), (64,)) for t in shots] + [hp], -1)
        wsd = None
        if curve:
            x_w = m.head_w(h).float()
            wsd = (torch.softmax(torch.nn.functional.linear(x_w, m.gru_w_weight, m.gru_w_bias), -1).numpy(),
                   torch.nn.functional.linear(x_w, m.gru_s_weight, m.gru_s_bias).numpy(),
                   torch.nn.functional.linear(x_w, m.gru_d_weight, m.gru_d_bias).numpy())
    return p, nxt, frozen, wsd


# ------------------------------------------------------------------ nested state helpers
def _leaves(st):
    """Nested ((shift, wkv), cstate) -> list of leaves (None kept)."""
    (a, b), c = st
    return [a, b, c]


def _nest(lv):
    return ((lv[0], lv[1]), lv[2])


# ------------------------------------------------------------------ warm-up
class Warm:
    """Every real review before the split, one row at a time (teacher forced, the training convention): states,
    builder, explicit features, the last commit curve and the FSRS-7 state of every card (part 1's warm-up, plus the
    outputs the policies need)."""

    def __init__(self, model, rev, user):
        self.model, self.rev, self.user = model, rev, user
        self.builder = S.Builder(day0=int(rev["day_offset"][0]), seed=[41, user])
        self.explicit = S.Explicit()
        self.states = {s: {} for s in model.streams}
        self.last_out, self.last_day = {}, {}
        self.fs = {}
        self.i = 0
        self.ids_of = {}
        for j in range(len(rev["card_id"])):
            c = int(rev["card_id"][j])
            self.ids_of.setdefault(c, (c, int(rev["note_id"][j]), int(rev["deck_id"][j]), int(rev["preset_id"][j])))
        self.seconds = 0.0

    def advance(self, end, fsrs):
        rev, m = self.rev, self.model
        t = time.perf_counter()
        for i in range(self.i, end):
            c = int(rev["card_id"][i])
            ids = self.ids_of[c]
            g, u = int(rev["rating"][i]), int(rev["duration"][i])
            es = int(rev["elapsed_seconds"][i])
            _, f = self.builder.row(*ids, int(rev["day_offset"][i]), int(rev["elapsed_days"][i]), es, g, u)
            keys = dict(zip(m.streams, (ids[0], ids[2], ids[1], ids[3], 0)))
            out, nxt, _ = m.step(f, {s: self.states[s].get(keys[s]) for s in m.streams})
            for s in m.streams:
                self.states[s][keys[s]] = nxt[s]
            self.explicit.push(c, g, u)
            self.last_out[c] = (out["w"], out["s"], out["d"])
            self.last_day[c] = int(rev["day_offset"][i])
            self.fs[c] = fsrs.step(np.array(max(0.0, es / DAY)), np.array(float(g)),
                                   self.fs.get(c, np.zeros(3)))
        self.seconds += time.perf_counter() - t
        self.i = end


def habit_rates(rev, end, recent_days=60, min_reviews=2000):
    """Context for counterfactual query rows (candidates, the horizon): the user's recent reviews and new cards per
    calendar day, and half the mean reviews / new cards per active day (a mid-session position), from the same recent
    window as sm41_sim.user_habits."""
    day = rev["day_offset"]
    lo = int(np.searchsorted(day, day[end - 1] - recent_days + 1, side="left"))
    lo = max(0, min(lo, end - min_reviews))
    span = max(1, int(day[end - 1] - day[lo] + 1))
    act = max(1, len(np.unique(day[lo:end])))
    n = end - lo
    nn = int((rev["elapsed_days"][lo:end] == -1).sum())
    return {"rev_per_day": n / span, "new_per_day": nn / span, "rev_today": n / act / 2, "new_today": nn / act / 2}


# ------------------------------------------------------------------ sessions
class PolicySession:
    """A policy lane's day: due cards (overdue first, ties in a seeded order), then the day's real new cards in real
    order; same-day steps by the user's own rules; the clock runs by answer times (stage 1's session loop)."""

    def __init__(self, key, due, new_by_day, rules):
        self.key, self.due, self.new_by_day, self.rules = key, due, new_by_day, rules
        self.main, self.learn, self.now, self.count = [], [], 0.0, {}

    def start_day(self, d, start):
        todays = [c for c, v in self.due.items() if v <= d]
        if todays:
            arr = np.asarray(todays, np.int64)
            tie = uniforms(self.key, 7, d, arr)
            order = np.lexsort((tie, np.asarray([self.due[c] for c in todays])))
            todays = arr[order].tolist()
        main = todays + self.new_by_day.get(d, [])
        main.reverse()
        # the clock never runs backwards: a session that ran past the next day's start time delays that start
        self.main, self.learn, self.now, self.count = main, [], max(start, self.now), {}

    def next(self):
        while True:
            if self.learn and self.learn[0][0] <= self.now:
                return heapq.heappop(self.learn)[2]
            if self.main:
                return self.main.pop()
            if self.learn:
                self.now = self.learn[0][0]
                continue
            return None

    def after(self, c, kind, rating, dur_ms, u_follow, u_gap, u_tie):
        """Advances the clock; True if the card leaves the day (the policy sets its next day)."""
        self.now += dur_ms / 1000.0
        p, gaps = self.rules[(kind, rating)]
        self.count[c] = self.count.get(c, 0) + 1
        if self.count[c] < S.MAX_SAME_DAY and u_follow < p:
            gap = float(gaps[min(int(u_gap * len(gaps)), len(gaps) - 1)])
            heapq.heappush(self.learn, (self.now + gap, u_tie, c))
            self.due.pop(c, None)
            return False
        return True


def session_start(key, D, d, sigma):
    u1, u2 = uniforms(key, 8, d, 0), uniforms(key, 8, d, 1)
    z = math.sqrt(-2.0 * math.log(max(float(u1), 1e-300))) * math.cos(2 * math.pi * float(u2))
    return (D + d) * DAY + float(np.clip(12 * 3600 + sigma * z, 3600, 20 * 3600))


# ------------------------------------------------------------------ the engine
class Engine:
    """One user, one split, B lanes (lane 0 must be 'tf' when any lane is re-anchored)."""

    def __init__(self, model, warm, D, N, lane_names, seeds_n=3, base_seed=0, trace=False, fsrs=None,
                 fsrs_source=""):
        install_fresh_hooks(model)
        self.model, self.warm, self.D, self.N = model, warm, D, N
        torch = model.torch
        rev = warm.rev
        self.rev = rev
        day = rev["day_offset"]
        self.end = int(np.searchsorted(day, D, side="left"))
        self.stop = int(np.searchsorted(day, D + N, side="left"))
        assert warm.i == self.end
        self.cfg = [lane_cfg(n) for n in lane_names]
        B = self.B = len(self.cfg)
        if any(c["anchor"] for c in self.cfg):
            assert self.cfg[0]["base"] == "tf", "re-anchored lanes need the tf lane first"
        # replay draws: one row per seed of part 1's stream (3 in part 1; more seeds add rows, the first 3 unchanged only
        # when seeds_n stays 3)
        self.seeds_n = max([seeds_n] + [c["seed"] + 1 for c in self.cfg if c["kind"] == "replay"])
        self.trace_on = trace
        self.rng_rep = np.random.default_rng([41, warm.user, D, N, base_seed])
        self.habits = S.user_habits(rev, self.end)
        self.rates = habit_rates(rev, self.end)
        self.fsrs, self.fsrs_source = fsrs, fsrs_source
        user = warm.user
        self.keys = [int(uniforms(np.uint64(user), D, N, c["seed"], 4141) * 2 ** 52) for c in self.cfg]
        # the card set: every card of the real window; entities of every stream
        win = range(self.end, self.stop)
        cards = []
        seen = set()
        for i in win:
            c = int(rev["card_id"][i])
            if c not in seen:
                seen.add(c); cards.append(c)
        self.cards = cards
        self.cidx = {c: k for k, c in enumerate(cards)}
        C = self.C = len(cards)
        streams = model.streams
        ids = np.array([warm.ids_of[c] for c in cards], np.int64)       # card, note, deck, preset
        col = {"card_id": 0, "note_id": 1, "deck_id": 2, "preset_id": 3}
        self.ent = {}       # stream -> entity index per card idx
        self.ekeys = {}
        for s in streams:
            if s == "user_id":
                self.ent[s] = np.zeros(C, np.int64); self.ekeys[s] = [0]
                continue
            keys = ids[:, col[s]]
            uniq = list(dict.fromkeys(keys.tolist()))
            pos = {k: j for j, k in enumerate(uniq)}
            self.ent[s] = np.array([pos[k] for k in keys.tolist()], np.int64)
            self.ekeys[s] = uniq
        # state arenas from the warm-up
        tmpl = {}
        for s in streams:
            ex = next(iter(warm.states[s].values()))
            tmpl[s] = {layer: _leaves(v) for layer, v in ex.items()}
        self.arena, self.fresh = {}, {}
        for s in streams:
            E = len(self.ekeys[s])
            self.arena[s] = {layer: [None if t is None else torch.zeros((E, B) + tuple(t.shape[1:]))
                                     for t in lv] for layer, lv in tmpl[s].items()}
            fr = np.ones((E, B), bool)
            for j, k in enumerate(self.ekeys[s]):
                st = warm.states[s].get(k)
                if st is None:
                    continue
                fr[j] = False
                for layer, v in st.items():
                    for a, t in zip(self.arena[s][layer], _leaves(v)):
                        if a is not None:
                            a[j] = t[0]
            self.fresh[s] = fr
        # id codes in the real window order (part 1's codes), the day baselines
        b = copy.deepcopy(warm.builder)
        for i in win:
            c = int(rev["card_id"][i])
            for (stream, dim), key in zip(S.ID_DIMS, warm.ids_of[c]):
                b.code(stream, key, dim)
        self.codes = np.stack([np.concatenate([b.codes[stream][key] for (stream, _), key in zip(S.ID_DIMS, warm.ids_of[c])])
                               for c in cards]).astype(np.float32)
        self.base = list(b.base)
        self.day0 = b.day0
        # per lane counters (sm41_sim.Builder), per lane x card history
        wb = warm.builder
        self.th = np.full(B, wb.th, np.int64); self.cum_new = np.full(B, wb.cum_new, np.int64)
        self.new_today = np.zeros(B, np.int64); self.rev_today = np.zeros(B, np.int64)
        self.prev_day = np.full(B, wb.prev_day if wb.prev_day is not None else 0, np.int64)
        self.has_prev = np.full(B, wb.prev_day is not None)
        z = lambda dtype=np.float64: np.zeros((B, C), dtype)   # noqa: E731
        self.seen, self.lth, self.cnl = z(bool), z(np.int64), z(np.int64)
        self.ed_cum, self.cum, self.first_day = z(), z(), z(np.int64)
        self.last_day = np.full((B, C), -10 ** 9, np.int64)
        self.last_end = np.full((B, C), np.nan)
        self.dprev = z()
        self.k_rev = z(np.int64)
        self.seen_win = z(bool)
        nc = 3
        self.cw, self.cs, self.cd = np.zeros((B, C, nc)), np.zeros((B, C, nc)), np.zeros((B, C, nc))
        self.fs = np.zeros((B, C, 3))
        self.aim = np.full((B, C), np.nan)
        self.old = np.zeros(C, bool)
        for k, c in enumerate(cards):
            st = wb.cards.get(c)
            if st is not None:
                self.seen[:, k] = True
                self.lth[:, k], self.cnl[:, k], self.ed_cum[:, k], self.cum[:, k], self.first_day[:, k] = st
            if c in warm.last_out:
                self.old[k] = True
                w_, s_, d_ = warm.last_out[c]
                self.cw[:, k], self.cs[:, k], self.cd[:, k] = w_, s_, d_
                self.last_day[:, k] = warm.last_day[c]
                self.last_end[:, k] = warm.last_day[c] * DAY + MU
            if c in warm.fs:
                self.fs[:, k] = warm.fs[c]
        self.expl = [copy.deepcopy(warm.explicit) for _ in range(B)]
        # the real window by day
        self.rev_card = np.array([self.cidx[int(c)] for c in rev["card_id"][self.end:self.stop]], np.int64)
        self.real_by_day = {}
        self.new_by_day = {}
        for j, i in enumerate(win):
            d = int(day[i]) - D
            self.real_by_day.setdefault(d, []).append(i)
            if rev["elapsed_days"][i] == -1:
                self.new_by_day.setdefault(d, []).append(self.cidx[int(rev["card_id"][i])])
        self.ticks, self.lane_reviews, self.cand_rows, self.wall = 0, 0, 0, 0.0
        self.pending = None
        # policy lanes: the first due day of every old card
        self.sess = [None] * B
        self.backlog = np.zeros(B, np.int64)
        self.init_p = None
        old_idx = np.nonzero(self.old)[0]
        need_ihr = any(c["kind"] == "policy" and c["policy"] == "ihr" for c in self.cfg)
        if need_ihr and len(old_idx):
            self.init_p = self.candidates_p(np.zeros(len(old_idx), np.int64), old_idx, None,
                                            self.last_day[0, old_idx], T_GRID)
        for bl, c in enumerate(self.cfg):
            if c["kind"] != "policy":
                continue
            r = np.full(len(old_idx), c["r"])
            if c["policy"] == "ihr":
                _, ti, aim = choose_interval(self.init_p, r)
            elif c["policy"] == "fsrs7":
                t = self.fsrs.interval(self.fs[bl, old_idx], r)
                ti = np.maximum(1, np.round(t)).astype(np.int64)
                aim = self.fsrs.recall(ti.astype(np.float64), self.fs[bl, old_idx, 0], self.fs[bl, old_idx, 1],
                                       self.fs[bl, old_idx, 2])
            else:
                t = time_at_vec(self.cw[bl, old_idx], self.cs[bl, old_idx], self.cd[bl, old_idx], r)
                ti = np.maximum(1, np.round(t / DAY)).astype(np.int64)
                aim = curve_np(self.cw[bl, old_idx], self.cs[bl, old_idx], self.cd[bl, old_idx], ti * float(DAY))
            dd = np.maximum(0, self.last_day[bl, old_idx] + ti - D)
            self.aim[bl, old_idx] = aim
            self.backlog[bl] = int((self.last_day[bl, old_idx] + ti - D < 0).sum())
            self.sess[bl] = PolicySession(self.keys[bl], dict(zip(old_idx.tolist(), dd.tolist())), self.new_by_day,
                                          self.habits["rules"])
        nb = (N + BUCKET - 1) // BUCKET
        zz = lambda *sh: np.zeros((B, nb) + sh)   # noqa: E731
        self.acc = {"later_old": zz(3), "later_new": zz(3), "first_touch": zz(3), "grades_same": zz(4),
                    "grades_later": zz(4), "time_later": zz(4), "implied_follow": zz(2), "grade_ll": zz(4),
                    "load": zz(6),      # n first / same-day / later, seconds first / same-day / later
                    "aim": zz(2)}       # later-day reviews with a policy aim: n, sum of the aimed recall
        self.traces = [[] for _ in range(B)]
        self.mid = (model.lo + model.hi) / 2

    # -------------------------------------------------------------- state gather / scatter
    def gather(self, A, e):
        torch = self.model.torch
        At = torch.from_numpy(A)
        st, fr = {}, {}
        for s in self.model.streams:
            ent = self.ent[s][e]
            Et = torch.from_numpy(ent)
            st[s] = {layer: _nest([None if a is None else a[Et, At] for a in lv]) for layer, lv in self.arena[s].items()}
            fr[s] = self.fresh[s][ent, A]
        return st, fr

    def scatter(self, A, e, nxt):
        torch = self.model.torch
        At = torch.from_numpy(A)
        for s in self.model.streams:
            ent = self.ent[s][e]
            Et = torch.from_numpy(ent)
            for layer, lv in self.arena[s].items():
                for a, t in zip(lv, _leaves(nxt[s][layer])):
                    if a is not None:
                        a[Et, At] = t
            self.fresh[s][ent, A] = False

    def anchor(self, d):
        lanes = [b for b, c in enumerate(self.cfg) if c["anchor"] and d % c["anchor"] == 0]
        if not lanes:
            return
        L = self.model.torch.tensor(lanes)
        for s in SHARED:
            for lv in self.arena[s].values():
                for a in lv:
                    if a is not None:
                        a[:, L] = a[:, 0:1].clone()
            self.fresh[s][:, lanes] = self.fresh[s][:, 0:1]

    # -------------------------------------------------------------- features
    def day_enc(self, rel, rel_first):
        out = np.empty((len(rel), 28), np.float32)
        for j, (p, b) in enumerate(zip(S.PERIODS, self.base)):
            fr = 2 * np.pi / p
            out[:, 4 * j] = np.sin(fr * ((b + rel) % p)); out[:, 4 * j + 1] = np.cos(fr * ((b + rel) % p))
            out[:, 4 * j + 2] = np.sin(fr * ((b + rel_first) % p)); out[:, 4 * j + 3] = np.cos(fr * ((b + rel_first) % p))
        return out

    def rows(self, A, e, d_abs, ed):
        """sm41_sim.Builder.row for rows (lane A, card e) on absolute day d_abs with elapsed days ed: the review row
        without columns 2-7 (intervals) and 8-12 (outcome); advances the lanes' histories."""
        n = len(A)
        first = ed == -1
        self.th[A] += 1
        self.cum_new[A] += first
        seen = self.seen[A, e]
        diff_new = np.where(seen, self.cum_new[A] - self.cnl[A, e], 0).astype(np.float64)
        diff_rev = np.where(seen, np.maximum(0, self.th[A] - self.lth[A, e] - 1), 0).astype(np.float64)
        self.first_day[A, e] = np.where(seen, self.first_day[A, e], d_abs)
        self.seen[A, e] = True
        cum_rev_today = self.rev_today[A].copy()
        self.rev_today[A] += 1
        self.new_today[A] += first
        day_diff = np.where(self.has_prev[A], d_abs - self.prev_day[A], 0).astype(np.float64)
        self.prev_day[A] = d_abs; self.has_prev[A] = True
        self.ed_cum[A, e] += ed
        self.lth[A, e] = self.th[A]; self.cnl[A, e] = self.cum_new[A]
        f = np.zeros((n, 92), np.float32)
        f[:, 0] = DP.scale_elapsed_days(ed.astype(np.float64))
        f[:, 1] = DP.scale_elapsed_days_cumulative(self.ed_cum[A, e])
        f[:, 16] = DP.scale_day_offset_diff(day_diff)
        f[:, 17] = (((d_abs - self.day0) % 7) - 3) / 3
        f[:, 18] = DP.scale_diff_new_cards(diff_new)
        f[:, 19] = DP.scale_diff_reviews(diff_rev)
        f[:, 20] = DP.scale_cum_new_cards_today(self.new_today[A].astype(np.float64))
        f[:, 21] = DP.scale_cum_reviews_today(cum_rev_today.astype(np.float64))
        f[:, 24:64] = self.codes[e]
        f[:, 64:92] = self.day_enc(np.full(n, d_abs - self.day0, np.int64), self.first_day[A, e] - self.day0)
        return f

    def _cand(self, A, e, base_day, T):
        """Counterfactual query rows: the card's next review T [n, K] days after base_day (its last review day), from
        the lane's post-commit history -> rows [n * K, 92]. Elapsed seconds = T whole days; the day gap since the lane's
        previous review 0 (mid-session); new cards / reviews since the card's last review = the user's recent rates x T;
        today's counters at a mid-session position (habit_rates)."""
        n, K = T.shape
        rt = self.rates
        Tf = T.astype(np.float64)
        q = np.zeros((n, K, 92), np.float32)
        es = Tf * DAY
        q[:, :, 0] = DP.scale_elapsed_days(Tf)
        q[:, :, 1] = DP.scale_elapsed_days_cumulative(self.ed_cum[A, e][:, None] + Tf)
        cols = DR.es_cols(es.ravel(), (self.cum[A, e][:, None] + es).ravel()).reshape(n, K, 6)
        q[:, :, 2:8] = cols
        q[:, :, 16] = DP.scale_day_offset_diff(np.zeros((n, K)))
        dd = base_day[:, None] + T
        q[:, :, 17] = (((dd - self.day0) % 7) - 3) / 3
        q[:, :, 18] = DP.scale_diff_new_cards(rt["new_per_day"] * Tf)
        q[:, :, 19] = DP.scale_diff_reviews(rt["rev_per_day"] * Tf)
        q[:, :, 20] = DP.scale_cum_new_cards_today(np.float64(rt["new_today"]))
        q[:, :, 21] = DP.scale_cum_reviews_today(np.float64(rt["rev_today"]))
        q[:, :, 23] = 1.0
        q[:, :, 24:64] = self.codes[e][:, None]
        rel = (dd - self.day0).ravel()
        relf = np.repeat(self.first_day[A, e] - self.day0, K)
        q[:, :, 64:92] = self.day_enc(rel, relf).reshape(n, K, 28)
        return q.reshape(n * K, 92)

    def candidates_p(self, A, e, nxt, base_day, grid, chunk=2048):
        """P(pass) [n, K] at the grid intervals for rows (lane A, card e): states = nxt (post-commit, [n] leaves) or,
        when None, the arena's (gathered here, in chunks)."""
        torch = self.model.torch
        K = len(grid)
        T = np.broadcast_to(np.asarray(grid, np.int64), (len(A), K))
        out = np.empty((len(A), K))
        step = max(1, chunk // K)
        for lo in range(0, len(A), step):
            sl = slice(lo, lo + step)
            X = self._cand(A[sl], e[sl], base_day[sl], T[sl])
            if nxt is None:
                st, fr = self.gather(A[sl], e[sl])
            else:
                st = {s: {layer: _nest([None if t is None else t[sl] for t in _leaves(v)]) for layer, v in nxt[s].items()}
                      for s in nxt}
                fr = {s: None for s in nxt}
            rep = {s: {layer: _nest([None if t is None else torch.repeat_interleave(t, K, 0) for t in _leaves(v)])
                       for layer, v in st[s].items()} for s in st}
            frr = {s: None if fr[s] is None else np.repeat(fr[s], K) for s in fr}
            p, _, _, _ = vstep(self.model, X, rep, frr)
            p = p / p.sum(-1, keepdims=True)
            out[sl] = (1.0 - p[:, 0]).reshape(-1, K)
            self.cand_rows += X.shape[0]
        return out

    # -------------------------------------------------------------- run
    def run(self):
        t0 = time.perf_counter()
        for d in range(self.N):
            self.day(d)
        self.wall = time.perf_counter() - t0
        return self.result()

    def day(self, d):
        D = self.D
        self.anchor(d)
        self.new_today[:] = 0; self.rev_today[:] = 0
        real = self.real_by_day.get(d, [])
        rep_ptr = {}
        active = []
        for b, c in enumerate(self.cfg):
            if c["kind"] == "replay":
                if real:
                    rep_ptr[b] = 0; active.append(b)
            else:
                self.sess[b].start_day(d, session_start(self.keys[b], D, d, self.habits["sigma"]))
                active.append(b)
        while active:
            lanes, cards, kinds, eds, ridx = [], [], [], [], []
            still = []
            for b in active:
                if self.cfg[b]["kind"] == "replay":
                    j = rep_ptr[b]
                    if j >= len(real):
                        continue
                    i = real[j]; rep_ptr[b] = j + 1
                    k = self.rev_card[i - self.end]
                    ed = int(self.rev["elapsed_days"][i])
                    ridx.append(i)
                else:
                    k = self.sess[b].next()
                    if k is None:
                        continue
                    ld = self.last_day[b, k]
                    ed = -1 if ld < -10 ** 8 else d + D - int(ld)
                    ridx.append(-1)
                still.append(b); lanes.append(b); cards.append(k); eds.append(ed)
                kinds.append(0 if ed == -1 else (1 if ed == 0 else 2))
            active = still
            if not lanes:
                break
            self.tick(d, np.asarray(lanes, np.int64), np.asarray(cards, np.int64), np.asarray(kinds),
                      np.asarray(eds, np.int64), np.asarray(ridx, np.int64))
        self.flush()

    def resolve(self, pend, pg):
        """Instant-head intervals of the cards that left the day at the previous tick (their candidate rows ran with
        this tick's query rows, in the same batched step)."""
        _, ti, aim = choose_interval(pg, pend["r"])
        for j, (b, k, t_, a_) in enumerate(zip(pend["A"], pend["e"], ti, aim)):
            self.sess[b].due[int(k)] = pend["d"] + int(t_)
            self.aim[b, k] = a_
            if self.trace_on:
                pos = pend["tpos"][j]
                tr = self.traces[b][pos]
                self.traces[b][pos] = tr[:5] + (int(t_),) + tr[6:]

    def flush(self):
        pend, self.pending = self.pending, None
        if pend is not None:
            mrows = len(pend["A"])
            pg = self.candidates_p(pend["A"], pend["e"], pend["st"], np.full(mrows, self.D + pend["d"], np.int64), T_GRID)
            self.resolve(pend, pg)

    def tick(self, d, A, e, kind, ed, ridx):
        torch = self.model.torch
        m = self.model
        D = self.D
        n = len(A)
        self.ticks += 1
        self.lane_reviews += n
        bkt = d // BUCKET
        d_abs = D + d
        rep = ridx >= 0
        pol = ~rep
        rev = self.rev
        # real data of replay rows
        es_r = np.where(rep, rev["elapsed_seconds"][np.maximum(ridx, 0)], 0).astype(np.float64)
        g_r = np.where(rep, rev["rating"][np.maximum(ridx, 0)], 0).astype(np.int64)
        u_r = np.where(rep, rev["duration"][np.maximum(ridx, 0)], 0).astype(np.int64)
        # draws: replay rows from part 1's stream (one draw per real review, shared by the replay lanes of a seed),
        # policy rows counter-based
        U = np.zeros((n, 5))
        if rep.any():
            assert len(set(ridx[rep].tolist())) == 1, "replay lanes out of step"
            Ur = self.rng_rep.random((self.seeds_n, 3))
            seeds = np.array([self.cfg[b]["seed"] for b in A[rep]])
            U[rep, :3] = Ur[seeds]
        if pol.any():
            key = np.array([self.keys[b] for b in A[pol]], np.uint64)
            kk = self.k_rev[A[pol], e[pol]]
            for s in range(5):
                U[pol, s] = uniforms(key, 1, e[pol], kk, s)
        teacher = np.array([self.cfg[b]["teacher"] for b in A])
        f = self.rows(A, e, d_abs, ed)
        lane_cum = self.cum[A, e].copy()
        dp = self.dprev[A, e].copy()
        now = np.array([self.sess[b].now if self.cfg[b]["kind"] == "policy" else 0.0 for b in A])
        last_end = self.last_end[A, e]
        x = np.where(rep, es_r - u_r / 1000.0 + dp, now - np.nan_to_num(last_end))
        q_es = np.where(kind == 0, 0.0, np.floor(np.maximum(x, 0.0)))
        Q = f.copy()
        Q[:, 2:8] = DR.es_cols(q_es, lane_cum + q_es)
        Q[:, 23] = 1.0
        st, fr = self.gather(A, e)
        nf = kind != 0
        p_all = np.full((n, 4), 0.25)
        frozen = None
        pend, self.pending = self.pending, None
        if nf.any() or pend is not None:
            if not np.isfinite(Q[nf]).all():
                raise FloatingPointError(f"non-finite query row: user {self.warm.user} split {D} day {d}")
            X, stq, frq = Q, st, fr
            K = len(T_GRID)
            if pend is not None:
                # the previous tick's instant-head candidates ride along in this query step (rows are independent)
                mrows = len(pend["A"])
                Cx = self._cand(pend["A"], pend["e"], np.full(mrows, D + pend["d"], np.int64),
                                np.broadcast_to(T_GRID.astype(np.int64), (mrows, K)))
                X = np.concatenate([Q, Cx])
                stq = {s: {layer: _nest([None if a is None else torch.cat([a, torch.repeat_interleave(c_, K, 0)])
                                         for a, c_ in zip(_leaves(st[s][layer]), _leaves(pend["st"][s][layer]))])
                           for layer in st[s]} for s in st}
                frq = {s: np.concatenate([fr[s], np.zeros(mrows * K, bool)]) for s in fr}
                self.cand_rows += Cx.shape[0]
            p_full, _, fz, _ = vstep(m, X, stq, frq, snaps=True)
            p_full = p_full / p_full.sum(-1, keepdims=True)
            p_all, frozen = p_full[:n], fz[:n]
            if pend is not None:
                self.resolve(pend, (1.0 - p_full[n:, 0]).reshape(-1, K))
        # grades
        rating = DR.sample_cat(p_all, U[:, 0]) + 1
        fm = self.habits["first_mix"]
        first_g = DR.sample_cat(np.repeat(fm[None], n, 0), U[:, 0]) + 1
        rating = np.where(kind == 0, np.where(rep, g_r, first_g), np.where(teacher, g_r, rating))
        # times
        dur = np.where(rep, u_r, 0).astype(np.int64)
        k2 = np.nonzero(kind == 2)[0]
        ppass = 1.0 - p_all[:, 0]
        if len(k2):
            ex = np.stack([self.expl[A[j]].features(self.cards[e[j]], int(ed[j])) for j in k2])
            tp = DR.time_probs_batch(m, frozen[torch.from_numpy(k2)], ex)
            pb = tp[np.arange(len(k2)), rating[k2] - 1]
            bins = DR.sample_cat(pb / pb.sum(-1, keepdims=True), U[k2, 1])
            lo, hi = m.lo[bins], m.hi[bins]
            ms_s = np.where(bins == 15, 60000, np.round(1000 * np.exp(lo + U[k2, 2] * (hi - lo)))).astype(np.int64)
            dur[k2] = np.where(teacher[k2], u_r[k2], ms_s)
            A2 = A[k2]
            self.acc["time_later"][A2, bkt, 3] += (pb * np.exp(self.mid)).sum(-1)
            old2 = self.old[e[k2]]
            for nm, sel in (("later_old", old2), ("later_new", ~old2)):
                rr = k2[sel]
                self.acc[nm][A[rr], bkt, 0] += 1; self.acc[nm][A[rr], bkt, 1] += rating[rr] > 1
                self.acc[nm][A[rr], bkt, 2] += ppass[rr]
            ft = k2[old2 & ~self.seen_win[A2, e[k2]]]
            self.acc["first_touch"][A[ft], bkt, 0] += 1; self.acc["first_touch"][A[ft], bkt, 1] += rating[ft] > 1
            self.acc["first_touch"][A[ft], bkt, 2] += ppass[ft]
            self.acc["grades_later"][A2, bkt, rating[k2] - 1] += 1
            self.acc["time_later"][A2, bkt, 0] += 1; self.acc["time_later"][A2, bkt, 1] += dur[k2] / 1000.0
            self.acc["time_later"][A2, bkt, 2] += np.log(np.maximum(dur[k2], 1) / 1000.0)
            r2 = k2[rep[k2]]
            self.acc["grade_ll"][A[r2], bkt, 0] += -np.log(np.maximum(p_all[r2, g_r[r2] - 1], 1e-7))
            self.acc["grade_ll"][A[r2], bkt, 1] += 1
            am = k2[np.isfinite(self.aim[A2, e[k2]])]
            self.acc["aim"][A[am], bkt, 0] += 1; self.acc["aim"][A[am], bkt, 1] += self.aim[A[am], e[am]]
        k1 = np.nonzero(kind == 1)[0]
        if len(k1):
            self.acc["grades_same"][A[k1], bkt, rating[k1] - 1] += 1
            r1 = k1[rep[k1]]
            self.acc["grade_ll"][A[r1], bkt, 2] += -np.log(np.maximum(p_all[r1, g_r[r1] - 1], 1e-7))
            self.acc["grade_ll"][A[r1], bkt, 3] += 1
        pools = self.habits["pools"]
        for j in np.nonzero(pol & (kind < 2))[0]:
            pool = pools[(int(kind[j]), int(rating[j]))]
            dur[j] = int(pool[min(int(U[j, 1] * len(pool)), len(pool) - 1)])
        # commit row intervals
        c_es = np.round(es_r + dp + (dur - u_r) / 1000.0)
        differs = (dp != 0) | (dur != u_r)
        c_es = np.where(differs, np.maximum(c_es, np.round(np.maximum(x, 0.0) + dur / 1000.0)), c_es)
        c_es = np.where(pol, np.round(now + dur / 1000.0 - np.nan_to_num(last_end)), c_es)
        c_es = np.where(kind == 0, np.where(rep, es_r, -1.0), c_es)
        F = f
        F[:, 2:8] = DR.es_cols(c_es, lane_cum + c_es)
        F[:, 8] = DP.scale_duration(dur.astype(np.float64)).astype(np.float32)
        F[np.arange(n), 8 + rating] = 1.0
        if not np.isfinite(F).all():
            raise FloatingPointError(f"non-finite review row: user {self.warm.user} split {D} day {d}")
        _, nxt, _, (cw, cs, cd) = vstep(m, F, st, fr, curve=True)
        self.scatter(A, e, nxt)
        for j in range(n):
            self.expl[A[j]].push(self.cards[e[j]], int(rating[j]), int(dur[j]))
        self.acc["implied_follow"][A, bkt, 0] += 1
        rules = self.habits["rules"]
        self.acc["implied_follow"][A, bkt, 1] += np.array([rules[(int(k), int(g))][0] for k, g in zip(kind, rating)])
        for kk_, (cn, cs_) in enumerate(((0, 3), (1, 4), (2, 5))):
            sel = kind == kk_
            self.acc["load"][A[sel], bkt, cn] += 1
            self.acc["load"][A[sel], bkt, cs_] += dur[sel] / 1000.0
        self.cum[A, e] = lane_cum + c_es
        self.dprev[A, e] = np.where(rep, (u_r - dur) / 1000.0, 0.0)
        self.k_rev[A, e] += 1
        self.seen_win[A, e] = True
        self.cw[A, e], self.cs[A, e], self.cd[A, e] = cw, cs, cd
        if self.fsrs is not None:
            self.fs[A, e] = self.fsrs.step(np.maximum(0.0, c_es / DAY), rating.astype(np.float64), self.fs[A, e])
        self.last_day[A, e] = d_abs
        self.last_end[A, e] = np.where(rep, d_abs * DAY + MU, now + dur / 1000.0)
        self.aim[A, e] = np.nan
        # policy lanes: the session; cards leaving the day get their next day from the policy
        need = []
        for j in np.nonzero(pol)[0]:
            b = A[j]
            if self.sess[b].after(int(e[j]), int(kind[j]), int(rating[j]), int(dur[j]), U[j, 3], U[j, 4],
                                  float(uniforms(self.keys[b], 9, e[j], self.k_rev[b, e[j]]))):
                need.append(j)
        ivl = np.full(n, -1, np.int64)
        pend_new = None
        if need:
            need = np.asarray(need)
            pols = np.array([self.cfg[A[j]]["policy"] for j in need])
            rr = np.array([self.cfg[A[j]]["r"] for j in need])
            ti = np.zeros(len(need), np.int64); aim = np.zeros(len(need))
            sel = np.nonzero(pols == "ihr")[0]
            if len(sel):
                # deferred to the next tick's query step (or the end of the day): the card does not come back today
                jj = need[sel]
                sub = {s: {layer: _nest([None if t is None else t[torch.from_numpy(jj)] for t in _leaves(v)])
                           for layer, v in nxt[s].items()} for s in nxt}
                pend_new = {"A": A[jj], "e": e[jj], "d": d, "r": rr[sel], "st": sub, "jj": jj}
            sel = np.nonzero(pols == "fsrs7")[0]
            if len(sel):
                jj = need[sel]
                st7 = self.fs[A[jj], e[jj]]
                t = self.fsrs.interval(st7, rr[sel])
                ti[sel] = np.maximum(1, np.round(t)).astype(np.int64)
                aim[sel] = self.fsrs.recall(ti[sel].astype(np.float64), st7[:, 0], st7[:, 1], st7[:, 2])
            sel = np.nonzero(pols == "curve")[0]
            if len(sel):
                jj = need[sel]
                t = time_at_vec(cw[jj], cs[jj], cd[jj], rr[sel])
                ti[sel] = np.maximum(1, np.round(t / DAY)).astype(np.int64)
                aim[sel] = curve_np(cw[jj], cs[jj], cd[jj], ti[sel] * float(DAY))
            for j, t_, a_, p_ in zip(need, ti, aim, pols):
                if p_ == "ihr":
                    continue
                b = A[j]
                self.sess[b].due[int(e[j])] = d + int(t_)
                self.aim[b, e[j]] = a_
                ivl[j] = t_
        if self.trace_on:
            for j in range(n):
                self.traces[A[j]].append((d, int(e[j]), int(kind[j]), int(rating[j]), int(dur[j]), int(ivl[j]),
                                          float(ppass[j]) if kind[j] else -1.0))
        if pend_new is not None:
            pend_new["tpos"] = [len(self.traces[A[j]]) - 1 for j in pend_new["jj"]] if self.trace_on else None
            self.pending = pend_new

    # -------------------------------------------------------------- the horizon and the result
    def memorised(self):
        """Sum over the card set of each card's recall at the horizon (12:00 of day N): the instant head (a query row
        at the horizon, the lane's final states) and the card's last commit curve."""
        B, C = self.B, self.C
        A = np.repeat(np.arange(B), C); e = np.tile(np.arange(C), B)
        t_end = (self.D + self.N) * DAY + MU
        tdays = np.maximum(1, self.D + self.N - self.last_day[A, e])
        out = np.empty(len(A))
        chunk = 2048
        for lo in range(0, len(A), chunk):
            sl = slice(lo, lo + chunk)
            X = self._cand(A[sl], e[sl], self.last_day[A[sl], e[sl]], tdays[sl][:, None])
            st, fr = self.gather(A[sl], e[sl])
            p, _, _, _ = vstep(self.model, X, st, fr)
            p = p / p.sum(-1, keepdims=True)
            out[sl] = 1 - p[:, 0]
        inst = out.reshape(B, C).sum(1)
        tt = np.maximum(1.0, t_end - self.last_end)
        crv = np.array([curve_np(self.cw[b], self.cs[b], self.cd[b], tt[b]).sum() for b in range(B)])
        return inst, crv

    def result(self):
        inst, crv = self.memorised()
        lanes = [c["name"] for c in self.cfg]
        return {"user": self.warm.user, "split_day": self.D, "days": self.N, "warmup_reviews": self.end,
                "window_reviews": self.stop - self.end, "cards": self.C, "old_cards": int(self.old.sum()),
                "lanes": lanes, "bucket_days": BUCKET, "acc": {k: v.round(6).tolist() for k, v in self.acc.items()},
                "memorised_instant": inst.round(3).tolist(), "memorised_curve": crv.round(3).tolist(),
                "backlog": self.backlog.tolist(), "fsrs7": self.fsrs_source, "grid_days": T_GRID.tolist(),
                "rates": self.rates, "wall_seconds": round(self.wall, 2), "ticks": self.ticks,
                "lane_reviews": self.lane_reviews, "candidate_rows": self.cand_rows,
                "lane_reviews_per_second": round(self.lane_reviews / max(self.wall, 1e-9), 1),
                "real": {"reviews": self.stop - self.end,
                         "minutes": round(float(self.rev["duration"][self.end:self.stop].sum()) / 60000, 2)}}


def setup(model, user, D, cache=None):
    """Warm-up for (user, D), reusing `cache` (user -> Warm) when it has not passed D yet."""
    rev, _ = S.load_user(user)
    w7, src = fsrs7_params(user)
    fsrs = FSRS7(w7)
    warm = cache.get(user) if cache is not None else None
    end = int(np.searchsorted(rev["day_offset"], D, side="left"))
    if warm is None or warm.i > end:
        warm = Warm(model, rev, user)
    warm.advance(end, fsrs)
    if cache is not None:
        cache[user] = warm
    return warm, fsrs, src


# ------------------------------------------------------------------ scalar reference (proof P2)
def es_cols_scalar(es, cum):
    def sc(v, mean, std):
        return ((0.0 if v == -1 else math.log(1 + 1e-5 + v)) - DP.STATISTICS[mean]) / DP.STATISTICS[std]
    return np.array([sc(es, "elapsed_seconds_mean", "elapsed_seconds_std"),
                     math.sin((es % DAY) * 2 * math.pi / DAY), math.cos((es % DAY) * 2 * math.pi / DAY),
                     sc(cum, "elapsed_seconds_cumulative_mean", "elapsed_seconds_cumulative_std"),
                     math.sin((cum % DAY) * 2 * math.pi / DAY), math.cos((cum % DAY) * 2 * math.pi / DAY)], np.float32)


class ScalarLane:
    """One lane, one row at a time: stage 1's Builder (codes pre-set to the engine's), dict states with None for new
    entities and the stock recurrent step (sm41_sim.Model.step), stage 1's curve inversion (sm41_sim.time_at),
    candidate rows built from the Builder's own history. Shares with the engine only the session rules, the draws,
    the FSRS-7 functions and the grid interpolation."""

    def __init__(self, eng, b):
        self.eng, self.b, self.cfg = eng, b, eng.cfg[b]
        w = eng.warm
        self.states = {s: dict(v) for s, v in w.states.items()}
        self.builder = copy.deepcopy(w.builder)
        for k, c in enumerate(eng.cards):
            off = 0
            for (stream, dim), key in zip(S.ID_DIMS, w.ids_of[c]):
                self.builder.codes[stream][key] = eng.codes[k, off:off + dim].copy()
                off += dim
        self.explicit = copy.deepcopy(w.explicit)
        self.last_day = {c: d for c, d in w.last_day.items()}
        self.last_end = {c: d * DAY + MU for c, d in w.last_day.items()}
        self.last_out = dict(w.last_out)
        self.fs = {c: v.copy() for c, v in w.fs.items()}
        self.k = {}
        self.trace = []
        if self.cfg["kind"] == "policy":
            # the engine's first due days (the split-day candidate step is checked separately in P2)
            src = eng.sess[b]
            self.sess = PolicySession(src.key, dict(eng._due0[b]), eng.new_by_day, eng.habits["rules"])

    def step(self, f):
        e = self.eng
        c, note, deck, preset = self._ids
        keys = dict(zip(e.model.streams, (c, deck, note, preset, 0)))
        return e.model.step(f, {s: self.states[s].get(keys[s]) for s in e.model.streams}, snaps=True), keys

    def cand_row(self, c, base_day, t):
        e, bld = self.eng, self.builder
        st = bld.cards[c]
        rt = e.rates
        q = np.zeros(92, np.float32)
        q[0] = DP.scale_elapsed_days(np.float64(t))
        q[1] = DP.scale_elapsed_days_cumulative(np.float64(st[2] + t))
        q[2:8] = es_cols_scalar(float(t * DAY), float(st[3] + t * DAY))
        q[16] = DP.scale_day_offset_diff(np.float64(0))
        q[17] = (((base_day + t - bld.day0) % 7) - 3) / 3
        q[18] = DP.scale_diff_new_cards(np.float64(rt["new_per_day"] * t))
        q[19] = DP.scale_diff_reviews(np.float64(rt["rev_per_day"] * t))
        q[20] = DP.scale_cum_new_cards_today(np.float64(rt["new_today"]))
        q[21] = DP.scale_cum_reviews_today(np.float64(rt["rev_today"]))
        q[23] = 1.0
        k = 24
        for (stream, dim), key in zip(S.ID_DIMS, e.warm.ids_of[c]):
            q[k:k + dim] = bld.codes[stream][key]
            k += dim
        rel, relf = base_day + t - bld.day0, st[4] - bld.day0
        for p, b in zip(S.PERIODS, bld.base):
            fr = 2 * np.pi / p
            q[k] = np.sin(fr * ((b + rel) % p)); q[k + 1] = np.cos(fr * ((b + rel) % p))
            q[k + 2] = np.sin(fr * ((b + relf) % p)); q[k + 3] = np.cos(fr * ((b + relf) % p))
            k += 4
        return q

    def review(self, d, c, ed, real=None, now=None):
        """One review; real = (es, rating, ms) for teacher forcing. Returns (rating, ms, interval or -1)."""
        e, m = self.eng, self.eng.model
        D = e.D
        d_abs = D + d
        self._ids = e.warm.ids_of[c]
        kind = 0 if ed == -1 else (1 if ed == 0 else 2)
        st = self.builder.cards.get(c)
        prev_cum = float(st[3]) if st is not None else 0.0
        q, f = self.builder.row(c, self._ids[1], self._ids[2], self._ids[3], d_abs, ed, 0, 1, 0)
        kidx = e.cidx[c]
        kk = self.k.get(c, 0)
        if real is None:
            u = [float(uniforms(e.keys[self.b], 1, kidx, kk, s)) for s in range(5)]
        if real is not None:
            es, g, ms = real
            q_es = 0.0 if kind == 0 else math.floor(max(es - ms / 1000.0, 0.0))
        else:
            q_es = 0.0 if kind == 0 else math.floor(max(now - self.last_end[c], 0.0))
        pp = -1.0
        cur_states = None
        if kind != 0:
            q[2:8] = es_cols_scalar(q_es, prev_cum + q_es)
            (out, _, frozen), keys = self.step(q)
            pg = out["p"].astype(np.float64); pg = pg / pg.sum()
            pp = 1 - pg[0]
        if real is not None:
            rating, dur = g, ms
            c_es = float(es)
        else:
            if kind == 0:
                rating = int(DR.sample_cat(e.habits["first_mix"][None], np.array([u[0]]))[0]) + 1
            else:
                rating = int(DR.sample_cat(pg[None], np.array([u[0]]))[0]) + 1
            if kind == 2:
                tp = m.time_probs(frozen, self.explicit.features(c, ed))
                pb = tp[rating - 1].astype(np.float64)
                b_ = int(DR.sample_cat((pb / pb.sum())[None], np.array([u[1]]))[0])
                dur = 60000 if b_ == 15 else int(np.round(1000 * np.exp(m.lo[b_] + u[2] * (m.hi[b_] - m.lo[b_]))))
            else:
                pool = e.habits["pools"][(kind, rating)]
                dur = int(pool[min(int(u[1] * len(pool)), len(pool) - 1)])
            c_es = -1.0 if kind == 0 else float(np.round(now + dur / 1000.0 - self.last_end[c]))
        f[2:8] = es_cols_scalar(c_es, prev_cum + c_es)
        f[8] = DP.scale_duration(np.float64(dur))
        f[9:13] = 0.0
        f[8 + rating] = 1.0
        self.builder.cards[c][3] = prev_cum + c_es
        (out, nxt, _), keys = self.step(f)
        for s in m.streams:
            self.states[s][keys[s]] = nxt[s]
        self.explicit.push(c, rating, dur)
        self.last_out[c] = (out["w"], out["s"], out["d"])
        self.fs[c] = e.fsrs.step(np.array(max(0.0, c_es / DAY)), np.array(float(rating)), self.fs.get(c, np.zeros(3)))
        self.last_day[c] = d_abs
        self.k[c] = kk + 1
        ivl = -1
        if real is None:
            self.last_end[c] = now + dur / 1000.0
            sess = self.sess
            leaves = sess.after(kidx, kind, rating, dur, u[3], u[4], float(uniforms(e.keys[self.b], 9, kidx, kk + 1)))
            if leaves:
                r = self.cfg["r"]
                pol = self.cfg["policy"]
                if pol == "ihr":
                    pgr = []
                    for t in T_GRID:
                        qq = self.cand_row(c, d_abs, int(t))
                        o, _, _ = m.step(qq, nxt)
                        pz = o["p"].astype(np.float64); pz = pz / pz.sum()
                        pgr.append(1 - pz[0])
                    _, ti, aim = choose_interval(np.asarray(pgr)[None], np.array([r]))
                    ivl = int(ti[0])
                elif pol == "fsrs7":
                    t = e.fsrs.interval(self.fs[c][None], np.array([r]))[0]
                    ivl = max(1, int(np.round(t)))
                else:
                    w_, s_, d_ = out["w"], out["s"], out["d"]
                    ivl = max(1, int(round(S.time_at(w_, s_, d_, r) / DAY)))
                sess.due[kidx] = d + ivl
        self.trace.append((d, kidx, kind, rating, dur, ivl, pp))
        return rating, dur


def scalar_run(eng, b):
    """Lane b of `eng` (built but not run) and the tf anchor lane 0, scalar, day by day: anchor at the start of every
    day (the deck / preset / user entries of lane 0's dicts), then lane 0's real reviews, then lane b's session."""
    tf = ScalarLane(eng, 0)
    ln = ScalarLane(eng, b)
    rev = eng.rev
    for d in range(eng.N):
        if ln.cfg["anchor"] and d % ln.cfg["anchor"] == 0:
            for s in SHARED:
                ln.states[s] = dict(tf.states[s])
        for i in eng.real_by_day.get(d, []):
            c = int(rev["card_id"][i])
            tf.review(d, c, int(rev["elapsed_days"][i]),
                      real=(int(rev["elapsed_seconds"][i]), int(rev["rating"][i]), int(rev["duration"][i])))
        sess = ln.sess
        sess.start_day(d, session_start(eng.keys[b], eng.D, d, eng.habits["sigma"]))
        while True:
            k = sess.next()
            if k is None:
                break
            c = eng.cards[k]
            ld = ln.last_day.get(c)
            ed = -1 if ld is None else eng.D + d - ld
            ln.review(d, c, ed, now=sess.now)
    return tf.trace, ln.trace


# ------------------------------------------------------------------ proof
def compare_traces(a, b, tol=1e-4):
    """Draws identical (everything but the predicted pass rate) and the predicted pass rate within tol."""
    if len(a) != len(b):
        return {"identical_draws": False, "len": [len(a), len(b)], "first_diff": 0, "max_p_gap": None}
    first = None
    gap = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x[:6] != y[:6] and first is None:
            first = i
        gap = max(gap, abs(x[6] - y[6]))
    return {"identical_draws": first is None, "len": len(a), "first_diff": first, "max_p_gap": float(gap),
            "ok": bool(first is None and gap <= tol)}


def proof(args):
    S.low_priority()
    model = S.Model(threads=1)
    install_fresh_hooks(model)
    res = {"days": args.days}
    cache = {}
    ok = True
    lanes = ["tf", "fr-shared1#0", "fr-shared1#1", "ihr-0.90#0", "ihr-0.80#1", "ihr-0.90-free#0", "fsrs7-0.90#0",
             "curve-0.90#0"]
    for u, D in args.users:
        DR.wait_clear(f"proof user {u}")
        warm, fsrs, src = setup(model, u, D, cache)
        t = time.perf_counter()
        full = Engine(model, warm, D, args.days, lanes, trace=True, fsrs=fsrs, fsrs_source=src)
        rf = full.run()
        tb = time.perf_counter() - t
        out = {"window_reviews": rf["window_reviews"], "lane_reviews": rf["lane_reviews"],
               "lane_reviews_per_second": rf["lane_reviews_per_second"], "lanes": {}}
        # P1: each lane alone (re-anchored lanes with the tf lane: B = 2)
        for k, name in enumerate(lanes):
            sub = [name] if (name == "tf" or not lane_cfg(name)["anchor"]) else ["tf", name]
            if lane_cfg(name)["kind"] == "replay" and not lane_cfg(name)["teacher"]:
                sub = ["tf", name]   # replay draws: the lane's seed row of the shared stream
            al = Engine(model, warm, D, args.days, sub, trace=True, fsrs=fsrs, fsrs_source=src)
            ra = al.run()
            j = len(sub) - 1
            cmp = compare_traces(full.traces[k], al.traces[j])
            A_, B_ = ({n: np.asarray(v)[k] for n, v in rf["acc"].items()}, {n: np.asarray(v)[j] for n, v in ra["acc"].items()})
            rel = max(float((np.abs(A_[n] - B_[n]) / np.maximum(1.0, np.abs(A_[n]))).max()) for n in A_)
            cmp["acc_max_rel_gap"] = rel
            cmp["memorised_gap"] = abs(rf["memorised_instant"][k] - ra["memorised_instant"][j])
            cmp["ok"] = bool(cmp.get("ok") and rel <= 1e-4 and cmp["memorised_gap"] <= 1e-2)
            out["lanes"][name] = cmp
            ok = ok and cmp["ok"]
            print(f"[sm41-engine-proof] P1 user {u} {name}: {cmp}", flush=True)
        # P2: the scalar reference, every policy kind (lane b against the full batch's trace)
        out["scalar"] = {}
        ref_eng = Engine(model, warm, D, args.days, lanes, trace=True, fsrs=fsrs, fsrs_source=src)
        ref_eng._due0 = {b: dict(ref_eng.sess[b].due) for b in range(ref_eng.B) if ref_eng.sess[b] is not None}
        for name in ("ihr-0.90#0", "fsrs7-0.90#0", "curve-0.90#0", "ihr-0.90-free#0"):
            b = lanes.index(name)
            t1 = time.perf_counter()
            tf_tr, ln_tr = scalar_run(ref_eng, b)
            c1 = compare_traces(full.traces[b], ln_tr)
            c0 = compare_traces(full.traces[0], tf_tr)
            out["scalar"][name] = {"lane": c1, "tf": c0, "seconds": round(time.perf_counter() - t1, 1)}
            ok = ok and bool(c1.get("ok") and c0.get("ok"))
            print(f"[sm41-engine-proof] P2 user {u} {name}: lane {c1}; tf {c0}", flush=True)
        # P2b: the split-day candidate step (batched, from the arena) == scalar steps, on 40 old cards
        if full.init_p is not None:
            sl = ScalarLane(ref_eng, 0)
            old_idx = np.nonzero(full.old)[0]
            pick = old_idx[np.linspace(0, len(old_idx) - 1, min(40, len(old_idx))).astype(int)]
            gap = 0.0
            for k in pick:
                c = full.cards[k]
                ids = warm.ids_of[c]
                keys = dict(zip(model.streams, (ids[0], ids[2], ids[1], ids[3], 0)))
                cur = {s: warm.states[s].get(keys[s]) for s in model.streams}
                for jt, t in enumerate(T_GRID):
                    o, _, _ = model.step(sl.cand_row(c, int(warm.last_day[c]), int(t)), cur)
                    pz = o["p"].astype(np.float64); pz = pz / pz.sum()
                    gap = max(gap, abs((1 - pz[0]) - full.init_p[np.searchsorted(old_idx, k), jt]))
            out["split_candidates_max_abs"] = gap
            ok = ok and gap <= 1e-4
            print(f"[sm41-engine-proof] P2b user {u}: split-day candidates max abs {gap:.2e}", flush=True)
        out["batched_seconds"] = round(tb, 1)
        res[str(u)] = out
    # P3: replay lanes reproduce part 1's rows (tf, fr-shared1 x 3 seeds), 60 days
    u3, D3 = args.p3
    row = next(json.loads(l) for l in (DR.OUT / "rows" / f"{u3}-60.jsonl").read_text().splitlines()
               if l.strip() and json.loads(l)["split_day"] == D3)
    DR.wait_clear(f"proof P3 user {u3}")
    warm, fsrs, src = setup(model, u3, D3, cache)
    names = ["tf", "fr-shared1#0", "fr-shared1#1", "fr-shared1#2"]
    eng = Engine(model, warm, D3, 60, names, seeds_n=3, fsrs=fsrs, fsrs_source=src)
    r3 = eng.run()
    p3 = {}
    for k, name in enumerate(names):
        j = row["lanes"].index(name)
        A_ = {n: np.asarray(row["acc"][n])[j] for n in row["acc"]}
        B_ = {n: np.asarray(r3["acc"][n])[k] for n in row["acc"]}
        ints = ["grades_same", "grades_later"]
        exact = all(np.array_equal(A_[n], B_[n]) for n in ints) and all(
            np.array_equal(A_[n][..., :2], B_[n][..., :2]) for n in ("later_old", "later_new", "first_touch")) and \
            np.array_equal(A_["time_later"][..., 0], B_["time_later"][..., 0])
        rel = max(float((np.abs(A_[n] - B_[n]) / np.maximum(1.0, np.abs(A_[n]))).max()) for n in A_)
        p3[name] = {"draws_identical": bool(exact), "max_rel_gap": rel, "ok": bool(exact and rel <= 1e-4)}
        ok = ok and p3[name]["ok"]
        print(f"[sm41-engine-proof] P3 user {u3} split {D3} {name}: {p3[name]}", flush=True)
    ft = {n: np.asarray(r3["acc"]["first_touch"])[k] for k, n in enumerate(names)}
    blk = slice(2, 4)
    st_drift = [float(ft[n][blk, 2].sum() / ft[n][blk, 0].sum() - ft["tf"][blk, 2].sum() / ft["tf"][blk, 0].sum())
                for n in names[1:]]
    res["P3"] = {"user": u3, "split": D3, "lanes": p3, "fr_shared1_first_touch_state_drift_days_30_59": st_drift,
                 "lane_reviews_per_second": r3["lane_reviews_per_second"]}
    res["ok"] = bool(ok)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[sm41-engine-proof] {'PASS' if ok else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ run
DEFAULT_LANES = ("tf,fr-shared1#0,fr-shared1#1,ihr-0.80#0,ihr-0.80#1,ihr-0.90#0,ihr-0.90#1,ihr-0.95#0,ihr-0.95#1,"
                 "fsrs7-0.80#0,fsrs7-0.80#1,fsrs7-0.90#0,fsrs7-0.90#1,fsrs7-0.95#0,fsrs7-0.95#1,curve-0.90#0,curve-0.90#1")


def run(args):
    S.low_priority()
    user, D, N = args.user, args.split, args.days
    if (user, D, N) in DR.done_keys(args.out):
        print(f"[sm41-engine] user {user} split {D}: done", flush=True)
        return
    DR.wait_clear(f"user {user}")
    model = S.Model(threads=args.threads)
    warm, fsrs, src = setup(model, user, D)
    print(f"[sm41-engine] user {user} split {D}: warm-up {warm.i} reviews {warm.seconds:.0f} s", flush=True)
    DR.wait_clear(f"user {user} split {D}")
    eng = Engine(model, warm, D, N, args.lanes.split(","), fsrs=fsrs, fsrs_source=src)
    res = eng.run()
    res["warmup_seconds"] = round(warm.seconds, 1)
    res["finished"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[sm41-engine] user {user} split {D}: {res['lane_reviews']} lane reviews in {res['wall_seconds']} s "
          f"({res['lane_reviews_per_second']}/s), {res['candidate_rows']} candidate rows", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ------------------------------------------------------------------ report
POLICY_TEXT = {"tf": "teacher forcing on the real schedule (the anchor)",
               "fr-shared1": "daily shared-stream re-anchoring (fr-shared1), real schedule, sampled grades and times",
               "ihr": "instant-head desired retention", "fsrs7": "FSRS-7 desired retention",
               "curve": "curve-head desired retention"}


def lane_summary(r, k, lo_b, hi_b):
    """Metrics of lane k over buckets [lo_b, hi_b)."""
    A = {n: np.asarray(v)[k, lo_b:hi_b].sum(0) for n, v in r["acc"].items()}
    lat = A["later_old"] + A["later_new"]
    load = A["load"]
    days = (hi_b - lo_b) * r["bucket_days"]
    gl = A["grades_later"]
    tl = A["time_later"]
    div = lambda a, b: float(a / b) if b else float("nan")   # noqa: E731
    return {"ret": div(lat[1], lat[0]), "pred": div(lat[2], lat[0]), "aim": div(A["aim"][1], A["aim"][0]),
            "later": lat[0], "reviews": load[:3].sum(), "min_day": load[3:].sum() / 60 / days,
            "minutes": load[3:].sum() / 60, "easy": div(gl[3], gl[1:].sum()), "s_later": div(tl[1], tl[0]),
            "ft_pred": div(A["first_touch"][2], A["first_touch"][0]), "ft_n": A["first_touch"][0]}


def report(args):
    import glob
    rows = []
    for path in sorted(glob.glob(args.rows)):
        rows += [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["user"], r["split_day"]))
    L = ["# The small model in the review simulator (SM41): engine validation tables", ""]
    out = {"users": [], "policies": {}}
    pol_rows = {}
    for r in rows:
        nb = len(r["acc"]["load"][0])
        burn = 30 // r["bucket_days"]
        groups = {}
        for k, name in enumerate(r["lanes"]):
            groups.setdefault(name.split("#")[0], []).append(k)
        tf = lane_summary(r, groups["tf"][0], burn, nb)
        tf_all = lane_summary(r, groups["tf"][0], 0, nb)
        for base, ks in groups.items():
            late = [lane_summary(r, k, burn, nb) for k in ks]
            whole = [lane_summary(r, k, 0, nb) for k in ks]
            mem = float(np.mean([r["memorised_instant"][k] for k in ks]))
            memc = float(np.mean([r["memorised_curve"][k] for k in ks]))
            mean = lambda xs, key: float(np.nanmean([x[key] for x in xs]))   # noqa: E731
            rec = {"user": r["user"], "split": r["split_day"], "policy": base,
                   "ret_30_59": mean(late, "ret"), "pred_30_59": mean(late, "pred"), "aim_30_59": mean(late, "aim"),
                   "ret_all": mean(whole, "ret"), "min_day_30_59": mean(late, "min_day"),
                   "min_day_all": mean(whole, "min_day"), "minutes_all": mean(whole, "minutes"),
                   "reviews_all": mean(whole, "reviews"), "memorised": mem, "memorised_curve": memc,
                   "min_per_memorised": mean(whole, "minutes") / max(mem, 1e-9),
                   "easy_drift": mean(late, "easy") - tf["easy"], "s_drift": mean(late, "s_later") - tf["s_later"],
                   "ft_state_drift": mean(late, "ft_pred") - tf["ft_pred"] if base.startswith("fr") else float("nan"),
                   "backlog": float(np.mean([r["backlog"][k] for k in ks])), "cards": r["cards"]}
            pol_rows.setdefault(base, []).append(rec)
        out["users"].append({"user": r["user"], "split": r["split_day"], "window_reviews": r["window_reviews"],
                             "cards": r["cards"], "lane_reviews_per_second": r["lane_reviews_per_second"],
                             "real_retention_all": tf_all["ret"], "fsrs7": r["fsrs7"]})
    order = [p for p in ["tf", "fr-shared1", "ihr-0.80", "ihr-0.90", "ihr-0.95", "fsrs7-0.80", "fsrs7-0.90",
                         "fsrs7-0.95", "curve-0.90"] if p in pol_rows] + sorted(set(pol_rows) - {
                             "tf", "fr-shared1", "ihr-0.80", "ihr-0.90", "ihr-0.95", "fsrs7-0.80", "fsrs7-0.90",
                             "fsrs7-0.95", "curve-0.90"})

    def label(p):
        if p in POLICY_TEXT:
            return f"{POLICY_TEXT[p]} ({p})" if p != "fr-shared1" else POLICY_TEXT[p]
        kind, _, rr = p.partition("-")
        return f"{POLICY_TEXT.get(kind, kind)} {rr} ({p})"
    users = sorted({(r['user'], r['split_day']) for r in rows})
    L += [f"{len(users)} users x 1 split, {rows[0]['days'] if rows else 0} days, 2 seeds per sampled lane (seeds "
          "averaged). Retention = pass share of later-day reviews; 'aim' = the policy's own predicted recall at the "
          "interval it chose (instant head, FSRS-7 or the curve head); 'pred' = the instant head's P(pass) at the "
          "actual review. Days 30-59 = after the 30-day burn-in of the policy switch. Workload = answer time of every "
          "review (first, same-day, later-day). Memorised = sum over the card set of the instant head's recall at day "
          "60. Drift columns: lane minus teacher forcing on the real schedule, days 30-59 (Easy share of later-day "
          "passes, seconds per later-day review; first-touch state drift for the replay lane).", ""]
    L += ["## Mean over users", "",
          "| policy | retention 30-59 | pred | aim | retention all | min/day 30-59 | min/day all | memorised | "
          "min per memorised card | Easy drift, points | seconds drift | first-touch state drift, points | backlog |",
          "|---|" + "---|" * 12]
    for p in order:
        v = pol_rows[p]
        m = lambda key: float(np.nanmean([x[key] for x in v]))   # noqa: E731
        fmt = lambda x, d=3: "—" if not np.isfinite(x) else f"{x:.{d}f}"   # noqa: E731
        L.append(f"| {label(p)} | {fmt(m('ret_30_59'))} | {fmt(m('pred_30_59'))} | {fmt(m('aim_30_59'))} | "
                 f"{fmt(m('ret_all'))} | {fmt(m('min_day_30_59'), 1)} | {fmt(m('min_day_all'), 1)} | "
                 f"{fmt(m('memorised'), 0)} | {fmt(m('min_per_memorised'), 2)} | {fmt(100 * m('easy_drift'), 1)} | "
                 f"{fmt(m('s_drift'), 1)} | {fmt(100 * m('ft_state_drift'), 1)} | {fmt(m('backlog'), 0)} |")
        out["policies"][p] = {k: m(k) for k in v[0] if k not in ("user", "split", "policy")}
    L += ["", "## Per user", "", "| user (split) | policy | retention 30-59 | aim | min/day all | memorised | "
          "min per memorised card |", "|---|---|---|---|---|---|---|"]
    for (u, sp) in users:
        for p in order:
            for x in pol_rows[p]:
                if x["user"] == u and x["split"] == sp:
                    L.append(f"| {u} ({sp}) | {p} | {x['ret_30_59']:.3f} | {x['aim_30_59']:.3f} | "
                             f"{x['min_day_all']:.1f} | {x['memorised']:.0f} | {x['min_per_memorised']:.2f} |")
    lr = sum(r["lane_reviews"] for r in rows); wall = sum(r["wall_seconds"] for r in rows)
    wu = sum(r.get("warmup_seconds", 0) for r in rows)
    L += ["", f"Speed: {lr:,} simulated lane reviews in {wall / 3600:.2f} process-hours ({lr / max(wall, 1):,.0f} "
          f"lane reviews/s per process, 1 thread, {len(rows[0]['lanes']) if rows else 0} lanes), "
          f"{sum(r['candidate_rows'] for r in rows):,} candidate query rows; warm-ups {wu / 3600:.2f} process-hours.", ""]
    out["speed"] = {"lane_reviews": lr, "wall_seconds": wall, "lane_reviews_per_second": lr / max(wall, 1),
                    "warmup_seconds": wu}
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print("\n".join(L))


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("proof"); q.add_argument("--days", type=int, default=20)
    q.add_argument("--users", type=lambda x: [tuple(map(int, y.split(":"))) for y in x.split(",")], default="3876:225")
    q.add_argument("--p3", type=lambda x: tuple(map(int, x.split(":"))), default="3876:225")
    q.add_argument("--out", default=str(OUT / "proof-cpu.json"))
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True)
    r.add_argument("--split", type=int, required=True); r.add_argument("--days", type=int, default=60)
    r.add_argument("--lanes", default=DEFAULT_LANES); r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", default=str(OUT / "validation.jsonl"))
    t = sub.add_parser("report"); t.add_argument("--rows", default=str(OUT / "rows" / "*.jsonl"))
    t.add_argument("--md", default=str(OUT / "engine-tables.md")); t.add_argument("--json", default=str(OUT / "engine-result.json"))
    a = p.parse_args()
    {"proof": proof, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
