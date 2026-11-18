#!/usr/bin/env python
"""Fluency vs desired retention in the simulator with answer-time feedback (SM127). Design (fixed before any result):
artifacts/small-model/sm127/design.md.

Question (user, 11-15): is a high desired retention enough for mastery, or do we need a scheduler that keeps answer
time? At equal daily review time, instant-head desired retention 0.90 / 0.95 / 0.98 against a fluency target
scheduler (the interval at which P(recalled AND answered within 8 s, reading the back included) falls to 0.80), one
simulated year per user; memorised and fluent cards at day 365.

The learner is the review simulator hosting the small model (SM41) lane engine (scripts/sm41_engine.py, unchanged:
this file subclasses it) with the model table swapped to the show-time retrain (SM92) + its time head from the time
heads for the family (SM117). Additions here, all opt-in and outside the SM41 code:
  - the fluency policy `flu-R` (grid policy): the same 16 candidate query rows as the instant-head desired-retention
    policy (ihr-R), plus the time head on each candidate row -> P(recalled and answered within S s) = sum over Hard /
    Good / Easy of P(g) x P(t <= S | g) (log-uniform inside the bin, as the sampler draws); interval = the first
    crossing of R; 1 day if already below R at 1 day (counted as a floor decision). Suffix -fb<RR>: floor cards get the
    instant-head desired-retention RR interval instead.
  - lane suffix -nofb: the trunk's duration input of every later-day review row = the user's pre-split median
    later-day answer time of that grade (no answer-time feedback into the memory model); the sampled time still
    drives the workload, the clock and the time head's explicit features (the sensitivity arm of design.md).
  - a daily time budget for every policy lane (the user's real mean review seconds per calendar day over the window):
    due cards overdue first, then deferred and today's new cards in real order; once the day's answer seconds reach
    the budget no new card and no further due card starts (same-day steps of cards already shown still run); unshown
    due cards stay due, unshown new cards are deferred.
  - read-outs: per later-day review the realised "passed and answered within S s" and the head's P of it at the show
    moment; at the horizon memorised (sum of P(recall)) and fluent (sum of P(recalled and within S s)) over the cards
    the lane has introduced, plus threshold counts.

  prepare   the combined master (the show-time retrain (SM92) trunk + its SM117 head) -> work/ (sha256 checked)
  select    users.json: the fixed user list (full design + pilot)
  check     the CPU proof -> artifacts/small-model/sm127/check-cpu.json
  run       --user U --split D --days N --out F   one user (resumable: F exists = done); holds in quiet windows
  report    rows -> sm127-tables.md / sm127-result.json
"""
from argparse import ArgumentParser
import copy
import datetime as dt
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RWKV_NO_JIT", "1")
from scripts import sm41_sim as S       # noqa: E402
from scripts import sm41_drift as DR    # noqa: E402
from scripts import sm41_engine as E    # noqa: E402
from rwkv import data_processing as DP  # noqa: E402

HERE = ROOT / "artifacts/small-model/sm127"
WORK = HERE / "work"
RUN = HERE / "run"
STOP = RUN / "STOP"
WT = Path("C:/Users/jonat/workspace/wt")
SM92_MASTER = WT / "sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth"   # read only
SM92_SHA = "2091f8a879a56cf96a3df3b820fdacfa64c6467e8502a3a5eb17436524100d7f"   # = hf-family rwkv-small-two-head manifest
SM117_HEAD = WT / "sm117/artifacts/small-model/sm117-sm92-head/head.pth"                                     # read only
COMBINED = WORK / "sm92-sm117head-master.pth"
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
U732 = ROOT / "artifacts/small-model/eval-users-732.json"
S_FAST = 8.0               # seconds: "answered within 8 s, reading the back included"
HORIZON = 365
POLICIES = ("ihr-0.90", "ihr-0.95", "ihr-0.98", "flu-0.50", "flu-0.70", "flu-0.80", "flu-0.80-fb95",
            "ihr-0.90-nofb", "ihr-0.98-nofb", "flu-0.80-nofb")
SEEDS = 2
LANES = ",".join(["tf", "fr-shared1#0"] + [f"{p}#{s}" for s in range(SEEDS) for p in POLICIES])
BOOT = 10000
SEED = 127
BAR = 0.02                 # practical bar of the decision rule: 2% of fluent (memorised) cards

E.GRID_POLICIES = ("ihr", "hyb", "flu")   # flu reads the candidate grid (tick defers its decisions like ihr)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ model table: SM92 + its SM117 time head
def prepare(args=None):
    """The combined master: the show-time retrain (SM92) trunk (sha256 = the package manifest) + its time head (SM117,
    keys hidden.* / out.*) under the SM82c prefix time_mlps.0., so sm41_sim.Model loads it unchanged."""
    import torch
    got = sha256(SM92_MASTER)
    if got != SM92_SHA:
        raise SystemExit(f"{SM92_MASTER}: sha256 {got} != {SM92_SHA}")
    trunk = torch.load(SM92_MASTER, map_location="cpu", weights_only=True)
    head = torch.load(SM117_HEAD, map_location="cpu", weights_only=True)
    assert not any(k.startswith(S.HEAD) for k in trunk), "the SM92 trunk already carries a time head"
    assert set(head) == {"hidden.weight", "hidden.bias", "out.weight", "out.bias"}, sorted(head)
    sd = dict(trunk)
    sd.update({S.HEAD + k: v for k, v in head.items()})
    WORK.mkdir(parents=True, exist_ok=True)
    if not COMBINED.exists():
        torch.save(sd, COMBINED)
    info = {"sm92_master": str(SM92_MASTER), "sm92_sha256": got, "sm117_head": str(SM117_HEAD),
            "sm117_head_sha256": sha256(SM117_HEAD), "combined": str(COMBINED), "combined_sha256": sha256(COMBINED)}
    print(json.dumps(info, indent=1))
    return info


class Model127(S.Model):
    """sm41_sim.Model on the combined master (show-time retrain (SM92) + its SM117 time head)."""

    def __init__(self, threads=1):
        if not COMBINED.exists():
            prepare()
        super().__init__(master=COMBINED, threads=threads)
        lnS = math.log(S_FAST)
        lo, hi = self.lo, self.hi
        # share of each bin at or below S under the sampler's log-uniform draw (bin 15 = the 60 s cap: 0 for S < 60)
        self.frac = np.where(hi <= lnS, 1.0, np.where(lo >= lnS, 0.0, (lnS - lo) / np.where(hi > lo, hi - lo, 1.0)))


# ------------------------------------------------------------------ vstep wrapper: no-feedback rows + captures
_VSTEP0 = E.vstep


def _vstep(model, X, st, fresh, snaps=False, curve=False):
    nf = getattr(model, "_nofb", None)
    if curve and nf is not None and nf.any():
        # the commit (review) row: duration column of the -nofb lanes' later-day rows = the user's median for the grade
        g = np.argmax(X[:, 9:13], 1) + 1
        X[nf, 8] = model._med[g[nf]]
        model._nofb_rows = getattr(model, "_nofb_rows", 0) + int(nf.sum())
    if curve and getattr(model, "_commit_log", None) is not None:
        model._commit_log.append((model._cur_A.copy(), model._cur_kind.copy(), X[:, 8].copy(),
                                  np.argmax(X[:, 9:13], 1) + 1))
    out = _VSTEP0(model, X, st, fresh, snaps=snaps, curve=curve)
    if snaps:
        model._last_p, model._last_fz = out[0], out[2]
    return out


E.vstep = _vstep


# ------------------------------------------------------------------ lane names
def split_name(name):
    """'flu-0.80-fb95-nofb#1' -> ('flu-0.80#1', {'nofb': True, 'fb': 0.95}): the SM127 suffixes are not SM41 options."""
    base, _, seed = name.partition("#")
    keep, flags = [], {"nofb": False, "fb": None}
    for p in base.split("-"):
        if p == "nofb":
            flags["nofb"] = True
        elif re.fullmatch(r"fb\d\d", p):
            flags["fb"] = int(p[2:]) / 100
        else:
            keep.append(p)
    return "-".join(keep) + (f"#{seed}" if seed else ""), flags


# ------------------------------------------------------------------ the budgeted session
class BudgetSession(E.PolicySession):
    """sm41_engine.PolicySession + a daily answer-time budget and a deferred new-card queue (design.md 4)."""

    def __init__(self, key, due, new_by_day, rules, budget_s):
        super().__init__(key, due, new_by_day, rules)
        self.budget = float(budget_s)
        self.spent, self.deferred, self.closed = 0.0, [], False
        self.hits, self.introduced, self.violations = 0, [], 0
        self.day_spent = []

    def start_day(self, d, start):
        if self.main or self.learn:
            raise RuntimeError("a day started before the previous one ended")
        if self.count:
            self.day_spent.append(round(self.spent, 1))
        todays = [c for c, v in self.due.items() if v <= d]
        if todays:
            arr = np.asarray(todays, np.int64)
            tie = E.uniforms(self.key, 7, d, arr)
            order = np.lexsort((tie, np.asarray([self.due[c] for c in todays])))
            todays = arr[order].tolist()
        main = todays + self.deferred + self.new_by_day.get(d, [])
        main.reverse()
        self.deferred = []
        self.main, self.learn, self.now, self.count = main, [], max(start, self.now), {}
        self.spent, self.closed = 0.0, False

    def close(self):
        rest = self.main[::-1]
        self.deferred = [c for c in rest if c not in self.due]   # never shown: new cards keep their real order
        self.main, self.closed = [], True
        self.hits += 1

    def next(self):
        while True:
            if self.learn and self.learn[0][0] <= self.now:
                return heapq.heappop(self.learn)[2]
            if self.main and self.spent >= self.budget:
                self.close()
            if self.main:
                if self.spent >= self.budget:
                    self.violations += 1
                return self.main.pop()
            if self.learn:
                self.now = self.learn[0][0]
                continue
            return None

    def after(self, c, kind, rating, dur_ms, u_follow, u_gap, u_tie):
        self.spent += dur_ms / 1000.0
        if kind == 0:
            self.introduced.append(int(c))
        return super().after(c, kind, rating, dur_ms, u_follow, u_gap, u_tie)

    def finish(self):
        if self.count:
            self.day_spent.append(round(self.spent, 1))


# ------------------------------------------------------------------ the engine
class Engine127(E.Engine):
    """sm41_engine.Engine + the fluency policy, the -nofb lanes, the budget and the fluency read-outs."""

    def __init__(self, model, warm, D, N, lane_names, budget_s, med_ms, pause=None, trace=False):
        bases, flags = zip(*[split_name(n) for n in lane_names])
        super().__init__(model, warm, D, N, list(bases), trace=trace, fsrs=None, fsrs_source="none (not used)")
        for b, n in enumerate(lane_names):
            self.cfg[b]["name"] = n
        B = self.B
        self.pause = pause
        self.nofb = np.array([f["nofb"] for f in flags])
        self.fb_r = np.array([f["fb"] if f["fb"] else np.nan for f in flags], np.float64)
        self.is_flu = np.array([c["kind"] == "policy" and c["policy"] == "flu" for c in self.cfg])
        assert not (self.nofb & ~self.is_pol).any() and not (np.isfinite(self.fb_r) & ~self.is_flu).any()
        self.med_ms = {int(g): float(v) for g, v in med_ms.items()}
        model._med = np.array([0.0] + [float(DP.scale_duration(np.array([self.med_ms[g]], np.float64))[0])
                                       for g in (1, 2, 3, 4)], np.float32)
        model._nofb = None
        model._nofb_rows = 0          # per engine (the model object is reused across engines)
        self.budget = float(budget_s)
        self.floor_n, self.dec_n, self.aimf_sum = np.zeros(B), np.zeros(B), np.zeros(B)
        nb = self.acc["load"].shape[1]
        # later-day reviews: n, realised passed-and-within-S, the head's P(recalled and within S) at the show moment,
        # realised passes
        self.acc["fluent_later"] = np.zeros((B, nb, 4))
        old_idx = np.nonzero(self.old)[0]
        dues = {}
        if self.is_flu.any() and len(old_idx):
            pg, pf = self.grid2(np.zeros(len(old_idx), np.int64), old_idx, None, self.last_day[0, old_idx])
            for bl in np.nonzero(self.is_flu)[0]:
                r = np.full(len(old_idx), self.cfg[bl]["r"])
                ti, aim = self._flu_choose(np.full(len(old_idx), bl), pg, pf, r)
                dd = np.maximum(0, self.last_day[bl, old_idx] + ti - D)
                self.aim[bl, old_idx] = aim
                self.backlog[bl] = int((self.last_day[bl, old_idx] + ti - D < 0).sum())
                dues[bl] = dict(zip(old_idx.tolist(), dd.tolist()))
        for bl, c in enumerate(self.cfg):
            if c["kind"] == "policy":
                due = dues.get(bl, self.sess[bl].due)
                self.sess[bl] = BudgetSession(self.keys[bl], due, self.new_by_day, self.habits["rules"], self.budget)

    # -------------------------------------------------------------- the fluency grid
    def _explicit_rows(self, A, e, T):
        """Explicit time features [n * K, 26] for card e of lane A at elapsed T days (the card's lane history)."""
        n, K = len(A), len(T)
        base = np.stack([self.expl[a].features(self.cards[k], 1) for a, k in zip(A, e)]) if n else np.zeros((0, 26))
        ex = np.repeat(base[:, None, :], K, 1).astype(np.float32)
        ex[:, :, 23] = (np.log1p(np.maximum(np.asarray(T, np.float64), 0)) / 3.0)[None, :]
        ex[:, :, 24] = 0.0
        return ex.reshape(n * K, 26)

    def _pfast(self, p, fz, ex):
        """P(recalled and answered within S s) per row: p [m, 4] (normalised), fz [m, 448], ex [m, 26]."""
        tp = DR.time_probs_batch(self.model, fz, ex)          # [m, 4, 16]
        cdf = tp @ self.model.frac                              # [m, 4]
        return (p[:, 1:] * cdf[:, 1:]).sum(-1)

    def grid2(self, A, e, nxt, base_day, chunk=2048):
        """sm41_engine.Engine.candidates_p with the frozen features kept: (P(pass), P(recalled and within S)) [n, K]."""
        torch = self.model.torch
        K = len(E.T_GRID)
        T = np.broadcast_to(np.asarray(E.T_GRID, np.int64), (len(A), K))
        pg, pf = np.empty((len(A), K)), np.empty((len(A), K))
        step = max(1, chunk // K)
        for lo in range(0, len(A), step):
            sl = slice(lo, lo + step)
            X = self._cand(A[sl], e[sl], base_day[sl], T[sl])
            if nxt is None:
                st, fr = self.gather(A[sl], e[sl])
            else:
                st = {s: {layer: E._nest([None if t is None else t[sl] for t in E._leaves(v)])
                          for layer, v in nxt[s].items()} for s in nxt}
                fr = {s: None for s in nxt}
            rep = {s: {layer: E._nest([None if t is None else torch.repeat_interleave(t, K, 0) for t in E._leaves(v)])
                       for layer, v in st[s].items()} for s in st}
            frr = {s: None if fr[s] is None else np.repeat(fr[s], K) for s in fr}
            p, _, fz, _ = E.vstep(self.model, X, rep, frr, snaps=True)
            p = p / p.sum(-1, keepdims=True)
            pg[sl] = (1.0 - p[:, 0]).reshape(-1, K)
            pf[sl] = self._pfast(p, fz, self._explicit_rows(A[sl], e[sl], E.T_GRID)).reshape(-1, K)
            self.cand_rows += X.shape[0]
        return pg, pf

    def _flu_choose(self, lanes, pg, pf, r):
        _, ti, _ = E.choose_interval(pf, r)
        floor = pf[:, 0] < r
        fb = self.fb_r[lanes]
        use = floor & np.isfinite(fb)
        if use.any():
            _, tb, _ = E.choose_interval(pg[use], fb[use])
            ti[use] = tb
        lg = np.log(E.T_GRID)
        aim = np.array([np.interp(math.log(x), lg, pg[i]) for i, x in enumerate(ti)])
        aimf = np.array([np.interp(math.log(x), lg, pf[i]) for i, x in enumerate(ti)])
        np.add.at(self.floor_n, lanes, floor)
        np.add.at(self.dec_n, lanes, 1)
        np.add.at(self.aimf_sum, lanes, aimf)
        return ti, aim

    def _decide(self, pend, pg, pf):
        A, e, r = pend["A"], pend["e"], pend["r"]
        assert not pend.get("special"), "hyb lanes are not part of SM127"
        _, ti, aim = E.choose_interval(pg, r)
        if pf is not None:
            idx = np.nonzero(self.is_flu[A])[0]
            if len(idx):
                ti_f, aim_f = self._flu_choose(A[idx], pg[idx], pf[idx], r[idx])
                ti[idx], aim[idx] = ti_f, aim_f
        for j, (b, k, t_, a_) in enumerate(zip(A, e, ti, aim)):
            self.sess[b].due[int(k)] = pend["d"] + int(t_)
            self.aim[b, k] = a_
            if self.trace_on:
                pos = pend["tpos"][j]
                tr = self.traces[b][pos]
                self.traces[b][pos] = tr[:5] + (int(t_),) + tr[6:]

    def resolve(self, pend, pg):
        """Called by tick right after the query step that carried the candidates (their frozen features are the last
        m x K rows of that step, captured by the vstep wrapper)."""
        pf = None
        A = pend["A"]
        idx = np.nonzero(self.is_flu[A])[0]
        if len(idx):
            torch = self.model.torch
            K = len(E.T_GRID)
            m = len(A)
            p = self.model._last_p[-m * K:]
            p = p / p.sum(-1, keepdims=True)
            fz = self.model._last_fz[-m * K:]
            rows = (idx[:, None] * K + np.arange(K)[None, :]).ravel()
            pf = np.full((m, K), np.nan)
            pf[idx] = self._pfast(p[rows], fz[torch.from_numpy(rows)],
                                  self._explicit_rows(A[idx], pend["e"][idx], E.T_GRID)).reshape(-1, K)
        self._decide(pend, pg, pf)

    def flush(self):
        pend, self.pending = self.pending, None
        if pend is None:
            return
        m = len(pend["A"])
        base = np.full(m, self.D + pend["d"], np.int64)
        if self.is_flu[pend["A"]].any():
            pg, pf = self.grid2(pend["A"], pend["e"], pend["st"], base)
        else:
            pg, pf = self.candidates_p(pend["A"], pend["e"], pend["st"], base, E.T_GRID), None
        self._decide(pend, pg, pf)

    # -------------------------------------------------------------- per tick: no-feedback rows and the read-outs
    def tick(self, d, A, e, kind, ed, ridx):
        m = self.model
        n = len(A)
        k2 = np.nonzero(kind == 2)[0]
        ex = (np.stack([self.expl[A[j]].features(self.cards[e[j]], int(ed[j])) for j in k2]) if len(k2) else None)
        m._last_fz = None
        m._nofb = (self.nofb[A] & (kind == 2)) if self.nofb.any() else None
        m._cur_A, m._cur_kind = A, kind
        try:
            super().tick(d, A, e, kind, ed, ridx)
        finally:
            m._nofb = None
        if not len(k2) or m._last_fz is None:
            return
        torch = m.torch
        p = m._last_p[:n]
        p = p / p.sum(-1, keepdims=True)
        pfl = self._pfast(p[k2], m._last_fz[:n][torch.from_numpy(k2)], ex)
        g = np.empty(len(k2), np.int64)
        ms = np.empty(len(k2))
        for jj, j in enumerate(k2):
            st = self.expl[A[j]].cards[self.cards[e[j]]]
            g[jj] = st[2]
            ms[jj] = float(self.rev["duration"][ridx[j]]) if ridx[j] >= 0 else 1000.0 * math.exp(st[0][0])
        fast = (g > 1) & (ms > 0) & (ms <= S_FAST * 1000.0 + 1e-6)
        bkt = d // E.BUCKET
        acc = self.acc["fluent_later"]
        np.add.at(acc, (A[k2], bkt, 0), 1.0)
        np.add.at(acc, (A[k2], bkt, 1), fast.astype(np.float64))
        np.add.at(acc, (A[k2], bkt, 2), pfl)
        np.add.at(acc, (A[k2], bkt, 3), (g > 1).astype(np.float64))

    def day(self, d):
        if self.pause is not None:
            self.pause(d)
        super().day(d)

    # -------------------------------------------------------------- the horizon
    def memorised(self):
        """At 12:00 of day N, over the card set: P(recall) and P(recalled and within S s) from a query row on the
        lane's final states. New cards the lane never showed count 0 (they were never learned)."""
        torch = self.model.torch
        B, C, D, N = self.B, self.C, self.D, self.N
        intro = self.old[None, :] | self.seen_win
        A = np.repeat(np.arange(B), C); e = np.tile(np.arange(C), B)
        ok = intro.ravel()
        last = np.where(ok, self.last_day[A, e], D + N - 1)
        tdays = np.maximum(1, D + N - last)
        pr, pf = np.zeros(len(A)), np.zeros(len(A))
        chunk = 2048
        for lo in range(0, len(A), chunk):
            sl = slice(lo, lo + chunk)
            sel = np.nonzero(ok[sl])[0] + lo
            if not len(sel):
                continue
            X = self._cand(A[sel], e[sel], last[sel], tdays[sel][:, None])
            st, fr = self.gather(A[sel], e[sel])
            p, _, fz, _ = E.vstep(self.model, X, st, fr, snaps=True)
            p = p / p.sum(-1, keepdims=True)
            pr[sel] = 1 - p[:, 0]
            ex = np.stack([self.expl[a].features(self.cards[k], int(t)) for a, k, t in zip(A[sel], e[sel], tdays[sel])])
            pf[sel] = self._pfast(p, fz, ex.astype(np.float32))
        PR, PF = pr.reshape(B, C), pf.reshape(B, C)
        old = self.old
        self.h = {"memorised": PR.sum(1), "fluent": PF.sum(1), "memorised_old": PR[:, old].sum(1),
                  "fluent_old": PF[:, old].sum(1), "n_mem90": (PR >= 0.9).sum(1), "n_flu80": (PF >= 0.8).sum(1),
                  "n_both": ((PR >= 0.9) & (PF >= 0.8)).sum(1), "introduced_new": (intro & ~old[None, :]).sum(1)}
        t_end = (D + N) * E.DAY + E.MU
        tt = np.maximum(1.0, t_end - np.nan_to_num(self.last_end, nan=t_end))
        crv = np.array([np.where(intro[b], E.curve_np(self.cw[b], self.cs[b], self.cd[b], tt[b]), 0.0).sum()
                        for b in range(B)])
        return PR.sum(1), crv

    def result(self):
        res = super().result()
        sess = [s if isinstance(s, BudgetSession) else None for s in self.sess]
        for s in sess:
            if s is not None:
                s.finish()
        res.update({
            "s_fast": S_FAST, "budget_seconds_per_day": round(self.budget, 2), "med_ms": self.med_ms,
            "new_cards": int((~self.old).sum()),
            **{f"h_{k}": np.round(np.asarray(v, np.float64), 3).tolist() for k, v in self.h.items()},
            "flu_decisions": self.dec_n.tolist(), "flu_floor_decisions": self.floor_n.tolist(),
            "flu_aim_fluent_sum": np.round(self.aimf_sum, 3).tolist(),
            "budget_hit_days": [s.hits if s else -1 for s in sess],
            "budget_violations": [s.violations if s else -1 for s in sess],
            "deferred_new_at_end": [len(s.deferred) if s else -1 for s in sess],
            "day_seconds_max_over_budget": [round(max([0.0] + [x - self.budget for x in s.day_spent]), 1) if s else -1
                                            for s in sess],
            "nofb_rows": int(getattr(self.model, "_nofb_rows", 0)),
        })
        return res


# ------------------------------------------------------------------ quiet windows (quiet.json all/cpu + daily break)
def quiet_blocks(now):
    blocks = []
    try:
        for w in json.loads(QUIET.read_text(encoding="utf-8")):
            if w.get("scope", "all") in ("all", "cpu"):
                blocks.append((dt.datetime.fromisoformat(w["start"]), dt.datetime.fromisoformat(w["end"])))
    except Exception:  # noqa: BLE001  unreadable: the daily break below still applies
        pass
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    blocks += [(day + dt.timedelta(days=k, hours=20), day + dt.timedelta(days=k + 1, hours=1)) for k in range(-1, 3)]
    blocks += [(day + dt.timedelta(days=k, hours=16), day + dt.timedelta(days=k, hours=20)) for k in range(-1, 3)
               if (day + dt.timedelta(days=k)).weekday() == 6]
    return blocks


class Stopped(Exception):
    pass


def quiet_wait(tag, margin_min=0.0):
    """Sleeps while inside a quiet window or within margin_min of the next one; raises Stopped on the STOP file."""
    said = False
    while True:
        if STOP.exists():
            raise Stopped(f"{tag}: STOP file")
        now = dt.datetime.now()
        inside = [b for a, b in quiet_blocks(now) if a - dt.timedelta(minutes=margin_min) <= now < b]
        if not inside:
            if said:
                print(f"[sm127] {tag}: resumed at {now:%m-%d %H:%M}", flush=True)
            return
        if not said:
            print(f"[sm127] {tag}: holding (quiet window until {max(inside):%m-%d %H:%M}) at {now:%m-%d %H:%M}",
                  flush=True)
            said = True
        time.sleep(60)


# ------------------------------------------------------------------ users
def select(args):
    """The fixed user list (design.md 2): the 732 set in numpy default_rng(4141) order (the SM41 order); users with 8k-150k
    reviews, >= 95% timed, later-day retention < 0.99 over the history (Good-only excluded); one 365-day window per
    user, the latest split day D on a 15-day grid back from (last day - 364) that passes: >= 3,000 reviews before D,
    3,000-30,000 reviews in the window, >= 150 active days, >= 1,000 later-day reviews with retention < 0.995 and >= 50
    Again, >= 300 new cards, <= 8,000 cards in the window (RAM). Full design: the first 150 that qualify; pilot: the
    first 12 of them."""
    import pandas as pd
    from scripts import compress_common as cc
    data = cc.data_path()
    pool = json.loads(U732.read_text())["users"]
    rng = np.random.default_rng(4141)
    order = [int(u) for u in rng.permutation(sorted(int(u) for u in pool))]
    chosen, looked = [], 0
    for u in order:
        if len(chosen) >= args.users:
            break
        looked += 1
        df = pd.read_parquet(data / "revlogs" / f"user_id={u}",
                             columns=["day_offset", "elapsed_days", "rating", "duration", "card_id"])
        n = len(df)
        if not 8000 <= n <= 150000 or (df["duration"] > 0).mean() < 0.95:
            continue
        day = df["day_offset"].to_numpy(); ed = df["elapsed_days"].to_numpy(); g = df["rating"].to_numpy()
        card = df["card_id"].to_numpy(); dur = df["duration"].to_numpy()
        if ((g[ed >= 1] > 1).mean() if (ed >= 1).any() else 1.0) >= 0.99:
            continue
        # get_rwkv_data shifts day_offset to start at 0: the engine's split day is relative to the first day
        d0 = int(day[0])
        for D in range(int(day[-1]) - HORIZON + 1, d0 + 59, -15):
            a = int(np.searchsorted(day, D)); b = int(np.searchsorted(day, D + HORIZON))
            if a < 3000:
                break
            w = b - a
            if not 3000 <= w <= 30000 or len(np.unique(day[a:b])) < 150:
                continue
            later = ed[a:b] >= 1
            if later.sum() < 1000 or (g[a:b][later] > 1).mean() >= 0.995 or (g[a:b][later] == 1).sum() < 50:
                continue
            newc = int((ed[a:b] == -1).sum()); C = len(np.unique(card[a:b]))
            if newc < 300 or C > 8000:
                continue
            chosen.append({"user": u, "reviews": n, "split_day": int(D - d0), "pre_reviews": a, "window_reviews": w,
                           "active_days": int(len(np.unique(day[a:b]))), "new_cards": newc, "cards": C,
                           "real_minutes_per_day": round(float(dur[a:b].sum()) / 60000 / HORIZON, 2)})
            break
    out = {"rule": select.__doc__.split("\n", 1)[1].strip().replace("\n    ", " "), "looked_at": looked,
           "users": chosen, "pilot": [c["user"] for c in chosen[:args.pilot]]}
    atomic_json(HERE / "users.json", out)
    print(f"[sm127-select] {len(chosen)} users (looked at {looked}); pilot {out['pilot']}; window reviews "
          f"{sum(c['window_reviews'] for c in chosen):,} (pilot {sum(c['window_reviews'] for c in chosen[:args.pilot]):,})")


def user_entry(user):
    for c in json.loads((HERE / "users.json").read_text())["users"]:
        if int(c["user"]) == int(user):
            return c
    raise KeyError(user)


def medians_later(rev, end, recent_days=60, min_n=100):
    """The user's pre-split median later-day answer time per grade (ms): the last `recent_days` days before the split
    when that grade has >= min_n timed later-day reviews there, else all pre-split ones, else all grades pooled."""
    day = rev["day_offset"][:end]; ed = rev["elapsed_days"][:end]; g = rev["rating"][:end]; u = rev["duration"][:end]
    lat = (ed >= 1) & (u > 0)
    recent = lat & (day >= day[end - 1] - recent_days + 1)
    pooled = float(np.median(u[lat])) if lat.any() else 8000.0
    out = {}
    for k in (1, 2, 3, 4):
        r = recent & (g == k); a = lat & (g == k)
        out[k] = float(np.median(u[r])) if r.sum() >= min_n else (float(np.median(u[a])) if a.sum() >= 20 else pooled)
    return out


def make_engine(model, user, D, N, lanes, budget=None, pause=None, trace=False, cache=None):
    """Warm-up (every real review before D, the training convention) + the engine. budget None = the user's real mean
    review seconds per calendar day over [D, D + N); float('inf') = no budget."""
    warm = cache.get(user) if cache is not None else None     # the engine only reads the warm-up (proof K6)
    if warm is None or warm.i != int(np.searchsorted(warm.rev["day_offset"], D, side="left")):
        rev = S.load_user(user)[0]
        warm = E.Warm(model, rev, user)
        end = int(np.searchsorted(rev["day_offset"], D, side="left"))
        warm.advance(end, E.FSRS7(E.FSRS7_DEFAULT))   # FSRS-7 states unused by SM127 policies
        if cache is not None:
            cache[user] = warm
    rev = warm.rev
    end = warm.i
    stop = int(np.searchsorted(rev["day_offset"], D + N, side="left"))
    real_s = float(rev["duration"][end:stop].sum()) / 1000.0
    b = real_s / N if budget is None else budget
    eng = Engine127(model, warm, D, N, lanes, b, medians_later(rev, end), pause=pause, trace=trace)
    return eng, warm


def run(args):
    S.low_priority()
    out = Path(args.out)
    if out.exists():
        print(f"[sm127] user {args.user}: done already ({out})", flush=True)
        return 0
    tag = f"user {args.user}"
    try:
        quiet_wait(tag, margin_min=args.margin)
        E.wait_memory(tag, args.min_free_gb)
        model = Model127(threads=args.threads)
        D = args.split if args.split is not None else int(user_entry(args.user)["split_day"])
        t0 = time.perf_counter()

        def pause(d):
            quiet_wait(f"{tag} day {d}", margin_min=0)
        eng, warm = make_engine(model, args.user, D, args.days, args.lanes.split(","), pause=pause)
        tw = time.perf_counter() - t0
        print(f"[sm127] {tag} split {D}: warm-up {warm.i} reviews {tw:.0f} s; {eng.C} cards, {eng.B} lanes, budget "
              f"{eng.budget:.0f} s/day", flush=True)
        res = eng.run()
    except Stopped as ex:
        print(f"[sm127] {ex}: exiting without output", flush=True)
        return 3
    res["warmup_seconds"] = round(tw, 1)
    res["finished"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[sm127] {tag}: {res['lane_reviews']} lane reviews in {res['wall_seconds']} s "
          f"({res['lane_reviews_per_second']}/s), {res['candidate_rows']} candidate rows", flush=True)
    atomic_json(out, res)
    return 0


# ------------------------------------------------------------------ CPU proof
def _strip(res):
    return {k: v for k, v in res.items() if k not in ("wall_seconds", "lane_reviews_per_second", "finished")}


def _lane_ret(res, k, lo_b, hi_b):
    a = np.asarray(res["acc"]["later_old"])[k, lo_b:hi_b].sum(0) + np.asarray(res["acc"]["later_new"])[k, lo_b:hi_b].sum(0)
    aim = np.asarray(res["acc"]["aim"])[k, lo_b:hi_b].sum(0)
    fl = np.asarray(res["acc"]["fluent_later"])[k, lo_b:hi_b].sum(0)
    return {"later_reviews": int(a[0]), "retention": round(a[1] / max(a[0], 1), 4),
            "aim": round(aim[1] / max(aim[0], 1), 4) if aim[0] else None,
            "fast_pass_share": round(fl[1] / max(fl[0], 1), 4), "pred_fast_pass": round(fl[2] / max(fl[0], 1), 4)}


def check(args):
    S.low_priority()
    import torch
    torch.set_num_threads(1)
    rep = {"started": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "s_fast": S_FAST}
    ok = {}
    # K1 the model table
    info = prepare()
    sd = torch.load(COMBINED, map_location="cpu", weights_only=True)
    tr = torch.load(SM92_MASTER, map_location="cpu", weights_only=True)
    hd = torch.load(SM117_HEAD, map_location="cpu", weights_only=True)
    same_trunk = set(tr) == {k for k in sd if not k.startswith(S.HEAD)} and all(torch.equal(sd[k], tr[k]) for k in tr)
    same_head = all(torch.equal(sd[S.HEAD + k], v) for k, v in hd.items())
    rep["K1_model"] = {**info, "trunk_bitwise": same_trunk, "head_bitwise": same_head}
    # C1-C3 of sm41_sim (builder == pipeline, recurrent == parallel incl. the 448 frozen time features, explicit
    # features) on the combined master
    S.MASTER = COMBINED
    S.Model.__init__.__defaults__ = (COMBINED, 1)
    c_out = WORK / "check-c123.json"
    S.check(type("A", (), {"user": args.c_user, "rows": args.c_rows, "out": str(c_out)})())
    c = json.loads(c_out.read_text())
    rep["K1_model"]["sm41_check_C1_C3"] = c
    ok["K1"] = bool(same_trunk and same_head and c["ok"] and info["sm92_sha256"] == SM92_SHA)
    torch.set_num_threads(1)
    model = Model127(threads=1)

    # K2 the fluent probability == the sampler's share at or below S (synthetic head outputs, the engine's draw rule)
    rng = np.random.default_rng(SEED)
    logits = rng.normal(0, 1.5, (64, 16))
    pb = np.exp(logits); pb /= pb.sum(1, keepdims=True)
    analytic = pb @ model.frac
    nd = 200000
    mc = []
    for row in pb:
        u = rng.random((nd, 3))
        bins = DR.sample_cat(np.repeat(row[None], nd, 0), u[:, 1])
        lo, hi = model.lo[bins], model.hi[bins]
        ms = np.where(bins == 15, 60000, np.round(1000 * np.exp(lo + u[:, 2] * (hi - lo))))
        mc.append(float((ms <= S_FAST * 1000).mean()))
    gap = float(np.abs(np.asarray(mc) - analytic).max())
    rep["K2_cdf_vs_sampler"] = {"rows": 64, "draws_per_row": nd, "max_abs": round(gap, 5), "bar": 0.004}
    ok["K2"] = gap < 0.004

    users = args.users
    cache = {}
    U0 = users[0]
    D0 = int(user_entry(U0)["split_day"])
    # K3 the feedback path (20 days): commit-row durations, and the -nofb twin draws as its feedback lane
    lanes3 = ["tf", "ihr-0.90#0", "ihr-0.90-nofb#0", "flu-0.80#0", "flu-0.80-nofb#0"]
    eng, _ = make_engine(model, U0, D0, args.short, lanes3, trace=True, cache=cache)
    model._commit_log, model._nofb_rows = [], 0
    eng.run()
    logs, model._commit_log = model._commit_log, None
    med = model._med
    nb_bad = fb_bad = n_nofb = n_fb = 0
    traces = {b: list(eng.traces[b]) for b in range(eng.B)}
    pos = {b: 0 for b in range(eng.B)}
    for A_, kind_, col8, g_ in logs:
        for j, (b, k_) in enumerate(zip(A_.tolist(), kind_.tolist())):
            tr_ = traces[b][pos[b]]; pos[b] += 1
            assert tr_[3] == g_[j], "trace / commit order"
            want = (med[g_[j]] if (eng.nofb[b] and k_ == 2) else
                    np.float32(DP.scale_duration(np.array([float(tr_[4])]))[0]))
            bad = abs(float(col8[j]) - float(want)) > 1e-6
            if eng.nofb[b] and k_ == 2:
                n_nofb += 1; nb_bad += bad
            else:
                n_fb += 1; fb_bad += bad
    twin = {}
    for a_, b_ in ((1, 2), (3, 4)):
        ta, tb = traces[a_], traces[b_]
        first_div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x[:5] != y[:5]), None)
        first_later = next((i for i, x in enumerate(ta) if x[2] == 2), None)
        twin[f"{eng.cfg[a_]['name']} vs {eng.cfg[b_]['name']}"] = {
            "reviews": [len(ta), len(tb)], "first_divergence": first_div, "first_later_day_review": first_later,
            "ok": first_div is None or (first_later is not None and first_div > first_later)}
    rep["K3_feedback_path"] = {"user": U0, "days": args.short, "nofb_later_rows": n_nofb, "nofb_wrong": nb_bad,
                               "other_rows": n_fb, "other_wrong": fb_bad, "twins": twin,
                               "medians_ms": eng.med_ms, "nofb_rows_counted": int(model._nofb_rows)}
    ok["K3"] = nb_bad == 0 and fb_bad == 0 and n_nofb > 0 and all(t["ok"] for t in twin.values())

    # K4 + K6 + K7: the full lane set with the budget (20 days): budget rules, determinism, batched == alone
    full = LANES.split(",")
    r1 = make_engine(model, U0, D0, args.short, full, trace=True, cache=cache)[0]
    res1 = r1.run()
    r2 = make_engine(model, U0, D0, args.short, full, cache=cache)[0].run()
    a1, a2 = _strip(res1), _strip(r2)
    same = json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    rep["K6_determinism"] = {"identical": same,
                             "differing_keys": sorted(k for k in a1 if json.dumps(a1[k]) != json.dumps(a2.get(k)))}
    ok["K6"] = same
    real_new = []
    for d in range(args.short):
        real_new += r1.new_by_day.get(d, [])
    prefix_ok, viol = True, 0
    for b, s in enumerate(r1.sess):
        if isinstance(s, BudgetSession):
            viol += s.violations
            prefix_ok &= s.introduced == real_new[:len(s.introduced)]
    rep["K4_budget"] = {"budget_s_per_day": round(r1.budget, 1), "violations": viol,
                        "introduced_is_real_order_prefix": bool(prefix_ok),
                        "hit_days": res1["budget_hit_days"], "max_day_over_budget_s": res1["day_seconds_max_over_budget"],
                        "deferred_new_at_end": res1["deferred_new_at_end"]}
    ok["K4"] = viol == 0 and prefix_ok
    alone = {}
    for nm in ("flu-0.80#1", "ihr-0.98-nofb#0", "flu-0.80-fb95#0"):
        b = full.index(nm)
        ra = make_engine(model, U0, D0, args.short, ["tf", nm], trace=True, cache=cache)[0]
        resa = ra.run()
        ta, tb = ra.traces[1], r1.traces[b]
        same_tr = len(ta) == len(tb) and all(x[:6] == y[:6] and abs(x[6] - y[6]) < 1e-4 for x, y in zip(ta, tb))
        acc_gap = max(float(np.max(np.abs(np.asarray(resa["acc"][k])[1] - np.asarray(res1["acc"][k])[b]) /
                                   np.maximum(1.0, np.abs(np.asarray(res1["acc"][k])[b])))) for k in res1["acc"])
        h_gap = max(abs(resa[f"h_{k}"][1] - res1[f"h_{k}"][b]) / max(1.0, abs(res1[f"h_{k}"][b]))
                    for k in ("memorised", "fluent"))
        alone[nm] = {"reviews": len(tb), "traces_identical": bool(same_tr), "acc_rel_gap": acc_gap, "horizon_rel_gap": h_gap,
                     "ok": bool(same_tr and acc_gap < 1e-4 and h_gap < 1e-4)}
    rep["K7_batched_vs_alone"] = alone
    ok["K7"] = all(v["ok"] for v in alone.values())
    rep["K8_speed_full_lanes"] = {"user": U0, "days": args.short, "lanes": len(full), "lane_reviews": res1["lane_reviews"],
                                  "wall_s": res1["wall_seconds"], "lane_reviews_per_s": res1["lane_reviews_per_second"],
                                  "candidate_rows": res1["candidate_rows"], "window_reviews": res1["window_reviews"],
                                  "cards": res1["cards"]}

    # K5 policies hit their targets (no budget, 60 days, days 30-59): instant-head desired retention within 0.02 of its
    # own aim, rising with the target; fluency lanes reported (realised passed-and-within-S share vs their aim)
    lanes5 = ["tf", "ihr-0.90#0", "ihr-0.95#0", "ihr-0.98#0", "flu-0.70#0", "flu-0.80#0"]
    k5 = {}
    for u in users:
        Du = int(user_entry(u)["split_day"])
        eng5 = make_engine(model, u, Du, args.long, lanes5, budget=float("inf"), cache=cache)[0]
        r5 = eng5.run()
        nb = len(r5["acc"]["load"][0])
        lo_b = 30 // E.BUCKET
        k5[str(u)] = {nm: _lane_ret(r5, k, lo_b, nb) for k, nm in enumerate(lanes5)}
        for k, nm in enumerate(lanes5):
            if nm.startswith("flu"):
                k5[str(u)][nm].update({"floor_share": round(r5["flu_floor_decisions"][k] / max(r5["flu_decisions"][k], 1), 4),
                                       "aim_fluent": round(r5["flu_aim_fluent_sum"][k] / max(r5["flu_decisions"][k], 1), 4)})
        k5[str(u)]["lane_reviews_per_s"] = r5["lane_reviews_per_second"]
        cache.pop(u, None)
    pool = {}
    for nm in lanes5[1:4]:
        rs = [k5[str(u)][nm] for u in users]
        n_ = sum(r["later_reviews"] for r in rs)
        pool[nm] = {"retention": round(sum(r["retention"] * r["later_reviews"] for r in rs) / n_, 4),
                    "aim": round(sum(r["aim"] * r["later_reviews"] for r in rs) / n_, 4)}
    rets = [pool[nm]["retention"] for nm in lanes5[1:4]]
    k5_ok = all(abs(pool[nm]["retention"] - pool[nm]["aim"]) <= 0.02 for nm in pool) and rets == sorted(rets)
    rep["K5_targets"] = {"users": users, "days": args.long, "read_days": "30-59", "pooled": pool, "per_user": k5,
                         "bar": "|realised - aim| <= 0.02 pooled per instant-head lane, realised rising with the target"}
    ok["K5"] = bool(k5_ok)
    rep["ok"] = ok
    rep["pass"] = all(ok.values())
    rep["finished"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    atomic_json(args.out, rep)
    print(json.dumps({"ok": ok, "pass": rep["pass"]}), flush=True)
    return 0 if rep["pass"] else 1


# ------------------------------------------------------------------ report
def boot_ci(x, rng, n=BOOT):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def per_user(res):
    """Seed-averaged metrics per policy for one user row."""
    lanes = res["lanes"]
    acc = {k: np.asarray(v) for k, v in res["acc"].items()}
    nb = acc["load"].shape[1]
    out = {}
    for k, nm in enumerate(lanes):
        pol = nm.split("#")[0]
        load = acc["load"][k].sum(0)
        lat = acc["later_old"][k].sum(0) + acc["later_new"][k].sum(0)
        fl = acc["fluent_later"][k].sum(0)
        fl_end = acc["fluent_later"][k, max(0, nb - 5):].sum(0)        # the last 5 buckets (days 300-364)
        ft = acc["first_touch"][k].sum(0)
        days = float(res["days"])
        m = {"minutes": load[3:].sum() / 60.0, "min_per_day": load[3:].sum() / 60.0 / days,
             "reviews_per_day": load[:3].sum() / days, "memorised": res["h_memorised"][k],
             "fluent": res["h_fluent"][k], "n_mem90": res["h_n_mem90"][k], "n_flu80": res["h_n_flu80"][k],
             "n_both": res["h_n_both"][k], "introduced_new": res["h_introduced_new"][k],
             "retention": lat[1] / max(lat[0], 1), "fast_pass_share": fl[1] / max(fl[0], 1),
             "pred_fast_pass": fl[2] / max(fl[0], 1), "fast_pass_share_end": fl_end[1] / max(fl_end[0], 1),
             "floor_share": res["flu_floor_decisions"][k] / max(res["flu_decisions"][k], 1),
             "hit_days": res["budget_hit_days"][k], "first_touch_pred": ft[2] / max(ft[0], 1)}
        out.setdefault(pol, []).append(m)
    return {p: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]} for p, ms in out.items()}


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted(Path(args.rows).glob("*.json"))]
    users = json.loads((HERE / "users.json").read_text())
    want = users["pilot"] if args.set == "pilot" else [c["user"] for c in users["users"]]
    rows = [r for r in rows if r["user"] in want]
    rng = np.random.default_rng(SEED)
    P = [per_user(r) for r in rows]
    pols = [p.split("#")[0] for p in LANES.split(",")]
    pols = list(dict.fromkeys(pols))
    L = [f"# Fluency vs desired retention (SM127): {args.set}, {len(rows)} of {len(want)} users", "",
         "Answer time = the whole time on the card, question shown to grade press, capped (\"answered within 8 s, "
         "reading the back included\"). Equal daily time: every policy lane has the user's real mean review seconds "
         "per day. Memorised / fluent = expected cards at day 365 (sums of P).", "",
         "| policy | min/day | reviews/day | memorised | fluent | P>=0.9 | fluent>=0.8 | both | new cards shown | "
         "retention | passed within 8 s (all / last 65 d) | floor share | budget-hit days |", "|" + "---|" * 13]
    summ = {}
    for p in pols:
        if not all(p in u for u in P):
            continue
        g = {k: float(np.mean([u[p][k] for u in P])) for k in P[0][p]}
        summ[p] = g
        L.append(f"| {p} | {g['min_per_day']:.1f} | {g['reviews_per_day']:.0f} | {g['memorised']:.0f} | "
                 f"{g['fluent']:.0f} | {g['n_mem90']:.0f} | {g['n_flu80']:.0f} | {g['n_both']:.0f} | "
                 f"{g['introduced_new']:.0f} | {g['retention']:.3f} | {g['fast_pass_share']:.3f} / "
                 f"{g['fast_pass_share_end']:.3f} | {g['floor_share']:.3f} | {g['hit_days']:.0f} |")
    out = {"set": args.set, "users": [r["user"] for r in rows], "summary": summ}

    def gap(a, b, key):
        x = [u[a][key] / max(u[b][key], 1e-9) - 1 for u in P]
        return {"mean": float(np.mean(x)), "ci": boot_ci(x, rng), "n": len(x)}
    dr = ["ihr-0.90", "ihr-0.95", "ihr-0.98"]
    best = max(dr, key=lambda p: summ[p]["fluent"]) if all(p in summ for p in dr) else None
    if best:
        L += ["", f"Best desired-retention arm by fluent cards (population): {best}.", "",
              "| comparison | fluent | memorised | minutes |", "|---|---|---|---|"]
        comps = {}
        for a, b in [("flu-0.80", best)] + [("flu-0.80", p) for p in dr if p != best] + \
                    [("flu-0.70", best), ("flu-0.80-fb95", best), ("ihr-0.98", "ihr-0.90"),
                     ("flu-0.80-nofb", "ihr-0.98-nofb"), ("flu-0.80-nofb", "ihr-0.90-nofb"),
                     ("ihr-0.98-nofb", "ihr-0.90-nofb"), ("flu-0.80", "flu-0.80-nofb"), ("ihr-0.98", "ihr-0.98-nofb")]:
            if a in summ and b in summ:
                c = {k: gap(a, b, k) for k in ("fluent", "memorised", "minutes")}
                comps[f"{a} vs {b}"] = c
                L.append(f"| {a} vs {b} | " + " | ".join(
                    f"{100 * c[k]['mean']:+.2f}% [{100 * c[k]['ci'][0]:+.2f}, {100 * c[k]['ci'][1]:+.2f}]"
                    for k in ("fluent", "memorised", "minutes")) + " |")
        out["comparisons"] = comps
        G = comps[f"flu-0.80 vs {best}"]
        f_, m_ = G["fluent"], G["memorised"]
        if f_["ci"][1] < BAR:
            verdict = "a high desired retention is enough (the fluency scheduler adds less than 2% fluent cards)"
        elif f_["mean"] >= BAR and f_["ci"][0] > 0:
            verdict = ("a fluency scheduler is needed" if m_["mean"] >= -BAR else
                       "a trade-off: the fluency scheduler gives more fluent cards but more than 2% fewer memorised")
        else:
            verdict = "inconclusive"
        # the same rule without answer-time feedback
        nof = [p for p in ("ihr-0.90-nofb", "ihr-0.98-nofb") if p in summ]
        v2 = None
        if nof and "flu-0.80-nofb" in summ:
            bn = max(nof, key=lambda p: summ[p]["fluent"])
            x = [u["flu-0.80-nofb"]["fluent"] / max(u[bn]["fluent"], 1e-9) - 1 for u in P]
            ci = boot_ci(x, rng)
            v2 = {"best": bn, "mean": float(np.mean(x)), "ci": ci,
                  "verdict": ("enough" if ci[1] < BAR else "needed" if np.mean(x) >= BAR and ci[0] > 0 else "inconclusive")}
        out.update({"best_dr": best, "verdict": verdict, "nofb": v2})
        L += ["", f"**Verdict (decision rule of design.md 7, feedback on): {verdict}.**"]
        if v2:
            L.append(f"Without answer-time feedback: flu-0.80-nofb vs {v2['best']} fluent {100 * v2['mean']:+.2f}% "
                     f"[{100 * v2['ci'][0]:+.2f}, {100 * v2['ci'][1]:+.2f}] -> {v2['verdict']}.")
    # drift check: replay lane fr-shared1 vs tf, first-touch predicted pass (state drift), all days
    if "fr-shared1" in summ and "tf" in summ:
        x = [u["fr-shared1"]["first_touch_pred"] - u["tf"]["first_touch_pred"] for u in P]
        out["drift_first_touch"] = {"mean": float(np.mean(x)), "ci": boot_ci(x, rng)}
        L += ["", f"Drift check (daily shared-stream re-anchoring (fr-shared1) minus teacher forcing, first-touch "
                  f"predicted pass, whole year): {100 * np.mean(x):+.2f}% [{100 * out['drift_first_touch']['ci'][0]:+.2f}, "
                  f"{100 * out['drift_first_touch']['ci'][1]:+.2f}]."]
    lr = [r["lane_reviews"] for r in rows]; ws = [r["wall_seconds"] + r.get("warmup_seconds", 0) for r in rows]
    out["cost"] = {"lane_reviews": int(sum(lr)), "process_hours": round(sum(ws) / 3600, 2),
                   "lane_reviews_per_s": round(sum(lr) / max(sum(r["wall_seconds"] for r in rows), 1e-9), 1)}
    L += ["", f"Cost: {sum(lr):,} lane reviews, {out['cost']['process_hours']} CPU process-hours "
              f"({out['cost']['lane_reviews_per_s']} lane reviews/s per process)."]
    Path(args.md).write_text("\n".join(L) + "\n", encoding="utf-8")
    atomic_json(args.json, out)
    print("\n".join(L))


def main():
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("select"); s.add_argument("--users", type=int, default=150); s.add_argument("--pilot", type=int, default=12)
    c = sub.add_parser("check")
    c.add_argument("--users", type=lambda x: [int(v) for v in x.split(",")], required=True)
    c.add_argument("--c-user", type=int, default=722); c.add_argument("--c-rows", type=int, default=3000)
    c.add_argument("--short", type=int, default=20); c.add_argument("--long", type=int, default=60)
    c.add_argument("--out", default=str(HERE / "check-cpu.json"))
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True); r.add_argument("--split", type=int)
    r.add_argument("--days", type=int, default=HORIZON); r.add_argument("--lanes", default=LANES)
    r.add_argument("--threads", type=int, default=1); r.add_argument("--margin", type=float, default=60.0,
                                                                     help="minutes before a quiet window: no new user")
    r.add_argument("--min-free-gb", type=float, default=6.0); r.add_argument("--out", required=True)
    t = sub.add_parser("report"); t.add_argument("--rows", default=str(HERE / "rows"))
    t.add_argument("--set", default="pilot", choices=("pilot", "full"))
    t.add_argument("--md", default=str(HERE / "sm127-tables.md")); t.add_argument("--json", default=str(HERE / "sm127-result.json"))
    a = ap.parse_args()
    fn = {"prepare": prepare, "select": select, "check": check, "run": run, "report": report}[a.cmd]
    rc = fn(a) if a.cmd != "prepare" else (fn(), 0)[1]
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
