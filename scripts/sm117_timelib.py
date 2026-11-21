"""Time heads for the family (SM117): the answer-time code of the SM82 line, copied here so that it runs on the
show-time code of this branch (the SM82 line, branches sm82 / sm82b / sm82c, has no show-time inputs; this branch has no
answer-time code). Provenance (branch sm82c at 7f5f4150, read only, nothing changed unless marked SM117):

  scripts/expertium_options.py : TIME_BIN_EDGES_MS, TIME_CAP_MS, TIME_LOW_CAP_SHARE, time_user_cap, time_labels
                                 (SM117: `same_day` argument, see below)
  scripts/sm82_time.py         : label_db, test_reviews (SM117: `same_day`), LOG_EDGES, median_log_s, censored_ll,
                                 RunningMedian, baseline_points, GRID, residual_cdf, baseline_prob, user_sums
  scripts/sm82b_head.py        : atomic_json, sha256, first, time_loss_rows, lr_at, GRADE_FIELDS, grade_compare, _paired
  scripts/sm82c_head.py        : explicit_features, EXPLICIT_NAMES, LN8
  expertium/model/variant_model.py : TimeMLP (the head module, parameter names hidden.* / out.*)

SM117 change of the label rule. SM82 dropped every review on the same day as the card's previous review because, with
answer-to-answer (normal) interval inputs, the instant row's interval holds that review's own answer time. On show-time
inputs (scripts/sm31_inputs.py e2s-query, what the family models are trained and scored on and what the fork feeds
them) the instant row's interval ends when the card is shown, so the duration is no longer in the inputs (CPU proof C4a
of scripts/sm117_head.py checks it). `same_day=True` (the SM117 rule) keeps those reviews: a review has a time label iff
it has a recorded time (duration > 0) and is not the card's first review (elapsed_days >= 0; first reviews have no
instant row). `same_day=False` is SM82's rule, bit for bit.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

BINS = 16
TIME_BIN_EDGES_MS = [500.0 * 120.0 ** (k / 14) for k in range(15)]
BIN_EDGES_MS = TIME_BIN_EDGES_MS
TIME_CAP_MS = 60000
CAP_MS = TIME_CAP_MS
TIME_LOW_CAP_SHARE = 0.005
LN8 = math.log(8.0)
LOG_EDGES = np.log(np.asarray(BIN_EDGES_MS) / 1000.0)   # ln seconds


# ------------------------------------------------------------------ labels (expertium_options.py)
def time_user_cap(duration):
    """The user's own cap in ms: TIME_CAP_MS, unless the user's max time is lower and holds >= 0.5% (and >= 20) of
    the timed reviews (Anki's per-preset "maximum answer seconds")."""
    timed = np.asarray(duration)[np.asarray(duration) > 0]
    if timed.size == 0:
        return TIME_CAP_MS
    top = int(timed.max())
    if top < TIME_CAP_MS and (timed == top).mean() >= TIME_LOW_CAP_SHARE and (timed == top).sum() >= 20:
        return top
    return TIME_CAP_MS


def time_labels(duration, elapsed_days=None, same_day=True):
    """(lower bin, censored, cap) per review of a user's history (revlog order = review_th - 1). Lower bin -1 = no
    label. SM82 (same_day False): duration 0, a first review or a review on the same day as the card's previous one.
    SM117 (same_day True, default): duration 0 or a first review (elapsed_days < 0). Censored = the time sits at a
    lower user cap: the label is "this bin or a later one"."""
    duration = np.asarray(duration, dtype=np.int64)
    cap = time_user_cap(duration)
    lower = np.searchsorted(np.asarray(TIME_BIN_EDGES_MS), duration.astype(np.float64), side="right").astype(np.int64)
    censored = ((duration >= cap) & (cap < TIME_CAP_MS)).astype(np.int64)
    drop = duration <= 0
    if elapsed_days is not None:
        drop |= np.asarray(elapsed_days) < (0 if same_day else 1)
    lower[drop] = -1
    censored[drop] = 0
    return lower, censored, cap


# ------------------------------------------------------------------ small helpers (sm82b_head.py)
def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first(items):
    return items[0]


def time_loss_rows(logits, grade, lower, cens):
    """Per-row censored cross-entropy of the true grade's 16-bin distribution."""
    import torch
    per = logits.float().view(-1, 4, BINS)[torch.arange(len(grade), device=logits.device), grade]
    log_p = torch.log_softmax(per, dim=-1)
    log_tail = torch.logcumsumexp(log_p.flip([-1]), dim=-1).flip([-1])
    index = lower.unsqueeze(-1)
    return -torch.where(cens.unsqueeze(-1) > 0, log_tail.gather(-1, index), log_p.gather(-1, index)).squeeze(-1)


def lr_at(step, total, peak, warmup_share):
    warm = max(1, round(warmup_share * total))
    if step < warm:
        return peak * (step + 1) / warm
    if step >= total:
        return 0.0
    return peak * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def make_time_mlp(inputs, hidden, outputs):
    """expertium/model/variant_model.py TimeMLP (sm82c): Linear + ReLU, zero-initialised output Linear."""
    import torch

    class TimeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(inputs, hidden)
            self.out = torch.nn.Linear(hidden, outputs)
            torch.nn.init.zeros_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    return TimeMLP()


# ------------------------------------------------------------------ explicit causal features (sm82c_head.py)
EXPLICIT_DIM = 26
EXPLICIT_NAMES = (["card_last_ln_t_1", "card_last_ln_t_2", "card_last_ln_t_3", "card_none_1", "card_none_2", "card_none_3",
                   "card_reviews_ln1p", "card_last_grade_1", "card_last_grade_2", "card_last_grade_3", "card_last_grade_4"]
                  + [f"user_median_ln_t_g{g}" for g in range(1, 5)] + [f"user_count_ln1p_g{g}" for g in range(1, 5)]
                  + ["user_median_ln_t_all", "user_prev_ln_t", "user_prev_none", "user_ema_ln_t", "elapsed_days_ln1p",
                     "new_card", "user_index_ln1p"])
assert len(EXPLICIT_NAMES) == EXPLICIT_DIM


class RunningMedian:
    """Exact running median (two heaps) of a stream; median() of the values pushed so far (None if empty)."""

    def __init__(self):
        self.low, self.high = [], []   # max-heap (negated), min-heap

    def push(self, x):
        import heapq
        if not self.low or x <= -self.low[0]:
            heapq.heappush(self.low, -x)
        else:
            heapq.heappush(self.high, x)
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if not self.low:
            return None
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2


def explicit_features(card, rating, duration, elapsed):
    """[n, 26] float32: row i = the features for predicting review i (review_th i + 1) from reviews 0..i-1 and review
    i's card id and elapsed days only."""
    card = np.asarray(card, np.int64); rating = np.asarray(rating, np.int64)
    duration = np.asarray(duration, np.int64); elapsed = np.asarray(elapsed, np.int64)
    n = len(duration)
    out = np.zeros((n, EXPLICIT_DIM), np.float64)
    logd = np.log(np.maximum(duration, 1) / 1000.0)
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    counts = [0, 0, 0, 0, 0]
    cards = {}
    prev_user, ema = None, LN8
    for i in range(n):
        c = int(card[i])
        st = cards.get(c)
        if st is None:
            st = [[], 0, 0]
            cards[c] = st
        last3, cnt, last_grade = st
        row = out[i]
        for k in range(3):
            if k < len(last3):
                row[k] = last3[k] - LN8
            else:
                row[3 + k] = 1.0
        row[6] = math.log1p(cnt) / 5.0
        if 1 <= last_grade <= 4:
            row[6 + last_grade] = 1.0
        em = every.median()
        fb = em if em is not None else LN8
        for g in (1, 2, 3, 4):
            m = per[g].median()
            row[10 + g] = (m if m is not None else fb) - LN8
            row[14 + g] = math.log1p(counts[g]) / 10.0
        row[19] = fb - LN8
        if prev_user is None:
            row[21] = 1.0
        else:
            row[20] = prev_user - LN8
        row[22] = ema - LN8
        e = int(elapsed[i])
        row[23] = math.log1p(max(e, 0)) / 3.0
        row[24] = 1.0 if e < 0 else 0.0
        row[25] = math.log1p(i) / 10.0
        r = int(rating[i])
        st[1] = cnt + 1
        st[2] = r
        if duration[i] > 0:
            x = float(logd[i])
            per.get(r, every).push(x)
            every.push(x)
            if r in per:
                counts[r] += 1
            last3.insert(0, x)
            del last3[3:]
            prev_user = x
            ema = 0.9 * ema + 0.1 * x
    return out.astype(np.float32)


def revlog(data, user, cap=None):
    import pandas as pd
    rev = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration", "elapsed_days"])
    return rev if cap is None else rev.iloc[:cap]


def explicit_table(rev):
    return explicit_features(rev["card_id"].to_numpy(), rev["rating"].to_numpy(), rev["duration"].to_numpy(),
                             rev["elapsed_days"].to_numpy())


# ------------------------------------------------------------------ test reviews, baselines, metrics (sm82_time.py)
def label_db():
    import lmdb
    from scripts import compress_common as cc
    return lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)


def test_reviews(data, user, txn, same_day=True):
    """The user's time-scored test reviews: the benchmark test reviews of the label DB (the grade scorer's "all") that
    carry a time label. Returns (review_th sorted, frame columns of the whole history, lower, censored, cap)."""
    import pandas as pd
    from scripts.compress_metrics import read_labels
    labels = read_labels(txn, user)
    frame = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration", "elapsed_days"])
    lower, cens, cap = time_labels(frame["duration"].to_numpy(), frame["elapsed_days"].to_numpy(), same_day=same_day)
    if labels is None:
        return None, frame, lower, cens, cap
    ths = np.unique(np.asarray(labels[0], dtype=np.int64))
    ths = ths[(ths >= 1) & (ths <= len(frame))]
    ths = ths[lower[ths - 1] >= 0]
    return ths, frame, lower, cens, cap


def median_log_s(prob):
    """Median of a [N, 16] bin distribution in ln seconds: linear in ln time inside the crossing bin (bin 0 from
    ln 0.25 s, bin 15 = its lower edge, 60 s)."""
    lo = np.r_[math.log(0.25), LOG_EDGES]
    hi = np.r_[LOG_EDGES, LOG_EDGES[-1]]
    cdf = np.cumsum(prob, axis=1)
    b = np.minimum((cdf < 0.5).sum(axis=1), BINS - 1)
    before = np.where(b > 0, np.take_along_axis(cdf, np.maximum(b - 1, 0)[:, None], 1)[:, 0], 0.0)
    pb = np.take_along_axis(prob, b[:, None], 1)[:, 0]
    frac = np.clip((0.5 - before) / np.maximum(pb, 1e-12), 0, 1)
    return lo[b] + frac * (hi[b] - lo[b])


def censored_ll(prob, lower, cens):
    """-ln P(label): the bin, or (censored) the bin and every later one; prob [N, 16], floored at 1e-7."""
    p = np.maximum(prob, 1e-7)
    p = p / p.sum(axis=1, keepdims=True)
    tail = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    picked = np.where(cens > 0, np.take_along_axis(tail, lower[:, None], 1)[:, 0], np.take_along_axis(p, lower[:, None], 1)[:, 0])
    return -np.log(picked)


def baseline_points(frame):
    """Per review (whole history, ln seconds): the card's last recorded time and the user's running median per grade
    of every earlier recorded time (falling back to the running median over all grades, then 8 s)."""
    import pandas as pd
    d = frame["duration"].to_numpy(np.int64)
    card = frame["card_id"].to_numpy(np.int64)
    rating = frame["rating"].to_numpy(np.int64)
    n = len(d)
    logd = np.log(np.maximum(d, 1) / 1000.0)
    last = np.full(n, np.nan)
    order = np.lexsort((np.arange(n), card))
    cs, ds = card[order], np.where(d[order] > 0, logd[order], np.nan)
    prev = pd.Series(ds).groupby(cs).shift(1)
    prev = prev.groupby(cs).ffill().to_numpy()
    last[order] = prev
    med = np.full(n, np.nan)
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    for i in range(n):
        m = per[rating[i]].median() if rating[i] in per else None
        if m is None:
            m = every.median()
        med[i] = m if m is not None else math.log(8.0)
        if d[i] > 0:
            per.get(rating[i], every).push(logd[i])
            every.push(logd[i])
    last = np.where(np.isnan(last), med, last)
    return last, med


GRID = np.arange(-10.0, 10.0 + 1e-9, 0.02)   # residual CDF grid, ln seconds


def residual_cdf(residuals):
    hist = np.histogram(np.clip(residuals, GRID[0], GRID[-1] - 1e-9), bins=GRID)[0].astype(np.float64) + 0.5
    return np.r_[0.0, np.cumsum(hist) / hist.sum()]


def baseline_prob(mu, cdf):
    """[N, 16] bin probabilities of a point prediction mu (ln s) with a residual CDF on GRID."""
    edges = LOG_EDGES[None, :] - mu[:, None]
    f = np.interp(edges, GRID, cdf)
    return np.diff(np.c_[np.zeros(len(mu)), f, np.ones(len(mu))], axis=1)


def user_sums(prob, t, mask=None):
    """Per-user sums: censored LogLoss and |ln error| of the median (uncensored), overall and by grade."""
    if mask is not None:
        prob = prob[mask]
        t = {k: v[mask] for k, v in t.items()}
    ll = censored_ll(prob, t["lower"].astype(np.int64), t["cens"])
    err = np.abs(median_log_s(prob) - t["logd"])
    unc = t["uncensored"]
    out = {"n": int(len(ll)), "ll": float(ll.sum()), "n_unc": int(unc.sum()), "ae": float(err[unc].sum())}
    for g in range(4):
        m = t["grade"] == g
        out[f"g{g + 1}"] = [int(m.sum()), float(ll[m].sum()), int((m & unc).sum()), float(err[m & unc].sum())]
    return out


def _paired(a_ll, b_ll, a_ae, b_ae, n, nu, idx):
    """Paired review-weighted differences a - b with user-bootstrap 95% CIs (idx [2000, users])."""
    row = {}
    for metric, x, y, w in (("time_logloss", a_ll, b_ll, n), ("mae_ln_s", a_ae, b_ae, nu)):
        delta = float((x - y).sum() / w.sum())
        boot = ((x - y)[idx]).sum(1) / w[idx].sum(1)
        row[metric] = {"delta": round(delta, 5), "ci": [round(float(np.percentile(boot, q)), 5) for q in (2.5, 97.5)],
                       "relative_pct": round(100 * delta / float(y.sum() / w.sum()), 2),
                       "users_better": int(((x - y) / np.maximum(w, 1) < 0).sum())}
    return row


# ------------------------------------------------------------------ grade check (sm82b_head.py)
GRADE_FIELDS = ("reviews", "curve_labels", "instant_labels", "curve_log_loss", "curve_raw_log_loss", "instant_log_loss",
                "instant_rating_cross_entropy", "benchmark", "benchmark_existing", "benchmark_deleted")


def grade_compare(rows_path, dump, rel_tag, new_tag, users, existing=None, existing_label=None, variant="w8+s8@e2s-query"):
    """Released vs frozen-after-training scorer rows (every grade field equal) and per-review dumps (arrays equal);
    with `existing`, the released model's recorded rows too. SM117: the label variant carries @e2s-query."""
    rows = [json.loads(x) for x in Path(rows_path).read_text().splitlines()]
    by = {(r["user_id"], r["label"]): r for r in rows if "label" in r}
    old = {}
    if existing is not None:
        old = {r["user_id"]: r for r in map(json.loads, Path(existing).read_text().splitlines()) if r.get("label") == existing_label}
    per, ok_all, ok_existing = [], True, True
    for u in users:
        a, b = by.get((u, f"{rel_tag}/{variant}")), by.get((u, f"{new_tag}/{variant}"))
        if a is None or b is None:
            per.append({"user_id": u, "missing": True}); ok_all = False
            continue
        fields = {f: a.get(f) == b.get(f) for f in GRADE_FIELDS}
        fa, fb = Path(dump) / f"{u}_{rel_tag}_{variant}.npz", Path(dump) / f"{u}_{new_tag}_{variant}.npz"
        if not fa.exists() or not fb.exists():
            per.append({"user_id": u, "dump_missing": [str(x) for x in (fa, fb) if not x.exists()]}); ok_all = False
            continue
        da = np.load(fa); db = np.load(fb)
        arrays = {k: bool(np.array_equal(da[k], db[k])) for k in da.files} if set(da.files) == set(db.files) else {"files": False}
        rec = {"user_id": u, "reviews": a.get("reviews"), "dump_rows": int(len(da["review_th"])), "fields_equal": all(fields.values()),
               "dump_arrays_equal": all(arrays.values()), "unequal_fields": [f for f, v in fields.items() if not v],
               "dump_files": [fa.name, fb.name]}
        ok_all &= rec["fields_equal"] and rec["dump_arrays_equal"]
        if existing is not None:
            e = old.get(u)
            ef = {f: (e or {}).get(f) == b.get(f) for f in GRADE_FIELDS}
            rec["existing_fields_equal"] = bool(e) and all(ef.values())
            rec["existing_unequal_fields"] = [f for f, v in ef.items() if not v]
            ok_existing &= rec["existing_fields_equal"]
        per.append(rec)
    out = {"pass": bool(ok_all), "users": users, "fields": list(GRADE_FIELDS), "variant": variant, "per_user": per}
    if existing is not None:
        out["existing_equal"] = bool(ok_existing)
        out["existing"] = {"rows": str(existing), "label": existing_label}
    return out
