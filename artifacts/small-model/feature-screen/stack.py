"""Learning-steps stacking test (user 2026-10-17: can rep number / a mini learning-steps model replace review type?).

A gradient-boosted corrector on top of a model's own logit, fitted on ALL scored reviews (no review type, so it would be
deployable for any scheduler), 2-fold by user. Gain of B over A = information in scheduler-independent card-history
features the model does not already use; C adds review type as a ceiling (what the SM-2 label still carries beyond them).
  A  logit
  B  logit + rep number, lapses, passes since last Again, ever-graduated (any interval >= 1 day before), previous
     rating, previous step (log s), this step (log s), step ratio, same-day reviews, FSRS-6 R/S/D, user's running
     median first step, user pass rate as of the previous review
  C  B + review type of the predicted and the previous review
Curve features are all known at the card's previous review (card-level history) or are the query time itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
D = Path(".")
MODELS = ("fork", "st")
users = sorted(int(p.stem) for p in (D / "fx").glob("*.npz") if all((D / "dump" / f"{p.stem}_{m}_w8+s8.npz").exists() for m in MODELS))
cols = {}
P = {(m, h): [] for m in MODELS for h in ("ahead", "instant")}
Y, U, ST = [], [], []


def add(k, v):
    cols.setdefault(k, []).append(v.astype(float))


for u in users:
    z = np.load(D / "fx" / f"{u}.npz")
    dd = {m: np.load(D / "dump" / f"{u}_{m}_w8+s8.npz") for m in MODELS}
    i = dd["st"]["review_th"] - 1
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset", "rating", "elapsed_days", "elapsed_seconds"])
    g = rv.groupby("card_id", sort=False)
    es = rv["elapsed_seconds"].to_numpy().astype(float)
    prev_es = g["elapsed_seconds"].shift(1).to_numpy().astype(float)
    prev_rating = g["rating"].shift(1).to_numpy().astype(float)
    grad = g["elapsed_days"].transform(lambda s: (s.shift(1) >= 1).cummax()).fillna(False).to_numpy().astype(float)
    # user's learning-step proxy: running median of first steps (gap first->second review) of cards whose second
    # review happened before this row
    second = (g.cumcount() == 1).to_numpy()
    fs = pd.Series(np.where(second, np.log1p(np.clip(es, 0, None)), np.nan))
    first_step_med = fs.expanding().median().shift(1).to_numpy()
    for k in ("card_prior_reviews", "card_prior_lapses", "card_streak", "interval_ratio", "fsrs_R", "fsrs_logS", "fsrs_D",
              "card_reviews_today", "user_pass_ewma200_atprev", "user_pass_ewma5000_atprev", "review_type", "prev_review_type"):
        add(k, z[k][i])
    add("prev_rating", prev_rating[i]); add("log_prev_step", np.log1p(np.clip(prev_es, 0, None))[i])
    add("log_step", np.log1p(np.clip(es, 0, None))[i]); add("ever_graduated", grad[i]); add("user_first_step_med", first_step_med[i])
    for (m, h) in P:
        P[(m, h)].append(dd[m][f"p_{h}"])
    Y.append(dd["st"]["y"]); U.append(np.full(len(i), u)); ST.append(z["review_type"][i])
X = {k: np.concatenate(v) for k, v in cols.items()}
P = {k: np.clip(np.concatenate(v), 1e-6, 1 - 1e-6) for k, v in P.items()}
y = np.concatenate(Y); user = np.concatenate(U); state = np.concatenate(ST)
fold = np.array([hash((uu * 2654435761) % 1000003) for uu in user]) % 4
# v2: user-level features (pass-rate EWMAs, running first-step median) act as user ids with ~50 training users and
# overfit (v1: corrector worse than raw by 2-4%); card-level history only, 4 folds by user
CARD = [k for k in X if k not in ("review_type", "prev_review_type") and not k.startswith("user_")]
SETS = {"A": [], "B": CARD, "C": CARD + ["review_type", "prev_review_type"]}
print(f"{len(users)} users, {len(y):,} scored reviews", flush=True)


def ll(p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


uidx = np.unique(user, return_inverse=True)[1]; NU = uidx.max() + 1
BOOT = np.random.default_rng(1).integers(0, NU, size=(500, NU))


def ci(d, s):
    a, n = np.bincount(uidx[s], d[s], NU), np.bincount(uidx[s], minlength=NU)
    m = np.array([a[b].sum() / max(n[b].sum(), 1) for b in BOOT])
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


out = {}
for h in ("ahead", "instant"):
    for m in MODELS:
        lg = np.log(P[(m, h)] / (1 - P[(m, h)]))
        for name, feats in SETS.items():
            Z = np.column_stack([lg] + [X[k] for k in feats])
            p = np.empty(len(y))
            for k in range(4):
                tr, te = fold != k, fold == k
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                     l2_regularization=1.0, random_state=0).fit(Z[tr], y[tr])
                p[te] = clf.predict_proba(Z[te])[:, 1]
            out[(h, m, name)] = ll(np.clip(p, 1e-6, 1 - 1e-6))
            print(f"[{h}] {m} {name} done", flush=True)
    raw = {m: ll(P[(m, h)]) for m in MODELS}
    print(f"\n[{h}] log loss vs the raw fork in each slice (review-weighted); gain = A - B (nats), 95% CI by user")
    for sl, s in (("all", np.ones(len(y), bool)), ("learning", state == 1), ("review", state == 2), ("relearning", state == 3)):
        f = raw["fork"][s].sum()
        cells = [f"st raw {100 * (raw['st'][s].sum() / f - 1):+.2f}%"]
        for m in MODELS:
            for name in SETS:
                cells.append(f"{m}{name} {100 * (out[(h, m, name)][s].sum() / f - 1):+.2f}%")
        for m in MODELS:
            lo, hi = ci(out[(h, m, "A")] - out[(h, m, "B")], s)
            cells.append(f"{m} B gain CI [{lo:+.5f},{hi:+.5f}]")
        print(f"  {sl:10s} share {s.mean():5.1%} | " + " | ".join(cells))
