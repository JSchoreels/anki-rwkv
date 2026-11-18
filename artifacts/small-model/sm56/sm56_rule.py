#!/usr/bin/env python
"""SM56 launcher helper (CPU, seconds): per-epoch 464-set losses, the adaptive stopping rule, the final summary.

  record K LABEL   L_K (review-weighted benchmark LogLoss, w8+s8, curve and instant) of LABEL over the 458 fit users of
                   the 464 set from heldout-sm56.jsonl; appended once to epochs.jsonl; prints the decision line and, as
                   the last line, "continue" or "stop" (exit 2 if users are missing)
  selftest         synthetic sequences through the rule
  summary DLABEL   SM56 (decay) vs SM55 and SM36 on the 464 set and the 732 pool (723 fit users), paired user bootstrap
                   (2,000 resamples, seed 0), and vs the fork; writes sm56-summary.{md,json}

Rule (user 2026-10-22, plateau on 2 consecutive epochs): per head, gain_k = L_{k-1} - L_k and total_k = L_1 - L_k.
Stop after epoch k (k >= 5) when gain_k < 0.05 total_k AND gain_{k-1} < 0.05 total_{k-1} on BOTH heads; minimum 5
epochs, hard cap 14. The decay starts from epoch k (the last one trained).
User 2026-10-23 ("faisons au max"): while the file FULL-14 exists next to this script, the plateau stop is off and
the run trains all 14 epochs (the cap still applies). User 2026-10-23 11:30 ("ok" to stopping after epoch 10): while the
file CAP-EPOCHS exists (one integer), it lowers the cap to that epoch.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
sys.path.insert(0, str(WT))
SM = WT / "artifacts/small-model"
EPOCHS = HERE / "epochs.jsonl"
FULL = HERE / "FULL-14"   # user 2026-10-23: no plateau stop, train to the cap
CAP = HERE / "CAP-EPOCHS"  # user 2026-10-23: lower cap (10), so the decay starts at 02:00 after the break
OUT464 = SM / "heldout-sm56.jsonl"
USERS464 = HERE / "eval-users-464-fit.json"
MIN_EPOCHS, MAX_EPOCHS, FRACTION = 5, 14, 0.05
HEADS = ("curve", "instant")


def decide(L, k, full=False, cap=MAX_EPOCHS):
    """L: {epoch: {"curve": x, "instant": y}} for epochs 1..k. Returns (stop, reason, detail per head)."""
    detail = {}
    for head in HEADS:
        rows = {}
        for j in (k, k - 1):
            if j >= 2 and j in L and (j - 1) in L:
                gain, total = L[j - 1][head] - L[j][head], L[1][head] - L[j][head]
                rows[j] = {"gain": gain, "total": total, "plateau": gain < FRACTION * total}
        detail[head] = rows
    if k >= cap:
        return True, f"cap {cap} epochs" + ("" if cap == MAX_EPOCHS else " (CAP-EPOCHS, user 2026-10-23)"), detail
    if k < MIN_EPOCHS:
        return False, f"minimum {MIN_EPOCHS} epochs", detail
    if full:
        return False, "no plateau stop (FULL-14, user 2026-10-23: train to the cap)", detail
    flat = all(detail[h].get(j, {}).get("plateau", False) for h in HEADS for j in (k, k - 1))
    if flat:
        return True, f"plateau: gain < {FRACTION:g} x total on epochs {k - 1} and {k}, both heads", detail
    return False, f"no plateau on epochs {k - 1} and {k} for both heads", detail


def describe(k, L, stop, reason, detail):
    parts = []
    for head in HEADS:
        cells = []
        for j in (k, k - 1):
            r = detail[head].get(j)
            if r:
                ratio = r["gain"] / r["total"] if r["total"] else float("nan")
                cells.append(f"e{j} gain {r['gain']:+.5f} total {r['total']:+.5f} ({ratio:.1%} {'<' if r['plateau'] else '>='} 5%)")
        parts.append(f"{head} [" + "; ".join(cells) + "]" if cells else f"{head} [-]")
    return (f"epoch {k}: curve {L[k]['curve']:.5f} instant {L[k]['instant']:.5f} gain " + " ".join(parts)
            + f" -> {'stop' if stop else 'continue'} ({reason})")


def load_rows(files, label, users):
    from scripts import small_model_gate as g
    merged = {}
    for f in files:
        if Path(f).exists():
            merged.update(g.load(f, [label], users)[label])
    return merged


def read_epochs():
    out = {}
    if EPOCHS.exists():
        for line in EPOCHS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["epoch"]] = r
    return out


def record(k, label):
    users = set(json.loads(USERS464.read_text())["users"])
    per = load_rows([OUT464], label, users)
    if set(per) != users:
        print(f"[sm56-rule] {label}: {len(per)} of {len(users)} users scored")
        print("missing")
        return 2
    order = sorted(users)
    L_k = {}
    for head, name in (("ahead", "curve"), ("instant", "instant")):
        v = np.array([per[u][head][0] for u in order]); s = np.array([per[u][head][1] for u in order], dtype=float)
        L_k[name] = float((v * s).sum() / s.sum())
    epochs = read_epochs()
    if k not in epochs:
        with EPOCHS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": k, "label": label, "users": len(order), **L_k}) + "\n")
        epochs = read_epochs()
    L = {e: {h: r[h] for h in HEADS} for e, r in epochs.items()}
    missing = [e for e in range(1, k + 1) if e not in L]
    if missing:
        print(f"[sm56-rule] epochs {missing} have no recorded loss")
        print("missing")
        return 2
    # user 10-24: CAP-EPOCHS may raise the cap above MAX_EPOCHS (15; the experiment's epochs were raised to match)
    cap = int(CAP.read_text().split()[0]) if CAP.exists() else MAX_EPOCHS
    stop, reason, detail = decide(L, k, FULL.exists(), cap)
    print(describe(k, L, stop, reason, detail))
    print("stop" if stop else "continue")
    return 0


def selftest():
    def run(seq_c, seq_i):
        L = {}
        for k, (c, i) in enumerate(zip(seq_c, seq_i), 1):
            L[k] = {"curve": c, "instant": i}
            stop, reason, _ = decide(L, k)
            if stop:
                return k, reason
        return None, "never"
    geo = [0.31 + 0.02 * 0.5 ** k for k in range(14)]          # halving gains: gain/total 6.7% at e5, 3.2% at e6, 1.6% at e7
    linear = [0.32 - 0.001 * k for k in range(14)]              # constant gains: gain/total = 1/(k-1), >= 5% up to e21
    flat = [0.32, 0.31, 0.305, 0.3049, 0.30489, 0.304889, 0.304888]  # plateau from epoch 4 on
    cases = [
        ("halving gains on both heads: stop at 7 (e6 and e7 < 5%, e5 6.7%)", geo, geo, 7),
        ("constant gains on both heads: cap 14", linear, linear, 14),
        ("plateau on both heads from e4: stop at 5 (min 5, e4 and e5 flat)", flat, flat, 5),
        ("curve flat but instant still improving: continue to the cap", flat + [flat[-1]] * 7, linear, 14),
        ("single flat epoch then a gain: no stop", [0.32, 0.31, 0.305, 0.304, 0.3039, 0.300, 0.2999, 0.29989],
         [0.32, 0.31, 0.305, 0.304, 0.3039, 0.300, 0.2999, 0.29989], 8),
    ]
    ok = True
    for name, c, i, want in cases:
        got, reason = run(c, i)
        good = got == want
        ok &= good
        print(f"[sm56-rule selftest] {'PASS' if good else 'FAIL'} {name}: stop at {got} ({reason}), expected {want}")
    return 0 if ok else 1


def summary(dlabel):
    from scripts import small_model_gate as g
    from scripts.eval_pools import pool_users

    class Every:
        def __contains__(self, _):
            return True

    e = os.environ
    s42, s55, srcwt = Path(e["S42"]), Path(e["S55"]), Path(e["SRC_WT"])
    fork_files = [srcwt / "artifacts/small-model/heldout.jsonl", srcwt / "artifacts/small-model/heldout-pools.jsonl"]
    ours = [SM / "heldout-sm56-pools.jsonl"]
    refs = {"SM55": ([s55 / "heldout-sm55-pools.jsonl"], "s-w64kd-final-sc-lanes-dec12-1ep32k-muon/w8+s8"),
            "SM36": ([s42 / "heldout-sm36.jsonl", s42 / "heldout-sm36-pools.jsonl"], "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8")}
    fork = load_rows(fork_files, "fork/w8+s8", Every())
    mine = load_rows(ours, dlabel, Every())
    out, lines = {}, []
    for pool in (464, 732):
        for ref, (files, label) in refs.items():
            other = load_rows(files, label, Every())
            users = sorted(set(mine) & set(other) & set(fork) & set(pool_users(pool)))
            if not users:
                lines.append(f"{pool} vs {ref}: no common users"); continue
            res = {"users": len(users), "delta": f"SM56 - {ref}"}
            for head, hname in g.HEADS.items():
                vb = np.array([mine[u][head][0] for u in users]); va = np.array([other[u][head][0] for u in users])
                vf = np.array([fork[u][head][0] for u in users]); sizes = np.array([mine[u][head][1] for u in users], dtype=float)
                eb, ea, ef = g.rw(vb, sizes), g.rw(va, sizes), g.rw(vf, sizes)
                idx = np.random.default_rng(0).integers(0, len(sizes), size=(2000, len(sizes)))
                w = sizes[idx]
                d = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
                res[hname] = {"sm56": eb, "ref": ea, "fork": ef, "delta": eb - ea,
                              "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                              "sm56_pct_vs_fork": (eb / ef - 1) * 100, "ref_pct_vs_fork": (ea / ef - 1) * 100,
                              "users_sm56_better": int((vb < va).sum())}
            c, i = res["curve"], res["instant"]
            lines.append(f"{pool} vs {ref} ({len(users)} users): SM56 curve {c['sm56']:.5f} ({c['sm56_pct_vs_fork']:+.2f}% vs fork) "
                         f"instant {i['sm56']:.5f} ({i['sm56_pct_vs_fork']:+.2f}%); {ref} curve {c['ref']:.5f} ({c['ref_pct_vs_fork']:+.2f}%) "
                         f"instant {i['ref']:.5f} ({i['ref_pct_vs_fork']:+.2f}%); SM56 - {ref} curve {c['delta']:+.5f} "
                         f"[{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] instant {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}]; "
                         f"fork curve {c['fork']:.5f} instant {i['fork']:.5f}")
            out[f"{pool} vs {ref}"] = res
    epochs = read_epochs()
    ep_lines = [f"e{k}: curve {r['curve']:.5f} instant {r['instant']:.5f} ({r['users']} users)" for k, r in sorted(epochs.items())]
    train = {}
    exp = Path(e["EXP"])
    for line in (exp / "phases.jsonl").read_text().splitlines() if (exp / "phases.jsonl").exists() else []:
        r = json.loads(line)
        if r.get("status") == "complete":
            train[f"epoch {r['epoch']}"] = {k: r.get(k) for k in ("training_seconds", "wall_seconds", "reviews_per_second", "review_presentations", "updates")}
    result = Path(e["D"]) / "expertium/E89-F65e-F77a/result.json"
    if result.exists():
        r = json.loads(result.read_text())
        train["decay"] = {k: r.get(k) for k in ("training_users", "updates", "review_presentations", "training_seconds", "reviews_per_second")}
    tl = [f"{k}: {v.get('review_presentations') or 0:,} reviews, {(v.get('training_seconds') or 0) / 60:.0f} min, "
          f"{v.get('reviews_per_second') or 0:,.0f} reviews/s" for k, v in train.items()]
    (SM / "sm56-summary.json").write_text(json.dumps({"pairs": out, "epochs": epochs, "training": train}, indent=1) + "\n")
    (SM / "sm56-summary.md").write_text(
        f"# SM56: {dlabel} vs SM55 and SM36 (w8+s8, review-weighted, user bootstrap 2,000 seed 0, 95% CI; delta = SM56 - ref)\n\n"
        + "\n\n".join(lines) + "\n\n## Constant-LR epochs (464 set, 458 fit users)\n\n" + "\n".join(f"- {l}" for l in ep_lines)
        + "\n\n## Training\n\n" + "\n".join(f"- {l}" for l in tl) + "\n")
    print("[sm56-summary]\n" + "\n".join(lines) + "\n" + "\n".join(ep_lines) + "\n" + "\n".join(tl))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "record":
        raise SystemExit(record(int(sys.argv[2]), sys.argv[3]))
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "summary":
        raise SystemExit(summary(sys.argv[2]))
    raise SystemExit(f"unknown command {cmd}")
