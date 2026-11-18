"""Training-free state check (SM116): the pre-registered rule (design.md, fixed before the score).

  pool       write users-score.json: the 722 users of the show-time retrain (SM92)'s rule pool (the 732 set's 723 fitted
             users that the big model (1 leak-free epoch) rows have), the users the projected model is scored on
  proof      the 100% projection == the unmodified model: work/proof-k100.jsonl vs work/proof-base.jsonl (CPU, same
             scorer) and vs SM92's GPU rows -> check-cpu-sm116.json
  decide     the 70% state (rows/sm116-k70.jsonl) - the unmodified model (SM92's GPU rows) on the 722 users, both heads,
             show-time inputs -> sm116-decision.json / .md (exit 2 = rows missing)
  self-test  the bootstrap / verdict logic on synthetic rows

Rule: cost = review-weighted benchmark LogLoss difference (projected - unmodified), in % of the big model (1 leak-free
epoch) show-time LogLoss on the same users (the 11-11 standard), paired user bootstrap 2,000 resamples, seed 0, 95% CI.
"starting point" iff BOTH heads have the upper CI bound below +0.05%; otherwise "drop".
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _load  # noqa: E402

S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")   # read only
U732 = S72 / "eval-users-732-fit.json"
BIG_NORMAL = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
              "big-leakfree-1ep/w8+s8")   # only to fix SM92's 722-user rule pool
BIG_E2S = (Path("C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl"),
           "big-leakfree-1ep/w8+s8@e2s-query")   # the % denominator (show-time)
SM92_GPU = (Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/gpu/sm92-inputs.jsonl"),
            "sm92-e26-dec1ep32k/w8+s8@e2s-query")   # the unmodified model
NEW = (HERE / "rows" / "sm116-k70.jsonl", "sm116k70/w8+s8@e2s-query")
HEADS = {"ahead": "curve", "instant": "instant"}
LIMIT = 0.05      # % of the big model (1 leak-free epoch) show-time LogLoss
PROOF_TOL = {"cpu": 1e-6, "gpu": 1e-5}   # max |LogLoss difference| per user and head


def load(source, label, users=None):
    return {u: r for u, r in _load(source, label, users).items() if users is None or u in users}


def rule_pool():
    users732 = set(json.loads(U732.read_text())["users"])
    big = load(*BIG_NORMAL, users732)
    return sorted(users732 & set(big))


SCORE_MAX_REVIEWS = 125_000   # CPU RAM: the stock CPU scorer peaked at 3.9 GB for 112k reviews, 6.9 GB for 158k, 14.8 GB for 531k


def cmd_pool(_):
    from scripts import compress_common as cc
    counts = cc.run_info()["review_counts"]
    rule = rule_pool()
    assert len(rule) == 722, len(rule)
    users = [u for u in rule if counts[str(u)] <= SCORE_MAX_REVIEWS]
    doc = {"note": "The training-free state check (SM116): the scored users = the show-time retrain (SM92)'s rule pool "
                   "(the 732 set's 723 fitted users that the big model (1 leak-free epoch) normal rows have, 722 users) "
                   f"with <= {SCORE_MAX_REVIEWS} reviews (the CPU scorer's memory under the 6 GB per-process limit). "
                   "Public benchmark user ids only.",
           "rule_pool": len(rule), "max_reviews": SCORE_MAX_REVIEWS, "users_count": len(users),
           "reviews": sum(counts[str(u)] for u in users), "reviews_rule_pool": sum(counts[str(u)] for u in rule),
           "users": users}
    (HERE / "users-score.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(len(users), doc["reviews"], doc["reviews_rule_pool"])


def compare(new, ref, big, users, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u]["all"][head][0] for u in users]); b = np.array([ref[u]["all"][head][0] for u in users])
        c = np.array([big[u]["all"][head][0] for u in users])
        w = np.array([ref[u]["all"][head][1] for u in users], float)
        a, b, c = (np.where(w > 0, x, 0.0) for x in (a, b, c))
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        boot_own = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "unmodified": round(float((b * w).sum() / w.sum()), 6),
                  "big_e2s": round(float((c * w).sum() / w.sum()), 6),
                  "delta": round(float(((a - b) * w).sum() / w.sum()), 7),
                  "pct": round(float(((a - b) * w).sum() / (c * w).sum() * 100), 4),
                  "pct_ci": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
                  "pct_of_own": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 4),
                  "pct_of_own_ci": [round(float(np.percentile(boot_own, q)), 4) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()), "users_worse": int((a > b).sum()), "reviews": int(w.sum())}
    return out


def verdict(pair):
    ok = {h: pair[h]["pct_ci"][1] < LIMIT for h in HEADS.values()}
    why = "; ".join(f"{h} {pair[h]['pct']:+.3f}% [{pair[h]['pct_ci'][0]:+.3f}, {pair[h]['pct_ci'][1]:+.3f}]"
                    for h in HEADS.values())
    if all(ok.values()):
        return "starting point", f"both heads have the upper 95% CI bound below +{LIMIT}% ({why})"
    bad = [h for h, v in ok.items() if not v]
    return "drop", f"upper 95% CI bound not below +{LIMIT}% on {' and '.join(bad)} ({why})"


def cmd_proof(_):
    base = HERE / "work" / "proof-base.jsonl"
    k100 = HERE / "work" / "proof-k100.jsonl"
    cpu = load(base, "sm92/w8+s8@e2s-query")
    proj = load(k100, "sm116k100/w8+s8@e2s-query")
    users = sorted(proj)
    gpu = load(*SM92_GPU, set(users))
    raw = {}
    for f, label in ((base, "sm92/w8+s8@e2s-query"), (k100, "sm116k100/w8+s8@e2s-query")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("label") == label:
                raw[(label, r["user_id"])] = (r["curve_log_loss"], r["instant_log_loss"])
    per_user, worst = [], {"cpu": 0.0, "gpu": 0.0, "cpu_full": 0.0}
    for u in users:
        d_cpu = max(abs(proj[u]["all"][h][0] - cpu[u]["all"][h][0]) for h in HEADS)
        d_gpu = max(abs(proj[u]["all"][h][0] - gpu[u]["all"][h][0]) for h in HEADS)
        a, b = raw[("sm116k100/w8+s8@e2s-query", u)], raw[("sm92/w8+s8@e2s-query", u)]
        d_full = max(abs(x - y) for x, y in zip(a, b))
        worst = {"cpu": max(worst["cpu"], d_cpu), "gpu": max(worst["gpu"], d_gpu), "cpu_full": max(worst["cpu_full"], d_full)}
        per_user.append({"user": u, "max_abs_benchmark_vs_cpu": d_cpu, "max_abs_benchmark_vs_gpu": d_gpu,
                         "max_abs_full_history_loss_vs_cpu": d_full})
    passed = (len(users) == len(cpu) == len(gpu) and len(users) > 0 and worst["cpu"] <= PROOF_TOL["cpu"]
              and worst["cpu_full"] <= PROOF_TOL["cpu"] and worst["gpu"] <= PROOF_TOL["gpu"])
    doc = {"check": "the 100% projection (every card / note stored state through mu + P P^T (x - mu), P = all directions) "
                    "reproduces the unmodified show-time retrain (SM92): CPU scorer, w8+s8, show-time inputs, leak-free",
           "users": users, "tolerance": PROOF_TOL, "worst": worst, "per_user": per_user, "passed": bool(passed)}
    (HERE / "check-cpu-sm116.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"proof {'PASS' if passed else 'FAIL'}: {len(users)} users, worst {worst}")
    return 0 if passed else 1


def cmd_decide(_):
    users = json.loads((HERE / "users-score.json").read_text())["users"]
    new = load(*NEW, set(users))
    missing = [u for u in users if u not in new]
    if missing:
        print(f"{len(missing)} of {len(users)} users not scored yet")
        return 2
    ref, big = load(*SM92_GPU, set(users)), load(*BIG_E2S, set(users))
    assert all(u in ref and u in big for u in users)
    pair = compare(new, ref, big, users)
    v, why = verdict(pair)
    directions = json.loads((HERE / "directions.json").read_text())
    doc = {"verdict": v, "why": why, "pair": pair, "limit_pct": LIMIT,
           "unit": f"% of the big model (1 leak-free epoch) show-time LogLoss, review-weighted, {len(users)} users",
           "rows": {"new": f"{NEW[0]} {NEW[1]}", "unmodified": f"{SM92_GPU[0]} {SM92_GPU[1]}",
                    "big_e2s": f"{BIG_E2S[0]} {BIG_E2S[1]}"},
           "variance_kept_at_70pct": {k: r["variance_kept_at_70pct"] for k, r in directions.items()}}
    (HERE / "sm116-decision.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = ["# Training-free state check (SM116): decision", "",
          f"**Verdict: {v}.** {why}.", "",
          "The show-time retrain (SM92) with its card and note state kept at 70% of the dimensions (per block PCA from 100 "
          "users outside the scored pool, no training) minus the unmodified show-time retrain (SM92); show-time inputs, "
          "w8+s8, leak-free scoring, " + f"{len(users)} users (the 722-user rule pool with <= {SCORE_MAX_REVIEWS:,} reviews); % of the big model (1 leak-free epoch) show-time LogLoss; paired user "
          "bootstrap 2,000, seed 0, 95% CI. Rule: starting point iff both upper bounds < +0.05%.", "",
          "| head | unmodified | 70% state | difference | % (CI) | % of own LogLoss (CI) | users better / worse |",
          "|---|---|---|---|---|---|---|"]
    for h in HEADS.values():
        x = pair[h]
        md.append(f"| {h} | {x['unmodified']:.6f} | {x['new']:.6f} | {x['delta']:+.7f} | {x['pct']:+.3f}% "
                  f"[{x['pct_ci'][0]:+.3f}, {x['pct_ci'][1]:+.3f}] | {x['pct_of_own']:+.3f}% "
                  f"[{x['pct_of_own_ci'][0]:+.3f}, {x['pct_of_own_ci'][1]:+.3f}] | {x['users_better']} / {x['users_worse']} |")
    (HERE / "sm116-decision.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def cmd_self_test(_):
    rng = np.random.default_rng(1)
    users = list(range(300))
    big = {u: {"all": {"ahead": (0.30, 1000), "instant": (0.28, 1000)}} for u in users}
    ref = {u: {"all": {"ahead": (0.30 + rng.normal(0, 0.01), 1000), "instant": (0.28 + rng.normal(0, 0.01), 1000)}} for u in users}
    fails = []

    def shifted(curve, instant, noise):
        return {u: {"all": {"ahead": (ref[u]["all"]["ahead"][0] + curve + rng.normal(0, noise), 1000),
                            "instant": (ref[u]["all"]["instant"][0] + instant + rng.normal(0, noise), 1000)}} for u in users}
    same = compare(ref, ref, big, users)
    if verdict(same)[0] != "starting point" or same["curve"]["pct_ci"] != [0.0, 0.0]:
        fails.append(f"identical rows: {verdict(same)}")
    cheap = compare(shifted(0.00003, 0.00003, 1e-6), ref, big, users)   # +0.01% / +0.011%, tight CI
    if verdict(cheap)[0] != "starting point":
        fails.append(f"cheap: {verdict(cheap)}")
    costly = compare(shifted(0.0003, 0.0, 1e-6), ref, big, users)       # +0.1% curve
    if verdict(costly)[0] != "drop" or "curve" not in verdict(costly)[1]:
        fails.append(f"costly curve: {verdict(costly)}")
    noisy = compare(shifted(0.0, 0.0, 0.003), ref, big, users)          # mean 0, CI wider than 0.05%
    if verdict(noisy)[0] != "drop":
        fails.append(f"noisy: {verdict(noisy)} {noisy['curve']['pct_ci']}")
    print("self-test " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit({"pool": cmd_pool, "proof": cmd_proof, "decide": cmd_decide, "self-test": cmd_self_test}[sys.argv[1]](sys.argv[2:]) or 0)
