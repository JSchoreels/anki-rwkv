#!/usr/bin/env python
"""The instant model for RWKV-Instant (SM93): decisive check of the instant-only small model (SM57) vs the release (SM63).

Candidate  the instant-only small model (SM57), stage 1 instant-only arm: the leak-free small model (SM72s) epoch 25 +
           epoch 26 + the release's one-epoch 32k Muon cool-down, "curve_loss_weight": 0.0. Label sm57-inst-dec1ep32k/w8+s8.
Release    the released small model (cool-down from epoch 25, SM63), label s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8.

Pool: the 1,500 validation users of the 8k split (scripts/eval_pools.py --pool 1500), held out from both models (both
are final-tier small models trained on the 8,000 training users). Not the 2,225 set: its 1,493 extra users are training
users of both models (artifacts/eval/2225-labels/DESIGN.md), so it would mix in-sample users into the read-out.
Rows: the earlier 732-set rows (the release: wt/sm72s small-lf-neighbour.jsonl; the candidate: wt/sm57 sm57-rows.jsonl)
plus this check's rows (gpu/sm93-rows.jsonl) for the other 768 users; 6 users of the 732 set are re-scored here and must
match the earlier rows (max |LogLoss difference| <= 1e-5 per head) before the rows are merged.
Unit: the big model (1 leak-free epoch) is trained on the ids >= 5000 of this pool, so the denominator is the release's
own LogLoss on the pool (% of the release's LogLoss); on the 722-user rule pool the release is +0.28% vs the big model,
so 1% of the release's LogLoss = 1.003 points of the big model's: applying the 0.05 minimum to it is slightly stricter.
Paired user bootstrap 2,000 resamples, seed 0, 95% CI, review-weighted (sm92_rule.compare with the release as ref and
denominator). Leak-free scoring (--split-existing --deleted-cards neighbour), normal inputs, w8+s8.

Bar (the user's ship bar, fixed 2026-11-07 before any 1,500-pool row): instant head, candidate - release, 95% CI upper < 0
AND gain >= 0.05 (% of the release's LogLoss). The curve head of the candidate is untrained (reported, never served).

  python artifacts/small-model/sm93/sm93_rule.py decide   -> sm93-decision.{json,md}; exit 2 if rows are missing
"""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _rows_load  # noqa: E402  (no user filter)
_spec = importlib.util.spec_from_file_location("sm92_rule", ROOT / "artifacts/small-model/sm92/sm92_rule.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

CAND = "sm57-inst-dec1ep32k/w8+s8"
REL = R.RELEASE_LABEL
ROWS = HERE / "gpu" / "sm93-rows.jsonl"
OLD = {CAND: Path("C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/sm57-rows.jsonl"),
       REL: R.REL_NORMAL[0]}
POOL = ROOT / "artifacts/small-model/eval-users-1500.json"
USERS = HERE / "users-sm93-score.json"
SIZE = 0.05
TOL = 1e-5


def rows(label):
    old = _rows_load(OLD[label], label) if OLD[label].exists() else {}
    new = _rows_load(ROWS, label) if ROWS.exists() else {}
    return old, new


def decide(_args):
    pool = set(json.loads(POOL.read_text())["users"])
    spec = json.loads(USERS.read_text())
    res = {"pool": "the 1,500 validation users (eval-users-1500.json)", "check_users": spec["check_users"],
           "never_fit_skipped": spec["never_fit"]}
    merged, check = {}, {}
    for label in (CAND, REL):
        old, new = rows(label)
        diffs = []
        for u in spec["check_users"]:
            if u in old and u in new:
                diffs.append(max(abs(old[u]["all"][h][0] - new[u]["all"][h][0]) for h in R.HEADS))
                if any(old[u]["all"][h][1] != new[u]["all"][h][1] for h in R.HEADS):
                    diffs[-1] = float("inf")
        check[label] = {"users": len(diffs), "max_abs_logloss_diff": max(diffs) if diffs else None}
        m = {u: r for u, r in old.items() if u in pool}
        m.update({u: r for u, r in new.items() if u in pool and u not in m})
        merged[label] = m
    res["row_check"] = check
    if any(c["users"] < len(spec["check_users"]) for c in check.values()):
        res["verdict"], res["why"] = "pending", "check users not re-scored yet"
    elif any(c["max_abs_logloss_diff"] > TOL for c in check.values()):
        res["verdict"], res["why"] = "invalid", f"re-scored rows differ from the earlier rows by more than {TOL}: {check}"
    users = sorted(set(merged[CAND]) & set(merged[REL]))
    expected = sorted(pool - set(spec["never_fit"]))
    res["users"] = len(users)
    res["missing"] = {lb: len([u for u in expected if u not in merged[lb]]) for lb in (CAND, REL)}
    if "verdict" not in res and len(users) < len(expected) - 12:
        res["verdict"], res["why"] = "pending", f"{len(users)} of {len(expected)} users scored by both models"
    if users:
        pair = R.compare(merged[CAND], merged[REL], merged[REL], users)   # ref = denominator = the release
        res["candidate_minus_release"] = pair
        u732 = [u for u in users if u < 5000]
        u768 = [u for u in users if u >= 5000]
        res["by_part"] = {"ids < 5000 (the 732 set)": R.compare(merged[CAND], merged[REL], merged[REL], u732),
                          "ids >= 5000 (new here)": R.compare(merged[CAND], merged[REL], merged[REL], u768) if u768 else None}
        if "verdict" not in res:
            x = pair["instant"]
            if x["points_ci"][1] < 0 and x["points"] <= -SIZE:
                res["verdict"], res["why"] = "go", f"instant {R.fmt(x)}: CI below zero and at least {SIZE:.2f}"
            else:
                res["verdict"], res["why"] = "stop", f"instant {R.fmt(x)}: not (CI below zero and at least {SIZE:.2f})"
    users722, big = R.pools()
    old_c, _ = rows(CAND)
    old_r, _ = rows(REL)
    res["rule_pool_722_vs_big"] = R.compare({u: old_c[u] for u in users722}, {u: old_r[u] for u in users722}, big, users722)
    res["unit"] = "% of the release's LogLoss on the same users (1% of it = 1.003 points of the big model (1 leak-free epoch) on the 722 users)"
    (HERE / "sm93-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# The instant model for RWKV-Instant (SM93): the instant-only small model (SM57) vs the release (SM63), 1,500 validation users", "",
          f"Verdict: **{res['verdict']}** ({res.get('why', '')}).", "",
          "Leak-free scoring (deleted cards: neighbour), normal inputs, w8+s8, review-weighted LogLoss, paired user bootstrap 2,000, seed 0, "
          "95% CI. Unit: % of the release's LogLoss on the same users (the big model (1 leak-free epoch) trained on the ids >= 5000; on the 722 "
          "users 1% of the release's LogLoss = 1.003 points of the big model's). Bar: instant CI below 0 and gain >= 0.05.", "",
          f"Row check (6 users of the 732 set re-scored in this tree vs the earlier rows): {json.dumps(check)}", "",
          "| users | comparison | head | candidate | release | % [95% CI] | users better / worse |", "|---|---|---|---|---|---|---|"]
    parts = [("all", res.get("candidate_minus_release"))] + list(res.get("by_part", {}).items())
    for name, p in parts:
        if not p:
            continue
        for h in R.HEADS.values():
            x = p[h]
            tag = " (untrained curve head, never served)" if h == "curve" else ""
            md.append(f"| {p['users']} ({name}) | candidate - release{tag} | {h} | {x['new']:.5f} | {x['ref']:.5f} | "
                      f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}] | {x['users_better']} / {x['users_worse']} |")
    x = res["rule_pool_722_vs_big"]["instant"]
    md += ["", f"Reference (722-user rule pool, points of the big model (1 leak-free epoch)): instant {R.fmt(x)}, "
           f"{x['users_better']} / {x['users_worse']} users (the SM57 result)."]
    (HERE / "sm93-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    p = res.get("candidate_minus_release")
    print(f"[sm93-rule] {res['users']} users: {res['verdict'].upper()} ({res.get('why', '')})"
          + (f"; curve {R.fmt(p['curve'])}" if p else ""))
    return 0 if res["verdict"] in ("go", "stop") else 2


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("cmd", choices=["decide"])
    sys.exit(decide(ap.parse_args()))
