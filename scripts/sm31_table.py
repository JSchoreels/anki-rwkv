#!/usr/bin/env python
"""Result tables for duration-leak scoring (SM31) and robustness to other apps (SM29) (CPU, seconds).

Rows: scripts/small_model_heldout.py --inputs V (labels LABEL@V) in --rows; the normal-input rows of both models are the
leak-free read-out files (read only; today's scorer reproduces them exactly, scheduling-time vs review-time accuracy
(SM81) check existing_rows_max_abs_diff = 0). Review-weighted benchmark LogLoss over all test reviews, curve and instant
heads (the answer-time head is not evaluated); paired user bootstrap 2,000 resamples, seed 0, 95% CI; deltas in points
= % of the big model (1 leak-free epoch) with normal inputs on the same users. Pools: 732 (rule) = the 732-set users
with a big-model row (722), 732+long = plus the 3 long users when every row has them.
  sm31_table.py --rows R.jsonl   writes artifacts/small-model/sm31/sm31-result.{md,json} and
                                 artifacts/small-model/sm29/sm29-result.{md,json}; prints one line each; exit 2 if rows
                                 are missing (the tables are still written for what is there, marked partial)
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _load  # noqa: E402

S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
SMALL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8"
BIG = "big-leakfree-1ep/w8+s8"
NORMAL = {SMALL: S72 / "gpu" / "small-lf-neighbour.jsonl",
          BIG: Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl")}
NAMES = {SMALL: "small model (released: the leak-free small model (SM72s), cool-down from epoch 25)",
         BIG: "big model (1 leak-free epoch)"}
SHORT = {SMALL: "small model", BIG: "big model (1 leak-free epoch)"}
HEADS = {"ahead": "curve", "instant": "instant"}
SM31 = {"e2s-query": "show-time intervals (deployment-realistic)",
        "e2s-all": "end-to-start on every row (the outside-research definition)"}
SM29 = {"passfail": "pass / fail ratings only", "notime": "no answer times", "passfail-notime": "pass / fail, no times"}


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users} if Path(source).exists() else {}


def arrays(rows, users, head):
    loss = np.array([rows[u]["all"][head][0] for u in users], float)
    size = np.array([rows[u]["all"][head][1] for u in users], float)
    return loss, size


def diff(a, b, base, users, head):
    """Review-weighted (a - b) in points of % of `base`, paired user bootstrap CI (2,000, seed 0)."""
    la, wa = arrays(a, users, head)
    lb, wb = arrays(b, users, head)
    lc, wc = arrays(base, users, head)
    if not (np.array_equal(wa, wb) and np.array_equal(wa, wc)):
        raise SystemExit(f"{head}: test review counts differ between rows on the same users")
    w = wc
    la, lb, lc = (np.where(w > 0, x, 0.0) for x in (la, lb, lc))
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    boot = ((la - lb)[idx] * w[idx]).sum(1) / (lc[idx] * w[idx]).sum(1) * 100
    return {"a": round(float((la * w).sum() / w.sum()), 6), "b": round(float((lb * w).sum() / w.sum()), 6),
            "delta": round(float(((la - lb) * w).sum() / w.sum()), 6),
            "points": round(float(((la - lb) * w).sum() / (lc * w).sum() * 100), 3),
            "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
            "users_worse": int((la > lb).sum()), "users_better": int((la < lb).sum()),
            "user_mean_delta": round(float(np.mean(la - lb)), 6), "reviews": int(w.sum())}


def pools(rows, need):
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    rule = sorted(u for u in users732 if u in rows[(BIG, "normal")])
    have = lambda u: all(u in rows[k] for k in need)  # noqa: E731
    out = {"732 (rule)": [u for u in rule if have(u)]}
    extra = sorted(u for u in long_users if have(u))
    if extra:
        out["732+long"] = sorted(out["732 (rule)"] + extra)
    missing = [u for u in rule if not have(u)]
    return out, missing, len(rule)


def fmt(x):
    return f"{x['points']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def build(rows, variants, title, intro, note):
    need = [(m, v) for m in (SMALL, BIG) for v in ("normal", *variants)]
    pool, missing, n_rule = pools(rows, need)
    res = {"partial": bool(missing), "missing_users": missing, "rule_pool_users": n_rule, "pools": {}}
    md = [f"# {title}", "", intro, ""]
    if missing:
        md += [f"**Partial:** {len(missing)} of {n_rule} rule-pool users are not scored under every variant yet "
               f"(first: {missing[:10]}).", ""]
    for name, users in pool.items():
        if not users:
            continue
        p = res["pools"][name] = {"users": len(users)}
        md += [f"## {name} pool, {len(users)} users", "",
               "| model | inputs | curve LogLoss | curve vs normal, points [95% CI] | instant LogLoss | instant vs normal, points [95% CI] |",
               "| --- | --- | --- | --- | --- | --- |"]
        base = rows[(BIG, "normal")]
        for m in (SMALL, BIG):
            p[m] = {}
            for v in ("normal", *variants):
                cells = []
                p[m][v] = {}
                for head, h in HEADS.items():
                    d = diff(rows[(m, v)], rows[(m, "normal")], base, users, head)
                    p[m][v][h] = d
                    cells += [f"{d['a']:.5f}", "(reference)" if v == "normal" else fmt(d)]
                label = "normal" if v == "normal" else f"{(SM31 | SM29)[v]} (`{v}`)"
                md.append(f"| {NAMES[m] if v == 'normal' else ''} | {label} | {' | '.join(cells)} |")
        md += ["", "Small model minus big model (1 leak-free epoch) under the same inputs, points [95% CI] (positive = the "
               "small model is worse):", "", "| inputs | curve gap | instant gap | curve gap change vs normal | instant gap change vs normal |",
               "| --- | --- | --- | --- | --- |"]
        p["gap"] = {}
        for v in ("normal", *variants):
            g = p["gap"][v] = {h: diff(rows[(SMALL, v)], rows[(BIG, v)], base, users, head) for head, h in HEADS.items()}
            if v == "normal":
                md.append(f"| normal | {fmt(g['curve'])} | {fmt(g['instant'])} | | |")
                continue
            ch = {}
            for head, h in HEADS.items():   # (small_V - big_V) - (small_N - big_N), paired
                a = {u: {"all": {head: (rows[(SMALL, v)][u]["all"][head][0] - rows[(BIG, v)][u]["all"][head][0]
                                       + rows[(BIG, "normal")][u]["all"][head][0], rows[(BIG, "normal")][u]["all"][head][1])}}
                     for u in users}
                ch[h] = diff(a, rows[(SMALL, "normal")], base, users, head)
            p["gap"][v]["change_vs_normal"] = ch
            md.append(f"| {(SM31 | SM29)[v]} (`{v}`) | {fmt(g['curve'])} | {fmt(g['instant'])} | {fmt(ch['curve'])} | "
                      f"{fmt(ch['instant'])} |")
        md.append("")
    md += note
    return res, md


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "small-model")
    a = ap.parse_args()
    users = set(json.loads(U732.read_text())["users"]) | set(json.loads(ULONG.read_text())["users"])
    rows = {}
    for m in (SMALL, BIG):
        rows[(m, "normal")] = load(NORMAL[m], m, users)
        for v in (*SM31, *SM29):
            rows[(m, v)] = load(a.rows, f"{m}@{v}", users)
    for k, r in rows.items():
        for u, rec in r.items():
            if "all" not in rec:
                raise SystemExit(f"{k} user {u}: row without benchmark metrics")
    common = ("Leak-free scoring (deleted cards: neighbour), w8+s8, the 732 set; review-weighted benchmark LogLoss over "
              "all test reviews, curve and instant heads (the answer-time head is not evaluated); deltas in points = % "
              "of the big model (1 leak-free epoch) with normal inputs on the same users, paired user bootstrap 2,000 "
              "resamples, seed 0, 95% CI. Every variant scores the same test reviews with the same labels; only inputs "
              "change. No retraining: the released models as they are.")
    rc = 0
    for key, variants, title, intro, note in (
        ("sm31", SM31, "Duration-leak scoring (SM31): normal vs deployment-realistic intervals",
         common + " Normal inputs: `elapsed_seconds` runs from the card's previous answer to this answer, so it holds "
         "the review's own answer time. Show-time intervals (`e2s-query`, deployment-realistic): everything asked before "
         "the answer uses the interval from the card's previous answer to the moment the card is shown, "
         "floor(max(elapsed_seconds - duration, 0)): the instant query row's interval, its sin / cos, the cumulative "
         "interval (earlier answer-to-answer intervals + the show-time one) and its sin / cos, and the curve query time; "
         "update rows (after the answer) keep the full interval. `e2s-all` = the outside-research definition, the same "
         "subtraction on every row (update rows and the cumulative interval too).",
         ["Inputs checked on CPU: artifacts/small-model/sm31/check-cpu-sm31-inputs.json (which columns move on which "
          "rows; normal bit-identical to the scorer) and cpu-proof/ (default scorer rows identical to the previous "
          "scorer)."]),
        ("sm29", SM29, "Robustness to other apps (SM29): pass / fail ratings and no answer times",
         common + " `passfail`: Hard and Easy reviews get the Good rating one-hot in the inputs (an app with two "
         "buttons); `notime`: every review's answer time set to missing (duration 0, as missing times are encoded); "
         "`passfail-notime`: both. Labels (pass / fail) are unchanged.",
         ["Inputs checked on CPU: artifacts/small-model/sm31/check-cpu-sm31-inputs.json."])):
        res, md = build(rows, variants, title, intro, note)
        res.update(rows=str(a.rows), normal_rows={k: str(v) for k, v in NORMAL.items()},
                   labels={"small": SMALL, "big": BIG}, unit="points of % of the big model (1 leak-free epoch), normal inputs")
        out = a.out_dir / key
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{key}-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        (out / f"{key}-result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        p = res["pools"].get("732 (rule)", {})
        bits = []
        for m in (SMALL, BIG):
            for v in variants:
                if m in p and v in p[m]:
                    bits.append(f"{SHORT[m]} {v}: curve {p[m][v]['curve']['points']:+.2f} instant "
                                f"{p[m][v]['instant']['points']:+.2f}")
        print(f"[sm31-table] {key}: {p.get('users', 0)} users{' (partial)' if res['partial'] else ''}; " + "; ".join(bits))
        rc = rc or (2 if res["partial"] else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
