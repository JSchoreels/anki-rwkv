#!/usr/bin/env python
"""Codebook state compression sweep (SM90): screen read-out + finalist pick, and the final decision (CPU, seconds).

Rows: scripts/small_model_heldout.py, leak-free (--deleted-cards neighbour --split-existing), w8 weights, model tag
`rel` = the released small model (the leak-free small model (SM72s), cool-down from epoch 25). LogLoss = review-weighted
benchmark LogLoss over the users scored in every compared label; curve = the ahead head. "points" = 100 x (LogLoss
difference) / LogLoss of the big model (1 leak-free epoch) on the same users. Paired user bootstrap: 2,000 resamples,
seed 0, 2.5 / 97.5 percentiles.

  screen   gpu/screen-sm90.jsonl on screen-users.json: every recipe vs rel/w8+s4c8n (the fork default state), w8+s8 vs
           the recorded released rows (reproduction), finalist pick (DESIGN.md "Finalist pick"), writes
           screen-summary.{md,json} and finalists.txt (one label per line, scoring order). Exit 3 if rows are missing.
  final    gpu/final-sm90.jsonl (+ the screen rows are NOT used) on the 732 set's 723 fit users: per finalist the
           pre-registered decision (DESIGN.md "Decision rule"), sizes and speed; writes sm90-summary.{md,json}.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
SCREEN_ROWS, FINAL_ROWS = G / "screen-sm90.jsonl", G / "final-sm90.jsonl"
USERS732 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
RELEASED = (Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl"),
            "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8")
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")
BASE, INT8 = "rel/w8+s4c8n", "rel/w8+s8"
HEADS = {"curve": "ahead", "instant": "instant"}
# ladders (DESIGN.md), largest budget first: coded card + int8 note, int4 card + coded note; SM46's both-coded recipe
CARD = ["vqC5902", "vqC4430", "vqC2958", "vqC1934", "vqC1470"]
NOTE = ["vqN3964", "vqN3020", "vqN1524"]
CONTINUITY = ["vq5902x3028"]
NEARLY_FREE_POINTS = 0.02            # final rule: extra cost vs int4 cards + int8 notes, both heads
BAR = {"curve": 0.5, "instant": 0.4}  # smaller-model bar: % vs the big model (1 leak-free epoch)
SCREEN_PASS = 0.04
PRACTICAL = {"codebook_mb_f16": 10.0, "encode_ms_per_review_est": 1.0}


def rows_of(path, label=None):
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "error" in r or "benchmark" not in r or (label is not None and r["label"] != label):
            continue
        out.setdefault(r["label"], {})[r["user_id"]] = r
    return out


def arr(rows, users, head):
    v = np.array([rows[u]["benchmark"][head]["LogLoss"] for u in users])
    w = np.array([rows[u]["benchmark"][head]["size"] for u in users], float)
    return v * w, w


def compare(a, b, big, users, seed=0, n=2000):
    """a vs b on users: LogLoss of each, delta, points (100 x delta / big), % vs big of a; bootstrap CI."""
    users = sorted(users)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(users), size=(n, len(users)))
    out = {"users": len(users)}
    for h, head in HEADS.items():
        (av, w), (bv, _), (gv, _) = arr(a, users, head), arr(b, users, head), arr(big, users, head)
        la, lb, lg = av.sum() / w.sum(), bv.sum() / w.sum(), gv.sum() / w.sum()
        W = w[idx].sum(1)
        A, B, Gb = av[idx].sum(1) / W, bv[idx].sum(1) / W, gv[idx].sum(1) / W
        pts = 100 * (A - B) / Gb
        tot = 100 * (A / Gb - 1)
        out[h] = {"logloss": la, "ref_logloss": lb, "big_logloss": lg, "delta": la - lb,
                  "delta_ci": [float(np.percentile(A - B, 2.5)), float(np.percentile(A - B, 97.5))],
                  "points": 100 * (la - lb) / lg, "points_ci": [float(np.percentile(pts, 2.5)), float(np.percentile(pts, 97.5))],
                  "pct_vs_big": 100 * (la / lg - 1), "pct_vs_big_ci": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))],
                  "reviews": int(w.sum())}
    return out


def speed(rows):
    rev = sum(r["reviews"] for r in rows.values())
    sec = sum(r.get("model_seconds", 0) for r in rows.values())
    return {"reviews": rev, "model_seconds": sec, "reviews_per_second": rev / sec if sec else None}


def costs():
    path = HERE / "cost-sm90.json"
    return {r["recipe"]: r for r in json.loads(path.read_text())["rows"]} if path.exists() else {}


def fmt(c, h):
    x = c[h]
    return (f"{x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] = {x['points']:+.3f} pts "
            f"[{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]; {x['pct_vs_big']:+.2f}% vs big")


def reproduction(rows_int8, users):
    rel = rows_of(*RELEASED).get(RELEASED[1], {})
    d = {h: max(abs(rows_int8[u]["benchmark"][head]["LogLoss"] - rel[u]["benchmark"][head]["LogLoss"]) for u in users)
         for h, head in HEADS.items()} if users else {}
    return {"users": len(users), "max_abs_delta": d, "passes_1e-5": bool(d) and all(v <= 1e-5 for v in d.values())}


def screen():
    users_file = HERE / "screen-users.json"
    want = set(json.loads(users_file.read_text())["users"])
    rows = rows_of(SCREEN_ROWS)
    big = rows_of(*BIG).get(BIG[1], {})
    labels = [BASE, INT8] + [f"rel/w8+{r}" for r in CONTINUITY + CARD + NOTE]
    missing = {l: len(want - set(rows.get(l, {}))) for l in labels if want - set(rows.get(l, {}))}
    users = want & set(big)
    for l in labels:
        users &= set(rows.get(l, {}))
    rep = {"screen_users": len(want), "users": len(users), "missing": missing}
    rep["reproduction_w8+s8_vs_released_rows"] = reproduction(rows.get(INT8, {}), sorted(want & set(rows.get(INT8, {}))))
    if missing:
        print(f"[sm90-screen] rows missing: {missing}")
    if len(users) < 0.9 * len(want):
        print(f"[sm90-screen] only {len(users)} of {len(want)} screen users in every label; not picking")
        (HERE / "screen-summary.json").write_text(json.dumps(rep, indent=1))
        return 3
    cmp = {l: compare(rows[l], rows[BASE], big, users) for l in labels if l != BASE}
    rep["compare_vs_s4c8n"] = cmp
    rep["base_vs_big"] = compare(rows[BASE], rows[BASE], big, users)
    e = {r: max(cmp[f"rel/w8+{r}"]["curve"]["points"], cmp[f"rel/w8+{r}"]["instant"]["points"]) for r in CARD + NOTE}
    card_pass = [r for r in CARD if e[r] <= SCREEN_PASS]
    note_pass = [r for r in NOTE if e[r] <= SCREEN_PASS]
    c_star = card_pass[-1] if card_pass else None    # the smallest card budget within the screen threshold
    n_star = note_pass[-1] if note_pass else None
    picks, why = [], []
    if c_star:
        picks.append(c_star)
        why.append(f"card: smallest rung within {SCREEN_PASS} points = {c_star}")
    else:
        picks.append(CARD[0])
        why.append(f"card: no rung within {SCREEN_PASS} points, the largest ({CARD[0]})")
    if c_star and n_star:
        picks.append(f"{c_star}N{n_star[3:]}")
        why.append(f"both streams coded: {c_star} + {n_star}")
    i = CARD.index(c_star) if c_star else 0
    if c_star and i > 0:
        picks.append(CARD[i - 1])
        why.append(f"card safety rung {CARD[i - 1]}")
    if n_star:
        picks.append(n_star)
        why.append(f"note: smallest rung within {SCREEN_PASS} points = {n_star}")
    else:
        why.append(f"note: no rung within {SCREEN_PASS} points; notes stay int8 (no note finalist)")
    finalists = [BASE] + [f"rel/w8+{r}" for r in picks]
    why = "; ".join(why)
    best = e
    rep.update({"screen_points_max_head": e, "finalists": finalists, "why": why})
    (HERE / "finalists.txt").write_text("\n".join(finalists) + "\n")
    cost = costs()
    md = [f"# Codebook state compression sweep (SM90): screen ({len(users)} users of the 732 set)", "",
          "Released small model (the leak-free small model (SM72s), cool-down from epoch 25), w8 weights, leak-free inputs. "
          "Delta vs int4 cards + int8 notes (`w8+s4c8n`, the fork default) on the same users; points = % of the big model "
          "(1 leak-free epoch) LogLoss; [ ] = 95% paired user bootstrap (2,000, seed 0). Screen threshold "
          f"{SCREEN_PASS} points (the final rule is {NEARLY_FREE_POINTS}).", "",
          f"- w8+s8 vs the recorded released rows: {rep['reproduction_w8+s8_vs_released_rows']}",
          f"- int4 cards + int8 notes vs big: curve {rep['base_vs_big']['curve']['pct_vs_big']:+.2f}%, instant "
          f"{rep['base_vs_big']['instant']['pct_vs_big']:+.2f}%", "",
          "| Recipe | Card / note bits | State MB | Codebooks MB (f16) | Encode ms / review (est.) | GPU reviews/s | Curve vs s4c8n | Instant vs s4c8n |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for l in labels[1:]:
        r = l.split("+", 1)[1]
        c = cost.get(r, {})
        sp = speed(rows[l])["reviews_per_second"]
        md.append(f"| {l} | {c.get('card_bits', '-')} / {c.get('note_bits', '-')} | {c.get('state_mb', 0):.1f} | "
                  f"{c.get('codebook_mb_f16', 0):.1f} | {c.get('encode_ms_per_review_est', 0):.2f} | {sp:,.0f} | "
                  f"{fmt(cmp[l], 'curve')} | {fmt(cmp[l], 'instant')} |")
    md += ["", "Screen extra cost, max of the two heads (points): " + ", ".join(f"{k} {v:+.3f}" for k, v in best.items()),
           f"Finalists ({why}): {finalists}"]
    (HERE / "screen-summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "screen-summary.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(md))
    return 0


def final():
    want = set(json.loads(USERS732.read_text())["users"])
    rows = rows_of(FINAL_ROWS)
    big = rows_of(*BIG).get(BIG[1], {})
    finalists = (HERE / "finalists.txt").read_text().split()
    have = [l for l in finalists if l in rows]
    users = want & set(big)
    for l in have:
        users &= set(rows[l])
    cost = costs()
    rep = {"pool_users": len(want), "users": len(users), "finalists": finalists, "scored": have,
           "rows_per_label": {l: len(rows.get(l, {})) for l in finalists}, "rule": {
               "nearly_free_points": NEARLY_FREE_POINTS, "bar_pct_vs_big": BAR, "practical": PRACTICAL}}
    if BASE not in have:
        print("[sm90-final] no int4 cards + int8 notes rows yet")
        return 3
    rep["base_vs_big"] = compare(rows[BASE], rows[BASE], big, users)
    md = [f"# Codebook state compression sweep (SM90): finalists on the 732 set ({len(users)} users)", "",
          "Released small model (the leak-free small model (SM72s), cool-down from epoch 25), w8 weights, leak-free inputs. "
          "Delta vs int4 cards + int8 notes (`w8+s4c8n`, the fork default) on the same users; points = % of the big model "
          "(1 leak-free epoch) LogLoss; [ ] = 95% paired user bootstrap (2,000, seed 0).", "",
          f"- int4 cards + int8 notes: curve {rep['base_vs_big']['curve']['logloss']:.5f} "
          f"({rep['base_vs_big']['curve']['pct_vs_big']:+.2f}% vs big), instant {rep['base_vs_big']['instant']['logloss']:.5f} "
          f"({rep['base_vs_big']['instant']['pct_vs_big']:+.2f}% vs big); state 48.1 MB (fork format)", "",
          "| Recipe | Card / note bits | State MB | Codebooks MB (f16) | Encode ms / review (est.) | GPU reviews/s | Curve vs s4c8n | Instant vs s4c8n | Nearly free | Practical |",
          "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    verdicts = {}
    for l in have:
        if l == BASE:
            continue
        r = l.split("+", 1)[1]
        c = compare(rows[l], rows[BASE], big, users)
        free = all(c[h]["points"] <= NEARLY_FREE_POINTS for h in HEADS) and all(c[h]["pct_vs_big"] <= BAR[h] for h in HEADS)
        k = cost.get(r, {})
        practical = all(k.get(key, float("inf")) <= lim for key, lim in PRACTICAL.items())
        sp = speed(rows[l])["reviews_per_second"]
        verdicts[l] = {"compare": c, "nearly_free": free, "practical": practical, "cost": k, "speed": speed(rows[l])}
        md.append(f"| {l} | {k.get('card_bits', '-')} / {k.get('note_bits', '-')} | {k.get('state_mb', 0):.1f} | "
                  f"{k.get('codebook_mb_f16', 0):.1f} | {k.get('encode_ms_per_review_est', 0):.2f} | {sp:,.0f} | "
                  f"{fmt(c, 'curve')} | {fmt(c, 'instant')} | {'yes' if free else 'no'} | {'yes' if practical else 'no'} |")
    def kind(l):
        r = l.split("+", 1)[1]
        return "card" if r in CARD else "note" if r in NOTE else "both"
    smallest = {}
    for k in ("card", "note", "both"):
        free = [l for l in verdicts if verdicts[l]["nearly_free"] and kind(l) == k]
        smallest[k] = min(free, key=lambda l: verdicts[l]["cost"].get("state_mb", 1e9)) if free else None
    rep.update({"verdicts": verdicts, "smallest_nearly_free": smallest})
    md += [""] + [f"Smallest nearly-free {k} budget: {v or 'none of the finalists'}"
                  + (f" (practical: {verdicts[v]['practical']})" if v else "") for k, v in smallest.items()] + [
           f"Rule: extra cost <= {NEARLY_FREE_POINTS} points on both heads and total <= +{BAR['curve']}% curve / "
           f"+{BAR['instant']}% instant vs the big model (point estimates); practical = codebooks <= "
           f"{PRACTICAL['codebook_mb_f16']} MB at float16 and encode <= {PRACTICAL['encode_ms_per_review_est']} ms per review (est.)."]
    if len(have) < len(finalists) or len(users) < 700:
        md.append(f"INCOMPLETE: scored {have} on {len(users)} users")
    (HERE / "sm90-summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "sm90-summary.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit({"screen": screen, "final": final}[sys.argv[1]]())
