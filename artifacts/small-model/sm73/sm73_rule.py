#!/usr/bin/env python
"""Smaller model (SM73) stage 1 helper (CPU, seconds): the released leak-free small model (SM72s, cool-down from
epoch 20) with a smaller stored card/note state, against the bar +0.5% curve / +0.4% instant vs the big model
(1 leak-free epoch) on the 732 set.

  decide LABEL   LABEL (e.g. TAG/w8+s4) vs the big model (1 leak-free epoch) on the users of the 732 set (723 fit users)
                 that every series has (722 in the SM72s readout): review-weighted benchmark LogLoss (all reviews,
                 leak-free scoring), % vs big, and the paired delta vs the int8-state release (TAG/w8+s8) with a
                 user bootstrap (2,000, seed 0, 95% CI). Appends one JSON line to sm73-formats.jsonl and prints one line.
                 exit 0 = PASS (both heads within the bar, point estimates), 1 = FAIL, 2 = users missing.
  summary        sm73-formats.md from sm73-formats.jsonl.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SM72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")         # read only
SM72G = Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big")   # read only
TAG = "s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon"
REF = (SM72S / "gpu/small-lf-neighbour.jsonl", f"{TAG}/w8+s8")
BIG = (SM72G / "bigleak-neighbour.jsonl", "big-leakfree-1ep/w8+s8")
OUT = HERE / "gpu/sm73-formats-732.jsonl"
USERS = SM72S / "eval-users-732-fit.json"
RESULTS = HERE / "sm73-formats.jsonl"
BAR = {"curve": 0.50, "instant": 0.40}
HEADS = {"ahead": "curve", "instant": "instant"}
# Bytes per card + its note in the fork's serialized state layout (INTEGRATION.md 11.1 formula: 74 B of headers and
# lengths, per layer rows of a 4-byte scale + codes, f16 token shift, f16 channel shift where a channel mixer exists;
# card layer 1 has none). int8 4,554 and int4 3,018 are the published values; s4c8n = int4 card layers + int8 note layer.
FORK_BYTES = {"w8+s8": 4554, "w8+s4c8n": 3530, "w8+s4": 3018}


def load(path, label):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("label") != label or "error" in r or "benchmark" not in r:
            continue
        out[r["user_id"]] = {h: (r["benchmark"][k]["LogLoss"], r["benchmark"][k]["size"]) for k, h in HEADS.items()}
    return out


def arrays(rows, users, head):
    v = np.array([rows[u][head][0] for u in users]); w = np.array([rows[u][head][1] for u in users], float)
    return np.where(w > 0, v, 0.0) * w, w


def decide(label):
    fit = set(json.loads(USERS.read_text())["users"])
    mine = load(OUT, label) if OUT.exists() else {}
    ref, big = load(*REF), load(*BIG)
    users = sorted(fit & set(ref) & set(big))
    missing = [u for u in users if u not in mine]
    if missing:
        print(f"[sm73-rule] {label}: {len(users) - len(missing)} of {len(users)} users scored")
        return 2
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(users), size=(2000, len(users)))
    rec = {"label": label, "users": len(users), "fork_bytes_per_card_and_note": FORK_BYTES.get(label.split("/")[-1])}
    ok = True
    for head in HEADS.values():
        (m, w), (r, _), (b, _) = arrays(mine, users, head), arrays(ref, users, head), arrays(big, users, head)
        L, R, B = m.sum() / w.sum(), r.sum() / w.sum(), b.sum() / w.sum()
        pct, ref_pct = (L / B - 1) * 100, (R / B - 1) * 100
        d = (m[boot].sum(1) - r[boot].sum(1)) / w[boot].sum(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rec[head] = {"logloss": L, "pct_vs_big_leakfree": pct, "ref_int8_pct": ref_pct, "delta_vs_int8": L - R,
                     "delta_ci": [lo, hi], "delta_points": pct - ref_pct, "room_points": BAR[head] - ref_pct}
        ok &= pct <= BAR[head]
    rec["pass"] = bool(ok)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    c, i = rec["curve"], rec["instant"]
    print(f"{label} ({len(users)} users, {rec['fork_bytes_per_card_and_note']} B per card + note): "
          f"curve {c['logloss']:.5f} {c['pct_vs_big_leakfree']:+.2f}% vs big (1 leak-free epoch), "
          f"{c['delta_vs_int8']:+.6f} [{c['delta_ci'][0]:+.6f}, {c['delta_ci'][1]:+.6f}] vs int8; "
          f"instant {i['logloss']:.5f} {i['pct_vs_big_leakfree']:+.2f}%, "
          f"{i['delta_vs_int8']:+.6f} [{i['delta_ci'][0]:+.6f}, {i['delta_ci'][1]:+.6f}]; "
          f"bar +0.50% / +0.40%: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def summary():
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()] if RESULTS.exists() else []
    last = {r["label"]: r for r in rows}
    md = ["# Smaller model (SM73) stage 1: smaller stored state for the leak-free small model (SM72s, cool-down from epoch 20)",
          "", "732 set, leak-free scoring, review-weighted LogLoss; bar +0.50% curve / +0.40% instant vs the big model "
          "(1 leak-free epoch). Delta = paired vs the int8-state release (user bootstrap 2,000, seed 0, 95% CI).", "",
          "| state | bytes per card + note | users | curve | vs big | delta vs int8 | instant | vs big | delta vs int8 | bar |",
          "|---|---:|---:|---:|---:|---|---:|---:|---|---|"]
    for label, r in last.items():
        c, i = r["curve"], r["instant"]
        md.append(f"| {label.split('/')[-1]} | {r['fork_bytes_per_card_and_note']:,} | {r['users']} | {c['logloss']:.5f} | "
                  f"{c['pct_vs_big_leakfree']:+.2f}% | {c['delta_vs_int8']:+.6f} [{c['delta_ci'][0]:+.6f}, {c['delta_ci'][1]:+.6f}] | "
                  f"{i['logloss']:.5f} | {i['pct_vs_big_leakfree']:+.2f}% | {i['delta_vs_int8']:+.6f} "
                  f"[{i['delta_ci'][0]:+.6f}, {i['delta_ci'][1]:+.6f}] | {'PASS' if r['pass'] else 'FAIL'} |")
    (HERE / "sm73-formats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm73-rule] summary: {len(last)} formats -> {HERE / 'sm73-formats.md'}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["decide"]:
        sys.exit(decide(sys.argv[2]))
    if sys.argv[1:2] == ["summary"]:
        sys.exit(summary())
    sys.exit(__doc__)
