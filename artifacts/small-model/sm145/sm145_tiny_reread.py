#!/usr/bin/env python
"""Tiny re-read under the user's 11-22 Tiny rule (not pre-registered). CPU only, aggregates only.
Each Tiny label (REF, AN16, Y2N, Y4C, Y4N) vs the matching Small model at int8 (REF), paired by user on the GPU rows of the
732 set (same inputs): ratio of summed LogLoss x size, 2,000-draw user bootstrap (sm145_report._bench_cmp). Pairs: six-layer
Tiny two-head model (SM130) vs show-time two-head cool-down for Small (SM118); six-layer Tiny Instant-Only model (SM130) vs
longer Small Instant-Only model (SM113). Rule: upper bound <= +1.00% on each head. R1, R3 (long cards: point <= +0.5%, upper
<= +1.0%) and R4 are read from sm145-result.json unchanged. Writes sm145-tiny-reread.json."""
import json
import sm145_report as R

CEIL = 1.00
rows, _, _ = R.read_rows(R.ROWS732)
res = json.loads((R.HERE / "sm145-result.json").read_text())
byt = json.loads(R.BYTES.read_text())
bytes_pct = res["go"]["Tiny"]["bytes_vs_REF_pct"]  # only to check the key exists
out = {"rule": f"Tiny label vs matching Small at int8, 95% CI upper bound <= +{CEIL:.2f}% per head; R1, R3 (point <= +0.5%, upper <= +1.0%), R4 unchanged", "labels": {}}
pairs = [("sm130th", "sm118"), ("sm130io", "sm113")]
for lab in ["REF", "AN16", "Y2N", "Y4C", "Y4N"]:
    x = {"models": {}}
    ok = True
    for tk, sk in pairs:
        ttag, stag = R.MODELS[tk][0], R.MODELS[sk][0]
        ref = rows[f"{stag}/{R.LABELS['REF']}{R.E2S}"]
        new = rows[f"{ttag}/{R.LABELS[lab]}{R.E2S}"]
        users = sorted(set(new) & set(ref))
        m = {"vs": R.MODELS[sk][1], "users": len(users), "heads": {}}
        for h in R.MODELS[tk][2]:
            r = R._bench_cmp(new, ref, R.KEY[h], users)
            m["heads"][h] = r
            ok &= r["ci"][1] <= CEIL
        if lab != "REF":
            s = res["models"][tk]["schemes"][lab]
            m["R1"], m["R3"], m["R4"] = s["R1_finite"], s["R3"], s["R4"]
            ok &= s["R1_finite"] and s["R3"] and s["R4"]
        x["models"][tk] = m
    x["pass_tiny_rule"] = bool(ok)
    out["labels"][lab] = x
json.dump(out, open(R.HERE / "sm145-tiny-reread.json", "w"), indent=1)
for lab, x in out["labels"].items():
    for tk, m in x["models"].items():
        for h, r in m["heads"].items():
            print(lab, tk, h, m["users"], f"{r['value']:+.3f}% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]", m.get("R1"), m.get("R3"), m.get("R4"))
    print(lab, "PASS" if x["pass_tiny_rule"] else "fail")
