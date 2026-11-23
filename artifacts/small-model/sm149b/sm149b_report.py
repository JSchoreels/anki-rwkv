"""Rank-2 states with a tail-energy penalty (SM149b): dataset-wide read-out (732 set) and the stage gate, a copy of
SM149's sm149_report.py with the new arm added (reuses the helpers of sm148_report.py, not edited). CPU only.
Prints and writes sm149b-stage1.json. The long-card rules (R3, G) are added by HANDOFF.md step 3 from the CPU rows.

  python -X utf8 sm149b_report.py [--arm sm149b-sm118-tail1]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

S148 = Path("C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148")
S149 = Path("C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sm148_report", S148 / "sm148_report.py")
m = importlib.util.module_from_spec(spec)
sys.modules["sm148_report"] = m
spec.loader.exec_module(m)

C_CTL = 0.079   # SM149's control: Q2 vs its own int8, instant head (sm149-stage1.json gate), fixed in design.md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="sm149b-sm118-tail1")
    p.add_argument("--rows", default=str(HERE / "gpu" / "sm149b-rows-732.jsonl"))
    p.add_argument("--out", default=str(HERE / "sm149b-stage1.json"))
    args = p.parse_args()
    st = m.showtime()
    E, L = m.E2S, m.LABELS
    o_rows, _, _ = m.read_rows(S148 / "gpu/sm148-rows-732.jsonl")
    s_rows, _, _ = m.read_rows(S149 / "gpu/sm149-rows-732.jsonl")
    n_rows, _, _ = m.read_rows(Path(args.rows))
    tag = "sm118-e26-dec2ep32k-t91s2"
    users732 = set(json.load(open(m.U732))["users"])

    def get(rows, pre, s):
        return rows[f"{pre}/{L[s]}{E}"]
    orig = {s: get(o_rows, tag, s) for s in ("REF", "Q4", "T24", "Q2")}
    new = {s: get(n_rows, args.arm, s) for s in ("REF", "Q2", "T24")}
    qat = {s: get(s_rows, "sm149-sm118-qat", s) for s in ("REF", "Q2", "T24")}
    ctl = {s: get(s_rows, "sm149-sm118-ctl", s) for s in ("REF", "Q2")}
    common = set(orig["REF"])
    for d in (new, qat, ctl, orig):
        for r in d.values():
            common &= set(r)
    common = sorted(u for u in common if u in users732)
    print("common users", len(common))
    H = ("curve", "instant")

    def cmp(a, b):
        return m.dataset_cmp(a, b, H, common, st)
    out = {"arm": args.arm, "users": len(common), "coverage": {s: len(set(new[s]) & users732) for s in new},
           "r2_vs_orig_int8": {}, "r2_vs_own_int8": {}, "vs_orig_same_label": {}, "vs_sm149_qat_same_label": {},
           "extra": {}}
    out["r2_vs_orig_int8"]["NEW_REF (R6)"] = cmp(new["REF"], orig["REF"])
    for s in ("Q2", "T24"):
        out["r2_vs_orig_int8"][f"NEW_{s}"] = cmp(new[s], orig["REF"])
        out["r2_vs_own_int8"][f"NEW_{s}"] = cmp(new[s], new["REF"])
        out["vs_orig_same_label"][f"NEW_{s}"] = cmp(new[s], orig[s])
        out["vs_sm149_qat_same_label"][f"NEW_{s}"] = cmp(new[s], qat[s])
        out["r2_vs_orig_int8"][f"SM149_QAT_{s}"] = cmp(qat[s], orig["REF"])
        out["r2_vs_own_int8"][f"SM149_QAT_{s}"] = cmp(qat[s], qat["REF"])
    out["vs_sm149_qat_same_label"]["NEW_REF"] = cmp(new["REF"], qat["REF"])
    out["r2_vs_orig_int8"]["SM149_CTL_Q2"] = cmp(ctl["Q2"], orig["REF"])
    out["r2_vs_own_int8"]["SM149_CTL_Q2"] = cmp(ctl["Q2"], ctl["REF"])
    out["extra"]["orig_Q2_vs_orig_REF"] = cmp(orig["Q2"], orig["REF"])
    out["extra"]["orig_T24_vs_orig_REF"] = cmp(orig["T24"], orig["REF"])
    out["extra"]["NEW_REF_vs_CTL_REF"] = cmp(new["REF"], ctl["REF"])
    cn = out["r2_vs_own_int8"]["NEW_Q2"]["instant"]["value"]
    cc = out["r2_vs_own_int8"]["SM149_CTL_Q2"]["instant"]["value"]
    r2 = {k: all(v[h]["ci"][1] <= m.CEIL_ALL for h in H) for k, v in out["r2_vs_orig_int8"].items()}
    r6 = r2["NEW_REF (R6)"]
    label_pass_dataset = {s: r2[f"NEW_{s}"] and r6 for s in ("Q2", "T24")}
    verdict = ("pass pending R3 / G (long-card CPU step)" if any(label_pass_dataset.values()) else
               "partial" if cn <= 0.5 * C_CTL else "no effect")
    out["gate"] = {"C_new": cn, "C_ctl_preregistered": C_CTL, "C_ctl_recomputed": cc, "ratio": cn / C_CTL,
                   "R2_pass_vs_orig_int8": r2, "R6_pass": r6, "dataset_wide_pass_by_label": label_pass_dataset,
                   "verdict_dataset_wide": verdict}
    bk = {}
    for nm, a, b in (("NEW_Q2_vs_orig_REF", new["Q2"], orig["REF"]), ("NEW_Q2_vs_NEW_REF", new["Q2"], new["REF"]),
                     ("SM149_QAT_Q2_vs_QAT_REF", qat["Q2"], qat["REF"]),
                     ("SM149_CTL_Q2_vs_CTL_REF", ctl["Q2"], ctl["REF"])):
        nn, rr = {u: a[u] for u in common}, {u: b[u] for u in common}
        bk[nm] = {f"{lo}-{hi - 1 if hi else ''}": {h: m.bucket_cmp(nn, rr, m.KEY[h], lo, hi) for h in H}
                  for lo, hi in m.BUCKETS}
    out["buckets"] = bk
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for sec in ("r2_vs_orig_int8", "r2_vs_own_int8", "vs_orig_same_label", "vs_sm149_qat_same_label", "extra"):
        print("##", sec)
        for k, v in out[sec].items():
            print(k, "| curve", m.fmt(v["curve"]), "| instant", m.fmt(v["instant"]))
    print(json.dumps(out["gate"], indent=1))
    for nm, b in bk.items():
        print("##", nm)
        for k, v in b.items():
            print(k, v["instant"] and (m.fmt(v["instant"]), v["instant"]["reviews"]), "|", v["curve"] and m.fmt(v["curve"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
