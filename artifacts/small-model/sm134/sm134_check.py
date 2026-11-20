#!/usr/bin/env python
"""Six-layer Tiny with the card-only state (SM134): CPU proofs and the GPU smoke gate (run with CUDA_VISIBLE_DEVICES= ).

`sm134_check.py` (CPU, seconds), K1-K4, writes check-cpu-sm134.json, exit 0 iff all pass:
  K1 bytes: Tiny card-only 619 B per card, 1,315 B per note (sm131-bytes.json, whose sm131_bytes.py proof is bit for bit).
  K2 CPU smoke rows (cpu/smoke-cpu-rows.jsonl: users 2388, 4676, 1712, the two-head six-layer Tiny (SM130), w8+s8, compact
     w8+s4c8nhx8 and card-only w8+s4hgx4c8hx8n, scored on CPU by this tree): every row finite; w8+s8 equals six-layer Tiny
     (SM130)'s published GPU rows within 1e-5 per head, compact within 1e-4 (CPU vs GPU, the SM131 tolerances: 2e-6 / 1e-5
     for w8+s8, 1e-4 for the other formats); card-only differs from compact.
  K3 the scorer built w64-c1-d2p1u1: 185,057 parameters in every CPU smoke row.
  K4 rule selftest (sm134_rule.py).
`sm134_check.py smoke` (the first step of the GPU ticket): gpu/smoke-gpu-rows.jsonl (same users and formats on the GPU) vs
the CPU rows: w8+s8 within 1e-5 per head of the CPU rows and the published rows, compact within 1e-4 of the CPU rows and of
the published GPU rows, card-only within 1e-4 of its CPU row -> gpu/smoke-check.json, exit 0 iff pass.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
PY = sys.executable
PUB = HERE / "inputs/sm130-rows.jsonl"
SMOKE = HERE / "cpu/smoke-cpu-rows.jsonl"
TAG = "sm130-th-dec2ep32k-t91s2"
FORMATS = ("w8+s8", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n")
USERS = (2388, 4676, 1712)
HEADS = ("ahead", "instant")


def rows_of(path, tag=TAG):
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "benchmark" in r and str(r.get("label", "")).startswith(tag + "/"):
            out[(r["user_id"], r["label"].split("/", 1)[1].split("@")[0])] = r["benchmark"]
    return out


def diff(a, b):
    return max(abs(a[h]["LogLoss"] - b[h]["LogLoss"]) for h in HEADS)


def finite(b):
    return all(b[h]["LogLoss"] == b[h]["LogLoss"] and abs(b[h]["LogLoss"]) != float("inf") for h in HEADS)


def k1():
    b = json.loads((ROOT / "artifacts/small-model/sm131/sm131-bytes.json").read_text())["arms"]["Tiny (w64-c1)"]
    got = (b["w8+s4hgx4c8hx8n"]["card"]["bytes"], b["w8+s4hgx4c8hx8n"]["note"]["bytes"], b["w8+s4c8nhx8"]["card"]["bytes"])
    return got == (619, 1315, 803), got


def k2():
    cpu, pub = rows_of(SMOKE), rows_of(PUB)
    res = {}
    for u in USERS:
        res[f"{u}/all_finite"] = all((u, f) in cpu and finite(cpu[(u, f)]) for f in FORMATS)
        res[f"{u}/w8+s8_vs_published_1e-5"] = diff(cpu[(u, "w8+s8")], pub[(u, "w8+s8")]) <= 1e-5
        res[f"{u}/compact_vs_published_1e-4"] = diff(cpu[(u, "w8+s4c8nhx8")], pub[(u, "w8+s4c8nhx8")]) <= 1e-4
        res[f"{u}/card_only_differs_from_compact"] = cpu[(u, "w8+s4hgx4c8hx8n")] != cpu[(u, "w8+s4c8nhx8")]
    detail = {f"{u}/{f}": diff(cpu[(u, f)], pub[(u, f)]) for u in USERS for f in FORMATS[:2]}
    return all(res.values()), {"checks": res, "max_abs_logloss_diff_cpu_vs_published": detail}


def k3():
    params = {json.loads(l).get("parameters") for l in open(SMOKE, encoding="utf-8") if '"benchmark"' in l}
    return params == {185057}, sorted(params)


def k4():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm134_rule.py"), "selftest"], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip().splitlines()[-1:]


def smoke():
    cpu, pub, gpu = rows_of(SMOKE), rows_of(PUB), rows_of(HERE / "gpu/smoke-gpu-rows.jsonl")
    res, diffs = {}, {}
    for u in USERS:
        for f in FORMATS:
            key = (u, f)
            if key not in gpu or key not in cpu or not finite(gpu[key]):
                res[f"{u}/{f}"] = False
                continue
            d = diff(gpu[key], cpu[key])
            diffs[f"{u}/{f}_vs_cpu"] = d
            ok = d <= (1e-5 if f == "w8+s8" else 1e-4)
            if f in pub_formats() and key in pub:
                dp = diff(gpu[key], pub[key])
                diffs[f"{u}/{f}_vs_published"] = dp
                ok &= dp <= (1e-5 if f == "w8+s8" else 1e-4)
            res[f"{u}/{f}"] = bool(ok)
    out = {"pass": all(res.values()), "checks": res, "max_abs_logloss_diff": diffs}
    (HERE / "gpu/smoke-check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm134-smoke] {'PASS' if out['pass'] else 'FAIL'} {json.dumps(diffs)}")
    return 0 if out["pass"] else 1


def pub_formats():
    return ("w8+s8", "w8+s4c8nhx8")


def main():
    if sys.argv[1:] == ["smoke"]:
        return smoke()
    t0 = time.time()
    out = {}
    for name, fn in (("K1_bytes", k1), ("K2_cpu_smoke_rows", k2), ("K3_parameters", k3), ("K4_rule_selftest", k4)):
        ok, detail = fn()
        out[name] = {"pass": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail)[:500]}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "check-cpu-sm134.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm134-check] {'ALL OK' if out['pass'] else 'FAILED'} ({out['seconds']} s)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
