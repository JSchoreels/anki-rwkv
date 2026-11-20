#!/usr/bin/env python
"""Long-card state drift on real users (SM136): CPU proofs (minutes; run with CUDA_VISIBLE_DEVICES= ).

K1 bytes: sm136_bytes.py ALL OK (SM131's encoder / decoder plus the int8-card arm w8+s8hx8; known fork totals).
K2 pool: sm136_pool.py --check (the committed pools and aggregate counts are reproduced from the revlogs).
K3 formats: parse_variant (+ fork_streams for the fork-kind Big models) maps the five formats to the intended state bits;
   w8 = no state rounding (float32 state).
K4 default output unchanged:
   a. the scorer diff against origin/sm131 only adds lines (scripts/small_model_heldout.py; nothing else under scripts/);
   b. without --prior-buckets, this tree's CPU rows of the show-time big model (SM91s2) for users 2388 and 4676 at the
      four SM131 formats equal the compact state for Small and Big (SM131)'s committed CPU rows bit for bit (every field
      except timing and provenance);
   c. with --prior-buckets the rows are the rows without it plus the field prior_buckets, for the show-time big model
      (SM91s2) and the show-time two-head cool-down for Tiny (SM124) (expertium path), users 2388 and 1866.
K5 bucket sums (every flag row of K4c and K6): benchmark scope: counts add up to the benchmark size and
   sum(ll) / sum(n) equals the benchmark LogLoss within its 6-decimal rounding (<= 5.1e-7), every benchmark review has
   1..127 prior answers; "all" scope: counts add up to the scorer's curve / instant label counts and the mean equals
   its curve_log_loss / instant_log_loss within 1e-4 (float32 model averages, clamping), prior answers >= 1.
K6 CPU smoke rows (cpu/smoke-cpu-rows.jsonl; users 1866 and 2883, the two smallest main-pool users, cards up to 147 and
   138 prior answers): the six models x five formats with --prior-buckets on CPU: every row finite; w8+s8 within 1e-4
   per head of the published GPU rows (max reported); the bucket counts equal across the five formats; each int4 format
   differs from w8+s8.
K7 rule selftest (sm136_rule.py).
-> check-cpu-sm136.json, exit 0 iff all pass.

`sm136_check.py smoke` (the GPU ticket's first step): gpu/smoke-gpu-rows.jsonl (the same 2 users, 30 labels, GPU) vs
the CPU rows: bucket counts identical (both scopes); benchmark LogLoss per head within 1e-5 for w8+s8 (also vs the
published GPU rows) and within 1e-4 for the other formats (the flag kernel / int kernels vs the CPU reference: summation
order differs, so a rare rounding flip is allowed, as in SM131); the "all" scope's mean LogLoss per bucket with >= 20
reviews within 1e-4 -> gpu/smoke-check.json, exit 0 iff pass.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
PY = sys.executable
SC = ROOT / "scripts/small_model_heldout.py"
CPU = HERE / "cpu"
SMOKE = CPU / "smoke-cpu-rows.jsonl"
SM131_CPU = HERE.parent / "sm131/cpu/smoke-cpu-rows.jsonl"
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
CK = {"sm91s2": ("fork", "", W / "sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"),
      "sm99": ("fork", "", W / "sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth"),
      "sm118": ("expertium", "w64", W / f"sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/{TRACK}/master.pth"),
      "sm113": ("expertium", "w64", W / f"sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/{TRACK}/master.pth"),
      "sm124": ("expertium", "w64-c1", W / f"sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/{TRACK}/master.pth"),
      "sm115": ("expertium", "w64-c1", W / f"sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/{TRACK}/master.pth")}
SMOKE_USERS = (1866, 2883)
PRIOR = "0,32,128,300,1000"
VOLATILE = ("prepare_seconds", "model_seconds", "provenance_commit", "provenance_dirty")
SM131_FORMATS = ("w8+s8", "w8+s4c8n", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n")


def rule():
    import sm136_rule
    return sm136_rule


def score(out, users, models, labels, flag, users_file):
    """CPU scorer run (show-time inputs, leak-free, as the GPU tickets)."""
    Path(users_file).write_text(json.dumps({"users": list(users)}), encoding="utf-8")
    cmd = [PY, "-X", "utf8", "-W", "ignore", str(SC), "--device", "cpu", "--pool", "732", "--users-file", str(users_file),
           "--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour", "--nonfinite-rows", "--out", str(out)]
    for m in models:
        kind, arch, path = CK[m]
        cmd += ["--model", f"{rule().MODELS[m]['tag']}={kind}:{arch}:{Path(path).as_posix()}"]
    for lab in labels:   # --only takes the label without the input suffix (the scorer appends @e2s-query)
        cmd += ["--only", lab.removesuffix("@e2s-query")]
    if flag:
        cmd += ["--prior-buckets", PRIOR]
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START", "SM11_INPUTS")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"scorer failed: {p.stderr[-800:]}")


def rows_of(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" not in r:
            out[(r["user_id"], r["label"])] = r
    return out


def strip(r, extra=()):
    return {k: v for k, v in r.items() if k not in VOLATILE and k not in extra}


def k1():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm136_bytes.py")], capture_output=True, text=True)
    return p.returncode == 0 and "ALL OK" in p.stdout, p.stdout.strip().splitlines()[-4:]


def k2():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm136_pool.py"), "--check"], capture_output=True, text=True, cwd=str(ROOT))
    return p.returncode == 0 and "check OK" in p.stdout, p.stdout.strip().splitlines()[-1:]


def k3():
    from scripts import small_model_heldout as smh
    from scripts.small_model_wkv import F16_SCALE as F
    res = {}
    _, b = smh.parse_variant("w8+s8")
    res["w8+s8 = 8"] = b == 8
    _, b = smh.parse_variant("w8+s4c8n")
    res["w8+s4c8n = (4, 8), fork ((4, 0), (8, 0))"] = b == (4, 8) and smh.fork_streams(b) == ((4, 0), (8, 0))
    _, b = smh.parse_variant("w8+s4c8nhx8")
    res["w8+s4c8nhx8 compact"] = isinstance(b, smh.IntState) and smh.fork_streams(b) == ((4 | F, 8 | F), (8 | F, 8 | F))
    _, b = smh.parse_variant("w8+s8hx8")
    res["w8+s8hx8 int8 cards + notes, f16 scales, int8 shifts"] = isinstance(b, smh.IntState) and b.wkv_bits == 8 | F and \
        b.shift_bits == 8 | F and b.scale_bits == 16 and smh.fork_streams(b) == ((8 | F, 8 | F), (8 | F, 8 | F))
    s, b = smh.parse_variant("w8")
    res["w8 = w8 weights, no state rounding"] = s == "w8-pc" and b == 0
    return all(res.values()), res


def k4():
    res = {}
    p = subprocess.run(["git", "-C", str(ROOT), "diff", "origin/sm131", "--numstat", "--", "scripts/"], capture_output=True, text=True)
    lines = [x.split("\t") for x in p.stdout.strip().splitlines()]
    res["a. scorer diff additive"] = p.returncode == 0 and len(lines) == 1 and lines[0][2] == "scripts/small_model_heldout.py" \
        and lines[0][1] == "0" and int(lines[0][0]) > 0
    tag = rule().MODELS["sm91s2"]["tag"]
    labs = [f"{tag}/{f}@e2s-query" for f in SM131_FORMATS]
    off = CPU / "k4-off.jsonl"
    off.unlink(missing_ok=True)
    score(off, (2388, 4676), ("sm91s2",), labs, False, CPU / "k4-users-a.json")
    old, new = rows_of(SM131_CPU), rows_of(off)
    same = []
    for u in (2388, 4676):
        for lab in labs:
            same.append((u, lab) in new and (u, lab) in old and strip(new[(u, lab)]) == strip(old[(u, lab)]))
    res["b. flag off == SM131 committed CPU rows (8 rows)"] = all(same) and len(same) == 8
    on, off2 = CPU / "k4-on.jsonl", CPU / "k4-off2.jsonl"
    for f in (on, off2):
        f.unlink(missing_ok=True)
    t124 = rule().MODELS["sm124"]["tag"]
    labs2 = [f"{tag}/w8+s8@e2s-query", f"{tag}/w8+s4c8nhx8@e2s-query", f"{t124}/w8+s8@e2s-query", f"{t124}/w8+s4c8nhx8@e2s-query"]
    score(on, (2388, 1866), ("sm91s2", "sm124"), labs2, True, CPU / "k4-users-c.json")
    score(off2, (2388, 1866), ("sm91s2", "sm124"), labs2, False, CPU / "k4-users-c.json")
    a, b = rows_of(on), rows_of(off2)
    eq = [(k in a and k in b and "prior_buckets" in a[k] and "prior_buckets" not in b[k]
           and strip(a[k], ("prior_buckets",)) == strip(b[k])) for k in [(u, l) for u in (2388, 1866) for l in labs2]]
    res["c. flag on == flag off + prior_buckets (8 rows, fork and expertium)"] = all(eq) and len(eq) == 8
    return all(res.values()), res


def bucket_sums_ok(r):
    pb = r["prior_buckets"]
    out = {}
    for head, size_key, mean_key in (("ahead", "curve_labels", "curve_log_loss"), ("instant", "instant_labels", "instant_log_loss")):
        bb, aa, bench = pb["benchmark"][head], pb["all"][head], r["benchmark"][head]
        out[f"{head}/bench_n"] = sum(bb["n"]) == bench["size"]
        out[f"{head}/bench_ll"] = abs(sum(bb["ll"]) / sum(bb["n"]) - bench["LogLoss"]) <= 5.1e-7
        out[f"{head}/bench_prior_1_127"] = bb["min_prior"] >= 1 and bb["max_prior"] <= 127
        out[f"{head}/all_n"] = sum(aa["n"]) == r[size_key]
        out[f"{head}/all_mean"] = abs(sum(aa["ll"]) / sum(aa["n"]) - r[mean_key]) <= 1e-4
        out[f"{head}/all_prior_ge_1"] = aa["min_prior"] >= 1
    return out


def k5():
    res, worst = {}, 0.0
    for path in (CPU / "k4-on.jsonl", SMOKE):
        for (u, lab), r in rows_of(path).items():
            if "prior_buckets" not in r:
                res[f"{path.name}/{u}/{lab}"] = False
                continue
            c = bucket_sums_ok(r)
            res[f"{path.name}/{u}/{lab}"] = all(c.values())
            for head in ("ahead", "instant"):
                bb = r["prior_buckets"]["benchmark"][head]
                worst = max(worst, abs(sum(bb["ll"]) / sum(bb["n"]) - r["benchmark"][head]["LogLoss"]))
    return all(res.values()) and len(res) >= 8 + 60, {"rows": len(res), "failed": [k for k, v in res.items() if not v],
                                                     "max_bench_ll_diff": worst}


def k6():
    R = rule()
    labels = [R.label(m, f) for m in R.MODELS for f in R.FORMATS]
    score(SMOKE, SMOKE_USERS, tuple(R.MODELS), labels, True, CPU / "smoke-users.json")
    rows = rows_of(SMOKE)
    res, pub_diff = {}, {}
    for u in SMOKE_USERS:
        for m, spec in R.MODELS.items():
            got = {f: rows.get((u, R.label(m, f))) for f in R.FORMATS}
            res[f"{u}/{m}/all_finite"] = all(R.finite(r, spec["heads"]) for r in got.values())
            if not res[f"{u}/{m}/all_finite"]:
                continue
            counts = {f: [r["prior_buckets"][s][h]["n"] for s in ("all", "benchmark") for h in ("ahead", "instant")]
                      for f, r in got.items()}
            res[f"{u}/{m}/counts_equal"] = all(c == counts["w8+s8"] for c in counts.values())
            res[f"{u}/{m}/int4_differs"] = all(got[f]["prior_buckets"]["all"] != got["w8+s8"]["prior_buckets"]["all"]
                                               for f in R.INT4)
            pub = None
            for line in open(spec["pub"], encoding="utf-8", errors="replace"):
                try:
                    p = json.loads(line)
                except ValueError:
                    continue
                if p.get("user_id") == u and p.get("label") == R.label(m, "w8+s8") and "benchmark" in p:
                    pub = p["benchmark"]
            d = max(abs(got["w8+s8"]["benchmark"][R.HKEY[h]]["LogLoss"] - pub[R.HKEY[h]]["LogLoss"]) for h in spec["heads"]) \
                if pub else None
            pub_diff[f"{u}/{m}"] = d
            res[f"{u}/{m}/w8s8_vs_published_1e-4"] = d is not None and d <= 1e-4
    return all(res.values()) and len(rows) == 60, {"failed": [k for k, v in res.items() if not v],
                                                   "w8s8_vs_published_max_abs": pub_diff, "rows": len(rows)}


def k7():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm136_rule.py"), "selftest"], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip().splitlines()[-1:]


def smoke():
    R = rule()
    cpu, gpu = rows_of(SMOKE), rows_of(HERE / "gpu/smoke-gpu-rows.jsonl")
    res, diffs = {}, {}
    for u in SMOKE_USERS:
        for m, spec in R.MODELS.items():
            pub = None
            for line in open(spec["pub"], encoding="utf-8", errors="replace"):
                try:
                    p = json.loads(line)
                except ValueError:
                    continue
                if p.get("user_id") == u and p.get("label") == R.label(m, "w8+s8") and "benchmark" in p:
                    pub = p["benchmark"]
            for f in R.FORMATS:
                k = (u, R.label(m, f))
                name = f"{u}/{m}/{f}"
                if k not in cpu or k not in gpu or not R.finite(gpu[k], spec["heads"]):
                    res[name] = False
                    continue
                c, g = cpu[k], gpu[k]
                same_n = all(c["prior_buckets"][s][h]["n"] == g["prior_buckets"][s][h]["n"]
                             for s in ("all", "benchmark") for h in ("ahead", "instant"))
                tol = 1e-5 if f == "w8+s8" else 1e-4
                d = max(abs(c["benchmark"][R.HKEY[h]]["LogLoss"] - g["benchmark"][R.HKEY[h]]["LogLoss"]) for h in spec["heads"])
                db = 0.0
                for h in spec["heads"]:
                    ca, ga = c["prior_buckets"]["all"][R.HKEY[h]], g["prior_buckets"]["all"][R.HKEY[h]]
                    for n, lc, lg in zip(ca["n"], ca["ll"], ga["ll"]):
                        if n >= 20:
                            db = max(db, abs(lc - lg) / n)
                ok = same_n and d <= tol and db <= 1e-4
                if f == "w8+s8":
                    dp = max(abs(g["benchmark"][R.HKEY[h]]["LogLoss"] - pub[R.HKEY[h]]["LogLoss"]) for h in spec["heads"]) if pub else 1
                    ok &= dp <= 1e-5
                    diffs[name + "/vs_published"] = dp
                diffs[name] = [d, db]
                res[name] = bool(ok)
    out = {"pass": all(res.values()) and len(res) == 60, "checks": res,
           "max_abs_diff_gpu_vs_cpu [benchmark LogLoss, all-scope bucket mean]": diffs}
    (HERE / "gpu/smoke-check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    worst = max((v[0] for k, v in diffs.items() if isinstance(v, list)), default=None)
    print(f"[sm136-smoke] {'PASS' if out['pass'] else 'FAIL'} ({sum(res.values())}/{len(res)} rows; worst benchmark diff {worst})")
    return 0 if out["pass"] else 1


def main():
    if sys.argv[1:] == ["smoke"]:
        return smoke()
    t0 = time.time()
    CPU.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (("K1_bytes", k1), ("K2_pool", k2), ("K3_formats", k3), ("K4_default_unchanged", k4),
                     ("K6_cpu_smoke_rows", k6), ("K5_bucket_sums", k5), ("K7_rule_selftest", k7)):
        t = time.time()
        try:
            ok, detail = fn()
        except Exception as error:   # a failed proof is a FAIL line, not a crash
            ok, detail = False, repr(error)[:600]
        out[name] = {"pass": bool(ok), "detail": detail, "seconds": round(time.time() - t, 1)}
        print(f"{'PASS' if ok else 'FAIL'} {name} ({out[name]['seconds']} s): {json.dumps(detail)[:500]}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "check-cpu-sm136.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm136-check] {'ALL OK' if out['pass'] else 'FAILED'} ({out['seconds']} s)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
