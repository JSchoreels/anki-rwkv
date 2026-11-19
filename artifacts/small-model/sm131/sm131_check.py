#!/usr/bin/env python
"""Compact state for Small and Big (SM131): CPU proofs (seconds; run with CUDA_VISIBLE_DEVICES= ).

K1 bytes: sm131_bytes.py ALL OK (Big layout 4 heads x 32, 3 card / 2 note layers; round trip bit for bit against the
   scorer's own quantizer; known totals Big 9,271 / 10,283 / 15,415 B, Small and Tiny as in SM129).
K2 helpers: the patched fork model's _quantize_rows / _round_shift (scripts/compress_quant.py, SM131 change) are bit for
   bit the old formulas for plain 8 / 4 and float16 shifts, and bit for bit scripts/state_vq.int_row_quant for the
   flagged bits (8|F16, 4|F16, 4|F16|HEAD on [B, H, 32, 32]; shift 8|F16, 4|F16 on [B, T, 128]); random states with
   zero rows, tiny (float16-subnormal scales) and huge (clamped scales) values.
K3 streams: parse_variant + fork_streams map the four formats to the intended (card, note) bits; set_state_streams sets
   modules 0 (card) and 2 (note) only; set_state_bits resets the shift bits (a model is reused across formats).
K4 CPU smoke rows (cpu/smoke-cpu-rows.jsonl, 2 users, the show-time big model (SM91s2), four formats, scored on CPU by
   this tree): w8+s8 equals the published GPU rows within 2e-6 per head; every format finite; compact != fork default.
K5 rule selftest (sm131_rule.py).
-> check-cpu-sm131.json, exit 0 iff all pass.

`sm131_check.py smoke` (after the GPU smoke ticket step): gpu/smoke-gpu-rows.jsonl (the same 2 users and four formats,
scored on the GPU) vs the CPU rows: w8+s8 within 1e-5 per head of the CPU rows and of the published rows, every other
format within 1e-4 per head of its CPU row (the GPU kernel's float16-scale / per-head flags vs the CPU reference;
summation order differs, so a rare rounding flip is allowed up to 1e-4) -> gpu/smoke-check.json, exit 0 iff pass.
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
PUB91 = Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl")
SMOKE = HERE / "cpu/smoke-cpu-rows.jsonl"
TAG = "sm91s2-big-showtime"
FORMATS = ("w8+s8", "w8+s4c8n", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n")


def k1():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm131_bytes.py")], capture_output=True, text=True)
    return p.returncode == 0 and "ALL OK" in p.stdout, p.stdout.strip().splitlines()[-1:]


def k2():
    import torch
    from scripts import compress_quant as cq
    mod = cq.install_state_quant_model()
    from scripts.small_model_wkv import F16_SCALE, HEAD_SCALE
    from scripts.state_vq import int_row_quant

    def old_rows(x, bits):
        qmax = 127.0 if bits == 8 else 7.0
        scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    gen = torch.Generator().manual_seed(0)
    res = {}
    for kind, s in (("normal", 1.0), ("tiny", 1e-6), ("huge", 1e7)):
        m = torch.randn(3, 4, 32, 32, generator=gen) * s
        m[0, 1, 5] = 0.0
        m[1, 2] *= 50.0
        v = torch.randn(2, 7, 128, generator=gen) * min(s, 1e4)
        v[0, 3] = 0.0
        for bits in (8, 4):
            res[f"{kind}/plain{bits}"] = torch.equal(mod._quantize_rows(m, bits), old_rows(m, bits))
        res[f"{kind}/shift0"] = torch.equal(mod._round_shift(v, 0), v.half().float())
        for bits in (8 | F16_SCALE, 4 | F16_SCALE, 4 | F16_SCALE | HEAD_SCALE):
            res[f"{kind}/wkv{bits}"] = torch.equal(mod._quantize_rows(m, bits), int_row_quant(bits)(m))
        for bits in (8 | F16_SCALE, 4 | F16_SCALE):
            res[f"{kind}/shift{bits}"] = torch.equal(mod._round_shift(v, bits), int_row_quant(bits)(v))
    return all(res.values()), res


def k3():
    from scripts import compress_quant as cq
    from scripts import small_model_heldout as smh
    from scripts.small_model_wkv import F16_SCALE as F, HEAD_SCALE as HS
    want = {"w8+s4c8n": ((4, 0), (8, 0)), "w8+s4c8nhx8": ((4 | F, 8 | F), (8 | F, 8 | F)),
            "w8+s4hgx4c8hx8n": ((4 | F | HS, 4 | F), (8 | F, 8 | F))}
    res = {}
    for name, w in want.items():
        _, bits = smh.parse_variant(name)
        res[f"map/{name}"] = isinstance(bits, tuple) and smh.fork_streams(bits) == w
    _, b8 = smh.parse_variant("w8+s8")
    res["map/w8+s8_plain"] = b8 == 8
    model = cq.load_model(path=Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"))

    def got(model):
        return [[(b.time_mixer.state_bits, b.time_mixer.shift_bits, b.channel_mixer.state_bits, b.channel_mixer.shift_bits)
                 for b in mo.blocks] for mo in model.rwkv_modules]
    cq.set_state_streams(model, *want["w8+s4hgx4c8hx8n"])
    g = got(model)
    res["streams/card_note_only"] = (all(t == (4 | F | HS, 4 | F, 4 | F | HS, 4 | F) for t in g[0])
                                     and all(t == (8 | F, 8 | F, 8 | F, 8 | F) for t in g[2])
                                     and all(t == (0, 0, 0, 0) for i in (1, 3, 4) for t in g[i])
                                     and [len(x) for x in g] == [3, 4, 2, 3, 4])
    cq.set_state_bits(model, 8)
    g = got(model)
    res["set_state_bits_resets_shift"] = all(t == (8, 0, 8, 0) for i in (0, 2) for t in g[i]) and \
        all(t == (0, 0, 0, 0) for i in (1, 3, 4) for t in g[i])
    return all(res.values()), res


def k4():
    pub = {}
    for line in open(PUB91, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") == f"{TAG}/w8+s8@e2s-query" and r["user_id"] in (2388, 4676):
            pub[r["user_id"]] = r["benchmark"]
    rows = {}
    for line in open(SMOKE, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["user_id"], r["label"].split("/", 1)[1].split("@")[0])] = r["benchmark"]
    res = {}
    for u in (2388, 4676):
        res[f"{u}/w8+s8_vs_published"] = all(abs(rows[(u, "w8+s8")][h]["LogLoss"] - pub[u][h]["LogLoss"]) <= 2e-6
                                              for h in ("ahead", "instant"))
        res[f"{u}/all_formats_finite"] = all((u, f) in rows and all(rows[(u, f)][h]["LogLoss"] == rows[(u, f)][h]["LogLoss"]
                                                                    for h in ("ahead", "instant")) for f in FORMATS)
        res[f"{u}/compact_differs_from_fork"] = rows[(u, "w8+s4c8nhx8")] != rows[(u, "w8+s4c8n")]
    return all(res.values()), res


def k5():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm131_rule.py"), "selftest"], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip().splitlines()[-1:]


def smoke():
    def rows_of(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "benchmark" in r:
                out[(r["user_id"], r["label"].split("/", 1)[1].split("@")[0])] = r["benchmark"]
        return out
    cpu, gpu = rows_of(SMOKE), rows_of(HERE / "gpu/smoke-gpu-rows.jsonl")
    pub = {}
    for line in open(PUB91, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") == f"{TAG}/w8+s8@e2s-query" and r["user_id"] in (2388, 4676):
            pub[(r["user_id"], "w8+s8")] = r["benchmark"]
    res, diffs = {}, {}
    for u in (2388, 4676):
        for f in FORMATS:
            key = (u, f)
            if key not in gpu:
                res[f"{u}/{f}"] = False
                continue
            tol = 1e-5 if f == "w8+s8" else 1e-4
            d = max(abs(gpu[key][h]["LogLoss"] - cpu[key][h]["LogLoss"]) for h in ("ahead", "instant"))
            diffs[f"{u}/{f}"] = d
            ok = d <= tol
            if f == "w8+s8":
                ok &= max(abs(gpu[key][h]["LogLoss"] - pub[key][h]["LogLoss"]) for h in ("ahead", "instant")) <= 1e-5
            res[f"{u}/{f}"] = bool(ok)
    out = {"pass": all(res.values()), "checks": res, "max_abs_logloss_diff_gpu_vs_cpu": diffs}
    (HERE / "gpu/smoke-check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm131-smoke] {'PASS' if out['pass'] else 'FAIL'} {json.dumps(diffs)}")
    return 0 if out["pass"] else 1


def main():
    if sys.argv[1:] == ["smoke"]:
        return smoke()
    t0 = time.time()
    out = {}
    for name, fn in (("K1_bytes", k1), ("K2_helpers", k2), ("K3_streams", k3), ("K4_cpu_smoke_rows", k4), ("K5_rule_selftest", k5)):
        ok, detail = fn()
        out[name] = {"pass": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail)[:400]}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "check-cpu-sm131.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm131-check] {'ALL OK' if out['pass'] else 'FAILED'} ({out['seconds']} s)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
