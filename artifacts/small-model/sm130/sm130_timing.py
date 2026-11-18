#!/usr/bin/env python
"""Six-layer Tiny (SM130): CPU latency of one review, Tiny (w64-c1) vs six-layer Tiny (w64-c1-d2p1u1), no installs.

The recurrent per-review step (`review()`: input encoder, the five streams, heads) of expertium.variants.build(arch,
rnn=True), float32 eager torch, batch 1 (the runtime case) and batch 256, 1 thread, timed with
scripts/small_model_cpu_timing.py's time_steps (SM04/SM18 protocol) in alternating order A B A B ... so drift hits both.
An indication of the layer-count effect only: the fork runs its own Rust engine (scripts/small_model_cpu_timing.py
docstring), so a native benchmark is still needed before any claim about the client.
Also: parameters, MACs per review (count_macs), float32 state-dict bytes.

  python sm130_timing.py --tiny CK --six CK [--rounds 6 --steps 300] --out sm130-timing.json
"""
import argparse
import io
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.small_model_cpu_timing import build_variant, count_macs, time_steps  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", required=True)
    p.add_argument("--six", required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default=str(HERE / "sm130-timing.json"))
    a = p.parse_args()
    torch.set_num_threads(1)
    models = {"tiny w64-c1": ("w64-c1", a.tiny), "six-layer w64-c1-d2p1u1": ("w64-c1-d2p1u1", a.six)}
    built, info = {}, {}
    for name, (arch, ck) in models.items():
        m, real = build_variant(arch, ck)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        built[name] = m
        info[name] = {"arch": arch, "checkpoint": ck, "real_weights": real, "parameters": sum(x.numel() for x in m.parameters()),
                      "fp32_state_dict_bytes": len(buf.getvalue()), "macs": count_macs(m), "batch1": [], "batch256": []}
    for _ in range(a.rounds):
        for name, m in built.items():
            info[name]["batch1"].append(time_steps(m, 1, a.steps, 1, 20)["step"]["median_ms"])
            info[name]["batch256"].append(time_steps(m, 256, max(20, a.steps // 10), 1, 3)["per_review_median_ms"])
    for name in built:
        for k in ("batch1", "batch256"):
            info[name][f"{k}_median_ms"] = statistics.median(info[name][k])
    t, s = (info[k] for k in models)
    out = {"threads": 1, "torch": torch.__version__, "models": info,
           "six_over_tiny_batch1": s["batch1_median_ms"] / t["batch1_median_ms"],
           "six_over_tiny_batch256": s["batch256_median_ms"] / t["batch256_median_ms"],
           "six_over_tiny_macs": s["macs"]["total"] / t["macs"]["total"]}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm130-timing] batch 1: tiny {t['batch1_median_ms']:.3f} ms, six-layer {s['batch1_median_ms']:.3f} ms "
          f"(x{out['six_over_tiny_batch1']:.3f}); batch 256 per review: {t['batch256_median_ms'] * 1e3:.1f} vs "
          f"{s['batch256_median_ms'] * 1e3:.1f} us; MACs x{out['six_over_tiny_macs']:.3f}; parameters {t['parameters']:,} vs {s['parameters']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
