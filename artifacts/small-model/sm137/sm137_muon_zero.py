#!/usr/bin/env python
"""Trained smaller state for Tiny (SM137): CPU proof that the cool-down's fresh Muon optimizer keeps the dropped W_k / W_v
rows of the lane cut exactly 0 (the constant-LR epochs used AdamW; the cool-down switches to Muon, whose Newton-Schulz
orthogonalisation mixes the rows of a matrix).

  python sm137_muon_zero.py --decay <pass-1 folder> --out report.json [--steps 5]

The optimizer is the one the trainer builds for the folder (CatchUp(options, full_run).optimizer(master), as
scripts/check_muon_fresh_decay.py), on the folder's start weights. Every step gets a random gradient (seeded) whose dropped
rows are 0, as the lane mask makes them (proof L2), at the peak learning rate 7e-4. Pass iff after every step the dropped
rows are exactly 0 and the kept rows changed. Why it must hold: Muon's update of a matrix G is a polynomial
a X + (b A + c A A) X with A = X X^T and X = G / |G|; a zero row of G gives a zero row of A, of A A and of every product,
and weight decay scales a zero row by (1 - lr wd). CPU only.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
torch.set_num_threads(2)


def main():
    p = ArgumentParser()
    p.add_argument("--decay", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()
    from expertium.model.variant_model import VariantSrsRWKV
    from expertium.variants import STATE_LANES, variant
    from scripts.expertium_options import CatchUp
    from sm137_lanes import H, K, STREAMS, dropped_rows_zero
    decay = json.loads((args.decay / "decay.json").read_text())
    protocol = json.loads((args.decay / "protocol.json").read_text())
    options, full_run = decay["options"], protocol["full_run"]
    m = STATE_LANES.match(options["architecture"])
    v_keep, k_keep = int(m.group(2)), int(m.group(3))
    master = VariantSrsRWKV(variant(options["architecture"])).float()
    master.load_state_dict(torch.load(full_run["init_master"], map_location="cpu", weights_only=True), strict=True)
    optimizer = CatchUp(options, full_run, [{"train_range": [0, 1000]}]).optimizer(master)
    names = {id(q): n for n, q in master.named_parameters()}
    in_muon = sorted(names[id(q)] for g in optimizer.muon.param_groups for q in g["params"])
    lane = torch.arange(H * K) % K
    dropped = {f"rwkv_modules.{i}.blocks.0.time_mixer.{n}": keep for i in STREAMS for n, keep in
               (("W_k.weight", k_keep), ("W_v.weight", v_keep))}
    params = dict(master.named_parameters())
    gen = torch.Generator().manual_seed(137)
    start = {n: params[n].detach().clone() for n in dropped}
    steps = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for n, q in master.named_parameters():
            g = torch.randn(q.shape, generator=gen)
            if n in dropped:
                g[lane >= dropped[n]] = 0
            q.grad = g
        for group in optimizer.param_groups:
            group["lr"] = 7e-4
        optimizer.step()
        bad = dropped_rows_zero(master.state_dict(), v_keep, k_keep)
        moved = all(bool((params[n].detach()[lane < keep] != start[n][lane < keep]).any()) for n, keep in dropped.items())
        steps.append({"step": step + 1, "dropped_rows_zero": not bad, "kept_rows_changed": moved, "bad": bad[:4]})
        print(json.dumps(steps[-1]))
    ok = all(s["dropped_rows_zero"] and s["kept_rows_changed"] for s in steps) and all(n in in_muon for n in dropped)
    report = {"decay": str(args.decay), "architecture": options["architecture"], "optimizer": options.get("optimizer"),
              "dropped_tensors_in_muon": {n: n in in_muon for n in dropped}, "steps": steps, "passed": ok}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[sm137-muon-zero] {'PASS' if ok else 'FAIL'}: {args.steps} Muon steps, dropped W_k / W_v rows exactly 0, kept rows moved")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
