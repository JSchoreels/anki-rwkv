#!/usr/bin/env python
"""CPU check of a Muon decay prepared with `expertium_decay.py prepare --fresh-optimizer` (SM36).

  check_muon_fresh_decay.py --decay DIR --out report.json

1. protocol.json has no `init_optimizer` (fresh state) and decay.json options hold "optimizer": "expertium-iter53";
2. the optimizer the trainer would build (CatchUp.optimizer -> make_optimizer) has 309,888 Muon + 21,623 AdamW parameters
   for the w64 student, an EMPTY state, and exactly the groups the written learning rate reaches;
3. loading the source epoch's AdamW optimizer.pth into it fails (so the fresh start is required, not cosmetic);
4. one CPU step on the source epoch's weights is finite, changes the weights and restores the written learning rate;
5. the optimizer state round-trips through state_dict (the decay saves and resumes optimizer.pth).
Everything runs on CPU, never touches the source files except for reading them.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)


def main():
    parser = ArgumentParser()
    parser.add_argument("--decay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decay = json.loads((args.decay / "decay.json").read_text())
    protocol = json.loads((args.decay / "protocol.json").read_text())
    options, full_run = decay["options"], protocol["full_run"]
    checks = {}

    def record(name, value, ok):
        checks[name] = {"value": value, "ok": bool(ok)}
        print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(value))

    record("protocol has no init_optimizer", sorted(full_run), "init_optimizer" not in full_run and full_run.get("optimizer_state") == "fresh")
    record("options optimizer", options.get("optimizer"), options.get("optimizer") == "expertium-iter53")
    from expertium.model.variant_model import VariantSrsRWKV
    from expertium.variants import variant
    from scripts.expertium_options import CatchUp
    master = VariantSrsRWKV(variant(options["architecture"])).float()
    master.load_state_dict(torch.load(full_run["init_master"], map_location="cpu", weights_only=True), strict=True)
    catch_up = CatchUp(options, full_run, [{"train_range": [0, 1000]}])
    optimizer = catch_up.optimizer(master)
    muon = sum(p.numel() for g in optimizer.muon.param_groups for p in g["params"])
    adamw = sum(p.numel() for g in optimizer.adamw.param_groups for p in g["params"])
    # SM77 additions (zero-init biases and initial state, names sm77_*) are not matrices: AdamW, Muon set unchanged
    extra = sum(p.numel() for n, p in master.named_parameters() if n.split(".")[-1].startswith("sm77_"))
    record("w64 parameters Muon / AdamW", [muon, adamw], [muon, adamw] == [309888, 21623 + extra])
    record("fresh state is empty", [len(optimizer.muon.state), len(optimizer.adamw.state)],
           len(optimizer.muon.state) == 0 and len(optimizer.adamw.state) == 0)
    source = Path(decay["source_case"]) / "optimizer.pth"
    try:
        optimizer.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
        record("source AdamW optimizer.pth cannot be loaded into Muon", "loaded without error", False)
    except Exception as error:  # the expected outcome
        record("source AdamW optimizer.pth cannot be loaded into Muon", f"{type(error).__name__}: {str(error)[:120]}", True)
    optimizer = catch_up.optimizer(master)
    before = [p.detach().clone() for p in master.parameters()]
    lr = 7e-4 * 0.5
    master.zero_grad(set_to_none=True)
    loss = sum((p ** 2).sum() for p in master.parameters())
    loss.backward()
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    finite = all(torch.isfinite(p).all() for p in master.parameters())
    changed = sum(int((a != p.detach()).any()) for a, p in zip(before, master.parameters()))
    record("one step finite, parameters changed, lr restored", [bool(finite), changed, len(before), sorted({g["lr"] for g in optimizer.param_groups})],
           finite and changed > 0 and {g["lr"] for g in optimizer.param_groups} == {lr})
    state = optimizer.state_dict()
    again = catch_up.optimizer(master)
    again.load_state_dict(state)
    record("state round trip (muon, adamw entries)", [len(again.muon.state), len(again.adamw.state)],
           len(again.muon.state) > 0 and len(again.adamw.state) > 0)
    report = {"decay": str(args.decay), "checks": checks, "passed": all(c["ok"] for c in checks.values()),
              "fails": [k for k, c in checks.items() if not c["ok"]]}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("passed" if report["passed"] else "FAILED " + str(report["fails"]))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
