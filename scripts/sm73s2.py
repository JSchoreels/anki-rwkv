#!/usr/bin/env python
"""Smaller model (SM73) stage 2 early check: the small model with its card stream cut from 2 layers to 1 (candidate B,
expertium.variants preset "w64-c1"). Design: artifacts/small-model/sm73/DESIGN.md, addendum
artifacts/small-model/sm73s2/DESIGN-ADDENDUM.md.

  cut --master M --optimizer O --out DIR [--arch w64-c1]
        DIR/master.pth = M without the tensors ARCH does not have (rwkv_modules.0.blocks.1.*, the card stream's second
        layer), every other tensor copied bit for bit; DIR/optimizer.pth = O's AdamW state re-indexed to ARCH's
        parameter order (removed parameters: their moments dropped); DIR/extend.json in the format
        `scripts/sm77_warm.py epoch --init DIR` reads (init "cut").
  cpu-check --out report.json [--work DIR]
        CPU proofs before any GPU ticket (~10-20 min, < 3 GB RAM):
        C0  the cut checkpoint: ARCH loads it strictly, the removed tensors are exactly one layer's, the rest bitwise;
            the AdamW state moved bitwise and loads into ARCH's optimizer; parameter count
        C1  layer steps: ARCH's 12 steps = w64's 13 minus (card, 1), same order
        C2  TorchScript: ARCH builds scripted (the trainer's mode) and eager (the scorer's)
        T1  trainer path (CatchUp loss + strided-lane state carry, train mode, seeded, 2 users of SM72s's epoch-21
            protocol): ARCH from the cut == w64 from SM72s epoch 20 with the layer step (card, 1) skipped: loss, outputs
            and every kept parameter's gradient bitwise; eager and scripted; plus the zero-shot change vs the full model
        S1  scorer path (small_model_heldout.py, CPU, fp32 and w8+s8, leak-free flags): ARCH from the cut == w64 from
            SM72s epoch 20 with the card layer's residual branch zeroed (W_o = 0): every metric and every dumped
            prediction bitwise; plus the full model's zero-shot rows
        M1  the Muon fresh-state check (scripts/check_muon_fresh_decay.py) passes on an ARCH cool-down layout
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sm77_warm as warm  # noqa: E402  (sets GPU_BATCH1_DATA_PATH before any scripts.gpu_batch1 import)

TRACK = warm.TRACK
RUN72S = warm.RUN72S
SM72S = warm.SM72S
CUT_PREFIX = "rwkv_modules.0.blocks.1."


def cut(args):
    import torch
    from expertium.variants import build
    old = torch.load(args.master, map_location="cpu", weights_only=True)
    base, model = build("w64").float(), build(args.arch).float()
    base.load_state_dict(old, strict=True)
    new_sd = model.state_dict()
    removed = [k for k in old if k not in new_sd]
    if set(new_sd) - set(old) or any(not k.startswith(CUT_PREFIX) for k in removed):
        raise SystemExit(f"{args.arch} is not w64 minus {CUT_PREFIX}*: {sorted(set(old) ^ set(new_sd))[:10]}")
    for key, value in new_sd.items():
        if value.shape != old[key].shape:
            raise SystemExit(f"{key}: shape {tuple(old[key].shape)} -> {tuple(value.shape)}")
    out_sd = {k: old[k].clone() for k in new_sd}
    model.load_state_dict(out_sd, strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out / "master.pth")
    record = {"source_master": str(args.master), "source_master_sha256": warm.sha256(args.master), "architecture": args.arch,
              "init": "cut", "removed_prefixes": [CUT_PREFIX + "*"],
              "removed": {k: list(old[k].shape) for k in removed}, "removed_values": sum(old[k].numel() for k in removed),
              "parameters": sum(p.numel() for p in model.parameters()), "master_sha256": warm.sha256(out / "master.pth")}
    state = torch.load(args.optimizer, map_location="cpu", weights_only=True)
    old_groups, _ = warm.adamw_names(base)
    new_groups, new_optimizer = warm.adamw_names(model)
    if [len(g) for g in old_groups] != [len(g["params"]) for g in state["param_groups"]]:
        raise SystemExit("the source optimizer.pth does not have the constant-LR AdamW group layout")
    old_index = {name: i for i, name in enumerate(n for g in old_groups for n in g)}
    flat_new = [n for g in new_groups for n in g]
    if [n for g in old_groups for n in g if not n.startswith(CUT_PREFIX)] != flat_new:
        raise SystemExit("the kept parameters do not keep their AdamW order")
    groups, start = [], 0
    for old_group, names in zip(state["param_groups"], new_groups):
        groups.append(dict(old_group, params=list(range(start, start + len(names)))))
        start += len(names)
    remapped = {"state": {i: state["state"][old_index[n]] for i, n in enumerate(flat_new) if old_index[n] in state["state"]},
                "param_groups": groups}
    new_optimizer.load_state_dict(remapped)   # refuses a malformed layout
    check = new_optimizer.state_dict()
    moved = all(torch.equal(check["state"][i][key], state["state"][old_index[n]][key])
                for i, n in enumerate(flat_new) if old_index[n] in state["state"] for key in state["state"][old_index[n]])
    if not moved:
        raise SystemExit("re-indexed AdamW state differs after loading")
    torch.save(remapped, out / "optimizer.pth")
    record.update(source_optimizer=str(args.optimizer), source_optimizer_sha256=warm.sha256(args.optimizer),
                  optimizer_sha256=warm.sha256(out / "optimizer.pth"), optimizer_states_moved=len(remapped["state"]),
                  optimizer_states_dropped=len(state["state"]) - len(remapped["state"]),
                  optimizer_group_sizes=[len(g) for g in new_groups], optimizer_states_equal_after_load=moved)
    warm.atomic_json(out / "extend.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "removed"}))
    return 0


# the sm77 trainer probe plus an optional skipped layer-step list (the "manual skip" of the full model)
PROBE = warm.PROBE.replace(
    'model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)\n',
    'model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)\n'
    'if spec.get("skip"):\n'
    '    model.layer_steps = [s for s in model.layer_steps if [int(s[0]), int(s[1])] not in spec["skip"]]\n'
    'print("STEPS " + json.dumps([list(s) for s in model.layer_steps]))\n', 1)
assert "STEPS" in PROBE


def run_probe(spec, tmp, jit=False):
    saved = warm.PROBE
    warm.PROBE = PROBE
    try:
        return warm.run_probe(ROOT, spec, tmp, jit=jit)
    finally:
        warm.PROBE = saved


def cpu_check(args):
    import torch
    from expertium.variants import build
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "arch": args.arch}
    e20 = RUN72S / "epoch-20" / "expertium" / TRACK
    ck = e20 / "master.pth"
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fails = []

    def check(name, ok, **detail):
        report["checks"][name] = dict(detail, ok=bool(ok))
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    # C0 the cut checkpoint (the one the launcher trains from)
    ext = Path(args.ext).resolve()
    if not (ext / "extend.json").exists():
        subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "cut", "--master", str(ck),
                        "--optimizer", str(e20 / "optimizer.pth"), "--arch", args.arch, "--out", str(ext)], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    rec = json.loads((ext / "extend.json").read_text())
    old = torch.load(ck, map_location="cpu", weights_only=True)
    new = torch.load(ext / "master.pth", map_location="cpu", weights_only=True)
    model = build(args.arch).float()
    model.load_state_dict(new, strict=True)
    kept_bitwise = all(torch.equal(new[k], old[k]) for k in new)
    removed = sorted(set(old) - set(new))
    check("C0_cut_checkpoint", kept_bitwise and all(k.startswith(CUT_PREFIX) for k in removed)
          and rec["master_sha256"] == warm.sha256(ext / "master.pth") and rec["source_master_sha256"] == warm.sha256(ck)
          and rec["optimizer_states_equal_after_load"],
          parameters_w64=sum(v.numel() for v in old.values()), parameters_cut=sum(p.numel() for p in model.parameters()),
          removed_tensors=len(removed), removed_values=rec["removed_values"], kept_tensors_bitwise=kept_bitwise,
          adamw_states_moved=rec["optimizer_states_moved"], adamw_states_dropped=rec["optimizer_states_dropped"],
          adamw_group_sizes=rec["optimizer_group_sizes"])
    # C1 layer steps
    full = build("w64")
    expect = [s for s in full.layer_steps if s != (0, 1)]
    check("C1_layer_steps", list(model.layer_steps) == expect and len(expect) == 12 and model.stream_depths[0] == 1
          and model.stream_depths[1:] == full.stream_depths[1:],
          w64=[list(s) for s in full.layer_steps], cut=[list(s) for s in model.layer_steps],
          depths_w64=list(full.stream_depths), depths_cut=list(model.stream_depths))
    # C2 TorchScript and eager builds (subprocesses: RWKV_NO_JIT is read at import)
    builds = {}
    for mode, env_jit in (("scripted", None), ("eager", "1")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="")
        env.pop("RWKV_NO_JIT", None)
        if env_jit:
            env["RWKV_NO_JIT"] = env_jit
        code = ("import sys, torch; sys.path.insert(0, r'%s'); from expertium.variants import build; m = build('%s'); "
                "m.load_state_dict(torch.load(r'%s', map_location='cpu', weights_only=True), strict=True); "
                "print('TYPE', type(m).__name__, isinstance(m, torch.jit.ScriptModule))") % (ROOT, args.arch, ext / "master.pth")
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
        builds[mode] = (r.returncode, next((x for x in r.stdout.splitlines() if x.startswith("TYPE")), r.stderr[-400:]))
    check("C2_torchscript_and_eager_build", builds["scripted"][0] == 0 and builds["scripted"][1].endswith("True")
          and builds["eager"][0] == 0, builds={k: list(v) for k, v in builds.items()})

    shared = list(dict(model.named_parameters()))
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
    rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
    config = json.loads((RUN72S / "run.json").read_text())
    options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    report["trainer_users"] = {str(r["user_id"]): r["train_range"][1] for r in rows}
    keys = ("user", "offset", "lanes", "loss_hex", "p_digest", "grad_digest")
    with tempfile.TemporaryDirectory() as tmp:
        # T1 trainer path: cut == full with (card, 1) skipped (eager and scripted); zero-shot change vs the full model
        base = {"options": options, "rows": rows, "span": 256, "shared": shared}
        for mode, jit in (("eager", False), ("scripted", True)):
            arm = run_probe(dict(base, arch=args.arch, checkpoint=str(ext / "master.pth")), tmp, jit=jit)
            skip = run_probe(dict(base, arch="w64", checkpoint=str(ck), skip=[[0, 1]]), tmp, jit=jit)
            fullp = run_probe(dict(base, arch="w64", checkpoint=str(ck)), tmp, jit=jit) if not jit else None
            same = [{k: u[k] for k in keys} for u in arm] == [{k: u[k] for k in keys} for u in skip]
            detail = {"updates": len(arm), "loss_outputs_gradients_bitwise": same,
                      "loss_cut": [u["loss"] for u in arm], "loss_skip": [u["loss"] for u in skip]}
            if fullp:
                detail["loss_full_w64"] = [u["loss"] for u in fullp]
                detail["zero_shot_loss_change"] = [round(a["loss"] - f["loss"], 6) for a, f in zip(arm, fullp)]
            check(f"T1_trainer_cut_equals_skipped_layer_{mode}", same and len(arm) > len(rows), **detail)
        # S1 scorer path: cut == full with the card layer's residual branch zeroed (W_o = 0), fp32 and w8+s8
        zero = {k: v.clone() for k, v in old.items()}
        zero[CUT_PREFIX + "time_mixer.W_o.weight"].zero_()
        zck = Path(tmp) / "w64-card1-wo0.pth"
        torch.save(zero, zck)
        users_file = SM72S / "artifacts/small-model/sm72s/eval-users-732-fit.json"
        dump = Path(tmp) / "dump"
        srows = warm.run_scorer(ROOT, {"a": f"expertium:w64:{zck}", "b": f"expertium:{args.arch}:{ext / 'master.pth'}",
                                       "c": f"expertium:w64:{ck}"}, users_file, Path(tmp) / "s.jsonl", dump,
                                args.max_reviews, 2)
        by = {(r["label"].split("/")[0], r["user_id"], r["variant"]): r for r in srows if "variant" in r}
        pairs = []
        for (tag, user, variant), r in by.items():
            if tag == "a" and ("b", user, variant) in by:
                x, y = warm.metric_view(r), warm.metric_view(by[("b", user, variant)])
                x.pop("kind", None); y.pop("kind", None)
                pairs.append((user, variant, x == y, r, by[("b", user, variant)]))
        same_dumps, n = warm.dumps_equal(dump, dump, "a", "b")
        zs = {f"{u}/{v}": {h: round(by[("b", u, v)][f"{h}_log_loss"] - by[("c", u, v)][f"{h}_log_loss"], 6)
                           for h in ("curve", "instant") if f"{h}_log_loss" in by[("b", u, v)]}
              for (u, v, _, _, _) in pairs if ("c", u, v) in by}
        check("S1_scorer_cut_equals_zeroed_branch", len(pairs) == 4 and all(p[2] for p in pairs) and same_dumps,
              rows=len(pairs), dumps=n, users=sorted({p[0] for p in pairs}), reviews=sorted({p[3].get("reviews") for p in pairs}),
              variants=sorted({p[1] for p in pairs}), metrics_equal=[p[2] for p in pairs],
              weight_bytes={"w64": {p[1]: by.get(("c", p[0], p[1]), {}).get("weight_bytes") for p in pairs},
                            args.arch: {p[1]: p[4].get("weight_bytes") for p in pairs}},
              parameters={"w64": next((r.get("parameters") for k, r in by.items() if k[0] == "c"), None),
                          args.arch: next((r.get("parameters") for k, r in by.items() if k[0] == "b"), None)},
              zero_shot_logloss_change_vs_full=zs)
        # M1 the Muon fresh-state check on an ARCH cool-down layout (start = the cut, source AdamW = SM72s epoch 21)
        dec = Path(tmp) / "muon-layout"
        dec.mkdir()
        extra = json.loads(args.extra_options)
        opts = dict(config["options"], **extra, architecture=args.arch)
        (dec / "decay.json").write_text(json.dumps({"options": opts, "source_case": str(RUN72S / "epoch-21" / "expertium" / TRACK)}))
        (dec / "protocol.json").write_text(json.dumps({"full_run": {"init_master": str(ext / "master.pth"), "optimizer_state": "fresh"}}))
        r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/check_muon_fresh_decay.py"), "--decay", str(dec),
                            "--out", str(work / "check-muon-layout.json")], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        mj = json.loads((work / "check-muon-layout.json").read_text()) if (work / "check-muon-layout.json").exists() else {}
        check("M1_muon_fresh_check_on_cut_layout", r.returncode == 0 and mj.get("passed"),
              split=mj.get("checks", {}).get("w64 parameters Muon / AdamW", {}).get("value"), fails=mj.get("fails"),
              tail=r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-300:])
    report["fails"] = fails
    report["ok"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    warm.atomic_json(args.out, report)
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cut")
    p.add_argument("--master", required=True)
    p.add_argument("--optimizer", required=True)
    p.add_argument("--arch", default="w64-c1")
    p.add_argument("--out", required=True)
    p = sub.add_parser("cpu-check")
    p.add_argument("--arch", default="w64-c1")
    p.add_argument("--ext", default=str(ROOT / "artifacts/small-model/sm73s2-cut-e20"))
    p.add_argument("--out", default=str(ROOT / "artifacts/small-model/sm73s2/check-cpu-sm73s2.json"))
    p.add_argument("--work", default=str(ROOT / "artifacts/small-model/sm73s2/work"))
    p.add_argument("--max-reviews", type=int, default=4200)
    p.add_argument("--extra-options", default=json.dumps({
        "state_carry": {"lanes": "strided"}, "chunk_reviews": 32768, "batch_chunks": 1, "optimizer": "expertium-iter53"}))
    args = parser.parse_args()
    return {"cut": cut, "cpu-check": cpu_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
