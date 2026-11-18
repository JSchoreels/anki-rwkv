#!/usr/bin/env python
"""Six-layer Tiny (SM130): the context-stream depth cut of the context-stream pruning (SM15) applied to Tiny (w64-c1),
design artifacts/small-model/sm130/design.md. CPU only.

  cut --master M --optimizer O --out DIR [--src-arch w64-c1 --arch w64-c1-d2p1u1]
        DIR/master.pth = M without the tensors ARCH does not have (the trailing deck / preset / user layers:
        rwkv_modules.1.blocks.2-3.*, rwkv_modules.3.blocks.1-2.*, rwkv_modules.4.blocks.1-2.*), every other tensor copied
        bit for bit; DIR/optimizer.pth = O's AdamW state re-indexed to ARCH's parameter order (removed parameters: their
        moments dropped); DIR/extend.json (init "cut"), the format scripts/sm73s2.py cut writes.
  quick                       seconds (allowed in a break): parameter counts, layer steps, strict load of both cuts,
                              one random forward per cut finite (eager, eval)
  cpu-check --out report.json   the CPU proofs before any SM130 GPU ticket (~5-20 min, < 3 GB RAM; the launcher runs it
                              under memguard.ps1, 5.9 GB kill):
        C0  both cut checkpoints (two-head from the Tiny two-head model (SM106)'s epoch 29, Instant-Only from the Tiny
            Instant-Only model (SM101)'s epoch 38): ARCH loads strictly, the removed tensors are exactly the dropped
            layers', the rest bitwise the source's; AdamW state moved bitwise and reloads; 185,057 parameters
        C1  layer steps: ARCH's 6 = w64-c1's 12 minus the dropped (stream, layer) steps, same order
        C2  TorchScript (the trainer's mode) and eager (the scorer's) builds
        T1  trainer path (CatchUp loss + strided-lane state carry, train mode, seeded, 2 users of SM72s's epoch-21
            protocol): ARCH from the two-head cut == w64-c1 from SM106 epoch 29 with the dropped layer steps skipped:
            loss, outputs and every kept parameter's gradient bitwise; eager and scripted
        S1  scorer path (small_model_heldout.py, CPU, fp32 and w8+s8, leak-free flags, 2 users): ARCH from the two-head
            cut == w64-c1 with the dropped layers' residual branches zeroed (time mixer W_o = 0 and channel mixer W_v = 0):
            every metric and every dumped prediction bitwise; plus the zero-shot change vs uncut w64-c1
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts import sm77_warm as warm  # noqa: E402
from scripts import sm73s2  # noqa: E402

TRACK = warm.TRACK
RUN72S = warm.RUN72S
SM72S = warm.SM72S
SRC_ARCH, ARCH = "w64-c1", "w64-c1-d2p1u1"
PARAMETERS = 185_057
W = Path("C:/Users/jonat/workspace/wt")
SOURCES = {   # read only
    "th": {"case": W / "sm106/artifacts/small-model/sm106-c1-e29-st/expertium" / TRACK,
           "master_sha256": "b3a694ca14cbd7a2291c7400203d5f9774bbdabc5d3e572a8408a196e5cb7be9",
           "out": ROOT / "artifacts/small-model/sm130-th-cut-e29"},
    "io": {"case": W / "sm101/artifacts/small-model/sm101-c1-e38-st/expertium" / TRACK,
           "master_sha256": "e6ea69124af00a3dcf2325047ae34580aaf96051d963ff6130dd06bde02d396c",
           "out": ROOT / "artifacts/small-model/sm130-io-cut-e38"},
}
STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]


def depths(arch):
    from expertium.variants import variant
    return [dict(variant(arch).base.modules)[s].n_layers for s in STREAMS]


def dropped_steps(src_arch=SRC_ARCH, arch=ARCH):
    a, b = depths(src_arch), depths(arch)
    return [[s, k] for s in range(len(STREAMS)) for k in range(b[s], a[s])]


def prefixes(src_arch=SRC_ARCH, arch=ARCH):
    return [f"rwkv_modules.{s}.blocks.{k}." for s, k in dropped_steps(src_arch, arch)]


def cut(args):
    import torch
    from expertium.variants import build
    old = torch.load(args.master, map_location="cpu", weights_only=True)
    base, model = build(args.src_arch).float(), build(args.arch).float()
    base.load_state_dict(old, strict=True)
    new_sd = model.state_dict()
    pre = prefixes(args.src_arch, args.arch)
    removed = [k for k in old if k not in new_sd]
    if set(new_sd) - set(old) or any(not any(k.startswith(p) for p in pre) for k in removed) or \
            any(not any(k.startswith(p) for k in removed) for p in pre):
        raise SystemExit(f"{args.arch} is not {args.src_arch} minus {pre}: {sorted(set(old) ^ set(new_sd))[:10]}")
    for key, value in new_sd.items():
        if value.shape != old[key].shape:
            raise SystemExit(f"{key}: shape {tuple(old[key].shape)} -> {tuple(value.shape)}")
    out_sd = {k: old[k].clone() for k in new_sd}
    model.load_state_dict(out_sd, strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out / "master.pth")
    record = {"source_master": str(args.master), "source_master_sha256": warm.sha256(args.master), "architecture": args.arch,
              "source_architecture": args.src_arch, "init": "cut", "removed_prefixes": [p + "*" for p in pre],
              "removed": {k: list(old[k].shape) for k in removed}, "removed_values": sum(old[k].numel() for k in removed),
              "parameters": sum(p.numel() for p in model.parameters()), "master_sha256": warm.sha256(out / "master.pth")}
    state = torch.load(args.optimizer, map_location="cpu", weights_only=True)
    old_groups, _ = warm.adamw_names(base)
    new_groups, new_optimizer = warm.adamw_names(model)
    if [len(g) for g in old_groups] != [len(g["params"]) for g in state["param_groups"]]:
        raise SystemExit("the source optimizer.pth does not have the constant-LR AdamW group layout")
    old_index = {name: i for i, name in enumerate(n for g in old_groups for n in g)}
    flat_new = [n for g in new_groups for n in g]
    if [n for g in old_groups for n in g if not any(n.startswith(p) for p in pre)] != flat_new:
        raise SystemExit("the kept parameters do not keep their AdamW order")
    groups, start = [], 0
    for old_group, names in zip(state["param_groups"], new_groups):
        groups.append(dict(old_group, params=list(range(start, start + len(names)))))
        start += len(names)
    remapped = {"state": {i: state["state"][old_index[n]] for i, n in enumerate(flat_new) if old_index[n] in state["state"]},
                "param_groups": groups}
    new_optimizer.load_state_dict(remapped)
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


def cut_ok(which):
    """(ok, detail) for the pinned cut of chain `which` (th / io)."""
    import torch
    from expertium.variants import build
    src = SOURCES[which]
    out = src["out"]
    rec = json.loads((out / "extend.json").read_text())
    old = torch.load(src["case"] / "master.pth", map_location="cpu", weights_only=True)
    new = torch.load(out / "master.pth", map_location="cpu", weights_only=True)
    model = build(ARCH).float()
    model.load_state_dict(new, strict=True)
    pre = prefixes()
    removed = sorted(set(old) - set(new))
    ok = (all(torch.equal(new[k], old[k]) for k in new) and all(any(k.startswith(p) for p in pre) for k in removed)
          and rec["master_sha256"] == warm.sha256(out / "master.pth") and rec["source_master_sha256"] == src["master_sha256"]
          and warm.sha256(src["case"] / "master.pth") == src["master_sha256"]
          and rec["optimizer_sha256"] == warm.sha256(out / "optimizer.pth")
          and rec["optimizer_states_equal_after_load"] and sum(p.numel() for p in model.parameters()) == PARAMETERS
          and rec["architecture"] == ARCH)
    return ok, {"parameters_source": sum(v.numel() for v in old.values()), "parameters_cut": sum(p.numel() for p in model.parameters()),
                "removed_tensors": len(removed), "removed_values": rec["removed_values"], "removed_prefixes": rec["removed_prefixes"],
                "adamw_states_moved": rec["optimizer_states_moved"], "adamw_states_dropped": rec["optimizer_states_dropped"],
                "master_sha256": rec["master_sha256"], "optimizer_sha256": rec["optimizer_sha256"]}


def quick(_args):
    import torch
    from expertium.variants import build
    t0 = time.perf_counter()
    out = {}
    for arch in ("w64", SRC_ARCH, ARCH):
        m = build(arch)
        out[arch] = {"parameters": sum(p.numel() for p in m.parameters()), "depths": depths(arch),
                     "layer_steps": [[int(x) for x in s] for s in m.layer_steps]}
    src_steps = [s for s in out[SRC_ARCH]["layer_steps"] if s not in dropped_steps()]
    ok = out[ARCH]["parameters"] == PARAMETERS and out[ARCH]["layer_steps"] == src_steps and len(src_steps) == 6
    for which in SOURCES:
        good, detail = cut_ok(which)
        out[f"cut_{which}"] = dict(detail, ok=good)
        ok &= good
    # one trainer-path forward + backward of each cut (eager, CPU, 1 short user of SM72s's epoch-21 protocol): finite loss
    import math
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    row = min((r for r in protocol["rows"] if r["train_range"][1] >= 200), key=lambda r: r["train_range"][1])
    config = json.loads((RUN72S / "run.json").read_text())
    options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    with tempfile.TemporaryDirectory() as tmp:
        for which in SOURCES:
            probe = sm73s2.run_probe({"options": options, "rows": [row], "span": 256, "arch": ARCH, "shared": [],
                                      "checkpoint": str(SOURCES[which]["out"] / "master.pth")}, tmp)
            losses = [u["loss"] for u in probe]
            out[f"forward_{which}"] = {"user": row["user_id"], "reviews": row["train_range"][1], "losses": losses}
            ok &= bool(losses) and all(math.isfinite(x) for x in losses)
    print(json.dumps(out))
    print(f"[sm130-cut] quick {'OK' if ok else 'FAILED'} ({time.perf_counter() - t0:.1f} s)")
    return 0 if ok else 1


def cpu_check(args):
    import torch
    from expertium.variants import build
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "arch": ARCH, "src_arch": SRC_ARCH}
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fails = []

    def check(name, ok, **detail):
        report["checks"][name] = dict(detail, ok=bool(ok))
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    for which in SOURCES:
        good, detail = cut_ok(which)
        check(f"C0_cut_checkpoint_{which}", good, **detail)
    th = SOURCES["th"]
    ck, ext = th["case"] / "master.pth", th["out"]
    model, full = build(ARCH).float(), build(SRC_ARCH)
    drop = dropped_steps()
    expect = [s for s in full.layer_steps if [int(s[0]), int(s[1])] not in drop]
    check("C1_layer_steps", [list(map(int, s)) for s in model.layer_steps] == [list(map(int, s)) for s in expect]
          and len(expect) == 6, src=[list(map(int, s)) for s in full.layer_steps], cut=[list(map(int, s)) for s in model.layer_steps],
          dropped=drop, depths_src=depths(SRC_ARCH), depths_cut=depths(ARCH))
    builds = {}
    for mode, env_jit in (("scripted", None), ("eager", "1")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="")
        env.pop("RWKV_NO_JIT", None)
        if env_jit:
            env["RWKV_NO_JIT"] = env_jit
        code = ("import sys, torch; sys.path.insert(0, r'%s'); from expertium.variants import build; m = build('%s'); "
                "m.load_state_dict(torch.load(r'%s', map_location='cpu', weights_only=True), strict=True); "
                "print('TYPE', type(m).__name__, isinstance(m, torch.jit.ScriptModule))") % (ROOT, ARCH, ext / "master.pth")
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
        base = {"options": options, "rows": rows, "span": 256, "shared": shared}
        for mode, jit in (("eager", False), ("scripted", True)):
            arm = sm73s2.run_probe(dict(base, arch=ARCH, checkpoint=str(ext / "master.pth")), tmp, jit=jit)
            skip = sm73s2.run_probe(dict(base, arch=SRC_ARCH, checkpoint=str(ck), skip=drop), tmp, jit=jit)
            fullp = sm73s2.run_probe(dict(base, arch=SRC_ARCH, checkpoint=str(ck)), tmp, jit=jit) if not jit else None
            same = [{k: u[k] for k in keys} for u in arm] == [{k: u[k] for k in keys} for u in skip]
            detail = {"updates": len(arm), "loss_outputs_gradients_bitwise": same,
                      "loss_cut": [u["loss"] for u in arm], "loss_skip": [u["loss"] for u in skip]}
            if fullp:
                detail["loss_full_w64_c1"] = [u["loss"] for u in fullp]
                detail["zero_shot_loss_change"] = [round(a["loss"] - f["loss"], 6) for a, f in zip(arm, fullp)]
            check(f"T1_trainer_cut_equals_skipped_layers_{mode}", same and len(arm) > len(rows), **detail)
        old = torch.load(ck, map_location="cpu", weights_only=True)
        zero = {k: v.clone() for k, v in old.items()}
        zeroed = []
        for p in prefixes():
            for name in ("time_mixer.W_o.weight", "channel_mixer.W_v.weight"):
                if p + name in zero:
                    zero[p + name].zero_()
                    zeroed.append(p + name)
        zck = Path(tmp) / "w64-c1-dropped-branches-zero.pth"
        torch.save(zero, zck)
        users_file = SM72S / "artifacts/small-model/sm72s/eval-users-732-fit.json"
        dump = Path(tmp) / "dump"
        srows = warm.run_scorer(ROOT, {"a": f"expertium:{SRC_ARCH}:{zck}", "b": f"expertium:{ARCH}:{ext / 'master.pth'}",
                                       "c": f"expertium:{SRC_ARCH}:{ck}"}, users_file, Path(tmp) / "s.jsonl", dump,
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
        check("S1_scorer_cut_equals_zeroed_branches", len(pairs) == 4 and all(p[2] for p in pairs) and same_dumps,
              rows=len(pairs), dumps=n, users=sorted({p[0] for p in pairs}), reviews=sorted({p[3].get("reviews") for p in pairs}),
              variants=sorted({p[1] for p in pairs}), metrics_equal=[p[2] for p in pairs], zeroed=zeroed,
              weight_bytes={SRC_ARCH: {p[1]: by.get(("c", p[0], p[1]), {}).get("weight_bytes") for p in pairs},
                            ARCH: {p[1]: p[4].get("weight_bytes") for p in pairs}},
              parameters={SRC_ARCH: next((r.get("parameters") for k, r in by.items() if k[0] == "c"), None),
                          ARCH: next((r.get("parameters") for k, r in by.items() if k[0] == "b"), None)},
              zero_shot_logloss_change_vs_uncut=zs)
        dec = Path(tmp) / "muon-layout"
        dec.mkdir()
        extra = json.loads(args.extra_options)
        opts = dict(config["options"], **extra, architecture=ARCH)
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
    report["all_ok"] = not fails
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
    p.add_argument("--src-arch", default=SRC_ARCH)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--out", required=True)
    sub.add_parser("quick")
    p = sub.add_parser("cpu-check")
    p.add_argument("--out", default=str(HERE / "check-cpu-sm130.json"))
    p.add_argument("--work", default=str(HERE / "work"))
    p.add_argument("--max-reviews", type=int, default=4200)
    p.add_argument("--extra-options", default=json.dumps({
        "state_carry": {"lanes": "strided"}, "chunk_reviews": 32768, "batch_chunks": 1, "optimizer": "expertium-iter53"}))
    args = parser.parse_args()
    return {"cut": cut, "quick": quick, "cpu-check": cpu_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
