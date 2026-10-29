#!/usr/bin/env python
"""Tiny architecture additions (SM77) stage 1: warm-start tooling and checks (artifacts/small-model/sm77/DESIGN.md on main).

The arm is the leak-free small model (SM72s) from its constant-LR epoch K checkpoint with the SM77 flags on
(expertium.variants preset "w64-sm77": key extras (A) + learned initial state for card and note (B), new parameters
at zero), trained for one 8k epoch at the constant LR exactly as SM72s's epoch K+1 (same protocol: user order,
seeds, options, AdamW state), then a quarter-epoch 32k Muon cool-down. The reference arm is the same with the flags
off: SM72s's own epoch K+1 when it exists (K = 20..24, free), else trained here.

  extend   --master M --optimizer O --arch ARCH --out DIR [--perturb SCALE --seed S]
           DIR/master.pth = M plus the ARCH's new parameters (zeros; --perturb: normal(0, SCALE), checks only) and
           DIR/optimizer.pth = O's AdamW state re-indexed to ARCH's parameter order (new parameters: no state = fresh
           AdamW moments); DIR/extend.json records names, shapes and hashes.
  epoch    --run RUN --epoch N --out DIR [--arch ARCH --init DIR2]
           DIR/protocol.json = RUN's epoch-N protocol rebuilt (scripts/run_full_training.write_protocol's order rule +
           accuracy_loop's fields + the label-window trims) from epoch N-1's protocol and result; with --init the
           start weights / optimizer are DIR2's (extend output); DIR/decay.json = RUN's options (+ architecture), so
           `scripts/expertium_decay.py smoke|run|stop --out DIR` trains it (snapshots every 400 users, resumable).
  check-protocol --run RUN --epochs 2,21    rebuild those epochs and compare with the protocols on disk (must be equal)
  cooldown --run RUN --epoch N --start DIR --out DIR3 --fraction 0.25 --extra-options JSON [--arch ARCH]
           expertium_decay.prepare from RUN epoch N (users, seed 1, schedule), then the start weights = DIR's trained
           master.pth and update offset (DIR = an `epoch` folder, or RUN's epoch folder for the free reference).
  cpu-check --out report.json     CPU (~10-20 min): flags-off bit identity against the wt/sm72s tree (trainer path
           and scorer path) and function preservation (zero-initialised w64-sm77 == w64) on small users.
  gpu-check --out report.json     the first GPU ticket (~5 min): the new backward kernel's dL/dS0 and gradients vs the
           CPU autograd reference, forward-with-state at S0 = 0 == stock, the scorer's long kernel from S0, and
           function preservation of the trainer path on GPU (2 users x 512 reviews).
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACK = "E89-F65e-F77a"
SM72S = Path("C:/Users/jonat/workspace/wt/sm72s")
RUN72S = SM72S / "artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


# ------------------------------------------------------------------ extend
def adamw_names(model):
    """Parameter names in the order of the constant-LR epochs' AdamW (rwkv.train_rwkv.get_optimizer groups)."""
    from argparse import Namespace
    from rwkv.train_rwkv import get_optimizer
    names = {id(p): n for n, p in model.named_parameters()}
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), model)
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups], optimizer


def extend(args):
    import torch
    from expertium.variants import build
    old = torch.load(args.master, map_location="cpu", weights_only=True)
    base, model = build("w64").float(), build(args.arch).float()
    base.load_state_dict(old, strict=True)
    new_sd = model.state_dict()
    added = [k for k in new_sd if k not in old]
    if set(old) - set(new_sd) or any(".sm77_" not in k for k in added):
        raise SystemExit(f"{args.arch} is not w64 plus sm77 parameters: {sorted(set(old) ^ set(new_sd))[:10]}")
    generator = torch.Generator().manual_seed(args.seed)
    out_sd = {}
    for key, value in new_sd.items():
        if key in old:
            out_sd[key] = old[key].clone()
        elif args.perturb:
            out_sd[key] = torch.randn(value.shape, generator=generator, dtype=value.dtype) * args.perturb
        else:
            out_sd[key] = torch.zeros_like(value)
    model.load_state_dict(out_sd, strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out / "master.pth")
    record = {"source_master": str(args.master), "source_master_sha256": sha256(args.master), "architecture": args.arch,
              "added": {k: list(new_sd[k].shape) for k in added}, "added_values": sum(new_sd[k].numel() for k in added),
              "init": f"normal(0, {args.perturb}) seed {args.seed} (check only)" if args.perturb else "zeros",
              "master_sha256": sha256(out / "master.pth")}
    if args.optimizer:
        state = torch.load(args.optimizer, map_location="cpu", weights_only=True)
        old_groups, _ = adamw_names(base)
        new_groups, new_optimizer = adamw_names(model)
        if [len(g) for g in old_groups] != [len(g["params"]) for g in state["param_groups"]]:
            raise SystemExit("the source optimizer.pth does not have the constant-LR AdamW group layout")
        old_index = {name: i for i, name in enumerate(n for g in old_groups for n in g)}
        flat_new = [n for g in new_groups for n in g]
        groups = []
        start = 0
        for old_group, names in zip(state["param_groups"], new_groups):
            groups.append(dict(old_group, params=list(range(start, start + len(names)))))
            start += len(names)
        remapped = {"state": {i: state["state"][old_index[n]] for i, n in enumerate(flat_new)
                              if n in old_index and old_index[n] in state["state"]},
                    "param_groups": groups}
        new_optimizer.load_state_dict(remapped)   # refuses a malformed layout
        check = new_optimizer.state_dict()
        moved = all(torch.equal(check["state"][i][key], state["state"][old_index[n]][key])
                    for i, n in enumerate(flat_new) if n in old_index and old_index[n] in state["state"]
                    for key in state["state"][old_index[n]])
        torch.save(remapped, out / "optimizer.pth")
        record.update(source_optimizer=str(args.optimizer), source_optimizer_sha256=sha256(args.optimizer),
                      optimizer_sha256=sha256(out / "optimizer.pth"), optimizer_states_moved=len(remapped["state"]),
                      optimizer_new_without_state=[n for n in flat_new if n not in old_index],
                      optimizer_old_without_state=[n for n in flat_new if n in old_index and old_index[n] not in state["state"]],
                      optimizer_states_equal_after_load=moved)
        if not moved:
            raise SystemExit("re-indexed AdamW state differs after loading")
    atomic_json(out / "extend.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "added"}))
    return 0


# ------------------------------------------------------------------ epoch protocol
def build_protocol(run, epoch):
    """Epoch `epoch`'s protocol of an accuracy-loop run from epoch-1's protocol (fields, trims) and result (offset)."""
    from scripts.run_full_training import rows_for
    config = json.loads((run / "run.json").read_text())
    previous = json.loads((run / f"epoch-{epoch - 1:02d}" / "protocol.json").read_text())
    result = json.loads((run / f"epoch-{epoch - 1:02d}" / config["model"] / config["track"] / "result.json").read_text())
    if result.get("status") != "complete":
        raise RuntimeError(f"epoch {epoch - 1} of {run} is not complete")
    order = config["train_users"][:]
    random.Random(config["split_seed"] * 1000 + epoch).shuffle(order)
    rows = rows_for(order, config["review_counts"])
    trims = {t["user_id"]: t for t in previous.get("label_window_trims", [])}
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
    protocol = json.loads(json.dumps(previous))
    protocol["rows"] = rows
    source = (run / f"epoch-{epoch - 1:02d}" / config["model"] / config["track"]).resolve().relative_to(ROOT_OF(run))
    protocol["full_run"].update(epoch=epoch, init_master=(source / "master.pth").as_posix(),
                                init_optimizer=(source / "optimizer.pth").as_posix(),
                                update_offset=result["update_offset_end"])
    return protocol, config


def ROOT_OF(run):
    """The checkout that holds `run` (protocol paths are relative to it)."""
    path = Path(run).resolve()
    for parent in path.parents:
        if (parent / "scripts").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError(f"no checkout above {run}")


def check_protocol(args):
    run = Path(args.run)
    out = {}
    for epoch in [int(e) for e in args.epochs.split(",")]:
        built, _ = build_protocol(run, epoch)
        actual = json.loads((run / f"epoch-{epoch:02d}" / "protocol.json").read_text())
        out[epoch] = built == actual
        if not out[epoch]:
            diff = {k for k in set(built) | set(actual) if built.get(k) != actual.get(k)}
            print(f"epoch {epoch}: differs in {sorted(diff)}")
    print(json.dumps({"rebuilt_equals_disk": out}))
    return 0 if all(out.values()) else 1


def epoch(args):
    run = Path(args.run).resolve()
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new output")
    protocol, config = build_protocol(run, args.epoch)
    root = ROOT_OF(run)
    full = protocol["full_run"]
    absolute = lambda p: (root / p).resolve().as_posix()  # noqa: E731
    full["init_master"], full["init_optimizer"] = absolute(full["init_master"]), absolute(full["init_optimizer"])
    full["benchmark_db"] = absolute(full["benchmark_db"])
    options = dict(config["options"])
    note = f"SM72s epoch {args.epoch} protocol (rebuilt), start = SM72s epoch {args.epoch - 1}"
    if args.init:
        record = json.loads((Path(args.init) / "extend.json").read_text())
        if record["architecture"] != args.arch or Path(record["source_master"]).resolve() != Path(full["init_master"]).resolve():
            raise SystemExit(f"{args.init} extends {record['source_master']} for {record['architecture']}, not "
                             f"{full['init_master']} for {args.arch}")
        if record["init"] != "zeros":
            raise SystemExit(f"{args.init} is a check checkpoint ({record['init']})")
        full["init_master"] = (Path(args.init) / "master.pth").resolve().as_posix()
        full["init_optimizer"] = (Path(args.init) / "optimizer.pth").resolve().as_posix()
        note += f", extended to {args.arch} (new parameters zero, AdamW state re-indexed)"
    if args.arch != "w64":
        options["architecture"] = args.arch
    out.mkdir(parents=True)
    full["control_dir"] = (out / "control").relative_to(ROOT).as_posix()
    decay = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": "sm77 constant-LR epoch",
             "source_run": str(run), "source_epoch": args.epoch - 1, "epoch": args.epoch, "model": config["model"],
             "track": config["track"], "options": options, "architecture": args.arch,
             "training_users": len(protocol["rows"]),
             "training_reviews": sum(r["train_range"][1] - r["train_range"][0] for r in protocol["rows"]),
             "lr": "7e-4 constant (no lr_schedule: rwkv default schedule past warmup)", "note": note,
             "init_master_sha256": sha256(full["init_master"]), "init_optimizer_sha256": sha256(full["init_optimizer"])}
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(json.dumps({k: decay[k] for k in ("epoch", "architecture", "training_users", "training_reviews", "note")}))
    return 0


# ------------------------------------------------------------------ cool-down
def cooldown(args):
    from scripts.expertium_decay import prepare
    run = Path(args.run).resolve()
    out = (ROOT / args.out).resolve()
    extra = json.loads(args.extra_options)
    if args.arch != "w64":
        extra["architecture"] = args.arch
    code = prepare(run, args.epoch, out, args.fraction, "one_minus_sin", "none", 1, extra, 0.0, fresh_optimizer=True,
                   train_tier="final")
    if code:
        return code
    start = Path(args.start).resolve()
    if (start / "decay.json").exists():   # an `epoch` folder trained here
        meta = json.loads((start / "decay.json").read_text())
        case = start / meta["model"] / meta["track"]
        if meta.get("architecture", "w64") != args.arch:
            raise SystemExit(f"{start} trained {meta.get('architecture')}, the cool-down is {args.arch}")
    else:                                   # an epoch folder of the run itself (the free reference)
        case = start / "expertium" / TRACK
        if args.arch != "w64":
            raise SystemExit("a run epoch folder is the flags-off reference")
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"{case} is not a complete epoch")
    protocol = json.loads((out / "protocol.json").read_text())
    decay = json.loads((out / "decay.json").read_text())
    protocol["full_run"].update(init_master=(case / "master.pth").as_posix(), update_offset=result["update_offset_end"])
    decay.update(source_case=str(case), source_update_offset_end=result["update_offset_end"], start=str(start),
                 start_master_sha256=sha256(case / "master.pth"), architecture=args.arch,
                 sm77=f"users / order / schedule from {run.name} epoch {args.epoch}; start weights {case}")
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "decay.json", decay)
    print(json.dumps({"out": out.name, "start": str(case), "update_offset": result["update_offset_end"],
                      "training_users": decay["training_users"], "training_reviews": decay["training_reviews"]}))
    return 0


# ------------------------------------------------------------------ CPU checks
PROBE = r'''
import hashlib, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import torch
torch.set_num_threads(2)
from concurrent.futures import ThreadPoolExecutor
from scripts.gpu_r01_prefetch import training_batches
from scripts.expertium_options import CatchUp
from expertium.variants import build
spec = json.loads(open(sys.argv[2]).read())
model = build(spec["arch"]).float()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
shared = spec["shared"]
params = dict(model.named_parameters())
catch_up = CatchUp(spec["options"], {}, spec["rows"])
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(spec["rows"], spec["span"], spec["options"]["training_seed"], pool, spec["options"], None, 2))
out = []
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, row["user_id"])
    stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
    stats.average_loss.backward()
    h = hashlib.sha256()
    for name in shared:
        g = params[name].grad
        h.update(b"-" if g is None else g.numpy().tobytes())
    new = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in params.items() if n not in shared}
    if spec.get("dump"):
        torch.save({n: params[n].grad for n in shared}, spec["dump"] + f"-{len(out)}.pt")
    out.append({"user": row["user_id"], "offset": item.offset, "lanes": list(getattr(item, "carry_lanes", []) or []),
                "loss": stats.average_loss.item(), "loss_hex": float(stats.average_loss).hex(),
                "p_digest": hashlib.sha256(stats.p_imm_all.detach().numpy().tobytes()).hexdigest(),
                "grad_digest": h.hexdigest(), "new_grad_norm_sum": sum(v for v in new.values() if v is not None),
                "new_with_grad": sum(v is not None and v > 0 for v in new.values())})
print("PROBE " + json.dumps(out))
'''


def run_probe(tree, spec, tmp, jit=False):
    probe = Path(tmp) / "sm77_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               RWKV_DELETED_CARDS="neighbour")
    if jit:
        env.pop("RWKV_NO_JIT", None)
    else:
        env["RWKV_NO_JIT"] = "1"
    result = subprocess.run([sys.executable, "-X", "utf8", str(probe), str(tree), str(spec_file)], cwd=tree,
                            capture_output=True, text=True, env=env)
    line = next((x for x in result.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {result.stderr[-3000:]}")
    return json.loads(line[6:])


def run_scorer(tree, models, users_file, out, dump, max_reviews, count):
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    cmd = [sys.executable, "-X", "utf8", str(Path(tree) / "scripts/small_model_heldout.py"), "--device", "cpu",
           "--pool", "732", "--users-file", str(users_file), "--max-reviews", str(max_reviews), "--count", str(count),
           "--variants", "fp32,w8+s8", "--split-existing", "--deleted-cards", "neighbour", "--out", str(out),
           "--dump", str(dump), "--time-limit", "3000"]
    for tag, spec in models.items():
        cmd += ["--model", f"{tag}={spec}"]
    result = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"scorer failed in {tree}: {result.stderr[-3000:]}")
    return [json.loads(line) for line in Path(out).read_text().splitlines()]


METRIC_SKIP = {"prepare_seconds", "model_seconds", "seconds", "provenance", "checkpoint", "tag", "label", "arch",
               "parameters", "weight_bytes", "scorer", "started", "finished", "host", "time", "utc", "process"}


def metric_view(row):
    return {k: v for k, v in row.items() if k not in METRIC_SKIP and not k.endswith("_seconds")}


def dumps_equal(a_dir, b_dir, a_tag, b_tag):
    import numpy as np
    files = sorted(Path(a_dir).glob(f"*_{a_tag}_*.npz"))
    same, compared = True, 0
    for path in files:
        other = Path(b_dir) / path.name.replace(f"_{a_tag}_", f"_{b_tag}_")
        if not other.exists():
            return False, compared
        x, y = np.load(path), np.load(other)
        same &= sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        compared += 1
    return same and compared > 0, compared


def cpu_check(args):
    import torch
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    e20 = RUN72S / "epoch-20" / "expertium" / TRACK
    ck = e20 / "master.pth"
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ext = work / "ext-zero"
    if not (ext / "extend.json").exists():
        subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "extend", "--master", str(ck),
                        "--optimizer", str(e20 / "optimizer.pth"), "--arch", "w64-sm77", "--out", str(ext)], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    report["extend"] = json.loads((ext / "extend.json").read_text())
    from expertium.variants import build
    shared = list(dict(build("w64").named_parameters()))
    # rows: the two smallest training users with >= 300 reviews of SM72s's epoch-21 protocol
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
    rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
    config = json.loads((RUN72S / "run.json").read_text())
    options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    span = 256
    report["users"] = {str(r["user_id"]): r["train_range"][1] for r in rows}
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        # T0 flags off: this tree vs the wt/sm72s tree (trainer path, every update: loss, outputs, all gradients)
        spec = {"arch": "w64", "checkpoint": str(ck), "options": options, "rows": rows, "span": span, "shared": shared}
        ours, theirs = run_probe(ROOT, spec, tmp), run_probe(SM72S, spec, tmp)
        t0 = ours == theirs and len(ours) > len(rows)
        report["checks"]["T0_flags_off_trainer_equals_sm72s_tree"] = {"pass": t0, "updates": len(ours),
                                                                      "losses": [u["loss"] for u in ours]}
        fails += [] if t0 else ["T0"]
        # T1 function preservation: w64-sm77 at zero == w64 (loss bits, rating outputs, gradients of every old weight)
        spec_b = dict(spec, arch="w64-sm77", checkpoint=str(ext / "master.pth"))
        arm = run_probe(ROOT, spec_b, tmp)
        # outputs and loss must be bitwise equal; the old weights' gradients only up to float summation order (the
        # removal and insertion keys are two branches of the graph, so autograd adds their contributions to k and a
        # in another order): max |diff| relative to the gradient's max |value| per tensor, bar 1e-5
        keys = ("user", "offset", "lanes", "loss_hex", "p_digest")

        def grad_gap(tag_a, tag_b, n):
            worst = 0.0
            for i in range(n):
                ga, gb = (torch.load(Path(tmp) / f"{tag}-{i}.pt", weights_only=True) for tag in (tag_a, tag_b))
                for name in shared:
                    x, y = ga[name], gb[name]
                    if x is None or y is None:
                        if (x is None) != (y is None):
                            return float("inf")
                        continue
                    scale = float(x.abs().max())
                    if scale > 0:
                        worst = max(worst, float((x - y).abs().max()) / scale)
            return worst

        spec_ref = dict(spec, dump=str(Path(tmp) / "ref"))
        ref_dump = run_probe(ROOT, spec_ref, tmp)
        arm = run_probe(ROOT, dict(spec_b, dump=str(Path(tmp) / "arm")), tmp)
        t1 = [{k: u[k] for k in keys} for u in arm] == [{k: u[k] for k in keys} for u in ours] == [
            {k: u[k] for k in keys} for u in ref_dump]
        gap = grad_gap("ref", "arm", len(arm))
        bitwise = [a["grad_digest"] == b["grad_digest"] for a, b in zip(arm, ours)]
        learn = all(u["new_with_grad"] > 0 for u in arm)
        report["checks"]["T1_zero_init_trainer_equals_w64"] = {
            "pass": t1 and learn and gap < 1e-5, "updates": len(arm), "loss_and_outputs_bitwise_equal": t1,
            "old_gradients_bitwise_equal_per_update": bitwise, "old_gradients_max_relative_diff": gap,
            "new_parameters_with_gradient_per_update": [u["new_with_grad"] for u in arm],
            "new_grad_norm_sum": [round(u["new_grad_norm_sum"], 6) for u in arm]}
        fails += [] if t1 and learn and gap < 1e-5 else ["T1"]
        # T1j the same under TorchScript (the trainer's mode)
        arm_jit = run_probe(ROOT, dict(spec_b, dump=str(Path(tmp) / "armj")), tmp, jit=True)
        ref_jit = run_probe(ROOT, dict(spec, dump=str(Path(tmp) / "refj")), tmp, jit=True)
        t1j = [{k: u[k] for k in keys} for u in arm_jit] == [{k: u[k] for k in keys} for u in ref_jit]
        gapj = grad_gap("refj", "armj", len(arm_jit))
        report["checks"]["T1j_zero_init_trainer_equals_w64_scripted"] = {
            "pass": t1j and gapj < 1e-5, "updates": len(arm_jit), "loss_and_outputs_bitwise_equal": t1j,
            "old_gradients_max_relative_diff": gapj}
        fails += [] if t1j and gapj < 1e-5 else ["T1j"]

        # S0 / S1 scorer path (full history, CPU): flags off vs the sm72s tree; zero-init sm77 vs w64
        users_file = SM72S / "artifacts/small-model/sm72s/eval-users-732-fit.json"
        d_ours, d_theirs = Path(tmp) / "dump-ours", Path(tmp) / "dump-theirs"
        rows_ours = run_scorer(ROOT, {"a": f"expertium:w64:{ck}", "b": f"expertium:w64-sm77:{ext / 'master.pth'}"},
                               users_file, Path(tmp) / "ours.jsonl", d_ours, args.max_reviews, 2)
        rows_theirs = run_scorer(SM72S, {"a": f"expertium:w64:{ck}"}, users_file, Path(tmp) / "theirs.jsonl", d_theirs,
                                 args.max_reviews, 2)
        a_ours = sorted((metric_view(r) for r in rows_ours if r["label"].startswith("a/")), key=json.dumps)
        a_theirs = sorted((metric_view(r) for r in rows_theirs if r["label"].startswith("a/")), key=json.dumps)
        same_dumps, n = dumps_equal(d_ours, d_theirs, "a", "a")
        s0 = a_ours == a_theirs and len(a_ours) == 4 and same_dumps
        report["checks"]["S0_flags_off_scorer_equals_sm72s_tree"] = {"pass": s0, "rows": len(a_ours), "dumps": n,
                                                                    "users": sorted({r["user_id"] for r in rows_ours})}
        fails += [] if s0 else ["S0"]
        b_rows = {(r["user_id"], r["variant"]): r for r in rows_ours if r["label"].startswith("b/")}
        pairs = [(metric_view(r), metric_view(b_rows[(r["user_id"], r["variant"])])) for r in rows_ours
                 if r["label"].startswith("a/") and (r["user_id"], r["variant"]) in b_rows]
        for x, y in pairs:
            x.pop("kind", None); y.pop("kind", None)
        same_dumps_b, nb = dumps_equal(d_ours, d_ours, "a", "b")
        s1 = len(pairs) == 4 and all(x == y for x, y in pairs) and same_dumps_b
        wb = {r["variant"]: r.get("weight_bytes") for r in rows_ours if r["label"].startswith("b/")}
        wa = {r["variant"]: r.get("weight_bytes") for r in rows_ours if r["label"].startswith("a/")}
        report["checks"]["S1_zero_init_scorer_equals_w64"] = {
            "pass": s1, "rows": len(pairs), "dumps": nb, "reviews": sorted({r["reviews"] for r in rows_ours}),
            "curve_log_loss": {f"{x['user_id']}/{x['variant']}": x["curve_log_loss"] for x, _ in pairs},
            "weight_bytes_w64": wa, "weight_bytes_w64_sm77": wb,
            "parameters": {"w64": rows_ours[0].get("parameters") if rows_ours[0]["label"].startswith("a/") else None,
                           "w64-sm77": next(r.get("parameters") for r in rows_ours if r["label"].startswith("b/"))}}
        fails += [] if s1 else ["S1"]
    report["fails"] = fails
    report["ok"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    atomic_json(args.out, report)
    print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in report["checks"].items()}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


# ------------------------------------------------------------------ GPU check (first launcher ticket)
def gpu_check(args):
    import torch
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    from rwkv.model import nvrtc
    from rwkv.model.wkv_state import wkv_state, wkv_state_grad, wkv_with_state_grad
    from scripts.gpu_row_tiles import row_tiled_backend
    nvrtc.register()
    previous = nvrtc.set_max_timesteps(4096)
    g = torch.Generator().manual_seed(0)
    fails = []
    for backend in ("generic", "row_tiles_exact_4"):
        for dtype in (torch.float32, torch.bfloat16):
            B, T, H, K = 6, 300, 4, 16
            r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
            kd = torch.nn.functional.normalize(kd, dim=-1)
            a = torch.sigmoid(a)
            w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
            skip = torch.rand(B, T, generator=g) < 0.3
            skip[:, 0] = False
            s0 = torch.randn(B, H, K, K, generator=g) * 0.2
            grad = torch.randn(B, T, H, K, generator=g)
            inputs = [x.to(dtype) for x in (r, k * a, v)] + [w] + [x.to(dtype) for x in (a, kd)]
            # CPU float32 autograd reference
            ref_in = [x.float().clone().requires_grad_(True) for x in inputs]
            ref_s0 = s0.clone().requires_grad_(True)
            out_ref, _ = wkv_with_state_grad(*ref_in, skip, ref_s0)
            (out_ref * grad).sum().backward()
            with (row_tiled_backend(rows=4, tile=32, exact=True) if backend != "generic" else _null()):
                cuda_in = [x.cuda().clone().requires_grad_(True) for x in inputs]
                cuda_s0 = s0.cuda().clone().requires_grad_(True)
                out, final = wkv_state_grad(*cuda_in, skip.cuda(), cuda_s0)
                (out.float() * grad.cuda()).sum().backward()
                # S0 = 0: the forward and the other gradients equal the SM21 path (forward-with-state + stock backward)
                z_in = [x.cuda().clone().requires_grad_(True) for x in inputs]
                z_out, _ = wkv_state_grad(*z_in, skip.cuda(), torch.zeros_like(cuda_s0).requires_grad_(True))
                (z_out.float() * grad.cuda()).sum().backward()
                y_in = [x.cuda().clone().requires_grad_(True) for x in inputs]
                y_out, _ = wkv_state(*y_in, skip.cuda(), torch.zeros_like(cuda_s0))
                (y_out.float() * grad.cuda()).sum().backward()
                torch.cuda.synchronize()
            tol = 2e-3 if dtype == torch.float32 else 6e-2
            errors = {"out": float((out.float().cpu() - out_ref).abs().max()),
                      "dS0": float((cuda_s0.grad.cpu() - ref_s0.grad).abs().max()),
                      "dS0_scale": float(ref_s0.grad.abs().max()),
                      **{f"d{n}": float((x.grad.float().cpu() - y.grad).abs().max()) for n, x, y in
                         zip("rkvwad", cuda_in, ref_in)}}
            same = (torch.equal(z_out, y_out) and all(torch.equal(x.grad, y.grad) for x, y in zip(z_in, y_in)))
            ok = errors["out"] < tol and errors["dS0"] < tol * max(1.0, errors["dS0_scale"]) and same
            name = f"K_{backend}_{str(dtype).replace('torch.', '')}"
            report["checks"][name] = {"pass": ok, "max_abs_error": errors, "zero_state_bitwise_equal_sm21_path": same}
            fails += [] if ok else [name]
    nvrtc.set_max_timesteps(previous)
    # P: function preservation of the trainer path on GPU (bf16 training copy, the trainer's exact kernels, row tiles,
    # strided-lane carry): w64 from SM72s epoch 20 vs w64-sm77 zero-extended, 2 users x 512 reviews, no optimizer step
    pair = gpu_trainer_pair(args)
    report["checks"]["P_gpu_trainer_zero_init_equals_w64"] = pair
    fails += [] if pair["pass"] else ["P"]
    from scripts.small_model_wkv import self_test_state
    long = self_test_state()
    lk = all(v < 1e-4 for k, v in long.items() if not k.endswith("bitwise")) and all(
        v for k, v in long.items() if k.endswith("bitwise"))
    report["checks"]["L_scorer_long_kernel_from_state"] = {"pass": lk, **long}
    fails += [] if lk else ["L"]
    report["fails"] = fails
    report["ok"] = not fails
    atomic_json(args.out, report)
    print(json.dumps({k: v["pass"] for k, v in report["checks"].items()}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


def gpu_trainer_pair(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from expertium.model.variant_model import VariantSrsRWKV
    from expertium.variants import variant
    from scripts.compare import training_models
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_row_tiles import row_tiled_backend
    os.environ.setdefault("GPU_BATCH1_DATA_PATH", DATA)
    ext = Path(args.ext).resolve() if args.ext else Path(args.work).resolve() / "ext-zero"
    if not (ext / "master.pth").exists():
        raise SystemExit(f"{ext} missing: run extend first (the launcher's preparation / cpu-check do)")
    record = json.loads((ext / "extend.json").read_text())
    if record["init"] != "zeros":
        raise SystemExit(f"{ext} is not zero-initialised")
    base_master = Path(record["source_master"])
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    rows = [dict(r, train_range=[0, 512]) for r in protocol["rows"] if r["train_range"][1] >= 2048][:2]
    config = json.loads((RUN72S / "run.json").read_text())
    options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    device = torch.device("cuda:0")
    with ThreadPoolExecutor(max_workers=1) as pool:
        prepared = list(training_batches(rows, 256, options["training_seed"], pool, options, None, 2))
    results = {}
    with row_tiled_backend(rows=4, tile=32, exact=True):
        for arch, ck in (("w64", base_master), (record["architecture"], ext / "master.pth")):
            master, base = training_models(VariantSrsRWKV, variant(arch), device, torch.bfloat16, 2025)
            master.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
            base.copy_downcast_(master, torch.bfloat16)
            for module in base.modules():
                for name in ("exact_shift", "exact_index", "exact_group_norm"):
                    if hasattr(module, name):
                        setattr(module, name, bool(options.get(name)))
            base.skip_padding_sources = bool(options.get("skip_padding_sources"))
            catch_up = CatchUp(dict(options, architecture=arch), {}, rows)
            shared = [n for n, _ in base.named_parameters() if ".sm77_" not in n]
            params = dict(base.named_parameters())
            out = []
            for row, item in prepared:
                batch = item.batch.to(device)
                batch.start = batch.start.to(torch.bfloat16)
                base.train()
                base.zero_grad(set_to_none=True)
                torch.manual_seed(item.seed)
                catch_up.carry.attach(item, batch, row["user_id"])
                stats = catch_up.loss(base, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
                stats.average_loss.backward()
                torch.cuda.synchronize()
                out.append({"loss": float(stats.average_loss).hex(), "p": stats.p_imm_all.float().cpu(),
                            "grads": [params[n].grad.float().cpu() if params[n].grad is not None else None for n in shared],
                            "new_grads": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0))
                                             for n, p in params.items() if ".sm77_" in n)})
            results[arch] = out
    a, b = results["w64"], results[record["architecture"]]
    loss_equal = [x["loss"] == y["loss"] for x, y in zip(a, b)]
    p_equal = [bool(torch.equal(x["p"], y["p"])) for x, y in zip(a, b)]
    # loss and outputs bitwise; the old weights' gradients up to summation order (the removal / insertion key branches
    # add their contributions to k and a in another order; CPU float32: 2.8e-6 relative). bf16 gradients: bar 2e-2 of
    # each tensor's max |value| (a few bf16 ulps)
    rel = 0.0
    same_presence = True
    for x, y in zip(a, b):
        for gx, gy in zip(x["grads"], y["grads"]):
            if gx is None or gy is None:
                same_presence &= (gx is None) == (gy is None)
                continue
            scale = float(gx.abs().max())
            if scale > 0:
                rel = max(rel, float((gx - gy).abs().max()) / scale)
    grad_equal = all(torch.equal(gx, gy) if gx is not None and gy is not None else gx is gy
                     for x, y in zip(a, b) for gx, gy in zip(x["grads"], y["grads"]))
    return {"pass": len(a) == len(b) > 2 and all(loss_equal) and all(p_equal) and same_presence and rel <= 2e-2
            and all(y["new_grads"] > 0 for y in b),
            "updates": len(a), "loss_bitwise_equal": loss_equal, "outputs_bitwise_equal": p_equal,
            "old_gradients_bitwise_equal": grad_equal, "old_gradients_max_relative_diff": rel,
            "new_parameters_with_gradient": [y["new_grads"] for y in b], "users": [r["user_id"] for r in rows]}


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extend")
    p.add_argument("--master", required=True)
    p.add_argument("--optimizer", default="")
    p.add_argument("--arch", default="w64-sm77")
    p.add_argument("--out", required=True)
    p.add_argument("--perturb", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=77)
    p = sub.add_parser("epoch")
    p.add_argument("--run", default=str(RUN72S))
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="w64")
    p.add_argument("--init", default="")
    p = sub.add_parser("check-protocol")
    p.add_argument("--run", default=str(RUN72S))
    p.add_argument("--epochs", default="2,11,21")
    p = sub.add_parser("cooldown")
    p.add_argument("--run", default=str(RUN72S))
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--extra-options", required=True)
    p.add_argument("--arch", default="w64")
    p = sub.add_parser("cpu-check")
    p.add_argument("--out", default=str(ROOT / "artifacts/small-model/sm77/check-cpu-sm77.json"))
    p.add_argument("--work", default=str(ROOT / "artifacts/small-model/sm77/work"))
    p.add_argument("--max-reviews", type=int, default=4200)
    p = sub.add_parser("gpu-check")
    p.add_argument("--out", default=str(ROOT / "artifacts/small-model/sm77/gpu/check-gpu-sm77.json"))
    p.add_argument("--work", default=str(ROOT / "artifacts/small-model/sm77/work"))
    p.add_argument("--ext", default="", help="a zero-initialised extend folder (default: WORK/ext-zero)")
    args = parser.parse_args()
    return {"extend": extend, "epoch": epoch, "check-protocol": check_protocol, "cooldown": cooldown,
            "cpu-check": cpu_check, "gpu-check": gpu_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
