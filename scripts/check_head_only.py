#!/usr/bin/env python
"""CPU check of SM50 "head_only" (scripts/expertium_options.freeze_except_head; docs/SMALL_MODEL.md, SM50).

Real training users of a prepared SM50 decay (its protocol rows, train split, the same options as SM36's decay:
32,768 x 1 windows, strided-lane carried state, replay KD targets, Muon expertium-iter53 with SM36's optimizer state),
SM36's weights, float32, CPU eager (RWKV_NO_JIT=1), the trainer's update order (MasterChildSync, clip, LR, step).

  A  head split: every parameter is in exactly one of curve head / instant head / shared; in eval mode, perturbing
     every curve-head parameter leaves the rating probabilities bitwise unchanged and perturbing every instant-head
     parameter leaves the curve predictions and curve weights bitwise unchanged (the heads share nothing past the
     shared prehead_norm), while both perturbations do change their own head's output;
  B  per arm (curve, instant), N updates through the trainer's update path: frozen parameters (child and master)
     never get a gradient, loss and gradient norm finite, after the updates every frozen master parameter is bitwise
     SM36's and the head's parameters moved; the trunk output (input of prehead_norm, i.e. what the stored card/note
     state produces) on an eval forward is bitwise SM36's;
  C  head-only loss: curve arm, first update: the curve head's gradient with the full loss equals (bitwise) the one with
     the instant term removed (instant_loss_weight 0); instant arm: covered by A (the curve terms do not depend on the
     instant head's parameters, so they contribute no gradient to it);
  D  option absent = old code: the same N updates with SM36's options (no head_only, a fresh Muon state as in SM36)
     from this tree and from the tree before SM50 (git archive of `git merge-base HEAD small-model`): loss, every
     gradient and the final parameters sha256 equal; with SM36's LOADED optimizer state the old tree leaves the
     parameters unchanged (HybridOptimizer.param_groups bug, fixed here) and this tree trains; and this tree's
     `expertium_decay.py prepare` with SM36's arguments rewrites SM36's own decay.json / protocol.json (except created_utc).

  .venv/Scripts/python.exe -X utf8 scripts/check_head_only.py --decay artifacts/small-model/sm50/probe-curve \
      --out artifacts/small-model/sm50/check-head-only.json
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SM36_DIR = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon")
V2_RUN = Path("C:/Users/jonat/workspace/wt/sm21v2/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont/run")

# Runs in this process (exec) and in a subprocess with sys.path[0] = the tree under test.
PROBE = r'''
import hashlib, json, os, sys
import torch


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(b"-" if t is None else t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(spec, keep=False, hook=None):
    """spec: options, full_run, rows, checkpoint, optimizer_state, updates. Returns per-update records (+ models)."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from rwkv.train_rwkv import CLIP
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_r01_update import MasterChildSync
    options, full_run, rows = spec["options"], spec["full_run"], spec["rows"]
    state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    master, child = build(options["architecture"]).float(), build(options["architecture"]).float()
    master.load_state_dict(state, strict=True)
    child.load_state_dict(state, strict=True)
    catch_up = CatchUp(options, full_run, rows)
    head = None
    if options.get("head_only") is not None:
        from scripts.expertium_options import freeze_except_head
        head = freeze_except_head(child, options["head_only"])
    optimizer = catch_up.optimizer(master)
    initial = digest(list(master.parameters()))
    if spec.get("optimizer_state"):
        optimizer.load_state_dict(torch.load(spec["optimizer_state"], map_location="cpu", weights_only=True))
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    sync = MasterChildSync(master, child)
    span = options["batch_chunks"] * options["chunk_reviews"]
    with ThreadPoolExecutor(max_workers=1) as pool:  # materialized first: the preparation thread reseeds the RNG
        prepared = list(training_batches(rows, span, options["training_seed"], pool, options, None, 2))
    prepared = prepared[:spec["updates"]]
    records, reviews_done, offset = [], 0, int(full_run.get("update_offset", 0))
    for step, (row, item) in enumerate(prepared):
        batch = item.batch.to(torch.device("cpu"))
        user = row["user_id"]
        for p in master.parameters():
            p.grad = None
        for p in child.parameters():
            p.grad = None
        child.train()
        sync.downcast()
        torch.manual_seed(item.seed)
        catch_up.carry.attach(item, batch, user)
        stats = catch_up.loss(child, batch, reviews_done, (user, item.offset, item.reviews, item.seed), item.batch.labels)
        loss = stats.average_loss
        loss.backward()
        child_grads = {n: p.grad is not None for n, p in child.named_parameters()}
        sync.transfer_grads()
        master_grads = {n: p.grad is not None for n, p in master.named_parameters()}
        grad_digest = digest([loss] + [p.grad for p in master.parameters()])
        if hook is not None:
            hook(step, child, master, batch, item, user, catch_up)
        norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=False)
        lr = catch_up.learning_rate(step, offset, reviews_done)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        reviews_done += item.reviews
        records.append({"step": step + 1, "user": user, "offset": item.offset, "reviews": item.reviews,
                        "loss": float(loss.detach()), "norm": float(norm), "lr": lr, "finite": bool(torch.isfinite(loss) and torch.isfinite(norm)),
                        "grad_digest": grad_digest,
                        "child_grads": sorted(n for n, g in child_grads.items() if g),
                        "master_grads": sorted(n for n, g in master_grads.items() if g)})
    out = {"records": records, "initial_parameters": initial, "final_parameters": digest(list(master.parameters())), "head": head,
           "kd": catch_up.teacher.cache_stats if catch_up.teacher is not None else None}
    if keep:
        out["models"] = (master, child, catch_up, prepared)
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    sys.path.insert(0, root)
    os.chdir(root)
    torch.set_num_threads(4)
    result = train(json.loads(open(sys.argv[2], encoding="utf-8").read()))
    print("PROBE " + json.dumps({k: v for k, v in result.items() if k != "models"}))
'''


def run_probes(jobs, tmp):
    """jobs: [(tree, spec)]; runs every probe at once (separate processes) and returns their results in order."""
    probe = Path(tmp) / "probe_head_only.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = dict(os.environ, RWKV_NO_JIT="1", PYTHONPATH="", CUDA_VISIBLE_DEVICES="")
    running = []
    for index, (tree, spec) in enumerate(jobs):
        spec_file = Path(tmp) / f"spec-{index}.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        running.append((tree, subprocess.Popen([sys.executable, "-X", "utf8", str(probe), str(tree), str(spec_file)], cwd=tree,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)))
        if os.environ.get("HEAD_ONLY_CHECK_SEQUENTIAL") == "1":  # low-memory machines: one probe at a time
            running[-1] += (running[-1][1].communicate(),)
    results = []
    for item in running:
        tree, process = item[:2]
        stdout, stderr = item[2] if len(item) > 2 else process.communicate()
        line = next((x for x in stdout.splitlines() if x.startswith("PROBE ")), None)
        if line is None:
            raise RuntimeError(f"probe failed in {tree}: {stderr[-3000:]}")
        results.append(json.loads(line[6:]))
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--decay", type=Path, required=True, help="a prepared SM50 decay (its rows and options)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--users", default="", help="comma list; default: two small users + one with two windows")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--parts", default="ABCD", help="subset of the checks to run, e.g. ABC or D (separate reports)")
    args = parser.parse_args()
    started = time.perf_counter()
    decay_dir = (ROOT / args.decay).resolve()
    decay = json.loads((decay_dir / "decay.json").read_text())
    protocol = json.loads((decay_dir / "protocol.json").read_text())
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(4)
    namespace = {}
    exec(compile(PROBE, "probe_head_only", "exec"), namespace)
    train, digest = namespace["train"], namespace["digest"]
    from scripts.expertium_options import HEAD_PREFIXES, head_parameter_names

    rows_all = protocol["rows"]
    if args.users:
        wanted = [int(u) for u in args.users.split(",")]
    else:
        size = lambda r: r["train_range"][1] - r["train_range"][0]  # noqa: E731
        small = sorted((r for r in rows_all if 3000 <= size(r) <= 8000), key=size)[:2]
        two = sorted((r for r in rows_all if 33000 <= size(r) <= 45000), key=size)[:1]
        wanted = [r["user_id"] for r in small + two]
    rows = [next(r for r in rows_all if r["user_id"] == u) for u in wanted]
    options = {k: v for k, v in decay["options"].items() if k != "packed_ipc"}
    full_run = protocol["full_run"]
    checkpoint, optimizer_state = full_run["init_master"], full_run["init_optimizer"]
    report = {"decay": str(decay_dir), "users": wanted, "updates": args.updates, "checkpoint": checkpoint,
              "optimizer_state": optimizer_state, "options_without": ["packed_ipc"], "checks": {}}
    fails = []

    def check(name, ok, **values):
        report["checks"][name] = {"pass": bool(ok), **values}
        print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(values, default=str)[:600], flush=True)
        if not ok:
            fails.append(name)

    spec = {"options": {k: v for k, v in options.items() if k != "head_only"}, "full_run": full_run, "rows": rows,
            "checkpoint": checkpoint, "optimizer_state": optimizer_state, "updates": args.updates}
    sm36 = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if "A" not in args.parts and ("B" in args.parts or "C" in args.parts):
        parser.error("B and C need A (they reuse its batch and split)")
    # ---------------- A: head split
    if "A" in args.parts:
        run_abc(args, spec, options, full_run, rows, sm36, train, digest, check, head_parameter_names, torch)
    # ---------------- D: option absent = old code
    if "D" in args.parts:
        run_d(args, spec, check)
    report["parts"] = args.parts
    report["passed"], report["fails"] = not fails, fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("ALL OK" if not fails else f"FAILED {fails}", f"({report['seconds']} s)")
    return 0 if not fails else 1


def run_abc(args, spec, options, full_run, rows, sm36, train, digest, check, head_parameter_names, torch):
    from expertium.variants import build
    model = build(options["architecture"]).float()
    model.load_state_dict(sm36, strict=True)
    names = [n for n, _ in model.named_parameters()]
    curve, instant = set(head_parameter_names(model, "curve")), set(head_parameter_names(model, "instant"))
    shared = [n for n in names if n not in curve and n not in instant]
    numel = dict((n, p.numel()) for n, p in model.named_parameters())
    split = {"curve": sum(numel[n] for n in curve), "instant": sum(numel[n] for n in instant),
             "shared": sum(numel[n] for n in shared),
             "shared_outside_trunk": [n for n in shared if not n.startswith(("features2card.", "rwkv_modules."))],
             "curve_tensors": sorted(curve), "instant_tensors": sorted(instant)}
    first = train(dict(spec, updates=1), keep=True)
    _, _, catch_up, prepared = first["models"]
    row0, item0 = prepared[0]
    batch0 = item0.batch.to(torch.device("cpu"))
    key0 = (row0["user_id"], item0.offset, item0.reviews, item0.seed)

    from scripts.expertium_options import CatchUp

    def outputs(m):
        # a fresh carry table each time (the table keeps the state a forward commits), no teacher (outputs only)
        m.eval()
        fresh = CatchUp(spec["options"], full_run, rows)
        fresh.carry.attach(item0, batch0, row0["user_id"])
        with torch.no_grad():
            s = fresh.loss(m, batch0, 0, key0, item0.batch.labels)
        return s.p_curve.clone(), s.w.clone(), s.p_imm_all.clone()

    base_out = outputs(model)
    effects = {}
    for head, group in (("curve", curve), ("instant", instant)):
        bumped = build(options["architecture"]).float()
        bumped.load_state_dict(sm36, strict=True)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for n, p in bumped.named_parameters():
                if n in group:
                    p.add_(1e-2 * torch.randn(p.shape, generator=g))
        o = outputs(bumped)
        effects[head] = {"curve_equal": bool(torch.equal(o[0], base_out[0]) and torch.equal(o[1], base_out[1])),
                         "rating_equal": bool(torch.equal(o[2], base_out[2]))}
    check("A_head_split", len(curve) and len(instant) and not curve & instant and len(shared) + len(curve) + len(instant) == len(names)
          and split["shared_outside_trunk"] == ["prehead_norm.weight", "prehead_norm.bias"]
          and effects["curve"] == {"curve_equal": False, "rating_equal": True}
          and effects["instant"] == {"curve_equal": True, "rating_equal": False},
          split=split, perturbation=effects)

    # ---------------- B + C: per arm
    trunk_ref = {}

    def trunk_output(m):
        captured = []
        handle = m.prehead_norm.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach().clone()))
        try:
            outputs(m)
        finally:
            handle.remove()
        return digest(captured)

    trunk_ref["sm36"] = trunk_output(model)
    for head in ("curve", "instant"):
        arm_spec = dict(spec, options=dict(spec["options"], head_only=head))
        grads_c = {}

        def hook(step, child, master, batch, item, user, cu):
            if step == 0 and head == "curve":
                grads_c["full"] = digest([master.get_parameter(n).grad for n in sorted(curve)])
        result = train(arm_spec, keep=True, hook=hook)
        master, child, _, _ = result["models"]
        group = curve if head == "curve" else instant
        frozen_equal = all(torch.equal(p.detach(), sm36[n]) for n, p in master.named_parameters() if n not in group)
        moved = sorted(n for n, p in master.named_parameters() if n in group and not torch.equal(p.detach(), sm36[n]))
        grads_only_head = all(set(r["child_grads"]) <= group and set(r["master_grads"]) <= group and r["master_grads"]
                              for r in result["records"])
        finite = all(r["finite"] for r in result["records"])
        trunk_same = trunk_output(master) == trunk_ref["sm36"]
        check(f"B_{head}_arm", frozen_equal and grads_only_head and finite and trunk_same and len(moved) > 0,
              frozen_bitwise_unchanged=frozen_equal, head_tensors_moved=f"{len(moved)}/{len(group)}",
              gradients_only_on_head=grads_only_head, finite=finite, trunk_output_bitwise_sm36=trunk_same,
              trainable=result["head"]["trainable_parameters"], frozen=result["head"]["frozen_parameters"],
              records=[{k: r[k] for k in ("step", "user", "offset", "reviews", "loss", "norm", "lr")} for r in result["records"]])
        if head == "curve":
            no_instant = dict(arm_spec, options=dict(arm_spec["options"], instant_loss_weight=0.0))

            def hook0(step, child, master, batch, item, user, cu):
                if step == 0:
                    grads_c["curve_loss_only"] = digest([master.get_parameter(n).grad for n in sorted(curve)])
            # instant_loss_weight is applied by the trainer as a model attribute; do the same here
            from expertium.model import variant_model
            original = variant_model.VariantSrsRWKV.__init__

            def init(self, *a, **k):
                original(self, *a, **k)
                self.instant_loss_weight = 0.0
            variant_model.VariantSrsRWKV.__init__ = init
            try:
                train(dict(no_instant, updates=1), hook=hook0)
            finally:
                variant_model.VariantSrsRWKV.__init__ = original
            check("C_curve_head_only_loss", grads_c["full"] == grads_c["curve_loss_only"], **grads_c)


def run_d(args, spec, check):
    if True:  # (kept indented: the old `if not args.skip_base` block)
        base = subprocess.run(["git", "merge-base", "HEAD", "small-model"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        fresh = dict(spec, optimizer_state=None)  # SM36's own setting: a fresh Muon state
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "base"
            tree.mkdir()
            archive = subprocess.run(["git", "archive", base, "scripts", "rwkv", "expertium"], cwd=ROOT,
                                     capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-f", "-", "-C", str(tree)], input=archive, check=True)
            ours, theirs, loaded_ours, loaded_theirs = run_probes([(ROOT, fresh), (tree, fresh), (ROOT, spec), (tree, spec)], tmp)
        same = [a["grad_digest"] == b["grad_digest"] and a["loss"] == b["loss"] for a, b in zip(ours["records"], theirs["records"])]
        check("D_option_absent_equals_before_sm50", all(same) and len(same) == args.updates
              and ours["final_parameters"] == theirs["final_parameters"] and ours["final_parameters"] != ours["initial_parameters"],
              base_revision=base, optimizer_state="fresh (as SM36)", updates_equal=same,
              final_parameters_equal=ours["final_parameters"] == theirs["final_parameters"],
              losses=[r["loss"] for r in ours["records"]])
        # The HybridOptimizer.param_groups fix (scripts/muon_optimizer.py): with a LOADED Muon state the old tree never
        # applies the written LR (the parameters keep SM36's values after the updates); this tree trains.
        first_same = loaded_ours["records"][0]["grad_digest"] == loaded_theirs["records"][0]["grad_digest"]
        check("D_loaded_muon_state_lr_fix", first_same and loaded_theirs["final_parameters"] == loaded_theirs["initial_parameters"]
              and loaded_ours["final_parameters"] != loaded_ours["initial_parameters"],
              first_update_gradients_equal=first_same,
              old_tree_parameters_unchanged=loaded_theirs["final_parameters"] == loaded_theirs["initial_parameters"],
              this_tree_parameters_moved=loaded_ours["final_parameters"] != loaded_ours["initial_parameters"])
    scratch = ROOT / "artifacts/small-model/sm50" / f"check-prepare-{os.getpid()}"
    sm36_decay = json.loads((SM36_DIR / "decay.json").read_text())
    extra = {k: sm36_decay["options"][k] for k in ("distillation", "state_carry", "chunk_reviews", "batch_chunks", "optimizer")}
    try:
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/expertium_decay.py"), "prepare", "--source-run", str(V2_RUN),
                        "--epoch", "12", "--out", str(scratch.relative_to(ROOT)), "--fraction", "1.0", "--shape", "one_minus_sin",
                        "--evaluate", "none", "--seed", "1", "--fresh-optimizer", "--extra-options", json.dumps(extra)],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        new_decay = json.loads((scratch / "decay.json").read_text())
        same_protocol = (scratch / "protocol.json").read_bytes() == (SM36_DIR / "protocol.json").read_bytes()
        new_decay.pop("created_utc"), sm36_decay.pop("created_utc")
        check("D_prepare_option_absent_rewrites_sm36", new_decay == sm36_decay and same_protocol,
              decay_json_equal=new_decay == sm36_decay, protocol_json_bytes_equal=same_protocol)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
