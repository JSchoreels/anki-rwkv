#!/usr/bin/env python
"""Checks for SM39 arm 1 (Muon LR x0.5 decay) and its opt-in weight snapshots (SM51 input).

  cpu --decay DIR --reference DIR --out report.json   (CPU only, minutes)
  gpu --decay DIR --out report.json                   (GPU, ~2 min: three trainer smokes; run inside a queue ticket)

cpu:
  A  configuration: decay.json options equal the reference decay's (SM36) except "optimizer", which is the expected
     Muon x0.5 spec; protocol.json equals the reference's except full_run.snapshot_users; same users, order, reviews,
     LR schedule, source epoch, fresh optimizer state;
  B  optimizer: the dict spec with SM36's lr_scale builds exactly SM36's "expertium-iter53" groups (so only lr_scale
     differs); arm 1: same parameter partition (309,888 Muon / 21,623 AdamW), Muon lr_scale exactly half, AdamW groups
     identical, Muon decoupled weight decay per unit of written LR (lr_scale * weight_decay) unchanged, empty state;
     one step on the source weights with the same gradient: every AdamW parameter bitwise equal to SM36's step, Muon
     updates about half as large;
  C  snapshots on a short real run (CPU eager, float32, 2 users of the decay, real prefetch batches with strided lanes,
     the decay's LR schedule and the arm-1 optimizer; options as the decay without the packed IPC transport, the teacher
     and the 32k chunk): run 1 without snapshots, run 2 saving master.pth + optimizer.pth exactly as the trainer's
     save_snapshot does (torch.save of master.state_dict() and optimizer.state_dict()) at every user boundary and after
     every update. Inputs, every loss and gradient norm, final weights and final optimizer state are bitwise equal; each
     saved snapshot reloads to the weights it was taken from. The trainer's save_snapshot block is checked to touch the
     model and optimizer only through those two state_dict() calls (sha256 recorded).
gpu: the real trainer (expertium_decay.run in smoke mode, 2 users x 512 reviews, 2 updates) three times on scratch copies
  of the prepared decay: snapshots off, off again, on (snapshot_users 1: a snapshot between the two updates). Pass when
  on == off bitwise (trained_parameter_sha256, every loss, gradient norm and LR) and the snapshot exists. If off != off
  (non-determinism unrelated to snapshots) the report says so and fails (the decay ticket stops).
"""
from argparse import ArgumentParser
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PEAK_LR = 7e-4
SM36_LR_SCALE = 0.0025 / PEAK_LR


def muon_spec(scale):
    """The dict form of make_optimizer's "expertium-iter53" with a Muon lr_scale (iter53 = 0.0025 / 7e-4)."""
    return {"kind": "muon", "lr_scale": scale, "momentum": 0.95, "adjust_lr_fn": "original", "groups": "matrices",
            "include_lora": True, "lora_weight_decay": 0.0, "wd_at_adamw_rate": True}


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(b"-" if t is None else t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def state_digest(state):
    """sha256 over a (nested) state_dict: keys, dtypes, shapes and bytes."""
    import torch
    h = hashlib.sha256()

    def walk(value, key=""):
        if isinstance(value, dict):
            for k in sorted(value, key=str):
                walk(value[k], f"{key}/{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{key}/{i}")
        elif isinstance(value, torch.Tensor):
            h.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            h.update(value.detach().contiguous().cpu().numpy().tobytes())
        else:
            h.update(f"{key}={value!r}".encode())
    walk(state)
    return h.hexdigest()


def digest_batch(batch):
    """sha256 of a training batch's inputs (as scripts/check_state_carry_trainer.py's probe)."""
    h = hashlib.sha256()
    for name in ("start", "labels", "label_review_th"):
        h.update(getattr(batch, name).contiguous().numpy().tobytes())
    for group in (batch.sub_gather, batch.time_shift_selects, batch.skips):
        for splits in group:
            for t in splits:
                h.update(t.contiguous().numpy().tobytes())
    h.update(json.dumps([int(batch.num_data), [[int(v) for v in s] for s in batch.sub_gather_lens]]).encode())
    return h.hexdigest()


class Report:
    def __init__(self):
        self.checks = {}

    def __call__(self, name, value, ok):
        self.checks[name] = {"value": value, "ok": bool(ok)}
        print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(value, default=str)[:600], flush=True)

    def write(self, out, extra):
        report = dict(extra, checks=self.checks, passed=all(c["ok"] for c in self.checks.values()),
                      fails=[k for k, c in self.checks.items() if not c["ok"]])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
        print("ALL OK" if report["passed"] else "FAILED " + str(report["fails"]), flush=True)
        return 0 if report["passed"] else 1


def trainer_snapshot_block():
    text = (ROOT / "scripts" / "gpu_r01_training.py").read_text(encoding="utf-8")
    start = text.index("            def save_snapshot(reason):")
    end = text.index("            try:", start)
    return text[start:end]


def cpu(args):
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    decay = json.loads((args.decay / "decay.json").read_text())
    protocol = json.loads((args.decay / "protocol.json").read_text())
    ref = json.loads((args.reference / "decay.json").read_text())
    ref_protocol = json.loads((args.reference / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    import torch
    torch.set_num_threads(4)
    report = Report()
    expected = muon_spec(args.lr_factor * SM36_LR_SCALE)

    # ---------------- A: configuration vs the reference decay (SM36)
    options, ref_options = dict(decay["options"]), dict(ref["options"])
    report("A reference optimizer is expertium-iter53", ref_options.get("optimizer"), ref_options.get("optimizer") == "expertium-iter53")
    report("A optimizer is the Muon x%g spec" % args.lr_factor, options.get("optimizer"), options.get("optimizer") == expected)
    diff = {k: (ref_options.get(k), options.get(k)) for k in set(options) | set(ref_options)
            if k != "optimizer" and options.get(k) != ref_options.get(k)}
    report("A options identical except optimizer", diff, not diff)
    same = {k: decay[k] == ref[k] for k in ("source_run", "source_epoch", "source_update_offset_end", "fraction_of_epoch_reviews",
                                             "training_users", "training_reviews", "shuffle_seed", "lr", "evaluation_set", "track")}
    report("A decay fields equal (source, users, reviews, seed, LR)", same, all(same.values()))
    fr, ref_fr = dict(protocol["full_run"]), dict(ref_protocol["full_run"])
    snapshot_users = fr.pop("snapshot_users", None)
    report("A protocol full_run equal except snapshot_users", {"snapshot_users": snapshot_users, "equal_rest": fr == ref_fr},
           fr == ref_fr and snapshot_users == args.snapshot_users and fr.get("optimizer_state") == "fresh" and "init_optimizer" not in fr)
    rest = {k: protocol[k] == ref_protocol[k] for k in protocol if k != "full_run"}
    report("A protocol rows/dataset/training equal", rest, all(rest.values()) and set(protocol) == set(ref_protocol))
    report("A snapshot points (users, review fraction)", decay.get("snapshots"), bool(decay.get("snapshots")))

    # ---------------- B: optimizer
    from expertium.model.variant_model import VariantSrsRWKV
    from expertium.variants import variant
    from scripts.expertium_options import CatchUp, make_optimizer
    master = VariantSrsRWKV(variant(options["architecture"])).float()
    source = torch.load(protocol["full_run"]["init_master"], map_location="cpu", weights_only=True)
    master.load_state_dict(source, strict=True)

    def groups(opt):
        out = []
        for name, part in (("adamw", opt.adamw), ("muon", opt.muon)):
            for g in part.param_groups:
                out.append({"part": name, "params": [id(p) for p in g["params"]],
                            **{k: v for k, v in g.items() if k != "params"}})
        return out
    iter53 = make_optimizer("expertium-iter53", master)
    as_dict = make_optimizer(muon_spec(SM36_LR_SCALE), master)
    report("B dict spec at SM36's lr_scale == expertium-iter53 (all groups and hyperparameters)", len(groups(iter53)),
           groups(iter53) == groups(as_dict) and iter53.muon.defaults == as_dict.muon.defaults and iter53.adamw.defaults == as_dict.adamw.defaults)
    arm = CatchUp(options, protocol["full_run"], protocol["rows"][:1]).optimizer(master)
    muon_n = sum(p.numel() for g in arm.muon.param_groups for p in g["params"])
    adamw_n = sum(p.numel() for g in arm.adamw.param_groups for p in g["params"])
    report("B Muon / AdamW parameters", [muon_n, adamw_n], [muon_n, adamw_n] == [309888, 21623])
    report("B fresh state empty", [len(arm.muon.state), len(arm.adamw.state)], len(arm.muon.state) == len(arm.adamw.state) == 0)
    ga, gr = groups(arm), groups(iter53)
    adamw_equal = [a for a in ga if a["part"] == "adamw"] == [r for r in gr if r["part"] == "adamw"] and arm.adamw.defaults == iter53.adamw.defaults
    report("B AdamW groups identical to SM36", sum(a["part"] == "adamw" for a in ga), adamw_equal)
    mu_a, mu_r = [a for a in ga if a["part"] == "muon"], [r for r in gr if r["part"] == "muon"]
    scales = [(a["lr_scale"], r["lr_scale"]) for a, r in zip(mu_a, mu_r)]
    decays = [(a["lr_scale"] * a["weight_decay"], r["lr_scale"] * r["weight_decay"]) for a, r in zip(mu_a, mu_r)]
    report("B Muon lr_scale exactly x%g (effective peak %g)" % (args.lr_factor, PEAK_LR * expected["lr_scale"]), scales,
           len(mu_a) == len(mu_r) and all(a == args.lr_factor * r for a, r in scales)
           and [a["params"] for a in mu_a] == [r["params"] for r in mu_r])
    rel = max(abs(a - r) / max(abs(r), 1e-30) for a, r in decays)
    report("B Muon weight decay per written LR unchanged (lr_scale * wd)", {"pairs": decays, "max_rel_diff": rel}, rel <= 1e-15)
    other = [{k: v for k, v in a.items() if k not in ("lr_scale", "weight_decay")} == {k: v for k, v in r.items() if k not in ("lr_scale", "weight_decay")}
             for a, r in zip(mu_a, mu_r)]
    report("B Muon groups otherwise identical (momentum, nesterov, adjust_lr_fn, lr)", other,
           all(other) and arm.muon.defaults == iter53.muon.defaults)
    # one step with the same gradient from the source weights
    after = {}
    lr = PEAK_LR * 0.5
    gen = torch.Generator().manual_seed(0)
    grads = [torch.randn(p.shape, generator=gen) * 1e-3 for p in master.parameters()]
    for name, spec in (("arm", options["optimizer"]), ("sm36", "expertium-iter53")):
        master.load_state_dict(source, strict=True)
        opt = make_optimizer(spec, master)
        for p, g in zip(master.parameters(), grads):
            p.grad = g.clone()
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        after[name] = [p.detach().clone() for p in master.parameters()]
        master.zero_grad(set_to_none=True)
    muon_ids = {id(p) for g in arm.muon.param_groups for p in g["params"]}
    params = list(master.parameters())
    named = dict(master.named_parameters())
    base = [source[n] for n in named]
    adamw_same = all(torch.equal(a, b) for p, a, b in zip(params, after["arm"], after["sm36"]) if id(p) not in muon_ids)
    report("B one step: AdamW parameters bitwise equal to SM36", sum(id(p) not in muon_ids for p in params), adamw_same)
    ratios = [float((a - s).norm() / (b - s).norm()) for p, a, b, s in zip(params, after["arm"], after["sm36"], base)
              if id(p) in muon_ids and float((b - s).norm()) > 0]
    report("B one step: Muon update norm ratio arm / SM36 (expect ~%g)" % args.lr_factor,
           {"min": min(ratios), "max": max(ratios), "n": len(ratios)},
           len(ratios) > 0 and all(abs(r - args.lr_factor) < 0.05 for r in ratios))

    # ---------------- C: snapshots on a short real run
    block = trainer_snapshot_block()
    touches = [line.strip() for line in block.splitlines() if "master" in line or "optimizer" in line]
    report("C trainer save_snapshot touches model/optimizer only via state_dict() saves",
           {"lines": touches, "sha256": hashlib.sha256(block.encode()).hexdigest()},
           touches == ['torch.save(master.state_dict(), folder / "master.pth")',
                       'torch.save(optimizer.state_dict(), folder / "optimizer.pth")'])
    from concurrent.futures import ThreadPoolExecutor
    from rwkv.train_rwkv import CLIP
    from scripts.gpu_r01_prefetch import training_batches
    counts = {r["user_id"]: r["train_range"][1] - r["train_range"][0] for r in protocol["rows"]}
    small = [r for r in protocol["rows"] if args.min_reviews <= counts[r["user_id"]] <= args.max_reviews][:args.users]
    plain = {k: v for k, v in options.items() if k not in ("packed_ipc", "distillation")}
    plain.update(chunk_reviews=args.chunk, batch_chunks=args.lanes)
    span = plain["chunk_reviews"] * plain["batch_chunks"]

    def batches():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return [(row, item, item.batch.to(torch.device("cpu")))
                    for row, item in training_batches(small, span, plain["training_seed"], pool, plain, None, 2)]

    def train(snapshots, scratch):
        prepared = batches()  # materialized first: the in-process preparation thread reseeds the global RNG
        model = VariantSrsRWKV(variant(plain["architecture"])).float()
        model.load_state_dict(source, strict=True)
        catch_up = CatchUp(plain, protocol["full_run"], small)
        opt = catch_up.optimizer(model)
        model.train()
        log, saved, last_user, updates, reviews = [], [], None, 0, 0

        def save(tag):
            folder = scratch / f"{len(saved):03d}-{tag}"
            folder.mkdir(parents=True)
            torch.save(model.state_dict(), folder / "master.pth")  # the trainer's two save_snapshot calls
            torch.save(opt.state_dict(), folder / "optimizer.pth")
            saved.append((folder, state_digest(model.state_dict())))
        for row, item, batch in prepared:
            user = row["user_id"]
            if snapshots and last_user is not None and user != last_user:
                save(f"user-{last_user}")
            last_user = user
            opt.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            catch_up.carry.attach(item, batch, user)
            stats = catch_up.loss(model, batch, reviews, (user, item.offset, item.reviews, item.seed), item.batch.labels)
            stats.average_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            lr = catch_up.learning_rate(updates, int(protocol["full_run"]["update_offset"]), reviews)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            updates += 1
            reviews += item.reviews
            log.append([user, item.offset, item.reviews, digest_batch(batch), float(stats.average_loss.detach()),
                        stats.average_loss.detach().numpy().tobytes().hex(), norm.detach().numpy().tobytes().hex(), lr])
            if snapshots:
                save(f"update-{updates}")
        return {"log": log, "weights": state_digest(model.state_dict()), "optimizer": state_digest(opt.state_dict()),
                "saved": saved, "model_keys": {k: str(v.dtype) for k, v in model.state_dict().items()}}

    scratch = ROOT / "artifacts" / "small-model" / "smoke" / f"check-decay-snapshots-{os.getpid()}"
    try:
        off = train(False, scratch / "off")
        on = train(True, scratch / "on")
        report("C users / updates of the short run", {"users": [r["user_id"] for r in small], "reviews": [counts[r["user_id"]] for r in small],
                                                   "chunk": args.chunk, "lanes": args.lanes, "updates": len(off["log"]),
                                                   "snapshots_saved": len(on["saved"])},
               len(off["log"]) >= 4 and len({r[0] for r in off["log"]}) == len(small) >= 2)
        report("C inputs, losses, gradient norms, LRs bitwise equal (off vs on)", [r[4] for r in off["log"]], off["log"] == on["log"])
        report("C final weights bitwise equal", [off["weights"][:16], on["weights"][:16]], off["weights"] == on["weights"])
        report("C final optimizer state bitwise equal", [off["optimizer"][:16], on["optimizer"][:16]], off["optimizer"] == on["optimizer"])
        reload_ok = [state_digest(torch.load(f / "master.pth", map_location="cpu", weights_only=True)) == d for f, d in on["saved"]]
        opt_load = torch.load(on["saved"][-1][0] / "optimizer.pth", map_location="cpu", weights_only=True)
        report("C every snapshot reloads to the weights it was taken from; optimizer.pth has adamw/muon",
               {"snapshots": len(reload_ok), "optimizer_keys": sorted(opt_load)}, all(reload_ok) and sorted(opt_load) == ["adamw", "muon"])
        report("C snapshot state_dict keys/dtypes = model's", len(on["model_keys"]),
               {k: str(v.dtype) for k, v in torch.load(on["saved"][0][0] / "master.pth", map_location="cpu", weights_only=True).items()} == on["model_keys"])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return report.write(args.out, {"mode": "cpu", "decay": str(args.decay), "reference": str(args.reference), "torch": torch.__version__})


PROBE = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
         "from scripts.expertium_decay import run; raise SystemExit(run(Path(sys.argv[2]), smoke=True))")


def gpu(args):
    decay = json.loads((args.decay / "decay.json").read_text())
    protocol = json.loads((args.decay / "protocol.json").read_text())
    report = Report()
    runs = {}
    base = ROOT / "artifacts" / "small-model" / "smoke" / f"{args.decay.name}-snapcheck-{os.getpid()}"
    for name, snap in (("off_a", 0), ("off_b", 0), ("on", 1)):
        scratch = base / name
        scratch.mkdir(parents=True)
        d = json.loads(json.dumps(decay))
        dist = d["options"].get("distillation")
        if isinstance(dist, dict):
            d["options"]["distillation"] = {k: v for k, v in dist.items() if k != "cache"}
        p = json.loads(json.dumps(protocol))
        p["full_run"].pop("snapshot_users", None)
        if snap:
            p["full_run"]["snapshot_users"] = snap
        (scratch / "decay.json").write_text(json.dumps(d, indent=1))
        (scratch / "protocol.json").write_text(json.dumps(p, indent=1))
        code = subprocess.run([sys.executable, "-X", "utf8", "-c", PROBE, str(ROOT), str(scratch)], cwd=ROOT).returncode
        case = scratch / "expertium" / decay["track"]
        result = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
        log = [json.loads(line) for line in (case / "updates.jsonl").read_text().splitlines()] if (case / "updates.jsonl").exists() else []
        runs[name] = {"exit": code, "status": result.get("status"), "sha": result.get("trained_parameter_sha256"),
                      "steps": [(r["step"], r["user_id"], r["offset"], r["reviews"], r["seed"], r["loss"], r["gradient_norm"], r["lr"]) for r in log],
                      "snapshots": sorted(x.name for x in (case / "snapshots").glob("users-*")) if (case / "snapshots").exists() else []}
        report(f"run {name} complete", {k: runs[name][k] for k in ("exit", "status", "snapshots")} | {"updates": len(log)},
               code == 0 and result.get("status") == "complete" and len(log) >= 2)
    deterministic = runs["off_a"]["sha"] == runs["off_b"]["sha"] and runs["off_a"]["steps"] == runs["off_b"]["steps"]
    report("off vs off bitwise (trainer determinism)", runs["off_a"]["sha"], deterministic)
    report("on vs off bitwise: final weights", [runs["on"]["sha"], runs["off_a"]["sha"]], runs["on"]["sha"] == runs["off_a"]["sha"] is not None)
    report("on vs off bitwise: every loss, gradient norm, LR", runs["on"]["steps"], runs["on"]["steps"] == runs["off_a"]["steps"])
    report("on run saved a snapshot between the updates; off runs none", [runs[n]["snapshots"] for n in runs],
           runs["on"]["snapshots"] == ["users-00001"] and not runs["off_a"]["snapshots"] and not runs["off_b"]["snapshots"])
    code = report.write(args.out, {"mode": "gpu", "decay": str(args.decay), "runs": runs})
    if code == 0:
        shutil.rmtree(base, ignore_errors=True)
    return code


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--decay", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="cpu: the reference decay folder (SM36)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--snapshot-users", type=int, default=1250)
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--min-reviews", type=int, default=2500)
    parser.add_argument("--max-reviews", type=int, default=4000)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--lanes", type=int, default=2)
    args = parser.parse_args()
    return cpu(args) if args.mode == "cpu" else gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
