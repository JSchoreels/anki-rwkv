#!/usr/bin/env python
"""SM15 first pruning step: depth cut of the context streams (docs/SMALL_MODEL_PRUNING.md 2.1-2.4, option b2).

Presets `w64-d{D}p{P}u{U}[-nocmix]` (expertium/variants.py) keep the leading D deck / P preset / U user layers of w64.
Everything here is CPU and read-only on the parent experiment; GPU work goes through the runbook's queue tickets.

  depth      --src master.pth --arch w64-d2p1u1 --out pruned.pth
             drop rwkv_modules.<s>.blocks.<k>.* for k >= the new depth, strict-load into build(arch), check every kept
             tensor is bit-identical (-nocmix: the stripped channel mixers keep the child's 5-parameter dummies).
  make-run   --parent-exp DIR --epoch N --arch ARCH --name NAME
             synthetic experiment artifacts/accuracy/experiments/NAME/ whose run/epoch-N holds the pruned master.pth, a
             fresh AdamW optimizer.pth and a complete result.json (update_offset_end 0); run.json/experiment.json are
             the parent's with options.architecture = ARCH and `pruned_from`. The parent is only read; refuses an
             existing NAME. Its options (chunk_reviews 2048, batch_chunks 16, state_carry strided lanes, replay KD) carry
             over unchanged, so a decay prepared from it with --extra-options keeps them.
  ft-prepare --run RUN_DIR --epoch N --out DIR [--fraction 0.5 --seed 2]
             expertium_decay.prepare, then the lr_schedule is deleted from protocol.json: constant 7e-4 after the 5-update
             warmup and KD alpha 0.9. TRAP: `--final 1.0` would also give a constant LR but LearningRate.decaying() is
             then true for the whole run and KD would use decay_alpha 0.5.
  promote    --ft DIR --run RUN_DIR --epoch N+1
             copy the finished fine-tune's master/optimizer/result into the run as epoch N+1, so
             expertium_decay.py prepare --source-run RUN_DIR --epoch N+1 can decay from it.
  decide     --jsonl ROWS --label NEW/w8+s8 --ref-jsonl ROWS --ref-label P_DEC/w8+s8 --fork-jsonl ROWS --out FILE.json
             design 2.3: paired review-weighted delta % vs the parent (curve, instant) with a 95% user bootstrap
             (2,000 draws, seed 0): PASS / EXTEND / FAIL.
  screen-report --jsonl screen.jsonl --parent P_dec/w8+s8 --out report   (design 2.1 CPU screen: table and the choice rule)
  --self-test  five checks of design 2.4 plus the carry-options check on CPU; writes artifacts/small-model/check-prune.json
"""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RWKV_NO_JIT", "1")  # eager modules, as the CPU scorers and checks

MODEL, TRACK = "expertium", "E89-F65e-F77a"
STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]
BLOCK_KEY = re.compile(r"^rwkv_modules\.(\d+)\.blocks\.(\d+)\.")
EXPERIMENTS = ROOT / "artifacts" / "accuracy" / "experiments"
MARGIN = 0.25  # design 2.3: percent, per head
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- depth
def convert_state(state, arch):
    """Parent (w64) state dict -> state dict of `arch`, plus a report. Raises on anything not an exact depth cut."""
    import torch
    from expertium.variants import build, variant
    child = build(arch).float()
    target = child.state_dict()
    depths = {name: config.n_layers for name, config in variant(arch).base.modules}
    out, kept, reinit = {}, 0, []
    for key, tensor in target.items():
        if key not in state:
            raise KeyError(f"{arch}: parent checkpoint has no {key}")
        if state[key].shape == tensor.shape:
            out[key] = state[key].detach().clone().to(tensor.dtype if state[key].dtype != tensor.dtype else state[key].dtype)
            kept += 1
        elif ".channel_mixer." in key:  # -nocmix: a real mixer becomes the stripped 5-parameter dummy (never called)
            out[key] = tensor.detach().clone()
            reinit.append(key)
        else:
            raise ValueError(f"{arch}: shape mismatch at {key}: {tuple(state[key].shape)} vs {tuple(tensor.shape)}")
    dropped = sorted(key for key in state if key not in target)
    for key in dropped:
        match = BLOCK_KEY.match(key)
        if not match or int(match[2]) < depths[STREAMS[int(match[1])]]:
            raise ValueError(f"{arch}: dropped key {key} is not a trailing layer")
    child.load_state_dict(out, strict=True)
    for key, tensor in out.items():
        if key not in reinit and not torch.equal(tensor, state[key]):
            raise AssertionError(f"{key} is not bit-identical after the strict load")
    layers = {STREAMS[s]: sorted({int(BLOCK_KEY.match(k)[2]) for k in dropped if int(BLOCK_KEY.match(k)[1]) == s}) for s in range(5)}
    return out, {"arch": arch, "parameters": sum(p.numel() for p in child.parameters()), "kept_tensors": kept,
                 "dropped_tensors": len(dropped), "dropped_layers": {k: v for k, v in layers.items() if v},
                 "reinitialised_dummies": len(reinit)}


def depth(src, arch, out):
    import torch
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    state = torch.load(src, map_location="cpu", weights_only=True)
    pruned, report = convert_state(state, arch)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned, out)
    report["src"], report["out"] = str(src), str(out)
    print(json.dumps(report))
    return report


# ---------------------------------------------------------------------------------------------------- make-run
def case_folder(run_dir, epoch):
    return Path(run_dir) / f"epoch-{epoch:02d}" / MODEL / TRACK


def fresh_optimizer_state(options, model):
    from scripts.expertium_options import CatchUp
    return CatchUp(options, {}, []).optimizer(model).state_dict()


def make_run(parent_exp, epoch, arch, name):
    import torch
    from expertium.variants import build
    parent_exp = Path(parent_exp).resolve()
    dst = EXPERIMENTS / name
    if dst.exists():
        raise FileExistsError(f"{dst} exists; pick a new name")
    config = json.loads((parent_exp / "run" / "run.json").read_text())
    experiment = json.loads((parent_exp / "experiment.json").read_text())
    folder = case_folder(parent_exp / "run", epoch)
    parent_result = json.loads((folder / "result.json").read_text())
    if parent_result.get("status") != "complete":
        raise RuntimeError(f"{folder} is not a complete epoch")
    state = torch.load(folder / "master.pth", map_location="cpu", weights_only=True)
    pruned, report = convert_state(state, arch)
    provenance = {"experiment": parent_exp.name, "experiment_path": str(parent_exp), "epoch": epoch, "parent_architecture":
                  config["options"].get("architecture"), "master_sha256": hashlib.sha256((folder / "master.pth").read_bytes()).hexdigest(),
                  "created_utc": datetime.now(timezone.utc).isoformat(), **{k: report[k] for k in ("parameters", "dropped_layers")}}
    options = dict(config["options"], architecture=arch)
    child_run = dict(config, options=options, pruned_from=provenance, continued_from={"experiment": parent_exp.name, "epoch": epoch},
                     epochs=epoch)
    child_experiment = dict(experiment, name=name, created_utc=provenance["created_utc"], pruned_from=provenance,
                            note=f"{name}: SM15 depth cut {arch} of {parent_exp.name} epoch {epoch} (scripts/small_model_prune.py); "
                                 "docs/SMALL_MODEL_PRUNING.md", options=dict(experiment["options"], architecture=arch),
                            continued_from={"experiment": parent_exp.name, "epoch": epoch}, epochs=epoch)
    model = build(arch).float()
    model.load_state_dict(pruned, strict=True)
    target = case_folder(dst / "run", epoch)
    target.mkdir(parents=True)
    torch.save(pruned, target / "master.pth")
    torch.save(fresh_optimizer_state(options, model), target / "optimizer.pth")
    atomic_json(target / "result.json", {"model": MODEL, "track": TRACK, "status": "complete", "smoke_only": False,
                                         "options": options, "update_offset_end": 0, "updates": 0, "pruned_from": provenance,
                                         "note": "synthetic epoch: pruned master and a fresh AdamW state"})
    atomic_json(dst / "run" / "run.json", child_run)
    atomic_json(dst / "experiment.json", child_experiment)
    print(json.dumps({"run": str(dst / "run"), "epoch": epoch, **report}))
    return dst


# ---------------------------------------------------------------------------------------------------- ft-prepare / promote
def ft_prepare(run, epoch, out, fraction, seed):
    from scripts import expertium_decay
    run, out = Path(run).resolve(), Path(out).resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        raise ValueError("--out must be under this checkout's artifacts/")
    expertium_decay.prepare(run, epoch, out, fraction, "one_minus_sin", "none", seed, None, 0.0)
    protocol = json.loads((out / "protocol.json").read_text())
    protocol["full_run"].pop("lr_schedule", None)
    atomic_json(out / "protocol.json", protocol)
    decay = json.loads((out / "decay.json").read_text())
    reviews = decay["training_reviews"]
    decay["lr"] = f"constant 7e-4 after the 5-update warmup (no lr_schedule), KD alpha {decay['options']['distillation']['alpha']}, {reviews:,} reviews"
    decay["kind"] = "ft"
    atomic_json(out / "decay.json", decay)
    print(json.dumps({"ft": str(out), "lr": decay["lr"], "training_users": decay["training_users"]}))


def promote(ft, run, epoch):
    ft, run = Path(ft).resolve(), Path(run).resolve()
    decay = json.loads((ft / "decay.json").read_text())
    source = ft / MODEL / decay["track"]
    result = json.loads((source / "result.json").read_text())
    if result.get("status") != "complete" or result.get("smoke_only"):
        raise RuntimeError(f"{source} is not a finished fine-tune (status {result.get('status')})")
    for name in ("master.pth", "optimizer.pth"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    target = case_folder(run, epoch)
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.mkdir(parents=True)
    for name in ("master.pth", "optimizer.pth", "result.json"):
        shutil.copy2(source / name, target / name)
    print(json.dumps({"promoted": str(target), "update_offset_end": result["update_offset_end"], "updates": result.get("updates"),
                      "reviews_per_second": result.get("reviews_per_second")}))


# ---------------------------------------------------------------------------------------------------- decide
HEADS = {"ahead": "curve", "instant": "instant"}


def load_rows(path, label):
    per = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        bench = row.get("benchmark", {})
        if row.get("label") == label and "ahead" in bench and "instant" in bench:
            per[row["user_id"]] = {"curve": (bench["ahead"]["LogLoss"], bench["ahead"]["size"], bench["ahead"]["RMSE(bins)"]),
                                   "instant": (bench["instant"]["LogLoss"], bench["instant"]["size"], bench["instant"]["RMSE(bins)"]),
                                   "reviews": row.get("reviews", 0)}
    return per


def boot_pct(new, ref, sizes, n=2000, seed=0):
    """Review-weighted (new - ref) / ref in percent: point estimate and 95% user-bootstrap interval."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n, len(sizes)))
    w = sizes[idx]
    a = (new[idx] * w).sum(1) / w.sum(1)
    b = (ref[idx] * w).sum(1) / w.sum(1)
    pct = (a / b - 1) * 100
    point = ((new * sizes).sum() / (ref * sizes).sum() - 1) * 100
    return float(point), [float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))]


def decide(args):
    import numpy as np
    child = load_rows(args.jsonl, args.label)
    parent = load_rows(args.ref_jsonl, args.ref_label)
    fork = load_rows(args.fork_jsonl, "fork/w8+s8") if args.fork_jsonl else {}
    users = sorted(set(child) & set(parent) & (set(fork) if fork else set(child)))
    out = {"child": args.label, "parent": args.ref_label, "users": len(users), "margin_pct": MARGIN,
           "rule": "PASS upper95 <= +0.25% on both heads; EXTEND point <= +0.25% on both, an upper above; FAIL point > +0.25% on either"}
    if len(users) < args.min_users:
        out.update(verdict="INCOMPLETE", reason=f"{len(users)} common users < {args.min_users}")
        atomic_json(args.out, out)
        print(json.dumps(out))
        return 3
    sizes = {h: np.array([child[u][h][1] for u in users], dtype=float) for h in HEADS.values()}
    verdict_pass, verdict_fail = True, False
    for head in HEADS.values():
        new = np.array([child[u][head][0] for u in users])
        ref = np.array([parent[u][head][0] for u in users])
        point, ci = boot_pct(new, ref, sizes[head])
        entry = {"child": float((new * sizes[head]).sum() / sizes[head].sum()), "parent": float((ref * sizes[head]).sum() / sizes[head].sum()),
                 "delta_pct": point, "ci95_pct": ci,
                 "rmse_bins_child": float((np.array([child[u][head][2] for u in users]) * sizes[head]).sum() / sizes[head].sum()),
                 "rmse_bins_parent": float((np.array([parent[u][head][2] for u in users]) * sizes[head]).sum() / sizes[head].sum())}
        if fork:
            fk = np.array([fork[u][head][0] for u in users])
            entry["child_vs_fork_pct"], entry["child_vs_fork_ci95_pct"] = boot_pct(new, fk, sizes[head])
            entry["parent_vs_fork_pct"], _ = boot_pct(ref, fk, sizes[head])
        order = np.argsort([child[u]["reviews"] for u in users])
        thirds = np.array_split(order, 3)
        entry["tertiles_delta_pct"] = {name: boot_pct(new[t], ref[t], sizes[head][t], n=500)[0] for name, t in zip(("light", "mid", "heavy"), thirds)}
        verdict_pass &= ci[1] <= MARGIN
        verdict_fail |= point > MARGIN
        out[head] = entry
    if args.fp32_label:
        fp32 = load_rows(args.jsonl, args.fp32_label)
        common = [u for u in users if u in fp32]
        out["int8_cost"] = {"users": len(common), "limit_points": 0.02}
        for head in HEADS.values():
            sz = np.array([child[u][head][1] for u in common], dtype=float)
            if len(common):
                q = np.array([child[u][head][0] for u in common]); f = np.array([fp32[u][head][0] for u in common])
                out["int8_cost"][head] = {"w8s8_minus_fp32_logloss": float(((q - f) * sz).sum() / sz.sum()),
                                          "pct": float(((q * sz).sum() / (f * sz).sum() - 1) * 100)}
    out["verdict"] = "FAIL" if verdict_fail else ("PASS" if verdict_pass else "EXTEND")
    atomic_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("child", "users", "verdict")} | {h: {k: out[h][k] for k in ("delta_pct", "ci95_pct")} for h in HEADS.values()}))
    return 0


# ---------------------------------------------------------------------------------------------------- screen
SCREEN_ARMS = [("D9 (d2p2u2)", "w64-d2p2u2"), ("D7 (d2p1u1)", "w64-d2p1u1"), ("D7-nocmix", "w64-d2p1u1-nocmix"),
               ("deck cut d2p3u3", "w64-d2p3u3"), ("preset cut d4p1u3", "w64-d4p1u3"), ("user cut d4p3u1", "w64-d4p3u1")]
SINGLE = {"deck": ("w64-d2p3u3", "w64-d3p1u1"), "preset": ("w64-d4p1u3", "w64-d2p2u1"), "user": ("w64-d4p3u1", "w64-d2p1u2")}
SHARE = 0.6


def screen_report(args):
    """Design 2.1 choice rule: the GPU arm is D7 (w64-d2p1u1); if one single-stream cut accounts for > 60% of D7's zero-shot
    increase on either head, that stream stays one layer deeper (D8: deck -> d3p1u1, preset -> d2p2u1, user -> d2p1u2)."""
    import numpy as np
    labels = {"parent": args.parent, **{arch: f"{arch}/w8+s8" for _, arch in SCREEN_ARMS}}
    rows = {name: load_rows(args.jsonl, label) for name, label in labels.items()}
    present = {name: len(r) for name, r in rows.items()}
    users = sorted(set.intersection(*(set(r) for r in rows.values() if r)))
    missing = [name for name, r in rows.items() if not r]
    if missing:
        raise SystemExit(f"no rows for {missing}")
    from expertium.variants import PRUNED_PARAMETERS, build
    params = {arch: sum(p.numel() for p in build(arch).parameters()) for _, arch in SCREEN_ARMS}
    params["parent"] = 331_511
    sizes = {h: np.array([rows["parent"][u][h][1] for u in users], dtype=float) for h in HEADS.values()}
    base = {h: np.array([rows["parent"][u][h][0] for u in users]) for h in HEADS.values()}
    table, delta = [], {}
    for title, arch in SCREEN_ARMS:
        entry = {"arm": title, "arch": arch, "parameters": params[arch]}
        values = {h: np.array([rows[arch][u][h][0] for u in users]) for h in HEADS.values()}
        for h in HEADS.values():
            pct, ci = boot_pct(values[h], base[h], sizes[h])
            ll = float((values[h] * sizes[h]).sum() / sizes[h].sum())
            entry[h] = {"logloss": ll, "delta_pct": pct, "ci95_pct": ci,
                        "delta_ll": ll - float((base[h] * sizes[h]).sum() / sizes[h].sum())}
        delta[arch] = entry
        table.append(entry)
    shares, d7 = {}, delta["w64-d2p1u1"]
    for stream, (single, deeper) in SINGLE.items():
        shares[stream] = {h: (delta[single][h]["delta_ll"] / d7[h]["delta_ll"]) if d7[h]["delta_ll"] > 0 else None for h in HEADS.values()}
    best = max(((max((v for v in shares[s].values() if v is not None), default=-1), s) for s in shares))
    arch = "w64-d2p1u1"
    reason = "no single-stream cut accounts for > 60% of D7's zero-shot increase on either head: D7"
    if best[0] > SHARE:
        arch = SINGLE[best[1]][1]
        reason = f"the {best[1]} cut accounts for {best[0] * 100:.0f}% of D7's zero-shot increase (> 60%): keep the {best[1]} stream one layer deeper, {arch}"
    parent_ll = {h: float((base[h] * sizes[h]).sum() / sizes[h].sum()) for h in HEADS.values()}
    result = {"users": len(users), "reviews": int(sizes["curve"].sum()), "parent": args.parent, "parent_logloss": parent_ll,
              "parent_parameters": 331_511, "arms": table, "single_stream_share_of_D7_increase": shares, "arch": arch, "reason": reason,
              "arch_parameters": PRUNED_PARAMETERS.get(arch)}
    lines = [f"# SM15 CPU screen: zero-shot depth cuts of P_dec ({len(users)} feature-screen users, w8+s8, review-weighted)", "",
             f"Parent {args.parent}: curve {parent_ll['curve']:.5f}, instant {parent_ll['instant']:.5f}. No fine-tuning: an upper bound "
             "of the post-fine-tune cost. Delta % vs the parent, 95% user-bootstrap CI in brackets.", "",
             "| arm | params | curve LogLoss | curve delta | instant LogLoss | instant delta |", "|---|---:|---:|---:|---:|---:|"]
    for e in table:
        c, i = e["curve"], e["instant"]
        lines.append(f"| {e['arm']} | {e['parameters']:,} | {c['logloss']:.5f} | {c['delta_pct']:+.2f}% [{c['ci95_pct'][0]:+.2f}, {c['ci95_pct'][1]:+.2f}] | "
                     f"{i['logloss']:.5f} | {i['delta_pct']:+.2f}% [{i['ci95_pct'][0]:+.2f}, {i['ci95_pct'][1]:+.2f}] |")
    lines += ["", "Share of D7's zero-shot LogLoss increase carried by each single-stream cut (curve / instant):", ""]
    for stream, v in shares.items():
        fmt = lambda x: "n/a" if x is None else f"{x * 100:.0f}%"
        lines.append(f"- {stream}: {fmt(v['curve'])} / {fmt(v['instant'])}")
    lines += ["", f"**Chosen ARCH: {arch}** ({reason})."]
    text = chr(10).join(lines) + chr(10)
    Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
    atomic_json(Path(args.out).with_suffix(".json"), result)
    print(text)
    return 0


# ---------------------------------------------------------------------------------------------------- self-test
def self_test(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    started = time.perf_counter()
    torch.set_num_threads(4)
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    from expertium.variants import PRUNED_PARAMETERS, build, variant
    parent_exp = Path(args.parent_exp)
    parent_ckpt = Path(args.parent_ckpt)
    report, fails = {"parent_experiment": str(parent_exp), "parent_epoch": args.epoch, "parent_checkpoint": str(parent_ckpt), "checks": {}}, []

    def record(name, ok, **info):
        report["checks"][name] = {"pass": bool(ok), **info}
        print(f"[prune-self-test] {'ok  ' if ok else 'FAIL'} {name} {json.dumps(info)[:300]}", flush=True)
        if not ok:
            fails.append(name)

    # ---- 2. strict loads and parameter counts
    state = torch.load(parent_ckpt, map_location="cpu", weights_only=True)
    counts = {}
    for arch, expected in PRUNED_PARAMETERS.items():
        pruned, info = convert_state(state, arch)
        counts[arch] = {"parameters": info["parameters"], "expected": expected, "dropped_layers": info["dropped_layers"],
                        "reinitialised_dummies": info["reinitialised_dummies"]}
    w64 = sum(p.numel() for p in build("w64").parameters())
    record("strict_loads_and_counts", all(c["parameters"] == c["expected"] for c in counts.values()) and w64 == 331_511,
           w64=w64, archs=counts)

    # ---- 1. child forward == parent forward with the dropped blocks skipped (identity), fp32, eval, one user <= 3k reviews
    from scripts import compress_common as cc
    run = cc.run_info()
    users = json.loads((parent_exp / "run" / "run.json").read_text())
    counts_by_user = users["review_counts"]
    candidates = sorted((int(u) for u, n in counts_by_user.items() if 1500 <= n <= 3000 and int(u) < 5000), key=lambda u: (counts_by_user[str(u)], u))
    user = candidates[len(candidates) // 2]
    batch = cc.full_history_batch(cc.data_path(run), user)
    equal = {}
    for arch in ("w64-d2p1u1", "w64-d2p2u2"):
        pruned, _ = convert_state(state, arch)
        child = build(arch).float().eval()
        child.load_state_dict(pruned, strict=True)
        parent = build("w64").float().eval()
        parent.load_state_dict(state, strict=True)
        parent.layer_steps = list(child.layer_steps)  # the dropped blocks never run = identity residual branches
        with torch.inference_mode():
            a = child.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects, batch.skips,
                                            batch.num_data)
            b = parent.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects, batch.skips,
                                             batch.num_data)
            full = build("w64").float().eval()
            full.load_state_dict(state, strict=True)
            c = full.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects, batch.skips,
                                           batch.num_data)
        diff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b) if torch.is_tensor(x))
        zero_shot = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, c) if torch.is_tensor(x))
        equal[arch] = {"max_abs_diff_vs_parent_with_identity": diff, "max_abs_diff_vs_full_parent (informational)": zero_shot,
                       "layer_steps": len(child.layer_steps), "parent_steps": len(full.layer_steps)}
    record("forward_equals_parent_with_identity_blocks", all(v["max_abs_diff_vs_parent_with_identity"] <= 1e-6 for v in equal.values()),
           user=user, reviews=counts_by_user[str(user)], **equal)

    # ---- make-run in a scratch experiment (read-only on the parent), then 3/4/6 from it
    name = f"selftest-prune-{os.getpid()}"
    arch = "w64-d2p1u1"
    ft_dir = ROOT / "artifacts" / "small-model" / "smoke" / f"{name}-ft"
    dec_dir = ROOT / "artifacts" / "small-model" / "smoke" / f"{name}-dec"
    try:
        dst = make_run(parent_exp, args.epoch, arch, name)
        run_dir = dst / "run"
        folder = case_folder(run_dir, args.epoch)
        # 3. fresh optimizer.pth loads and its first step equals a fresh AdamW's
        child_run = json.loads((run_dir / "run.json").read_text())
        model_a = build(arch).float()
        model_a.load_state_dict(torch.load(folder / "master.pth", map_location="cpu", weights_only=True), strict=True)
        model_b = copy.deepcopy(model_a)
        from scripts.expertium_options import CatchUp
        opt_a = CatchUp(child_run["options"], {}, []).optimizer(model_a)
        saved = torch.load(folder / "optimizer.pth", map_location="cpu", weights_only=True)
        opt_a.load_state_dict(saved)
        opt_b = CatchUp(child_run["options"], {}, []).optimizer(model_b)
        gen = torch.Generator().manual_seed(7)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            g = torch.randn(pa.shape, generator=gen)
            pa.grad, pb.grad = g.clone(), g.clone()
        for opt in (opt_a, opt_b):
            for group in opt.param_groups:
                group["lr"] = 7e-4 / 5
            opt.step()
        same = all(torch.equal(x, y) for x, y in zip(model_a.state_dict().values(), model_b.state_dict().values()))
        record("fresh_optimizer_first_step_equals_fresh_adamw", same and not saved["state"], saved_state_entries=len(saved["state"]),
               param_groups=len(saved["param_groups"]), step_equal=same)
        # 4. ft-prepare output has no lr_schedule and CatchUp.loss sees decaying false (so KD alpha 0.9 is used)
        ft_prepare(run_dir, args.epoch, ft_dir, 0.5, 2)
        protocol = json.loads((ft_dir / "protocol.json").read_text())
        decay = json.loads((ft_dir / "decay.json").read_text())
        seen = {}

        class StubTeacher:
            def targets(self, batch, decaying, key, labels):
                seen["decaying"] = decaying
                return "mix"

        class StubModel:
            def get_loss(self, batch, kd_mix=None):
                seen["kd_mix"] = kd_mix
                return "loss"

        catch_up = CatchUp(decay["options"], protocol["full_run"], protocol["rows"][:3])
        catch_up.teacher, catch_up.carry = StubTeacher(), None
        catch_up.loss(StubModel(), None, 10**9, None, None)  # reviews_done far past any schedule end
        d = decay["options"]["distillation"]
        record("ft_prepare_constant_lr_kd_alpha_09", "lr_schedule" not in protocol["full_run"] and catch_up.schedule is None
               and seen == {"decaying": False, "kd_mix": "mix"} and d["alpha"] == 0.9 and d["kd_source"] == "replay"
               and protocol["full_run"]["update_offset"] == 0 and decay["fraction_of_epoch_reviews"] == 0.5 and decay["shuffle_seed"] == 2
               and catch_up.learning_rate(0, 0, 0) < catch_up.learning_rate(10, 0, 0) == 7e-4,
               seen=seen, alpha=d["alpha"], decay_alpha=d["decay_alpha"], warmup_lr_update0=catch_up.learning_rate(0, 0, 0),
               lr_update10=catch_up.learning_rate(10, 0, 0), training_users=decay["training_users"], training_reviews=decay["training_reviews"])
        # 6a. lanes / state_carry / replay / chunk options are carried into the child run and a decay prepared from it
        from scripts import expertium_decay
        expertium_decay.prepare(run_dir, args.epoch, dec_dir, 1.0, "one_minus_sin", "none", 1, {"chunk_reviews": 32768, "batch_chunks": 1}, 0.0)
        dopt = json.loads((dec_dir / "decay.json").read_text())["options"]
        popt = json.loads((parent_exp / "run" / "run.json").read_text())["options"]
        carried = {k: child_run["options"].get(k) == popt.get(k) for k in popt if k != "architecture"}
        record("options_carried_into_child_and_decay", all(carried.values()) and child_run["options"]["architecture"] == arch
               and dopt["architecture"] == arch and dopt["state_carry"] == {"lanes": "strided"} and dopt["chunk_reviews"] == 32768
               and dopt["batch_chunks"] == 1 and dopt["gpu_memory_fraction"] == 0.8 and dopt["distillation"]["kd_source"] == "replay",
               state_carry=dopt["state_carry"], chunk_reviews=dopt["chunk_reviews"], batch_chunks=dopt["batch_chunks"],
               architecture=dopt["architecture"], differing_keys=[k for k, v in carried.items() if not v])
        # 6b. strided lanes carry + replay spec work with the pruned arch on CPU (2 lanes of 2,048 on a 5-window user)
        from scripts.expertium_options import ReplayTargets
        from scripts.gpu_r01_prefetch import training_batches
        from scripts.state_carry_train import validate_options
        validate_options(dopt)
        ReplayTargets(dopt["distillation"])
        protocol_rows = json.loads((parent_exp / "run" / f"epoch-{args.epoch:02d}" / "protocol.json").read_text())["rows"]
        row = min((r for r in protocol_rows if 3 * 2048 <= r["train_range"][1] - r["train_range"][0] <= 5 * 2048),
                  key=lambda r: r["train_range"][1] - r["train_range"][0])
        options = {k: v for k, v in child_run["options"].items() if k not in ("packed_ipc", "distillation")}
        options["batch_chunks"] = 2
        child = build(arch).float()
        child.load_state_dict(torch.load(folder / "master.pth", map_location="cpu", weights_only=True), strict=True)
        child.train()
        cu = CatchUp(options, {}, [row])
        with ThreadPoolExecutor(max_workers=1) as pool:
            prepared = list(training_batches([row], 2 * options["chunk_reviews"], options["training_seed"], pool, options, None, 2))
        losses, grads_finite, lanes = [], True, []
        for r, item in prepared:
            b = item.batch.to(torch.device("cpu"))
            child.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            cu.carry.attach(item, b, r["user_id"])
            stats = cu.loss(child, b, 0, (r["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
            stats.average_loss.backward()
            losses.append(float(stats.average_loss))
            grads_finite &= all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in child.parameters())
            lanes.append(getattr(item, "carry_lanes", None))
        record("lanes_carry_runs_on_pruned_arch_cpu", len(losses) >= 2 and all(l == l and l < 1e3 for l in losses) and grads_finite
               and cu.carry is not None, user=row["user_id"], updates=len(losses), losses=losses, gradients_finite=grads_finite,
               carry=cu.carry.describe() if cu.carry else None)
    finally:
        for path in (EXPERIMENTS / name, ft_dir, dec_dir):
            shutil.rmtree(path, ignore_errors=True)

    # ---- 5. zero_features and the 92 -> 128 input layer on every asserted preset
    shapes = {}
    for arch in list(PRUNED_PARAMETERS):
        v = variant(arch)
        m = build(arch)
        first = m.features2card[0]
        shapes[arch] = {"zero_features": list(v.zero_features), "input": [first.in_features, first.out_features]}
    record("zero_features_22_and_input_92_128", all(s["zero_features"] == [22] and s["input"] == [92, 128] for s in shapes.values()), **shapes)

    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1))
    atomic_json(args.report, report)
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("depth")
    p.add_argument("--src", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("make-run")
    p.add_argument("--parent-exp", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--name", required=True)
    p = sub.add_parser("ft-prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2)
    p = sub.add_parser("promote")
    p.add_argument("--ft", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p = sub.add_parser("decide")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--ref-jsonl", required=True)
    p.add_argument("--ref-label", required=True)
    p.add_argument("--fork-jsonl", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--min-users", type=int, default=464)
    p.add_argument("--fp32-label", default="")
    p = sub.add_parser("screen-report")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--parent", default="P_dec/w8+s8")
    p.add_argument("--out", required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--parent-exp", default="C:/Users/jonat/workspace/wt/sm21v2/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont")
    parser.add_argument("--parent-ckpt", default="C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k/"
                        "expertium/E89-F65e-F77a/master.pth")
    parser.add_argument("--epoch", type=int, default=12)
    parser.add_argument("--report", default=str(ROOT / "artifacts/small-model/check-prune.json"))
    args = parser.parse_args()
    if args.self_test:
        return self_test(args)
    if args.command == "depth":
        depth(args.src, args.arch, args.out)
    elif args.command == "make-run":
        make_run(args.parent_exp, args.epoch, args.arch, args.name)
    elif args.command == "ft-prepare":
        ft_prepare(args.run, args.epoch, args.out, args.fraction, args.seed)
    elif args.command == "promote":
        promote(args.ft, args.run, args.epoch)
    elif args.command == "decide":
        return decide(args)
    elif args.command == "screen-report":
        return screen_report(args)
    else:
        parser.error("a command or --self-test is needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
