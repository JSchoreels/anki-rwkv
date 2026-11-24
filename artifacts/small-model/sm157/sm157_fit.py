#!/usr/bin/env python
"""Output-aware low-rank truncation (SM157): fit the key-side statistics of the card / note WKV states on TRAINING users
(design.md "Fit"). CPU only, forward passes only (the training users have no rows in the label DB, so the scorer itself
cannot run on them).

  PYTORCH_JIT=0 python sm157_fit.py --model TAG=KIND:ARCH:PATH [--model ...] --users-file users-fit.json --out stats.json

Per model: the scorer's CPU model (scripts/small_model_heldout.py make_model, CPU branch), w8 weights, the stored state
under FIT_LABEL (REF's int8 rows through the SM148 step loop, `lrRq8x8`: the int8 reference the costs are measured
against), show-time inputs (e2s-query), leak-free (--deleted-cards neighbour): the scorer's LeakBatches. At every write
(non-skipped step) of every card / note WKV layer, scripts/state_lowrank.record_stats accumulates r r^T, kd kd^T, w and
log w per head. The output JSON holds the sums per (model, "stream,layer"); M is derived from it at scoring time
(state_lowrank.weight_matrix). The users must be training users with id >= 5000 (checked against the 8k split).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts import small_model_heldout as smh  # noqa: E402  (installs the scorer's patched modules)
import torch  # noqa: E402

FIT_LABEL = "w8+lrRq8x8"


def check_users(users):
    from scripts import eval_pools as ep
    _, split, _ = ep.split_info()
    train = set(split["train_users"])
    held = set(split["validation_users"]) | set(split["final_test_users"]) | set(split.get("quick_validation_users", []))
    pools = set()
    for p in (732, 2225, 1500):
        pools |= set(ep.pool_users(p))
    bad = [u for u in users if u not in train or u < 5000 or u in held or u in pools]
    if bad:
        raise SystemExit(f"SM157 fit: not training users with id >= 5000 outside every pool: {bad[:10]}")


def build(tag, kind, arch, path):
    """The scorer's CPU make_model + the score_user CPU branch of an SM148 label."""
    from scripts import compress_quant as cq
    from scripts import state_lowrank, state_vq
    base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    layout = None
    if kind == "fork":
        model = cq.load_model(state=base, reference_wkv=True)
        for module in model.rwkv_modules:
            for block in module.blocks:
                block.time_mixer.use_reference_wkv = True
        state_lowrank.install_fork(model)
    else:
        model = smh.build_expertium(arch, base)
        layout = state_vq.install(model)
    scheme, codec = smh.parse_variant(FIT_LABEL)
    bits, group = cq.WEIGHT_SCHEMES[scheme]
    model.load_state_dict(cq.quantize_state_dict(base, cq.linear_weight_names(base), bits, group))
    state_lowrank.SPECTRUM.tag = tag
    if kind == "fork":
        state_lowrank.set_fork(codec)
        cq.set_state_streams(model, (codec.rows_bits, codec.shift_bits), (codec.rows_bits, codec.shift_bits))
    else:
        smh.set_expertium_state_bits(model, 0)
        state_lowrank.configure_expertium_sm148(layout, codec)
    return model.eval()


def merge(paths, out):
    """Sum the statistics of several fit files into one document: a model present in several files (the two halves) gets
    the sum of its statistics (same layers required), the others are copied."""
    docs = [(Path(p).name, json.loads(Path(p).read_text())) for p in paths]
    add = lambda a, b: [add(x, y) for x, y in zip(a, b)] if isinstance(a, list) else a + b   # noqa: E731
    res = {"models": {}, "provenance": {}, "parts": [n for n, _ in docs]}
    for name, d in docs:
        for tag, layers in d["models"].items():
            prov = dict(d["provenance"][tag], file=name)
            if tag not in res["models"]:
                res["models"][tag] = layers
                res["provenance"][tag] = [prov]
                continue
            if sorted(layers) != sorted(res["models"][tag]):
                raise SystemExit(f"SM157 merge: {tag} has different layers in {name}")
            res["models"][tag] = {key: {k: add(res["models"][tag][key][k], layers[key][k]) for k in layers[key]}
                                  for key in layers}
            res["provenance"][tag].append(prov)
    Path(out).write_text(json.dumps(res), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=[], help="TAG=KIND:ARCH:PATH (as the scorer)")
    p.add_argument("--users-file")
    p.add_argument("--out", required=True)
    p.add_argument("--merge", nargs="+", help="sum these fit files into --out instead of fitting")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    args = p.parse_args()
    if args.merge:
        return merge(args.merge, args.out)
    if not args.model or not args.users_file:
        raise SystemExit("--model and --users-file (or --merge)")
    if torch.jit._state._enabled:
        raise SystemExit("SM157 fit needs PYTORCH_JIT=0 (the fork model's eager modules)")
    torch.set_num_threads(args.threads)
    from scripts import compress_common as cc
    from scripts import state_lowrank
    users = json.loads(Path(args.users_file).read_text())["users"]
    check_users(users)
    run = cc.run_info()
    counts = run["review_counts"]
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {"models": {}, "provenance": {}}
    for item in args.model:
        tag, _, rest = item.partition("=")
        kind, arch, path = rest.split(":", 2)
        if tag in doc["models"]:
            print(f"[sm157-fit] {tag}: already fitted, skipped", flush=True)
            continue
        model = build(tag, kind, arch, path)
        state_lowrank._W.stats = {}
        data = smh.LeakBatches(users, cc.data_path(run), counts, "neighbour", 0, "e2s-query")
        started, reviews, failed = time.perf_counter(), 0, []
        for i in range(len(data)):
            user, batch, error, _ = data[i]
            if batch is None:
                failed.append([user, error])
                continue
            with torch.inference_mode():
                stats = model.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                failed.append([user, "non-finite loss"])
            reviews += counts[str(user)]
            if (i + 1) % 10 == 0 or i + 1 == len(data):
                el = time.perf_counter() - started
                print(f"[sm157-fit] {tag} {i + 1}/{len(data)} users, {reviews:,} reviews, {el:.0f}s, "
                      f"{reviews / max(el, 1e-9):,.0f} reviews/s", flush=True)
        doc["models"][tag] = state_lowrank.export_stats()[tag]
        doc["provenance"][tag] = {"checkpoint": path, "kind": kind, "arch": arch, "fit_label": FIT_LABEL,
                                  "inputs": "e2s-query", "deleted_cards": "neighbour", "users": len(users),
                                  "users_file": Path(args.users_file).name, "reviews": reviews, "failed": failed,
                                  "seconds": round(time.perf_counter() - started, 1), "threads": args.threads}
        state_lowrank._W.stats = None
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        os.replace(tmp, out)
        print(f"[sm157-fit] {tag} done, {reviews:,} reviews, failed {len(failed)}", flush=True)
        del model


if __name__ == "__main__":
    main()
