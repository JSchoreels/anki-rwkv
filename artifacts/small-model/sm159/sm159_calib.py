#!/usr/bin/env python
"""Energy-triggered rank growth (SM159): rank read-out (end-of-history rank of every answered card / note head) of
SM148 codec labels on users WITHOUT scoring (design.md "Calibration"). CPU only, forward passes only: the loss of each
pass is discarded, never read. Used on TRAINING users (the threshold search) and, in the fallback step only, on the
probe users for bytes alone.

  PYTORCH_JIT=0 python sm159_calib.py --model TAG=KIND:ARCH:PATH --label LABEL [--label ...] --users-file U.json \
      --out calib.json [--training]

Per model: the scorer's CPU model (sm157_fit.build: w8 weights, the fork model's eager modules or the expertium Hooks),
show-time inputs (e2s-query), leak-free (--deleted-cards neighbour): the scorer's LeakBatches. Per user and label the
codec is configured as the scorer's CPU branch does (state_lowrank.set_fork + cq.set_state_streams for the fork model,
configure_expertium_sm148 for expertium), one forward pass, then state_lowrank.RANK_READOUT.export() is summed per
(model, label). Output: aggregates only (no ids). Finished (model, label) pairs are skipped on a rerun.
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
sys.path.insert(0, str(HERE.parent / "sm157"))

from scripts import small_model_heldout as smh  # noqa: E402  (installs the scorer's patched modules)
import torch  # noqa: E402
import sm157_fit  # noqa: E402

SUMS = ("entities", "entities_n1", "entities_ge16", "sum_rank")
HISTS = ("rank_hist", "rank_hist_ge16", "entity_max_hist", "entity_max_hist_ge16")


def add_readout(acc, ro):
    """Sum one row's RANK_READOUT export into acc (same layers, heads and K required)."""
    for name, st in ro.items():
        a = acc.get(name)
        if a is None:
            acc[name] = json.loads(json.dumps(st))
            continue
        if (a["layers"], a["H"], a["K"]) != (st["layers"], st["H"], st["K"]):
            raise SystemExit(f"SM159 calib: stream {name} layout changed between users")
        for k in SUMS:
            a[k] += st[k]
        for k in HISTS:
            a[k] = [x + y for x, y in zip(a[k], st[k])]


def configure(model, kind, layout, codec):
    from scripts import compress_quant as cq
    from scripts import state_lowrank
    if kind == "fork":
        state_lowrank.set_fork(codec)
        cq.set_state_streams(model, (codec.rows_bits, codec.shift_bits), (codec.rows_bits, codec.shift_bits))
    else:
        smh.set_expertium_state_bits(model, 0)
        state_lowrank.configure_expertium_sm148(layout, codec)


def build(tag, kind, arch, path):
    """sm157_fit.build without its fixed label: the model and (expertium) layout."""
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
    scheme, _ = smh.parse_variant("w8+lrRq8x8")
    bits, group = cq.WEIGHT_SCHEMES[scheme]
    model.load_state_dict(cq.quantize_state_dict(base, cq.linear_weight_names(base), bits, group))
    state_lowrank.SPECTRUM.tag = tag
    return model.eval(), layout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=[], help="TAG=KIND:ARCH:PATH (as the scorer)")
    p.add_argument("--label", action="append", default=[], help="w8+lr... SM148 codec label (any rank-only codec)")
    p.add_argument("--model-label", action="append", default=[], help="TAG/LABEL: a label for one model only")
    p.add_argument("--users-file", required=True)
    p.add_argument("--training", action="store_true", help="check the users are training users outside every pool")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    args = p.parse_args()
    if torch.jit._state._enabled:
        raise SystemExit("SM159 calib needs PYTORCH_JIT=0 (the fork model's eager modules)")
    torch.set_num_threads(args.threads)
    from scripts import compress_common as cc
    from scripts import state_lowrank
    users = json.loads(Path(args.users_file).read_text())["users"]
    if args.training:
        sm157_fit.check_users(users)
    run = cc.run_info()
    counts = run["review_counts"]
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {"models": {}, "provenance": {}}
    for item in args.model:
        tag, _, rest = item.partition("=")
        kind, arch, path = rest.split(":", 2)
        labels = list(args.label) + [ml.split("/", 1)[1] for ml in args.model_label if ml.split("/", 1)[0] == tag]
        todo = [lab for lab in dict.fromkeys(labels) if lab not in doc["models"].get(tag, {})]
        if not todo:
            print(f"[sm159-calib] {tag}: all labels done, skipped", flush=True)
            continue
        model, layout = build(tag, kind, arch, path)
        codecs = {lab: smh.parse_variant(lab)[1] for lab in todo}
        acc = {lab: {} for lab in todo}
        secs = {lab: 0.0 for lab in todo}
        data = smh.LeakBatches(users, cc.data_path(run), counts, "neighbour", 0, "e2s-query")
        started, reviews, failed = time.perf_counter(), 0, []
        for i in range(len(data)):
            user, batch, error, _ = data[i]
            if batch is None:
                failed.append(error)
                continue
            for lab in todo:
                configure(model, kind, layout, codecs[lab])
                state_lowrank.RANK_READOUT.reset()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model.get_loss(batch)   # the loss is discarded (bytes only)
                secs[lab] += time.perf_counter() - t0
                add_readout(acc[lab], state_lowrank.RANK_READOUT.export())
                state_lowrank.RANK_READOUT.reset()
            reviews += counts[str(user)]
            el = time.perf_counter() - started
            print(f"[sm159-calib] {tag} {i + 1}/{len(data)} users, {reviews:,} reviews, {el:.0f}s, "
                  f"{reviews * len(todo) / max(el, 1e-9):,.0f} label-reviews/s", flush=True)
        state_lowrank.set_fork(None)
        doc["models"].setdefault(tag, {})
        for lab in todo:
            doc["models"][tag][lab] = {"streams": acc[lab], "seconds": round(secs[lab], 1)}
        doc["provenance"][tag] = {"checkpoint": path, "kind": kind, "arch": arch, "inputs": "e2s-query",
                                  "deleted_cards": "neighbour", "users": len(users),
                                  "users_file": Path(args.users_file).name, "reviews": reviews,
                                  "failed": len(failed), "training": bool(args.training), "threads": args.threads}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, out)
        print(f"[sm159-calib] {tag} done ({len(todo)} labels), {reviews:,} reviews, failed {len(failed)}", flush=True)
        del model


if __name__ == "__main__":
    main()
