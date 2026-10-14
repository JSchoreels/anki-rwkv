"""Held-out evaluation B (2026-10-14 user request): weight + state quantization variants of the fork's 2.76M RWKV on
real dataset users the fork never saw, paired on identical inputs, full-history protocol.

Users: compress_common.b_users() (train + validation users of the 8k/2k run outside 5000-10000, sealed final-test users
excluded; stratified by history-length decile, round-robin order so any prefix of 10k users is stratified).
Variants: `<weights>[+s8|+s4]` (compress_quant.WEIGHT_SCHEMES; +sN = the fork's int8/int4 card+note state precision).
Model: fp32 weights after fake quantization. The card and note modules run the reference recurrence (needed for the
per-answer state rounding) in every variant, including fp32, so all variants share the same arithmetic path; deck,
preset and user modules use the NVRTC WKV kernel. Labels: the run's label DB (validation users) and
artifacts/compression/label-db-b (train users, built by compress_labels.py). Resumable JSONL, one row per (user, variant).

usage: compress_heldout_b.py --count 1000 --variants fp32,fp32+s8,w8-pc+s8,w4-g32+s8,w4-g64+s8,w4-g32+s4 --time-limit 3200
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import compress_quant as cq  # noqa: E402

cq.install_state_quant_model()

from scripts import compress_common as cc  # noqa: E402
from scripts.compress_score_heldout import UserBatches, first  # noqa: E402

LABEL_DB_B = ROOT / "artifacts" / "compression" / "label-db-b"


def parse_variant(name):
    weights, _, state = name.partition("+")
    if weights not in cq.WEIGHT_SCHEMES or state not in ("", "s8", "s4"):
        raise SystemExit(f"unknown variant {name}")
    return weights, {"": 0, "s8": 8, "s4": 4}[state]


# CPU reference (compress_heldout_b --device cpu, 2026-10-14) for dataset user 1745, full history.
PARITY_1745 = {"fp32": (0.412768, 0.391849), "fp32+s8": (0.412765, 0.39184), "w4-g32+s8": (0.414157, 0.392666)}


def check_parity(model, states, run, counts, user, device):
    from scripts.compress_metrics import metrics, read_labels
    from scripts.full_run_metrics import evaluate_with_predictions
    import lmdb
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    with env.begin() as txn:
        labels = read_labels(txn, user)
    env.close()
    batch = cc.full_history_batch(cc.data_path(run), user).to(device)
    for variant, (curve, instant) in PARITY_1745.items():
        weights, bits = parse_variant(variant)
        model.load_state_dict(states[weights][0] if weights in states else
                              cq.quantize_state_dict(*_base_and_names(), *cq.WEIGHT_SCHEMES[weights]))
        cq.set_state_bits(model, bits)
        with torch.inference_mode():
            _, predictions = evaluate_with_predictions(model, [{"batch": batch}])
        got = metrics(user, labels, predictions)
        c, i = got["ahead"]["LogLoss"], got["instant"]["LogLoss"]
        print(f"[heldout-b] parity user {user} {variant}: GPU {c:.6f} / {i:.6f} vs CPU {curve:.6f} / {instant:.6f}",
              flush=True)
        if abs(c - curve) > 2e-4 or abs(i - instant) > 2e-4:
            raise SystemExit("GPU full-history scores disagree with the CPU reference")


def _base_and_names():
    base = {k: v.float() for k, v in torch.load(cq.FORK_WEIGHTS, map_location="cpu", weights_only=True).items()}
    return base, cq.linear_weight_names(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="fp32,fp32+s8,w8-pc+s8,w4-g32+s8,w4-g64+s8,w4-g32+s4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=3200, help="stop starting new users after N seconds")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", default="artifacts/compression/heldout-b.jsonl")
    parser.add_argument("--parity-user", type=int, default=1745, help="0 = skip the GPU vs CPU parity check")
    args = parser.parse_args()
    cpu = args.device == "cpu"
    if cpu:
        torch.set_num_threads(2)
        args.workers = 0
    else:
        from scripts import compress_wkv_cuda
        compress_wkv_cuda.register_cuda()
        errors = compress_wkv_cuda.self_test()
        print(f"[heldout-b] CUDA recurrence vs float32 reference, max abs error by state bits: {errors}", flush=True)
        if errors[0] > 1e-4:
            raise SystemExit("CUDA recurrence disagrees with the reference")
    import lmdb
    from scripts.compress_metrics import metrics, read_labels
    from scripts.experiment_provenance import process_provenance
    from scripts.full_run_metrics import evaluate_with_predictions

    device = torch.device(args.device)
    run = cc.run_info()
    counts = run["review_counts"]
    users = cc.b_users(run=run)[args.start:args.start + args.count]
    variants = args.variants.split(",")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for row in map(json.loads, out.read_text().splitlines()):
            done.setdefault(row["user_id"], set()).add(row["variant"])
    todo = [u for u in users if not set(variants) <= done.get(u, set())]
    base = {k: v.float() for k, v in torch.load(cq.FORK_WEIGHTS, map_location="cpu", weights_only=True).items()}
    names = cq.linear_weight_names(base)
    states = {}
    for variant in variants:
        weights, _ = parse_variant(variant)
        if weights not in states:
            bits, group = cq.WEIGHT_SCHEMES[weights]
            states[weights] = (cq.quantize_state_dict(base, names, bits, group),
                               cq.weight_bytes(base, names, bits, group)["total"])
    model = cq.load_model(state=base, reference_wkv=cpu).to(device)
    # Every module uses the patched recurrence (CPU: reference loop; CUDA: compress_wkv_cuda, any length) in every
    # variant, so all variants share one arithmetic path; state rounding applies to card and note only.
    for module in model.rwkv_modules:
        for block in module.blocks:
            block.time_mixer.use_reference_wkv = True
    if not cpu and args.parity_user:
        check_parity(model, states, run, counts, args.parity_user, device)
    envs = [lmdb.open(str(path), readonly=True, lock=False, map_size=10_000_000_000) for path in (cc.RUN / "label-db", LABEL_DB_B)
            if path.exists()]
    prov = process_provenance()
    print(f"[heldout-b] {len(users)} users ({args.start}..), {len(todo)} to score, variants {variants}", flush=True)
    loader = torch.utils.data.DataLoader(UserBatches(todo, cc.data_path(run), counts), batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    started = time.perf_counter()
    scored = 0
    for index, (user, batch, error, prep_seconds) in enumerate(loader):
        if time.perf_counter() - started > args.time_limit:
            print(f"[heldout-b] time limit reached after {index} users", flush=True)
            break
        labels = None
        for env in envs:
            with env.begin() as txn:
                labels = read_labels(txn, user)
            if labels is not None:
                break
        rows = []
        if batch is None or labels is None:
            rows.append({"user_id": user, "variant": "error", "error": error or "no labels"})
        else:
            gpu_batch = batch.to(device)
            for variant in variants:
                if variant in done.get(user, set()):
                    continue
                weights, state_bits = parse_variant(variant)
                state, nbytes = states[weights]
                model.load_state_dict(state)
                cq.set_state_bits(model, state_bits)
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        values, predictions = evaluate_with_predictions(model, [{"batch": gpu_batch}])
                except torch.OutOfMemoryError as oom:
                    torch.cuda.empty_cache()
                    rows.append({"user_id": user, "variant": variant, "error": f"OOM: {oom}"[:300]})
                    continue
                if not cpu:
                    torch.cuda.synchronize()
                rows.append({"user_id": user, "variant": variant, "weights": weights, "state_bits": state_bits or 32,
                             "weight_bytes": nbytes, "reviews": counts[str(user)],
                             "prepare_seconds": round(prep_seconds, 2),
                             "model_seconds": round(time.perf_counter() - t0, 3), **values,
                             "benchmark": metrics(user, labels, predictions)})
                scored += counts[str(user)]
            del gpu_batch
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                row["provenance_commit"] = prov["git_commit"]
                row["provenance_dirty"] = prov["git_dirty"]
                f.write(json.dumps(row) + "\n")
        elapsed = time.perf_counter() - started
        print(f"[heldout-b] {index + 1}/{len(todo)} user {user} ({counts[str(user)]} reviews) | {elapsed:.0f}s, "
              f"{scored / max(elapsed, 1e-9):,.0f} review-evals/s", flush=True)
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
