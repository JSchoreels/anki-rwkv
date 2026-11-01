"""CPU check of the SM21 carried-state prototype (scripts/state_carry.py); no GPU, no queue.

On one real gate-tier training user, the w64 checkpoint of the ctx2048 run (epoch 6), eval mode, float32:
  A  carry disabled = today's batches: the batch the trainer builds (`gpu_batch1.prepare_job`, the run's own options
     minus the packed IPC transport) equals, tensor for tensor, the batch the prototype builds (`samples_for` +
     `prepare`) for the same (user, span, seed);
  B  the carry forward with no carried state equals the stock `forward_batch_variant` bit for bit on that batch;
  C  exact carry: the user's history cut into 2,048-review windows, inputs taken from the full-history sample
     (`consistent_start`), state carried window to window (FreshCarry, fp32) reproduces the single full-history
     forward on every row (tolerance 1e-4 on probabilities); the same windows without carry show the gap carry closes;
  D  int8 carried state (FreshCarry bits=8) vs fp32, and option 3 (StaleCache, int8): bytes per window and per review.

  RWKV_NO_JIT=1 .venv/Scripts/python.exe scripts/check_state_carry.py [--user 1340] [--out artifacts/small-model/check-state-carry.json]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ.setdefault("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

RUN = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048")
CKPT = RUN / "run" / "epoch-06" / "expertium" / "E89-F65e-F77a" / "master.pth"


def digest(batch):
    h = hashlib.sha256()
    for name in ("start", "labels", "label_review_th"):
        h.update(getattr(batch, name).contiguous().numpy().tobytes())
    for group in (batch.sub_gather, batch.time_shift_selects, batch.skips):
        for splits in group:
            for t in splits:
                h.update(t.contiguous().numpy().tobytes())
    h.update(json.dumps([int(batch.num_data), [[int(v) for v in s] for s in batch.sub_gather_lens]]).encode())
    return h.hexdigest()


def probabilities(outputs):
    """Rating probabilities (p head) and the curve weights: what the loss and the scorer read."""
    _, out_w, _, p_logits, _, _ = outputs
    return torch.softmax(p_logits.float(), dim=-1), out_w.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=1340)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--out", default="artifacts/small-model/check-state-carry.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    from expertium.variants import build
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import DATA, prepare_job, split_frame
    from scripts.gpu_r01_samples import samples_for
    from scripts import state_carry as sc

    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    options = dict(json.loads((RUN / "experiment.json").read_text())["options"])
    row = next(r for r in protocol["rows"] if r["user_id"] == args.user)
    seed = options.get("training_seed", 2025) + args.user + 0
    model = build("w64").float()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    report = {"user": args.user, "reviews": row["train_range"][1] - row["train_range"][0], "chunk": args.chunk,
              "checkpoint": str(CKPT), "checks": {}}
    fails = []
    frame = split_frame(get_rwkv_data(DATA, args.user), row, "train")
    end = len(frame)

    # A: carry disabled reproduces today's batch
    plain = {k: v for k, v in options.items() if k != "packed_ipc"}
    trainer = prepare_job(row, 0, end, seed, plain, section=frame).batch
    samples = samples_for(frame, args.user, plain)
    ours = prepare(samples, seed=seed, padding_factor=plain.get("padding_factor", 0.9),
                   allowed_excess_in_one_step=plain.get("split_excess", 20000))
    a = digest(trainer) == digest(ours)
    report["checks"]["A_batches_identical"] = {"pass": a, "sha256": digest(trainer), "windows": len(samples)}
    fails += [] if a else ["A"]

    # B: zero carried state = stock forward, bit for bit
    global_T = ours.start.size(0) // ours.num_data
    ids, windows = sc.canonical_ids(samples, global_T, frame), sc.canonical_windows(samples, global_T)
    t0 = time.perf_counter()
    with torch.no_grad():
        stock = model.forward_batch_variant(ours.start, ours.sub_gather, ours.sub_gather_lens, ours.time_shift_selects,
                                            ours.skips, ours.num_data)
        t_stock = time.perf_counter() - t0
        t0 = time.perf_counter()
        carried, finals = sc.carry_forward(model, ours, ids, windows, sc.FreshCarry())
        t_carry = time.perf_counter() - t0
    b = all(torch.equal(x, y) for x, y in zip(stock, carried))
    report["checks"]["B_zero_state_bitwise"] = {"pass": b, "stock_seconds": round(t_stock, 1),
                                                "carry_seconds": round(t_carry, 1)}
    fails += [] if b else ["B"]

    # C: exact carry over consecutive windows == one full-history forward
    full_samples = samples_for(frame, args.user, dict(plain, chunk_reviews=len(frame)))
    assert len(full_samples) == 1
    full = prepare(full_samples, seed=seed)
    with torch.no_grad():
        full_out = model.forward_batch_variant(full.start, full.sub_gather, full.sub_gather_lens, full.time_shift_selects,
                                               full.skips, full.num_data)
    full_p, full_w = probabilities(full_out)
    keys = (full_samples[0].review_ths.long() * 2 + full_samples[0].skips.long()).tolist()
    where = {k: i for i, k in enumerate(keys)}
    window_samples = samples_for(frame, args.user, dict(plain, chunk_reviews=args.chunk))
    labelled = full.labels[0, :, 4] > 0
    results = {}
    for mode, source in (("fresh_fp32", sc.FreshCarry()), ("fresh_int8", sc.FreshCarry(bits=8)), ("no_carry", None)):
        diff_p, diff_w, gap_labelled, n_labelled = 0.0, 0.0, 0.0, 0
        started = time.perf_counter()
        for index, sample in enumerate(window_samples):
            batch = sc.consistent_start(prepare([sample], seed=seed + index), [sample], full, full_samples[0])
            gT = batch.start.size(0)
            with torch.no_grad():
                out, finals = sc.carry_forward(model, batch, sc.canonical_ids([sample], gT, frame),
                                               sc.canonical_windows([sample], gT, index), source)
            if source is not None:
                source.commit(finals)
            p, w = probabilities(out)
            n = sample.review_ths.numel()
            rows = torch.tensor([where[k] for k in (sample.review_ths.long() * 2 + sample.skips.long()).tolist()])
            dp = (p[0, :n] - full_p[0, rows]).abs()
            diff_p = max(diff_p, float(dp.max()))
            diff_w = max(diff_w, float((w[0, :n] - full_w[0, rows]).abs().max()))
            mask = labelled[rows]
            gap_labelled += float(dp.max(dim=-1).values[mask].sum())
            n_labelled += int(mask.sum())
        results[mode] = {"max_abs_dp": diff_p, "max_abs_dw": diff_w,
                         "mean_max_dp_labelled": gap_labelled / max(n_labelled, 1),
                         "seconds": round(time.perf_counter() - started, 1)}
        if source is not None and hasattr(source, "nbytes"):
            results[mode]["state_bytes_in_memory"] = source.nbytes()
    c = results["fresh_fp32"]["max_abs_dp"] < 1e-4 and results["fresh_fp32"]["max_abs_dw"] < 1e-4
    report["checks"]["C_exact_carry_equals_full_history"] = {"pass": c, "windows": len(window_samples), **results}
    fails += [] if c else ["C"]

    # D: option 3 storage, two passes (pass 2 reads pass 1's end-of-window states)
    stale = sc.StaleCache(bits=8)
    for pass_index in range(2):
        for index, sample in enumerate(window_samples):
            batch = sc.consistent_start(prepare([sample], seed=seed + index), [sample], full, full_samples[0])
            gT = batch.start.size(0)
            with torch.no_grad():
                _, finals = sc.carry_forward(model, batch, sc.canonical_ids([sample], gT, frame),
                                             sc.canonical_windows([sample], gT, index), stale)
            stale.commit(finals)
        stale.end_pass()
    stored = stale.nbytes()
    report["checks"]["D_stale_cache_int8"] = {"bytes": stored, "bytes_per_window": stored / len(window_samples),
                                              "bytes_per_review": stored / len(frame)}
    report["passed"] = not fails
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
