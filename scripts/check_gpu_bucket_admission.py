"""CPU-only F05 admission/structural checks on fingerprinted saved real inputs.

This is a two-input diagnostic, not whole-cohort admission, numerical model
parity, CUDA graph execution or predictive/speed validation.
"""

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def marker_hierarchy(batch, start, seed):
    """Integer marker recurrence tests gather/shift/skip/order, not RWKV numerics."""
    import torch
    prime = 2147483647
    values = start
    for gathers, lens, shifts, skips in zip(batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects, batch.skips):
        outputs = []
        for gather, length, shift, skip in zip(gathers, lens, shifts, skips):
            selected = values[gather.clamp_min(0)].view(-1, length)
            shifted = selected.gather(1, shift.view_as(selected))
            skip = skip.view_as(selected)
            memory = torch.zeros(selected.shape[0], dtype=torch.long)
            result = torch.empty_like(selected)
            for time in range(length):
                mixed = (selected[:, time] * 17 + shifted[:, time] * 19 + seed) % prime
                following = (memory * 23 + mixed) % prime
                result[:, time] = (following * 29 + mixed * 31) % prime
                memory = torch.where(skip[:, time], memory, following)
            outputs.append(result.flatten())
        values = torch.cat(outputs)
    return values


def check_admitted(original, result):
    import torch
    bucket = result.batch
    positions = result.output_positions
    torch.testing.assert_close(bucket.labels.flatten(0, 1)[positions], original.labels.flatten(0, 1), rtol=0, atol=0)
    assert torch.equal(bucket.label_review_th.flatten()[positions], original.label_review_th.flatten())
    n, old_length = original.labels.shape[:2]
    new_length = bucket.labels.shape[1]
    old = torch.arange(n * old_length)
    start_positions = old.div(old_length, rounding_mode="floor") * new_length + old.remainder(old_length)
    torch.testing.assert_close(bucket.start[start_positions], original.start, rtol=0, atol=0)
    old_markers = old + 1
    new_markers = torch.zeros(bucket.start.shape[0], dtype=torch.long)
    new_markers[start_positions] = old_markers
    real = original.sub_gather[-1][0] >= 0
    # Original padded outputs gather row zero and are not semantic reviews.
    # The bucket guard forbids feeding such padding into a real position.
    assert not original.labels[..., 4].flatten()[~real].count_nonzero()
    assert (original.label_review_th.flatten()[~real] < 0).all()
    padding_changes = []
    for seed in (13, 131):
        expected = marker_hierarchy(original, old_markers, seed)
        actual = marker_hierarchy(bucket, new_markers, seed)[positions]
        assert torch.equal(actual[real], expected[real]), "Bucket changed a real gather/shift/skip/order marker"
        padding_changes.append(int(((actual != expected) & ~real).sum()))
    old_curve = ((1 - original.labels[..., 6]) * original.labels[..., 4]).sum()
    old_instant = (original.labels[..., 6] * original.labels[..., 4]).sum()
    assert ((1 - bucket.labels[..., 6]) * bucket.labels[..., 4]).sum() == old_curve
    assert (bucket.labels[..., 6] * bucket.labels[..., 4]).sum() == old_instant
    return {"labels_review_ids_and_features": "exact on mapped original positions",
            "marker_hierarchy": "exact final markers at every real review/query for two independent seeds; gather/time-shift/skip/order simulation, not RWKV model parity",
            "original_padding_positions": int((~real).sum()), "changed_unused_padding_markers": padding_changes,
            "curve_labels": int(old_curve), "instant_labels": int(old_instant)}


def checks(inputs, compare):
    import torch
    from rwkv.model.srs_model import PreparedBatch
    from scripts.check_gpu_reproducibility import batch_hash
    from scripts.gpu_shape_buckets import BucketPolicy, bucket_batch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    source = json.loads((inputs / "result.json").read_text())
    if source.get("status") != "passed" or source.get("mode") != "CPU inputs":
        raise ValueError("Use a successful saved-input diagnostic")
    policies = {"default": BucketPolicy()}
    if compare:
        policies = {
            "default": BucketPolicy(),
            "legacy8": BucketPolicy((8, 16, 32, 64, 128, 256), count_quantum=None),
            "short": BucketPolicy((1, 2, 4, 8, 16, 32, 64, 128, 256), count_quantum=None),
            "dense": BucketPolicy((1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256), count_quantum=None),
            "short-count8": BucketPolicy((1, 2, 4, 8, 16, 32, 64, 128, 256), count_quantum=8),
            "dense-count8": BucketPolicy((1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256), count_quantum=8),
            "short-count4": BucketPolicy((1, 2, 4, 8, 16, 32, 64, 128, 256), count_quantum=4),
            "dense-count4": BucketPolicy((1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256), count_quantum=4),
        }
    rows = []
    for case in source["cases"]:
        path = inputs / case["input_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_file_sha256"]:
            raise ValueError("Saved real input file fingerprint differs")
        batch = PreparedBatch(**torch.load(path, weights_only=True, map_location="cpu"))
        if batch_hash(batch) != case["hashes"]["serial_source_load"]:
            raise ValueError("Saved real input tensor fingerprint differs")
        for name, policy in policies.items():
            result = bucket_batch(batch, policy)
            if not result.admitted:
                assert result.batch is batch
            rows.append({"user_id": case["user_id"], "offset": case["offset"], "raw_reviews": case["raw_reviews"],
                         "input_file_sha256": case["input_file_sha256"], "input_tensor_sha256": case["hashes"]["serial_source_load"],
                         "samples": batch.num_data, "original_history_length": batch.labels.shape[1],
                         "policy_name": name, "policy": asdict(policy), "admitted": result.admitted,
                         "reason": result.reason, "work": result.metrics,
                         "structural_checks": check_admitted(batch, result) if result.admitted else "original batch retained; remapping not executed"})
    assert not torch.cuda.is_initialized()
    return {"rows": rows, "input_provenance": source["provenance"], "input_folder": inputs.relative_to(ROOT).as_posix(),
            "dataset_path": source["dataset_path"], "dataset_revision": source["dataset_revision"],
            "runtime": {"torch": torch.__version__, "cuda_build": torch.version.cuda}, "cuda_context_initialized": False}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "artifacts/benchmarks/gpu-future/reproducibility-cpu-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    inputs, target = args.inputs.resolve(), args.output.resolve()
    if not inputs.is_relative_to(ROOT / "artifacts") or not target.is_relative_to(ROOT / "artifacts") or target.exists():
        parser.error("Use saved workspace inputs and a new result JSON below artifacts")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "provenance": process_provenance(),
              "scope": "two saved real training inputs only; admission and integer structural simulation, not whole-cohort coverage, model gradients, graph performance or held-out quality"}
    try:
        result.update(checks(args.inputs.resolve(), args.compare), status="passed")
        print(json.dumps([{**{key: row[key] for key in ("user_id", "offset", "policy_name", "admitted", "reason")},
                           "work_ratios": row["work"].get("work_ratios", row["work"].get("proposed_ratios"))}
                          for row in result["rows"]], indent=2))
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
