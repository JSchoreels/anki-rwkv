"""Parity checks for scripts/fast_evaluation.py against the full run's original evaluation.

    inputs   CPU: per section, build_batch(samples_for, section_features) == prepare(make_samples) bitwise
    metrics  CPU: fast benchmark metrics == full_run_metrics.benchmark_metrics on real label-db users
    forward  GPU (or CPU): per-position forward outputs, one section per forward vs merged sections
    compare  two result.json files (reference vs fast): per-user fields, exact-equality counts, max |delta|
"""
from argparse import ArgumentParser
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _flat(batch):
    out = [batch.start, batch.labels, batch.label_review_th]
    for group in (batch.sub_gather, batch.time_shift_selects, batch.skips):
        for sub in group:
            out.extend(sub)
    return out


def check_inputs(args):
    import os
    os.environ["GPU_BATCH1_DATA_PATH"] = args.data_path
    import torch
    torch.set_num_threads(1)
    from rwkv.prepare_batch import prepare
    from scripts import fast_evaluation as fe
    from scripts.gpu_batch1 import make_samples
    from scripts.gpu_r01_samples import samples_for
    run = json.loads(Path(args.run_json).read_text())
    users = [int(u) for u in run[args.users_key]][args.offset:args.offset + args.users]
    rows = fe.evaluation_rows(users, run["review_counts"])
    checked, mismatches, timing = 0, [], {"original": 0.0, "fast": 0.0}
    for row in rows:
        parts = [p for p in fe.section_parts([row])]
        # First, last and a few middle sections of each user (the last one is usually short).
        picks = sorted({0, len(parts) - 1, *range(0, len(parts), max(1, len(parts) // args.sections))})
        for index in picks:
            _, offset, end, seed = parts[index]
            section = fe._section_frame(row).iloc[offset:end]
            started = time.perf_counter()
            reference = prepare(make_samples(section, row["user_id"], 128), seed=seed)
            timing["original"] += time.perf_counter() - started
            started = time.perf_counter()
            samples = samples_for(section, row["user_id"], fe.SAMPLE_OPTIONS)
            fast = fe.build_batch(samples, fe.section_features(samples, seed))
            timing["fast"] += time.perf_counter() - started
            a, b = _flat(reference), _flat(fast)
            equal = (len(a) == len(b) and reference.num_data == fast.num_data
                     and [list(map(int, x)) for x in reference.sub_gather_lens] == fast.sub_gather_lens
                     and all(x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)))
            checked += 1
            if not equal:
                mismatches.append({"user": row["user_id"], "offset": offset})
    # The packed IPC form must decode to the same tensors.
    batch, _ = fe.prepare_job([fe.section_parts([rows[0]]).__next__()], pack=False)
    packed = fe.pack_aligned(batch)
    packed_equal = all(torch.equal(x, y) for x, y in zip(_flat(batch), _flat(packed)))
    return {"check": "inputs", "users": users, "sections_checked": checked, "mismatches": mismatches,
            "packed_equal": packed_equal, "passed": not mismatches and packed_equal and checked > 0,
            "seconds_per_section": {k: v / max(checked, 1) for k, v in timing.items()}}


def check_metrics(args):
    import numpy as np
    from scripts import fast_evaluation as fe
    from scripts.full_run_metrics import benchmark_metrics
    run = json.loads(Path(args.run_json).read_text())
    db = Path(args.run_json).resolve().parent / "label-db"
    users = [int(u) for u in run[args.users_key]][args.offset:args.offset + args.users]
    rng = np.random.default_rng(args.seed)
    mismatches, timing, checked = [], {"original": 0.0, "fast": 0.0}, 0
    for user in users:
        ths, _ = fe.benchmark_info(db, fe.DEFAULT_DB_SIZE, user)
        if not ths:
            continue
        n = run["review_counts"][str(user)]
        # Synthetic labelled positions: every benchmark review in both heads, plus extra review_ths,
        # a few dropped ones and repeated keys (the last write must win), in a shuffled order.
        keys = np.concatenate([np.asarray(ths), np.asarray(ths), rng.integers(1, n + 1, size=len(ths) // 4)])
        keys = np.delete(keys, rng.integers(0, len(keys), size=5))
        query = np.concatenate([np.zeros(len(ths), bool), np.ones(len(ths), bool), rng.random(len(keys) - 2 * len(ths)) < 0.5])[:len(keys)]
        perm = rng.permutation(len(keys))
        keys, query = keys[perm].astype(np.int64), query[perm]
        ratings = rng.integers(0, 4, size=len(keys)).astype(np.float32)
        preds = rng.random(len(keys)).astype(np.float32).clip(1e-5, 1 - 1e-5)
        predictions = {"ahead": {}, "instant": {}, "rating": {}}
        for key, q, r, p in zip(keys.tolist(), query.tolist(), ratings, preds):
            predictions["rating"][key] = float(r)
            predictions["instant" if q else "ahead"][key] = float(p)
        started = time.perf_counter()
        with contextlib.redirect_stderr(io.StringIO()):
            expected = benchmark_metrics(user, db, fe.DEFAULT_DB_SIZE, predictions)
        timing["original"] += time.perf_counter() - started
        started = time.perf_counter()
        actual = fe.benchmark_from_arrays(user, db, fe.DEFAULT_DB_SIZE, keys, query, ratings, preds)
        timing["fast"] += time.perf_counter() - started
        checked += 1
        if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
            mismatches.append({"user": user, "expected": expected, "actual": actual})
    return {"check": "metrics", "users_checked": checked, "mismatches": mismatches[:5], "mismatch_count": len(mismatches),
            "passed": checked > 0 and not mismatches, "seconds_per_user": {k: v / max(checked, 1) for k, v in timing.items()}}


def check_forward(args):
    """Which per-position values change when sections share one forward (bitwise and max |delta|)."""
    import os
    os.environ["GPU_BATCH1_DATA_PATH"] = args.data_path
    from scripts import fast_evaluation as fe
    fe.runtime_setup(args.threads)
    import torch
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import make_samples
    from scripts.gpu_r01_samples import samples_for
    run = json.loads(Path(args.run_json).read_text())
    if args.user_list:
        users = [int(u) for u in args.user_list.split(",")]
    else:
        users = [int(u) for u in run[args.users_key]][args.offset:args.offset + args.users]
    rows = fe.evaluation_rows(users, run["review_counts"])
    model = fe.load_model(args.checkpoint, device=args.device)
    names = ["out_ahead_logits", "out_w", "out_p_logits", "p_curve", "p_imm", "curve", "curve_raw", "rating", "instant"]

    def outputs(batch):
        with torch.no_grad():
            gpu = batch.to(args.device)
            out = model.forward_batch(gpu.start, gpu.sub_gather, gpu.sub_gather_lens, gpu.time_shift_selects,
                                      gpu.skips, gpu.num_data)
            terms = fe.loss_terms(model, gpu)
        values = dict(zip(names[:3], out))
        values.update({name: terms[name] for name in names[3:]})
        return {k: v.float().cpu() for k, v in values.items()}

    stats = {name: {"positions": 0, "exact": 0, "max_abs": 0.0} for name in names}
    parts = [p for p in fe.section_parts(rows)][:args.max_sections]
    merged_samples, merged_features, spans = [], [], []
    for row, offset, end, seed in parts:
        section = fe._section_frame(row).iloc[offset:end]
        samples = samples_for(section, row["user_id"], fe.SAMPLE_OPTIONS)
        spans.append((len(merged_samples), samples))
        merged_features.extend(fe.section_features(samples, seed))
        merged_samples.extend(samples)
    merged = outputs(fe.build_batch(merged_samples, merged_features))
    for (row, offset, end, seed), (first, samples) in zip(parts, spans):
        section = fe._section_frame(row).iloc[offset:end]
        single = outputs(prepare(make_samples(section, row["user_id"], 128), seed=seed))
        for index, sample in enumerate(samples):
            length = int(sample.card_features.size(0))
            for name in names:
                a = single[name].reshape(len(samples), -1, *single[name].shape[2:])[index, :length]
                b = merged[name].reshape(len(merged_samples), -1, *merged[name].shape[2:])[first + index, :length]
                equal = ((a == b) | (torch.isnan(a) & torch.isnan(b))).reshape(length, -1).all(dim=1)
                stats[name]["positions"] += length
                stats[name]["exact"] += int(equal.sum())
                stats[name]["max_abs"] = max(stats[name]["max_abs"], float((a - b).abs().max()))
    return {"check": "forward", "device": args.device, "users": users, "sections": len(parts),
            "merged_chunks": len(merged_samples), "values": stats,
            "passed": all(v["exact"] == v["positions"] for v in stats.values())}


FIELDS = ["curve_labels", "instant_labels", "curve_log_loss", "curve_raw_log_loss", "instant_log_loss",
          "instant_rating_cross_entropy"]
BENCH = ["LogLoss", "RMSE(bins)", "AUC", "RMSE", "size"]


def compare(args):
    a = json.loads(Path(args.reference).read_text())
    b = json.loads(Path(args.candidate).read_text())
    ua = {row["user_id"]: row for row in a["per_user"]}
    ub = {row["user_id"]: row for row in b["per_user"]}
    if set(ua) != set(ub):
        raise SystemExit("User sets differ")
    stats = {}

    def note(name, x, y):
        entry = stats.setdefault(name, {"values": 0, "exact": 0, "max_abs": 0.0, "max_rel": 0.0})
        entry["values"] += 1
        if x == y or (x is None and y is None):
            entry["exact"] += 1
            return
        if x is None or y is None:
            entry["max_abs"] = math.inf
            return
        delta = abs(x - y)
        entry["max_abs"] = max(entry["max_abs"], delta)
        entry["max_rel"] = max(entry["max_rel"], delta / abs(x) if x else math.inf)

    for user, x in ua.items():
        y = ub[user]
        for field in FIELDS:
            note(field, x[field], y[field])
        for key in ("size", "ahead_coverage", "instant_coverage"):
            note(f"benchmark.{key}", x["benchmark"].get(key), y["benchmark"].get(key))
        for head in ("ahead", "instant"):
            for metric in BENCH:
                note(f"benchmark.{head}.{metric}", x["benchmark"].get(head, {}).get(metric), y["benchmark"].get(head, {}).get(metric))
    aggregates = {}
    for section in ("evaluation", "benchmark"):
        flat_a, flat_b = _flatten(a[section]), _flatten(b[section])
        aggregates[section] = {key: {"reference": flat_a[key], "candidate": flat_b.get(key),
                                     "exact": flat_a[key] == flat_b.get(key)} for key in flat_a}
    exact_users = sum(json.dumps({k: ua[u][k] for k in FIELDS + ["benchmark"]}, sort_keys=True)
                      == json.dumps({k: ub[u][k] for k in FIELDS + ["benchmark"]}, sort_keys=True) for u in ua)
    return {"check": "compare", "reference": args.reference, "candidate": args.candidate, "users": len(ua),
            "users_all_fields_exact": exact_users, "fields": stats, "aggregates": aggregates,
            "timing": {name: {"evaluation_seconds": r["evaluation_seconds"], "reviews_per_second": r["reviews_per_second"],
                              "validation_reviews": r["validation_reviews"]} for name, r in (("reference", a), ("candidate", b))},
            "passed": exact_users == len(ua)}


def _flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out.update(_flatten(item, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = item
    return out


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=["inputs", "metrics", "forward", "compare"])
    parser.add_argument("--checkpoint", default=r"C:\Users\jonat\workspace\anki-rwkv\artifacts\full-run\run-8k-20261013\epoch-07\expertium\E89-F65e-F77a\master.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--user-list")
    parser.add_argument("--max-sections", type=int, default=64)
    parser.add_argument("--run-json", default=r"C:\Users\jonat\workspace\anki-rwkv\artifacts\full-run\run-8k-20261013\run.json")
    parser.add_argument("--data-path", default=r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
    parser.add_argument("--users-key", default="quick_validation_users")
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sections", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"inputs": check_inputs, "metrics": check_metrics, "forward": check_forward,
              "compare": compare}[args.check](args)
    from scripts.experiment_provenance import process_provenance
    result["provenance"] = process_provenance()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k not in ("fields", "aggregates", "provenance", "mismatches")}
    if "values" in summary:
        summary["values"] = {k: f"{v['exact']}/{v['positions']} max|d| {v['max_abs']:.3g}" for k, v in summary["values"].items()}
    print(json.dumps(summary)[:3000])
    if args.check == "compare":
        for name, entry in result["fields"].items():
            if entry["exact"] != entry["values"]:
                print(f"  {name}: exact {entry['exact']}/{entry['values']}, max|d| {entry['max_abs']:.3g}, max rel {entry['max_rel']:.3g}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
