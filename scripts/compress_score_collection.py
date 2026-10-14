"""Score quantized variants of the fork's 2.76M RWKV on the user's own collection (CPU only).

Re-uses the 2026-10-14 collection evaluation inputs read-only from the main checkout (prepared batches and the
benchmark label DB, see artifacts/collection-eval/2026-10-14/README.md there), so results are directly comparable
with its Model B rows (fork fp32, full protocol: curve 0.4417, instant 0.4366 on 173,690 reviews).

Variants: a weight scheme from compress_quant.WEIGHT_SCHEMES, optionally `+s8` / `+s4` for the fork's int8/int4
card+note state emulation, e.g. `fp32`, `w4-g32`, `fp32+s8`, `w4-g32+s8`.

usage: compress_score_collection.py --variants fp32,w8-pc,w4-g32 [--checkpoint qat.pth --tag qat1] [--protocol full]
"""
import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--variants", required=True)
parser.add_argument("--checkpoint", help="weights to quantize (default: the fork's RWKV_trained_on_5000_10000.pth)")
parser.add_argument("--tag", default="fork", help="label prefix for the weights, e.g. fork or qat-w4g32-1")
parser.add_argument("--protocol", choices=("window", "full"), default="full")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--out", default="artifacts/compression/collection-scores.jsonl")
parser.add_argument("--save-predictions", action="store_true")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(args.threads)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(args.threads)
assert not torch.cuda.is_available()

from scripts import compress_quant as cq  # noqa: E402

cq.install_state_quant_model()

import lmdb  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.compress_metrics import metrics  # noqa: E402
from scripts.full_run_metrics import evaluate_with_predictions  # noqa: E402
from scripts.experiment_provenance import process_provenance  # noqa: E402

MAIN_EVAL = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14")
USER = 1


def labels():
    env = lmdb.open(str(MAIN_EVAL / "collection-label-db"), readonly=True, lock=False)
    with env.begin() as txn:
        ths = torch.load(BytesIO(txn.get(f"{USER}_review_ths".encode())), weights_only=True).tolist()
        bins = torch.load(BytesIO(txn.get(f"{USER}_rmse_bins".encode())), weights_only=True).tolist()
    env.close()
    return ths, bins


def parse_variant(name):
    weights, _, state = name.partition("+")
    if weights not in cq.WEIGHT_SCHEMES or state not in ("", "s8", "s4"):
        raise SystemExit(f"unknown variant {name}")
    return weights, {"": 0, "s8": 8, "s4": 4}[state]


def main():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["label"], r["protocol"]) for r in map(json.loads, out.read_text().splitlines())}
    variants = [v for v in args.variants.split(",") if (f"{args.tag}/{v}", args.protocol) not in done]
    if not variants:
        print("nothing to do")
        return
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else cq.FORK_WEIGHTS
    base = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base = {k: v.float() for k, v in base.items()}
    names = cq.linear_weight_names(base)
    started = time.perf_counter()
    batches = torch.load(MAIN_EVAL / f"prepared-{args.protocol}-s0.pt", weights_only=False)
    prep = time.perf_counter() - started
    label_data = labels()
    prov = process_provenance()
    model = cq.load_model(state=base)
    for variant in variants:
        weights, state_bits = parse_variant(variant)
        bits, group = cq.WEIGHT_SCHEMES[weights]
        model.load_state_dict(cq.quantize_state_dict(base, names, bits, group))
        cq.set_state_bits(model, state_bits)
        t0 = time.perf_counter()
        with torch.inference_mode():
            values, predictions = evaluate_with_predictions(model, ({"batch": b} for b in batches))
        seconds = time.perf_counter() - t0
        bench = metrics(USER, label_data, predictions)
        row = {"label": f"{args.tag}/{variant}", "weights": weights, "state_bits": state_bits or 32,
               "checkpoint": str(checkpoint), "protocol": args.protocol, "threads": torch.get_num_threads(),
               "weight_bytes": cq.weight_bytes(base, names, bits, group),
               "card_state_bytes": cq.state_bytes(3, precision={0: "f32", 8: "int8", 4: "int4"}[state_bits]),
               "note_state_bytes": cq.state_bytes(2, precision={0: "f32", 8: "int8", 4: "int4"}[state_bits]),
               "benchmark_test_reviews": bench["size"], "curve": bench.get("ahead"), "instant": bench.get("instant"),
               "coverage": {"curve": bench.get("ahead_coverage"), "instant": bench.get("instant_coverage")},
               "r01_all_labels": {k: values[k] for k in ("curve_labels", "instant_labels", "curve_log_loss",
                                                         "instant_log_loss")},
               "prepare_seconds": round(prep, 1), "model_seconds": round(seconds, 1),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "provenance": prov}
        if args.save_predictions:
            ths = sorted(predictions["rating"])
            path = out.parent / "preds" / f"collection-{args.tag}-{variant}-{args.protocol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"review_th": ths, "rating": [predictions["rating"][t] for t in ths],
                          "p_curve": [predictions["ahead"].get(t) for t in ths],
                          "p_instant": [predictions["instant"].get(t) for t in ths]}).to_parquet(path)
            row["predictions"] = str(path.relative_to(ROOT))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        c, i = row["curve"], row["instant"]
        print(f"{row['label']} [{args.protocol}] {row['weight_bytes']['total']:,} B: curve LL {c['LogLoss']:.4f} "
              f"RMSE(bins) {c['RMSE(bins)']:.4f} AUC {c['AUC']:.4f} | instant LL {i['LogLoss']:.4f} "
              f"RMSE(bins) {i['RMSE(bins)']:.4f} AUC {i['AUC']:.4f} | {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
