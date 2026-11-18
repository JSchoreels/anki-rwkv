"""Load the package, run the NumPy reference over a test-vector file and print the outputs.

    python run_reference.py ../testvectors/user_4676.npz                  # fp32 weights, first 12 rows
    python run_reference.py ../testvectors/user_4676.npz --weights int8 --rows 40
    python run_reference.py ../testvectors/user_4676.npz --form parallel  # the batch form (small_model_numpy.py)
    python run_reference.py ../testvectors/user_4676.npz --state-bits 8   # int8 card/note state (no golden values)

Needs only numpy. Prints, per row: row index, query flag, skip flag, p_imm, p_rating (again/hard/good/easy),
p_curve at the row's label time, and the difference to the expected value stored in the test-vector file.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from small_model_numpy import NumpyModel  # noqa: E402
from small_model_recurrent import RecurrentModel  # noqa: E402
from weights_io import load_package  # noqa: E402

STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]


def run_recurrent(model, vec, rows=None):
    """Recurrent form over the first `rows` rows. Returns dict of arrays: p_curve (at label time), p_imm,
    p_rating, p_curve_sweep (rows x len(sweep_seconds))."""
    n = int(vec["n_rows"]) if rows is None else min(rows, int(vec["n_rows"]))
    store = model.new_store()
    sweep = vec["sweep_seconds"]
    out = {"p_curve": np.zeros(n), "p_imm": np.zeros(n), "p_rating": np.zeros((n, 4)),
           "p_curve_sweep": np.zeros((n, len(sweep)))}
    ids = {s: vec["id_" + s] for s in STREAMS}
    for i in range(n):
        heads = model.step(vec["features"][i], {s: int(ids[s][i]) for s in STREAMS}, bool(vec["skip"][i]), store)
        out["p_imm"][i], out["p_rating"][i] = model.instant(heads)
        out["p_curve"][i] = model.p_curve(heads, float(vec["label_elapsed_seconds"][i]))
        for j, t in enumerate(sweep):
            out["p_curve_sweep"][i, j] = model.p_curve(heads, float(t))
    return out


def run_parallel(model, vec):
    """Batch form: small_model_numpy.NumpyModel on the stored gather / select / skip arrays."""
    k = int(vec["par_num_streams"])
    gathers, lens, selects, skips = [], [], [], []
    for i in range(k):
        lens.append([int(x) for x in vec[f"par_lens_{i}"]])
        gathers.append([vec[f"par_gather_{i}_{j}"] for j in range(len(lens[-1]))])
        selects.append([vec[f"par_select_{i}_{j}"] for j in range(len(lens[-1]))])
        skips.append([vec[f"par_skip_{i}_{j}"].astype(bool) for j in range(len(lens[-1]))])
    labels = vec["par_labels"]
    base = dict(start=vec["par_start"], sub_gather=gathers, sub_gather_lens=lens, time_shift_selects=selects,
                skips=skips, num_data=1)
    res = model.predict(labels=labels, **base)
    out = {"p_curve": res["p_curve"][0], "p_imm": res["p_imm"][0], "p_rating": res["p_rating"][0]}
    sweep = []
    for t in vec["sweep_seconds"]:
        lab = labels.copy()
        lab[..., 0] = t
        sweep.append(model.predict(labels=lab, **base)["p_curve"][0])
    out["p_curve_sweep"] = np.stack(sweep, axis=1)
    return out


def max_diffs(got, vec, prefix, n):
    return {key: float(np.abs(got[key][:n].astype(np.float64) - vec[f"{prefix}_{key}"][:n]).max())
            for key in ("p_curve", "p_imm", "p_rating", "p_curve_sweep")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vectors")
    ap.add_argument("--package", default=str(HERE.parent))
    ap.add_argument("--weights", choices=("fp32", "int8"), default="fp32")
    ap.add_argument("--form", choices=("recurrent", "parallel"), default="recurrent")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--state-bits", type=int, choices=(0, 8), default=0)
    args = ap.parse_args()
    vec = np.load(args.vectors)
    tensors, spec = load_package(args.package, args.weights)
    started = time.perf_counter()
    if args.form == "recurrent":
        out = run_recurrent(RecurrentModel(tensors, spec, args.state_bits), vec, args.rows)
        n = len(out["p_imm"])
    else:
        out = run_parallel(NumpyModel(tensors, spec), vec)
        n = min(args.rows, len(out["p_imm"]))
    print(f"user {int(vec['user'])}, {args.weights} weights, {args.form} form, {time.perf_counter() - started:.1f} s")
    print("row query skip   p_imm   p_again p_hard  p_good  p_easy | label_s    p_curve(label_s)  expected")
    for i in range(n):
        p = out["p_rating"][i]
        print(f"{i:4d} {int(vec['is_query'][i]):5d} {int(vec['skip'][i]):4d}  {out['p_imm'][i]:.5f}  "
              f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {p[3]:.5f} | {vec['label_elapsed_seconds'][i]:9.0f}  "
              f"{out['p_curve'][i]:.6f}        {vec[args.weights + '_p_curve'][i]:.6f}")
    print("max |difference to the stored expected values| over these rows:")
    for key, value in max_diffs(out, vec, args.weights, n).items():
        print(f"  {key:14s} {value:.2e}")


if __name__ == "__main__":
    main()
