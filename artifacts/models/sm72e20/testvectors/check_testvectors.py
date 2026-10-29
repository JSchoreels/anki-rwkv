"""Compare the NumPy reference with the golden test vectors (tolerance 1e-5). Exit code 0 = pass.

    python check_testvectors.py                # all users, fp32 + int8, recurrent and parallel form
    python check_testvectors.py --rows 500     # quick run: recurrent form on the first 500 rows only

Needs only numpy. Run from any folder.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference"))
from run_reference import max_diffs, run_parallel, run_recurrent  # noqa: E402
from small_model_numpy import NumpyModel  # noqa: E402
from small_model_recurrent import RecurrentModel  # noqa: E402
from weights_io import load_package  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--rows", type=int, default=None, help="limit the recurrent form to the first N rows")
ap.add_argument("--no-parallel", action="store_true")
args = ap.parse_args()

failed = False
for path in sorted(HERE.glob("user_*.npz")):
    vec = np.load(path)
    for weights in ("fp32", "int8"):
        tensors, spec = load_package(HERE.parent, weights)
        forms = [("recurrent", lambda: run_recurrent(RecurrentModel(tensors, spec), vec, args.rows))]
        if not args.no_parallel:
            forms.append(("parallel", lambda: run_parallel(NumpyModel(tensors, spec), vec)))
        for form, run in forms:
            started = time.perf_counter()
            out = run()
            n = len(out["p_imm"]) if form == "recurrent" else int(vec["n_rows"])
            diffs = max_diffs(out, vec, weights, n)
            ok = max(diffs.values()) < args.tol
            failed |= not ok
            print(f"{'PASS' if ok else 'FAIL'} user {int(vec['user'])} {weights:4s} {form:9s} rows {n:5d} "
                  f"max|d| curve {diffs['p_curve']:.1e} sweep {diffs['p_curve_sweep']:.1e} imm {diffs['p_imm']:.1e} "
                  f"rating {diffs['p_rating']:.1e} ({time.perf_counter() - started:.0f} s)", flush=True)
print("RESULT:", "FAIL" if failed else "PASS", f"(tolerance {args.tol:g})")
sys.exit(1 if failed else 0)
