"""Check the answer-time head test vectors with the NumPy reference only (time_head.json format_version 2).

    python check_time_vectors.py            # run from any folder; needs only numpy

For every synthetic user (testvectors/time_user_*.npz):
  - the 26 explicit features recomputed from the stored revlog columns equal the stored ones (max abs error <= 1e-6);
  - the head on the stored inputs (frozen trunk inputs, kept for the first 500 time rows, + the explicit features)
    reproduces the stored logits: time_head.f32.bin -> fp32_time_logits, time_head.int8.bin -> int8_time_logits
    (grade-bin probabilities within 1e-5);
  - every time row is a labelled query (instant) row of the package's own test vectors (testvectors/user_*.npz).
Prints one line per user and weight file, then "RESULT: PASS" or "RESULT: FAIL"; exit code 0 = pass.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE / "reference"))
from answer_time_numpy import explicit_features, grade_probabilities, head_logits, read_weights, summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=1e-5)
args = ap.parse_args()

heads = {"fp32": read_weights(PACKAGE / "time_head.f32.bin"), "int8": read_weights(PACKAGE / "time_head.int8.bin")}
for w in heads.values():
    version = w.pop("time_head.format_version")
    assert version.shape == (1,) and version[0] == 2.0, version
failed = False
paths = sorted(HERE.glob("time_user_*.npz"))
if not paths:
    print("no time_user_*.npz found")
    failed = True
for path in paths:
    d = np.load(path, allow_pickle=False)
    user = int(d["user"])
    table = explicit_features(d["rev_card_id"], d["rev_rating"], d["rev_duration"], d["rev_elapsed_days"])
    e_err = float(np.abs(table[d["time_review_th"] - 1] - d["time_explicit"]).max())
    base = HERE / f"user_{user}.npz"
    link_ok = True
    if base.exists():
        b = np.load(base, allow_pickle=False)
        rows = d["time_row"]
        link_ok = bool(b["is_query"][rows].all() and b["has_label"][rows].all()
                       and np.array_equal(b["review_th"][rows], d["time_review_th"]))
    for name, w in heads.items():
        frozen = d[f"{name}_time_frozen"] if f"{name}_time_frozen" in d.files else d["fp32_time_frozen"]
        k = len(frozen)
        logits = head_logits(w, np.concatenate([frozen, d["time_explicit"][:k]], -1))
        want = d[f"{name}_time_logits"][:k]
        p_err = float(np.abs(grade_probabilities(logits) - grade_probabilities(want)).max())
        med, fast, _ = summary(d[f"{name}_time_logits"])
        ok = e_err <= 1e-6 and p_err <= args.tol and link_ok
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} user {user} {name} time rows {len(d['time_row'])} (head checked on {k}) "
              f"max|d| explicit {e_err:.1e} prob {p_err:.1e} rows-linked {link_ok} "
              f"mean median Good {med[:, 2].mean():.2f} s, mean P(Good under 3.9 s) {fast[:, 2].mean():.3f}", flush=True)
print("RESULT:", "FAIL" if failed else "PASS", f"(tolerance {args.tol:g})")
sys.exit(1 if failed else 0)
