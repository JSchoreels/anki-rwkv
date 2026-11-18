"""Check the answer-time test vectors with the NumPy reference only.

    python check_time_vectors.py <package dir>

For every user: the 26 explicit features recomputed from the stored revlog columns equal the stored ones, and the
head on the stored inputs (fp32_time_frozen, kept for the first 500 time rows, + time_explicit) with time_head.f32.bin reproduces fp32_time_logits.
Prints one JSON line with the maximum errors; exits 1 above 1e-5 (probabilities) or 1e-6 (explicit features).
"""
import json
import sys
from pathlib import Path

import numpy as np

package = Path(sys.argv[1])
sys.path.insert(0, str(package / "reference"))
from answer_time_numpy import explicit_features, head_logits, grade_probabilities, read_weights, summary  # noqa: E402

weights = read_weights(package / "time_head.f32.bin")
version = weights.pop("time_head.format_version")
assert version.shape == (1,) and version[0] == 1.0
report = {}
ok = True
for path in sorted((package / "testvectors").glob("user_*.npz")):
    d = np.load(path, allow_pickle=False)
    table = explicit_features(d["rev_card_id"], d["rev_rating"], d["rev_duration"], d["rev_elapsed_days"])
    explicit = table[d["time_review_th"] - 1]
    e_err = float(np.abs(explicit - d["time_explicit"]).max())
    # the frozen inputs are stored for the first rows only (fp32_time_frozen is a prefix of the time rows)
    k = len(d["fp32_time_frozen"])
    logits = head_logits(weights, np.concatenate([d["fp32_time_frozen"], d["time_explicit"][:k]], -1))
    want = d["fp32_time_logits"][:k]
    p_err = float(np.abs(grade_probabilities(logits) - grade_probabilities(want)).max())
    l_err = float(np.abs(logits - want).max())
    med, fast, _ = summary(d["fp32_time_logits"])
    report[path.name] = {"rows": int(len(d["time_row"])), "frozen_rows": k, "explicit_max_abs": e_err, "logit_max_abs": l_err,
                         "prob_max_abs": p_err, "median_good_s_mean": float(med[:, 2].mean()),
                         "p_fast_good_mean": float(fast[:, 2].mean())}
    ok &= e_err <= 1e-6 and p_err <= 1e-5
print(json.dumps(report))
sys.exit(0 if ok else 1)
