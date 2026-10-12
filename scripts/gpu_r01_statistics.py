"""R01 paired-user relative-loss arithmetic; no run authorization or promotion.

Caller must establish the frozen reference, matched protocol and training seeds.
These conditional single-seed bounds do not establish seed variability, cohort
eligibility, repeated-control stability, checkpoint selection or speed gains.
"""

import math

import numpy as np

METRICS = tuple(task + "_log_loss_" + kind for task in ("curve", "instant")
                for kind in ("label_weighted", "user_average"))


def paired_relative_losses(candidate, reference, *, resamples, bootstrap_seed,
                           family_tests, alpha, relative_limit):
    """Resample paired users, then take the ratio of each pair of aggregates.

    Relative fractions (0.01 = 1%) remain separate from absolute loss deltas.
    All statistical settings are explicit and must be predeclared by the round.
    """
    if (type(resamples) is not int or resamples < 2 or type(family_tests) is not int
            or family_tests < 1 or type(bootstrap_seed) is not int
            or not math.isfinite(alpha) or not 0 < alpha < 1
            or not math.isfinite(relative_limit) or not 0 < relative_limit < 1):
        raise ValueError("Invalid predeclared relative-loss bootstrap settings")

    def index(rows):
        result = {}
        for row in rows:
            user = row["user_id"]
            if type(user) is not int or user <= 0 or user in result:
                raise ValueError("Distinct positive integer user IDs required")
            result[user] = row
        return result

    a, b = index(candidate), index(reference)
    if not a or set(a) != set(b):
        raise ValueError("Paired user sets differ or are empty")
    users = sorted(a)
    arrays, samples = {}, {key: [] for key in METRICS}
    for task in ("curve", "instant"):
        weights, left, right = [], [], []
        for user in users:
            label_key, loss_key = task + "_labels", task + "_log_loss"
            labels = a[user][label_key]
            if (type(labels) is not int or labels <= 0
                    or type(b[user][label_key]) is not int or labels != b[user][label_key]):
                raise ValueError("Paired positive held-out label counts required")
            values = (a[user][loss_key], b[user][loss_key])
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("Both tasks require finite nonnegative binary losses")
            # A zero reference would make some bootstrap denominators undefined.
            if values[1] == 0:
                raise ValueError("Relative loss is undefined for a zero reference")
            weights.append(labels)
            left.append(values[0])
            right.append(values[1])
        arrays[task] = tuple(np.asarray(values, dtype=float) for values in (weights, left, right))
    rng = np.random.default_rng(bootstrap_seed)
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(users), (min(256, resamples - start), len(users)))
        for task, (weights, left, right) in arrays.items():
            for kind in ("label_weighted", "user_average"):
                mass = weights[indices] if kind == "label_weighted" else np.ones(indices.shape)
                candidate_sum = (left[indices] * mass).sum(axis=1)
                reference_sum = (right[indices] * mass).sum(axis=1)
                samples[task + "_log_loss_" + kind].append(candidate_sum / reference_sum - 1)
    metrics = {}
    for task, (weights, left, right) in arrays.items():
        for kind in ("label_weighted", "user_average"):
            key = task + "_log_loss_" + kind
            mass = weights if kind == "label_weighted" else np.ones(len(users))
            candidate_loss = float(np.average(left, weights=mass))
            reference_loss = float(np.average(right, weights=mass))
            distribution = np.concatenate(samples[key])
            upper = float(np.quantile(distribution, 1 - alpha / family_tests))
            metrics[key] = {"candidate": candidate_loss, "reference": reference_loss,
                            "absolute_delta": candidate_loss - reference_loss,
                            "relative_delta": candidate_loss / reference_loss - 1,
                            "relative_delta_percent": 100 * (candidate_loss / reference_loss - 1),
                            "relative_ci95": [float(value) for value in np.quantile(distribution, [0.025, 0.975])],
                            "relative_familywise_upper": upper,
                            "strict_upper_below_limit": upper < relative_limit}
    return {"scope": "paired-user arithmetic conditional on one training seed; not validation or promotion",
            "paired_users": len(users), "resamples": resamples, "bootstrap_seed": bootstrap_seed,
            "family_tests": family_tests, "alpha": alpha, "relative_limit": relative_limit,
            "expected_tail_resamples": resamples * alpha / family_tests,
            "metrics": metrics,
            "all_four_strict_upper_bounds_below_limit": all(value["strict_upper_below_limit"] for value in metrics.values())}
