"""F11 schedules and observed quality versus work/time; no interpolated accuracy."""

from dataclasses import dataclass
import math

METRICS = ("curve_log_loss_label_weighted", "curve_log_loss_user_average",
           "instant_log_loss_label_weighted", "instant_log_loss_user_average")


def learning_rate(step, total_steps, peak=7e-4, warmup=5, schedule="constant", decay_fraction=0.2):
    if not 1 <= step <= total_steps or warmup < 1 or not 0 < decay_fraction <= 1 or peak <= 0:
        raise ValueError("Invalid schedule or update range")
    if schedule not in ("constant", "cosine", "wsd"):
        raise ValueError("Schedule must be constant, cosine or wsd")
    if step <= warmup:
        return peak * step / warmup
    if schedule == "constant":
        return peak
    start = warmup if schedule == "cosine" else max(warmup, total_steps * (1 - decay_fraction))
    progress = max(0.0, min(1.0, (step - start) / max(1.0, total_steps - start)))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class AccuracyStopper:
    """Predeclare all four scores on a development set; never monitor final test."""
    targets: dict
    consecutive: int = 2
    hits: int = 0
    first_hit_seconds: float | None = None

    def __post_init__(self):
        if set(self.targets) != set(METRICS) or self.consecutive < 1:
            raise ValueError("Predeclare targets for both losses, weighted and user-averaged")
        if any(not math.isfinite(value) or value < 0 for value in self.targets.values()):
            raise ValueError("Log-loss targets must be finite and nonnegative")

    def observe(self, metrics, elapsed_seconds):
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            raise ValueError("Elapsed time must be finite and nonnegative")
        reached = all(name in metrics and math.isfinite(metrics[name]) and metrics[name] <= target
                      for name, target in self.targets.items())
        self.hits = self.hits + 1 if reached else 0
        if reached and self.first_hit_seconds is None:
            self.first_hit_seconds = elapsed_seconds
        return self.hits >= self.consecutive


def checkpoint_at(history, budget, axis):
    if axis not in ("training_seconds", "elapsed_seconds", "raw_review_presentations") or budget < 0:
        raise ValueError("Use a nonnegative observed time or raw-review budget")
    eligible = [row for row in history if row[axis] <= budget]
    return max(eligible, key=lambda row: (row[axis], row["update"])) if eligible else None


def compare_budgets(control, candidate):
    """Return actual checkpoints no later than shared limits, including gaps."""
    if not control or not candidate:
        return {"status": "pending measured checkpoints"}
    result = {"status": "observed checkpoints; different checkpoint gaps remain explicit"}
    for axis in ("training_seconds", "elapsed_seconds", "raw_review_presentations"):
        if any(axis not in row for row in [*control, *candidate]):
            result[axis] = {"status": "pending observed checkpoints with this timing scope"}
            continue
        budget = min(max(row[axis] for row in control), max(row[axis] for row in candidate))
        left, right = checkpoint_at(control, budget, axis), checkpoint_at(candidate, budget, axis)
        result[axis] = {"common_limit": budget, "control": left, "candidate": right,
                        "budget_gap": left[axis] - right[axis] if left and right else None}
    return result
