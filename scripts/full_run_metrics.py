"""Held-out benchmark metrics for full training runs (LogLoss, RMSE(bins), AUC).

The benchmark test reviews and their RMSE bins come from the upstream label database
(`rwkv/find_equalize_test_reviews.py`, built by `scripts/run_full_training.py prepare`), and the
per-user metrics from upstream `rwkv/get_result.get_stats`, so the numbers follow the FSRS
benchmark definitions. Aggregates are review-weighted (by benchmark test-review count) and
user-averaged, like the benchmark's two reporting modes.
"""
import contextlib
import io
import math

import numpy as np
import torch
import torch.nn.functional as F


def collect_predictions(stats, out):
    """Merge one batch's labelled predictions into out = {"ahead": {}, "instant": {}, "rating": {}} by review_th."""
    has_label = stats.has_label.bool().reshape(-1).cpu().numpy()
    is_query = stats.is_query.bool().reshape(-1).cpu().numpy()
    review_th = stats.label_review_th.reshape(-1).cpu().numpy()
    rating = stats.label_rating.reshape(-1).float().cpu().numpy()
    p_curve = stats.p_curve.reshape(-1).float().cpu().numpy()
    p_imm = stats.p_imm.reshape(-1).float().cpu().numpy()
    for index in np.flatnonzero(has_label):
        key = int(review_th[index])
        out["rating"][key] = float(rating[index])
        if is_query[index]:
            out["instant"][key] = float(p_imm[index])
        else:
            out["ahead"][key] = float(p_curve[index])


def evaluate_with_predictions(model, batches):
    """scripts.compare.evaluate's held-out losses plus every labelled prediction for benchmark metrics."""
    model.eval()
    totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0,
              "curve_raw_log_loss": 0.0, "instant_log_loss": 0.0,
              "instant_rating_cross_entropy": 0.0}
    predictions = {"ahead": {}, "instant": {}, "rating": {}}
    with torch.no_grad():
        for item in batches:
            batch = item["batch"]
            stats = model.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Non-finite evaluation loss")
            mask = stats.is_query.bool() & stats.has_label.bool()
            losses = F.binary_cross_entropy(stats.p_imm.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none")
            totals["curve_labels"] += stats.ahead_n
            totals["instant_labels"] += stats.imm_n
            totals["curve_log_loss"] += stats.ahead_avg.item() * stats.ahead_n
            totals["curve_raw_log_loss"] += stats.ahead_raw_avg.item() * stats.ahead_n
            totals["instant_rating_cross_entropy"] += stats.imm_avg.item() * stats.imm_n
            totals["instant_log_loss"] += losses[mask].sum().item()
            collect_predictions(stats, predictions)
    assert totals["curve_labels"] > 0 and totals["instant_labels"] > 0
    for key in ["curve_log_loss", "curve_raw_log_loss"]:
        totals[key] /= totals["curve_labels"]
    for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
        totals[key] /= totals["instant_labels"]
    return totals, predictions


def benchmark_metrics(user_id, db_path, db_size, predictions):
    """Upstream per-user metrics on the benchmark test reviews, for the curve (ahead) and instant heads."""
    from rwkv.get_result import get_benchmark_info, get_stats
    review_ths, bins = get_benchmark_info(str(db_path), db_size, user_id)
    if not review_ths:
        return {"size": 0, "skipped": "no benchmark test reviews"}
    result = {"size": len(review_ths)}
    for head in ("ahead", "instant"):
        preds = predictions[head]
        kept = [i for i, th in enumerate(review_ths) if th in preds and th in predictions["rating"]]
        result[f"{head}_coverage"] = len(kept) / len(review_ths)
        if not kept:
            continue
        ths = [review_ths[i] for i in kept]
        bins_dict = {review_ths[i]: bins[i] for i in kept}
        with contextlib.redirect_stdout(io.StringIO()):
            stats, _ = get_stats(user_id, ths, bins_dict, {th: np.float64(preds[th]) for th in ths},
                                 {th: np.float64(predictions["rating"][th]) for th in ths})
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result


def aggregate(per_user, user_ids=None):
    """Review-weighted and user-averaged benchmark metrics over the selected users (None = all)."""
    rows = [row for row in per_user if user_ids is None or row["user_id"] in user_ids]
    out = {"users": len(rows)}
    for head in ("ahead", "instant"):
        metrics = [row["benchmark"][head] for row in rows if head in row.get("benchmark", {})]
        if not metrics:
            continue
        weights = np.array([m["size"] for m in metrics], dtype=float)
        summary = {"users": len(metrics), "reviews": int(weights.sum())}
        for name in ("LogLoss", "RMSE(bins)", "AUC", "RMSE"):
            values = [(m[name], w) for m, w in zip(metrics, weights) if m.get(name) is not None and math.isfinite(m[name])]
            if values:
                v, w = np.array([x for x, _ in values]), np.array([y for _, y in values])
                summary[name] = {"weighted": float((v * w).sum() / w.sum()), "user_average": float(v.mean())}
        out[head] = summary
    return out
