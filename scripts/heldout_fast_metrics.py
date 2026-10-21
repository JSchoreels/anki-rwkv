"""SM48: vectorised held-out prediction collection and benchmark metrics, bitwise equal to the dict path.

The dict path (scripts/full_run_metrics.collect_predictions + scripts/compress_metrics.metrics + upstream
rwkv/get_result.get_stats) walks every labelled review in Python: ~1.8 dict entries per review in collect_predictions
and ~1.4 benchmark entries per review in get_stats (a scalar np.clip, list appends and a per-bin row list each, plus
an unused `raw` dict). On the 732 pool that loop was about half of the scorer's wall time
(artifacts/small-model/sm48/profile.md). Here the same quantities come from numpy arrays:

  collect_arrays(stats)              -> ArrayPredictions (per head: sorted unique review_th + value at the LAST
                                        labelled position with that review_th, i.e. the dict path's overwrite rule)
  evaluate_with_arrays(model, batch) -> (totals, ArrayPredictions)   # = full_run_metrics.evaluate_with_predictions
  metrics_arrays(user, labels, P)    -> compress_metrics.metrics(...)  (same dict, same float bits)
  get_stats_arrays(...)              -> get_stats(...)[0]

Exactness: sklearn and pandas get the same float64 inputs in the same order (numpy arrays instead of lists of
np.float64, which sklearn converts to the same arrays; the RMSE(bins) frame keeps get_stats' row order: bins by first
appearance, reviews in order within a bin). Duplicate benchmark review_ths (get_stats' bins dict would keep the last
bin for all copies) fall back to upstream get_stats. scripts/check_heldout_fast_metrics.py compares both paths.
"""
import numpy as np
import torch
import torch.nn.functional as F


class ArrayPredictions:
    """Labelled predictions by review_th as sorted arrays: heads 'ahead', 'instant', 'rating' -> (keys, values)."""

    def __init__(self, heads):
        self.heads = heads

    def lookup(self, head, review_ths):
        """(present mask, float64 values; 0 where absent) for an int64 array of review_ths."""
        keys, values = self.heads[head]
        if len(keys) == 0:
            return np.zeros(len(review_ths), dtype=bool), np.zeros(len(review_ths))
        pos = np.searchsorted(keys, review_ths)
        pos_clipped = np.minimum(pos, len(keys) - 1)
        present = (pos < len(keys)) & (keys[pos_clipped] == review_ths)
        return present, np.where(present, values[pos_clipped], 0.0)

    def to_dicts(self):
        """The dict path's {head: {review_th: float}} (for --dump and checks)."""
        return {head: dict(zip(keys.tolist(), values.tolist())) for head, (keys, values) in self.heads.items()}


def _last_by_key(keys, values):
    """Sorted unique keys and the value at each key's last occurrence (stable sort keeps occurrence order)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    last = np.ones(len(sorted_keys), dtype=bool)
    last[:-1] = sorted_keys[1:] != sorted_keys[:-1]
    return sorted_keys[last], values[order][last]


def collect_arrays(stats):
    """full_run_metrics.collect_predictions for one batch, as ArrayPredictions (float32 -> float64 like float())."""
    fields = [stats.has_label.bool(), stats.is_query.bool(), stats.label_review_th, stats.label_rating.float(),
              stats.p_curve.float(), stats.p_imm.float()]
    has_label, is_query, review_th, rating, p_curve, p_imm = (x.reshape(-1).cpu().numpy() for x in fields)
    index = np.flatnonzero(has_label)
    keys = review_th[index].astype(np.int64)
    query = is_query[index]
    return ArrayPredictions({
        "rating": _last_by_key(keys, rating[index].astype(np.float64)),
        "instant": _last_by_key(keys[query], p_imm[index][query].astype(np.float64)),
        "ahead": _last_by_key(keys[~query], p_curve[index][~query].astype(np.float64)),
    })


def evaluate_with_arrays(model, batches):
    """full_run_metrics.evaluate_with_predictions with collect_arrays (one batch per user in the scorer)."""
    model.eval()
    totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0,
              "curve_raw_log_loss": 0.0, "instant_log_loss": 0.0,
              "instant_rating_cross_entropy": 0.0}
    if len(batches) != 1:
        raise ValueError("evaluate_with_arrays takes one batch")
    with torch.no_grad():
        batch = batches[0]["batch"]
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
        predictions = collect_arrays(stats)
    assert totals["curve_labels"] > 0 and totals["instant_labels"] > 0
    for key in ["curve_log_loss", "curve_raw_log_loss"]:
        totals[key] /= totals["curve_labels"]
    for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
        totals[key] /= totals["instant_labels"]
    return totals, predictions


def get_stats_arrays(user_id, review_ths, bins, preds, ratings):
    """rwkv/get_result.get_stats(...)[0] for arrays in the benchmark order (unique review_ths)."""
    import pandas as pd
    from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
    y = np.clip(ratings, 0, 1)
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=preds)
    logloss = log_loss(y_true=y, y_pred=preds, labels=[0, 1])
    try:
        auc = round(roc_auc_score(y_true=y, y_score=preds), 6)
    except Exception:
        auc = None
    if auc is not None and np.isnan(auc):
        auc = None
    # get_stats' row list: bins in first-appearance order, reviews in benchmark order within each bin
    _, first, inverse = np.unique(bins, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    order = np.argsort(rank[inverse.reshape(-1)], kind="stable")
    tmp = pd.DataFrame({"bin": bins[order], "y": y[order], "p": preds[order],
                        "weights": np.ones(len(order), dtype=np.int64)})
    tmp = tmp.groupby("bin").agg({"y": "mean", "p": "mean", "weights": "sum"}).reset_index()
    rmse_bins = root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])
    if len(review_ths) >= 5e5:
        torch.cuda.empty_cache()   # upstream get_stats does this too
    return {"metrics": {"RMSE": round(rmse_raw, 6), "LogLoss": round(logloss, 6), "RMSE(bins)": round(rmse_bins, 6),
                        "AUC": auc}, "user": int(user_id), "size": len(review_ths)}


def metrics_arrays(user, labels, predictions):
    """scripts/compress_metrics.metrics for ArrayPredictions."""
    review_ths, bins = labels
    if not review_ths:
        return {"size": 0, "skipped": "no benchmark test reviews"}
    ths_all = np.asarray(review_ths, dtype=np.int64)
    bins_all = np.asarray(bins)
    unique = len(np.unique(ths_all)) == len(ths_all)
    result = {"size": len(review_ths)}
    has_rating, ratings_all = predictions.lookup("rating", ths_all)
    for head in ("ahead", "instant"):
        has_pred, preds_all = predictions.lookup(head, ths_all)
        kept = has_pred & has_rating
        n_kept = int(kept.sum())
        result[f"{head}_coverage"] = n_kept / len(review_ths)
        if not n_kept:
            continue
        if unique:
            stats = get_stats_arrays(user, ths_all[kept], bins_all[kept], preds_all[kept], ratings_all[kept])
        else:   # duplicate benchmark review_ths: upstream semantics (bins dict keeps the last bin)
            import contextlib
            import io
            from rwkv.get_result import get_stats
            ths = ths_all[kept].tolist()
            bins_dict = dict(zip(ths, bins_all[kept].tolist()))
            with contextlib.redirect_stdout(io.StringIO()):
                stats, _ = get_stats(user, ths, bins_dict, dict(zip(ths, map(np.float64, preds_all[kept]))),
                                     dict(zip(ths, map(np.float64, ratings_all[kept]))))
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result

