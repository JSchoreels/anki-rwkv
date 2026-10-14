"""Upstream per-user benchmark metrics (rwkv.get_result.get_stats) from given label lists.

Copy of artifacts/collection-eval/2026-10-14/score_heldout.metrics in the main checkout (not importable from here:
its module pulls the main checkout onto sys.path)."""
import contextlib
import io
from io import BytesIO

import numpy as np
import torch


def metrics(user, labels, predictions):
    from rwkv.get_result import get_stats
    review_ths, bins = labels
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
            stats, _ = get_stats(user, ths, bins_dict, {th: np.float64(preds[th]) for th in ths},
                                 {th: np.float64(predictions["rating"][th]) for th in ths})
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result


def read_labels(txn, user):
    """(review_ths, rmse_bins) for one user from an open label-DB transaction (None if absent)."""
    raw = txn.get(f"{user}_review_ths".encode())
    if raw is None:
        return None
    ths = torch.load(BytesIO(raw), weights_only=True).tolist()
    bins = torch.load(BytesIO(txn.get(f"{user}_rmse_bins".encode())), weights_only=True).tolist()
    return ths, bins
