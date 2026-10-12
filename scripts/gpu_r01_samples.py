"""F30: NumPy sample construction that reproduces rwkv.data_processing.create_sample exactly.

The per-chunk pandas work (query rows, ~30 column assignments, concat, two-key
sort, groupby indices and Python inverse permutations) dominates CPU preparation.
This path keeps add_segment_features unchanged, then rebuilds the same arrays with
NumPy: stable lexsort for sort_values, insertion-ordered sets for group order and
identical dtype conversions into the RWKVSample tensors.
"""

import numpy as np
import torch

from rwkv.config import RWKV_SUBMODULES
from rwkv.data_processing import CARD_FEATURE_COLUMNS, ModuleData, RWKVSample, add_segment_features

# Copied from rwkv.data_processing.add_queries; every column must be in exactly one list.
KEEP = ["card_id", "day_offset", "day_offset_first", "elapsed_days", "elapsed_days_cumulative", "elapsed_seconds",
        "elapsed_seconds_sin", "elapsed_seconds_cos", "elapsed_seconds_cumulative", "elapsed_seconds_cumulative_sin",
        "elapsed_seconds_cumulative_cos", "user_id", "review_th", "note_id", "deck_id", "preset_id", "note_id_is_nan",
        "deck_id_is_nan", "preset_id_is_nan", "i", "is_first_review", "cum_new_cards", "diff_new_cards", "diff_reviews",
        "cum_new_cards_today", "cum_reviews_today", "scaled_elapsed_days", "scaled_elapsed_days_cumulative",
        "scaled_elapsed_seconds", "scaled_elapsed_seconds_cumulative", "day_offset_diff", "day_of_week", "is_query"]
REJECT = ["rating", "duration", "scaled_duration", "state", "scaled_state", "y", "rating_1", "rating_2", "rating_3",
          "rating_4", "skip", "has_label", "label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating",
          "label_review_th", "label_is_equalize"]
LABELS = ["label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating", "has_label", "label_is_equalize", "is_query"]


def _matrix(columns):
    """Float64 columns: bool, integer and float values round to float32 exactly as DataFrame.to_numpy() does."""
    return np.column_stack([column.astype(np.float64) for column in columns])


def create_sample_fast(user_id, section_df, equalize_review_ths, dtype, device):
    frame = add_segment_features(section_df.copy()).reset_index(drop=True)
    data = {name: frame[name].to_numpy() for name in frame.columns}
    data["skip"] = np.zeros(len(frame), dtype=bool)
    if set(data) != set(KEEP) | set(REJECT) or len(data) != len(KEEP) + len(REJECT):
        raise ValueError("Sample columns differ from the add_queries keep/reject lists")
    queries = np.flatnonzero(data["is_first_review"] == False)  # noqa: E712 - same comparison as add_queries
    if len(queries):
        query = {name: values[queries] for name, values in data.items()}
        rating, review_th = query["rating"], query["review_th"]
        for name in REJECT:
            query[name] = np.zeros(len(queries), dtype=np.int64)
        query["skip"] = np.ones(len(queries), dtype=bool)
        query["label_rating"] = rating
        if not np.isin(rating, (1, 2, 3, 4)).all():
            raise KeyError("Rating outside the add_queries label map")
        query["label_y"] = np.where(rating == 1, 0, 1).astype(np.int64)
        query["label_review_th"] = review_th
        query["has_label"] = np.ones(len(queries), dtype=bool)
        query["is_query"] = np.ones(len(queries), dtype=np.int64)
        query["label_is_equalize"] = np.isin(query["label_review_th"], list(set(equalize_review_ths))).astype(np.int64)
        data = {name: np.concatenate([data[name], query[name]]) for name in data}
        # sort_values(by=["review_th", "skip"], ascending=[True, False]) is a stable lexsort.
        order = np.lexsort((~data["skip"].astype(bool), data["review_th"]))
        data = {name: values[order] for name, values in data.items()}
    label_rating = data["label_rating"]
    if not (label_rating.min() >= 1 and label_rating.max() <= 4):
        raise AssertionError("label_rating outside 1-4")
    length = len(data["review_th"])

    ids = {name: torch.tensor(data[name], dtype=torch.int32, device=device, requires_grad=False) for name in RWKV_SUBMODULES}
    modules = {}
    for submodule in RWKV_SUBMODULES:
        values = data[submodule]
        ordering = np.argsort(values, kind="stable")
        keys, starts = np.unique(values[ordering], return_index=True)
        bounds = list(starts) + [len(values)]
        indices = {key: ordering[bounds[i]:bounds[i + 1]] for i, key in enumerate(keys)}
        by_length = {}
        # Same insertion-ordered set iteration as _module_group_locations.
        for name in set(values):
            locations = indices[name]
            by_length.setdefault(len(locations), []).append(locations)
        locs, split_len, split_B = [], [], []
        for size in sorted(by_length):
            split_len.append(size)
            split_B.append(len(by_length[size]))
            locs.extend(by_length[size])
        flat = np.concatenate(locs)
        inverse = np.empty_like(flat)
        inverse[flat] = np.arange(len(flat))
        modules[submodule] = ModuleData(split_len=np.array(split_len, dtype=np.int32), split_B=np.array(split_B, dtype=np.int32),
                                        from_perm=torch.tensor(flat, dtype=torch.int32, device=device),
                                        to_perm=torch.tensor(inverse, dtype=torch.int32, device=device))
    return RWKVSample(
        user_id=user_id, start_th=int(data["review_th"].min()), end_th=int(data["review_th"].max()), length=length,
        modules=modules, ids=ids,
        card_features=torch.tensor(_matrix([data[c] for c in CARD_FEATURE_COLUMNS]), dtype=dtype, device=device, requires_grad=False),
        global_labels=torch.tensor(_matrix([data[c] for c in LABELS]), dtype=dtype, device=device, requires_grad=False),
        review_ths=torch.tensor(data["review_th"], dtype=torch.int32, device=device, requires_grad=False),
        label_review_ths=torch.tensor(data["label_review_th"], dtype=torch.int32, device=device, requires_grad=False),
        day_offsets=torch.tensor(data["day_offset"], dtype=torch.int32, device=device, requires_grad=False),
        day_offsets_first=torch.tensor(data["day_offset_first"], dtype=torch.int32, device=device, requires_grad=False),
        skips=torch.tensor(data["skip"].astype(bool), dtype=torch.bool, device=device, requires_grad=False))


def samples_for(section, user, options, chunk=128):
    """F30 when the frozen options select it; otherwise the original per-chunk construction."""
    if options.get("fast_samples"):
        return make_samples_fast(section, user, chunk)
    from scripts.gpu_batch1 import make_samples
    return make_samples(section, user, chunk)


def make_samples_fast(section, user, chunk):
    return [create_sample_fast(user, section.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"))
            for i in range(0, len(section), chunk)]
