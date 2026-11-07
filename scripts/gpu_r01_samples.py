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


def _matrix(columns, direct=False):
    """Float64 columns: bool, integer and float values round to float32 exactly as DataFrame.to_numpy() does."""
    if direct:
        # F45: the same float64 conversions directly into the final matrix,
        # without one temporary array per column followed by column_stack.
        result = np.empty((len(columns[0]), len(columns)), dtype=np.float64)
        for index, column in enumerate(columns):
            result[:, index] = column
        return result
    return np.column_stack([column.astype(np.float64) for column in columns])


# Grade-order sanity (SM33): track option "grade_order": {"weight": W, "density": D, "horizons_days": [...], "seed": S}.
# Absent (default) = nothing here runs and samples, batches, losses and gradients are bit for bit today's.
PROBE_ORDER_QUERY, PROBE_ORDER_REAL = 0, 5


def add_grade_probes(data, spec, user_id):
    """Insert four what-if rows (rating 1..4) right before a seeded `density` share of the committed review rows:
    copies of the review row with the rating one-hot (and rating / y) replaced, skip = True (no state commit, like
    the fork's answer_heads and the query rows), has_label = False (no curve / instant / KD target) and
    label_elapsed_days = -g (the probe grade; real rows hold elapsed days >= 0 or pi there, and the loss never reads
    that column otherwise). Order inside a review_th: query row, probes 1..4, the real row, so probes g and g + 1
    are adjacent rows and every real row keeps its committed history and token-shift source."""
    density = float(spec.get("density", 0.0))
    real = np.flatnonzero(~data["skip"].astype(bool))
    if density <= 0 or not len(real):
        return data
    rng = np.random.default_rng([int(spec.get("seed", 33)), int(user_id), int(data["review_th"].min())])
    pick = real[rng.random(len(real)) < density]
    if not len(pick):
        return data
    order = np.where(data["is_query"].astype(np.int64) == 1, PROBE_ORDER_QUERY, PROBE_ORDER_REAL)
    parts, orders = [data], [order]
    for g in (1, 2, 3, 4):
        probe = {name: values[pick].copy() for name, values in data.items()}
        probe["rating"] = np.full(len(pick), g, dtype=data["rating"].dtype)
        for i in (1, 2, 3, 4):
            probe[f"rating_{i}"] = np.full(len(pick), int(i == g), dtype=data[f"rating_{i}"].dtype)
        probe["y"] = np.full(len(pick), int(g > 1), dtype=data["y"].dtype)
        probe["skip"] = np.ones(len(pick), dtype=data["skip"].dtype)
        probe["has_label"] = np.zeros(len(pick), dtype=data["has_label"].dtype)
        probe["label_elapsed_days"] = np.full(len(pick), -g, dtype=np.float64)
        parts.append(probe)
        orders.append(np.full(len(pick), g, dtype=np.int64))
    merged = {name: np.concatenate([part[name] for part in parts]) for name in data}
    key = np.lexsort((np.concatenate(orders), merged["review_th"]))
    return {name: values[key] for name, values in merged.items()}


# The grade order by construction (SM95): track option "grade_rectify": {"density": D, "seed": S}. Absent (default) =
# nothing here runs and samples, batches, losses and gradients are bit for bit the code before.
def add_rectify_probes(data, spec, user_id):
    """Before a seeded `density` share of the labelled committed rows (density 1.0 = all, the scorer's setting), insert
    the three what-if rows of the OTHER grades: copies of the review row with the rating one-hot (and rating / y)
    replaced, skip = True (no state commit, like the fork's answer_heads), has_label = False (no target of their own)
    and label_elapsed_days = -g (the probe grade; real rows hold >= 0 or pi there). Each probe keeps the row's label
    time, so its curve is evaluated at the real row's t. Order inside a review_th: query row, the three probes in grade
    order, the real row; expertium.model.grade_rectify.rectify_curve finds them as the three rows right before a real
    row, and the real row itself is the slot of its own grade."""
    density = float(spec.get("density", 0.0))
    real = np.flatnonzero(~data["skip"].astype(bool) & data["has_label"].astype(bool))
    if density <= 0 or not len(real):
        return data
    rng = np.random.default_rng([int(spec.get("seed", 95)), int(user_id), int(data["review_th"].min())])
    pick = real[rng.random(len(real)) < density]
    if not len(pick):
        return data
    order = np.where(data["is_query"].astype(np.int64) == 1, PROBE_ORDER_QUERY, PROBE_ORDER_REAL)
    parts, orders = [data], [order]
    rating = data["rating"][pick].astype(np.int64)
    for g in (1, 2, 3, 4):
        rows = pick[rating != g]
        probe = {name: values[rows].copy() for name, values in data.items()}
        probe["rating"] = np.full(len(rows), g, dtype=data["rating"].dtype)
        for i in (1, 2, 3, 4):
            probe[f"rating_{i}"] = np.full(len(rows), int(i == g), dtype=data[f"rating_{i}"].dtype)
        probe["y"] = np.full(len(rows), int(g > 1), dtype=data["y"].dtype)
        probe["skip"] = np.ones(len(rows), dtype=data["skip"].dtype)
        probe["has_label"] = np.zeros(len(rows), dtype=data["has_label"].dtype)
        probe["label_elapsed_days"] = np.full(len(rows), -g, dtype=np.float64)
        parts.append(probe)
        orders.append(np.full(len(rows), g, dtype=np.int64))
    merged = {name: np.concatenate([part[name] for part in parts]) for name in data}
    key = np.lexsort((np.concatenate(orders), merged["review_th"]))
    return {name: values[key] for name, values in merged.items()}


def fix_leading_shift(batch, probes=4):
    """In a sub-sequence that starts with the 4 probe rows of its first committed row (a card's first review, or the
    first row of a deck / note / preset in the window), those rows and that real row read their own input in the token
    shift: prepare's rule for row 0, which the real row would get without the probes (under state carry, entities
    with a carried state read the carried vector there instead). Without it they would read probe 1's input. Query-led
    sequences (1 or 5 leading skipped rows) keep prepare's rule. A no-op (returns 0) on every batch without probes.
    SM95 (`probes=3`): three probes before a probed row; query-led sequences then start with 1 or 4 skipped rows."""
    changed = 0
    for module_shifts, module_skips, lens in zip(batch.time_shift_selects, batch.skips, batch.sub_gather_lens):
        for i, (shift, skip, r) in enumerate(zip(module_shifts, module_skips, lens)):
            s2 = skip.view(-1, r)
            first = (~s2).to(torch.int8).argmax(dim=1)
            t = torch.arange(r, device=shift.device)
            # only probe-led sequences: their first committed row follows exactly the 4 probes of its review (a
            # query-led sequence starts with 1 or 1 + 4 skipped rows and keeps prepare's rule, bit for bit)
            lead = (t[None, :] <= first[:, None]) & (first[:, None] == probes)
            old = shift.view(-1, r)
            new = torch.where(lead, t[None, :].to(old.dtype).expand_as(old), old)
            changed += int((new != old).sum())
            if changed:
                module_shifts[i] = new.reshape(shift.shape).contiguous()
    return changed


def create_sample_fast(user_id, section_df, equalize_review_ths, dtype, device, column_arrays=None,
                       segment_features_precomputed=False, direct_matrix=False, int64_ids=False, grade_probes=None,
                       rectify_probes=None):
    if column_arrays is None:
        frame = add_segment_features(section_df.copy()).reset_index(drop=True)
        data = {name: frame[name].to_numpy() for name in frame.columns}
    else:
        # F42: extract unchanged columns once per batch, preserving their dtypes.
        # Keep the original pandas arithmetic/groupby for segment-local features,
        # without constructing and copying a DataFrame for each 128-review chunk.
        import pandas as pd
        data = dict(column_arrays)
        if not segment_features_precomputed:
            days = pd.Series(data["day_offset"], copy=False)
            days = days - days.min()
            data["day_offset"] = days.to_numpy()
            data["day_offset_first"] = days.groupby(data["card_id"]).transform("first").to_numpy()
            data["day_of_week"] = (((days % 7) - 3) / 3).to_numpy()
    data["skip"] = np.zeros(len(section_df), dtype=bool)
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
    if grade_probes:
        data = add_grade_probes(data, grade_probes, user_id)
    if rectify_probes:
        data = add_rectify_probes(data, rectify_probes, user_id)
    label_rating = data["label_rating"]
    if not (label_rating.min() >= 1 and label_rating.max() <= 4):
        raise AssertionError("label_rating outside 1-4")
    length = len(data["review_th"])

    # SM21 state carry keys entities by int64 ids (the int32 ids collide for placeholder note ids).
    id_dtype = torch.int64 if int64_ids else torch.int32
    ids = {name: torch.tensor(data[name], dtype=id_dtype, device=device, requires_grad=False) for name in RWKV_SUBMODULES}
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
        card_features=torch.tensor(_matrix([data[c] for c in CARD_FEATURE_COLUMNS], direct_matrix), dtype=dtype, device=device, requires_grad=False),
        global_labels=torch.tensor(_matrix([data[c] for c in LABELS], direct_matrix), dtype=dtype, device=device, requires_grad=False),
        review_ths=torch.tensor(data["review_th"], dtype=torch.int32, device=device, requires_grad=False),
        label_review_ths=torch.tensor(data["label_review_th"], dtype=torch.int32, device=device, requires_grad=False),
        day_offsets=torch.tensor(data["day_offset"], dtype=torch.int32, device=device, requires_grad=False),
        day_offsets_first=torch.tensor(data["day_offset_first"], dtype=torch.int32, device=device, requires_grad=False),
        skips=torch.tensor(data["skip"].astype(bool), dtype=torch.bool, device=device, requires_grad=False))


def samples_for(section, user, options, chunk=128):
    """F30 when the frozen options select it; otherwise the original per-chunk construction.
    The track option "chunk_reviews" (long-context training, docs/SMALL_MODEL.md 5.3) overrides the chunk length."""
    chunk = options.get("chunk_reviews", chunk)
    if options.get("state_carry"):
        # SM21: the section already holds user-level segment features (gpu_r01_prefetch); int64 entity ids.
        if not (options.get("fast_samples") and options.get("columnar_samples")) or options.get("batch_segment_features"):
            raise ValueError("state_carry needs fast columnar samples without F44 batch segment features")
        return make_samples_fast(section, user, chunk, columnar=True, user_segment_features=True,
                                 direct_matrix=options.get("direct_sample_matrix", False),
                                 grade_probes=options.get("grade_order"),
                                 **({"rectify_probes": options["grade_rectify"]} if options.get("grade_rectify") else {}))
    if options.get("grade_order"):
        raise ValueError("grade_order (SM33) probe rows are implemented for the state_carry sample path only")
    if options.get("grade_rectify"):
        raise ValueError("grade_rectify (SM95) probe rows are implemented for the state_carry sample path only")
    if options.get("fast_samples"):
        return make_samples_fast(section, user, chunk, columnar=options.get("columnar_samples", False),
                                 batch_segment_features=options.get("batch_segment_features", False),
                                 direct_matrix=options.get("direct_sample_matrix", False))
    if options.get("columnar_samples"):
        raise ValueError("F42 columnar sample construction requires fast_samples")
    from scripts.gpu_batch1 import make_samples
    return make_samples(section, user, chunk)


def make_samples_fast(section, user, chunk, *, columnar=False, batch_segment_features=False, direct_matrix=False,
                      user_segment_features=False, grade_probes=None, rectify_probes=None):
    if batch_segment_features and not columnar:
        raise ValueError("F44 requires columnar samples")
    if user_segment_features:
        # SM21: day_offset, day_offset_first and day_of_week were computed over the user's whole training section
        # (scripts.state_carry_train.user_segment_features), so windows keep them; ids are int64.
        if not columnar or batch_segment_features:
            raise ValueError("user-level segment features need columnar samples without F44")
        columns = {name: section[name].to_numpy() for name in section.columns}
        return [create_sample_fast(user, section.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"),
                                   column_arrays={name: values[i:i + chunk] for name, values in columns.items()},
                                   segment_features_precomputed=True, direct_matrix=direct_matrix, int64_ids=True,
                                   grade_probes=grade_probes,
                                   **({"rectify_probes": rectify_probes} if rectify_probes else {}))
                for i in range(0, len(section), chunk)]
    if columnar:
        columns = {name: section[name].to_numpy() for name in section.columns}
        if batch_segment_features and len(section):
            import pandas as pd
            chunks = np.arange(len(section)) // chunk
            days = pd.Series(columns["day_offset"], copy=False)
            days = days - days.groupby(chunks).transform("min")
            columns["day_offset"] = days.to_numpy()
            columns["day_offset_first"] = days.groupby([chunks, columns["card_id"]]).transform("first").to_numpy()
            columns["day_of_week"] = (((days % 7) - 3) / 3).to_numpy()
        return [create_sample_fast(user, section.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"),
                                   column_arrays={name: values[i:i + chunk] for name, values in columns.items()},
                                   segment_features_precomputed=batch_segment_features, direct_matrix=direct_matrix)
                for i in range(0, len(section), chunk)]
    return [create_sample_fast(user, section.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"), direct_matrix=direct_matrix)
            for i in range(0, len(section), chunk)]
