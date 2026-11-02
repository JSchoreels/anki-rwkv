"""Scheduling-time vs review-time accuracy (SM81): the start-of-day batch for the held-out scorer.

Start of day = the instant head p_imm for each review of day D computed from the state after every review before day D
only (no update of day D yet), with the review's own query-row inputs (elapsed time of the actual review, counters,
codes). Day = the dataset's `day_offset` (anki-revlogs-10k: the Anki day with the user's rollover hour).

How: the query rows never commit state (skip = 1: rwkv/data_processing.add_queries, the WKV kernels keep the state on
skip rows, the token shift reads the last non-skipped row). So moving every query row of day D in front of day D's
first update row gives each of them exactly the state after the last update before day D, and leaves the update rows
(and so the curve head) unchanged. The scorer's normal row order is (review_th, query first); here it is
(day_offset, query rows first, review_th).

Exactness details:
  * the sample is permuted (rows) and its per-entity groups are rebuilt in the SAME group order as the normal sample
    (only the row order inside each group changes), so the bucketing / gathers have the same shapes;
  * prepare() draws the entity codes in `set(ids)` iteration order, which can depend on insertion order: the start
    tensor (features + codes + calendar encodings) of the start-of-day batch is therefore the normal batch's start
    tensor permuted, so every row has exactly the same inputs as in the normal pass;
  * an entity whose first rows are now query rows (a card / note / deck first seen on day D and asked again that day;
    the user stream on the user's first day) has no state at the start of day D: those leading query rows get a zero
    state (the WKV start) and their own input as token shift (time_shift_select = itself, the convention for a new
    entity), and the entity's first update row also shifts from itself. prepare() would point all of them at row 0;
    in the normal order every entity starts with an update row, so this changes nothing there.
Check (CPU, `check` below): the normal pass is unchanged, the curve head of the start-of-day pass equals the normal
pass, and start of day == review time exactly for the user's first review of each day (and differs otherwise).
"""
import numpy as np
import torch


def sod_order(sample):
    """Permutation `perm` (new row i = old row perm[i]): (day_offset, query rows first, review_th)."""
    day = sample.day_offsets.cpu().numpy().astype(np.int64)
    query = sample.skips.cpu().numpy().astype(bool)
    th = sample.review_ths.cpu().numpy().astype(np.int64)
    return np.lexsort((th, (~query).astype(np.int64), day))


def permute_sample(sample, perm):
    """The RWKVSample with rows in `perm` order and every entity group kept in the same group order."""
    from rwkv.data_processing import ModuleData, RWKVSample
    n = sample.length
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    modules = {}
    for name, md in sample.modules.items():
        old = md.from_perm.cpu().numpy().astype(np.int64)
        locs, at = [], 0
        for length, count in zip(md.split_len, md.split_B):
            for _ in range(int(count)):
                locs.append(np.sort(inv[old[at:at + int(length)]]))
                at += int(length)
        assert at == n, (name, at, n)
        locs = np.concatenate(locs)
        to_perm = np.empty(n, dtype=np.int64)
        to_perm[locs] = np.arange(n)
        modules[name] = ModuleData(split_len=md.split_len.copy(), split_B=md.split_B.copy(),
                                   from_perm=torch.tensor(locs, dtype=torch.int32),
                                   to_perm=torch.tensor(to_perm, dtype=torch.int32))
    return RWKVSample(
        user_id=sample.user_id, start_th=sample.start_th, end_th=sample.end_th, length=n,
        card_features=sample.card_features[perm_t], ids={k: v[perm_t] for k, v in sample.ids.items()},
        modules=modules, global_labels=sample.global_labels[perm_t], review_ths=sample.review_ths[perm_t],
        label_review_ths=sample.label_review_ths[perm_t], day_offsets=sample.day_offsets[perm_t],
        day_offsets_first=sample.day_offsets_first[perm_t], skips=sample.skips[perm_t])


def fix_leading_skips(batch):
    """Entities whose sequence starts with skipped (query) rows: those rows and the first update row shift from
    themselves (prepare() points them at position 0). Returns the number of sequences changed."""
    changed = 0
    for m, splits in enumerate(batch.sub_gather):
        for s, take in enumerate(splits):
            r = batch.sub_gather_lens[m][s]
            take2 = take.view(-1, r)
            skip2 = batch.skips[m][s].view(-1, r)
            ts2 = batch.time_shift_selects[m][s].view(-1, r)
            upd = (~skip2.bool()) & (take2 >= 0)
            assert bool(upd.any(dim=1).all()), "an entity sequence without any update row"
            first = upd.to(torch.int8).argmax(dim=1)                       # first update position per sequence
            rows = torch.nonzero(first > 0).flatten()
            if rows.numel() == 0:
                continue
            pos = torch.arange(r, dtype=ts2.dtype).expand(len(rows), r)
            lead = pos <= first[rows, None].to(ts2.dtype)
            sub = ts2[rows]
            sub[lead] = pos[lead]
            ts2[rows] = sub
            changed += int(rows.numel())
    return changed


def sod_batch(sample, normal_batch, seed):
    """The start-of-day PreparedBatch of a one-sample normal batch (`normal_batch` = prepare([sample], seed=seed))."""
    from rwkv.prepare_batch import prepare
    perm = sod_order(sample)
    sod = prepare([permute_sample(sample, perm)], seed=seed)
    assert sod.start.shape == normal_batch.start.shape == (sample.length, normal_batch.start.shape[1])
    sod.start = normal_batch.start[torch.as_tensor(perm, dtype=torch.long)].clone()   # same inputs per row
    sod.sod_leading_fixed = fix_leading_skips(sod)
    return sod


def first_of_day_ths(data_path, user):
    """review_ths (1-based) of the user's first review of each day_offset."""
    import pandas as pd
    day = pd.read_parquet(data_path / "revlogs" / f"user_id={user}", columns=["day_offset"])["day_offset"].to_numpy()
    first = np.ones(len(day), dtype=bool)
    first[1:] = day[1:] != day[:-1]
    return np.flatnonzero(first) + 1, day
