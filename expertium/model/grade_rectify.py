"""The grade order by construction (SM95): the four what-if curves of a review, pooled so that
R_Again(t) <= R_Hard(t) <= R_Good(t) <= R_Easy(t) holds exactly at every t.

The rectifier is the unweighted isotonic (non-decreasing) regression of the four values at the same t, written in its
max-min form

    out_i = max_{j <= i} min_{k >= i} m(j, k),     m(j, k) = (x_j + ... + x_k) / (k - j + 1),

applied only to rows that are not already ordered (ordered rows are returned bit for bit). Why the order is exact in
floating point, not just in exact arithmetic: every m(j, k) is computed once per row, and out_{i+1} takes its max over a
superset of j and its min over a subset of k compared with out_i, so out_i <= out_{i+1} holds for whatever values the
rounded m(j, k) table holds (min and max are exact operations). Each rectified curve stays a forgetting curve: means,
minima and maxima of non-increasing functions of t are non-increasing, and values stay inside [min_g R_g, max_g R_g].
With one target retention for all four buttons this is exactly the fork's PAVA on the margins R_g(t) - target
(rslib/src/rwkv/engine.rs pava_non_decreasing at each point of the interval grid), so on these curves the fork's PAVA
changes nothing.
"""
import numpy as np
import torch

SLOTS = 4


def _means(columns, lo, hi):
    total = columns[lo]
    for k in range(lo + 1, hi + 1):
        total = total + columns[k]
    return total if hi == lo else total / float(hi - lo + 1)


def pava4_torch(values):
    """[..., 4] -> [..., 4] non-decreasing along the last axis; differentiable (sub-gradients of min / max)."""
    cols = values.unbind(-1)
    m = {(j, k): _means(cols, j, k) for j in range(SLOTS) for k in range(j, SLOTS)}
    out = []
    for i in range(SLOTS):
        best = None
        for j in range(i + 1):
            low = None
            for k in range(i, SLOTS):
                low = m[(j, k)] if low is None else torch.minimum(low, m[(j, k)])
            best = low if best is None else torch.maximum(best, low)
        out.append(best)
    pooled = torch.stack(out, dim=-1)
    ordered = (values[..., 1:] >= values[..., :-1]).all(dim=-1, keepdim=True)
    return torch.where(ordered, values, pooled)


def pava4_numpy(values):
    """The same operator in NumPy (any float dtype), for reports and independent recomputes."""
    v = np.asarray(values)
    cols = [v[..., i] for i in range(SLOTS)]
    m = {(j, k): _means(cols, j, k) for j in range(SLOTS) for k in range(j, SLOTS)}
    out = np.empty_like(v)
    for i in range(SLOTS):
        best = None
        for j in range(i + 1):
            low = None
            for k in range(i, SLOTS):
                low = m[(j, k)] if low is None else np.minimum(low, m[(j, k)])
            best = low if best is None else np.maximum(best, low)
        out[..., i] = best
    ordered = np.all(v[..., 1:] >= v[..., :-1], axis=-1, keepdims=True)
    return np.where(ordered, v, out)


def pava4_scalar(v):
    """Reference: literal stack PAVA (pool adjacent violators, equal weights) on 4 python floats."""
    stack = []  # (sum, count)
    for x in v:
        stack.append((float(x), 1))
        while len(stack) >= 2 and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]:
            s1, c1 = stack.pop()
            s0, c0 = stack.pop()
            stack.append((s0 + s1, c0 + c1))
    out = []
    for s, c in stack:
        out.extend([s / c] * c)
    return out


def rectify_slots(curve_probs_raw, global_labels):
    """(flat indices of the rectified real rows, their rectified curve values) for a batch with SM95 probe rows.

    Probe layout (scripts/gpu_r01_samples.add_rectify_probes): before a probed real row r with rating g sit exactly the
    three probes of the other grades, in grade order, marked label_elapsed_days = -grade; each probe copies row r's
    label time, so all four curve values are already evaluated at row r's t. Slot g of the 4-vector is row r itself
    (its own dropout mask and gradient path), the other slots are the probes."""
    marker = global_labels[..., 1].reshape(-1)
    probe = marker < 0
    n = probe.numel()
    prev = [torch.nn.functional.pad(probe[:n - s], (s, 0), value=False) for s in (1, 2, 3)]
    real = torch.nonzero(~probe & prev[0] & prev[1] & prev[2]).squeeze(1)
    if real.numel() == 0:
        return real, curve_probs_raw.reshape(-1)[real]
    grades = [-marker.index_select(0, real - s) for s in (3, 2, 1)]          # ascending, the real grade left out
    g = (10.0 - grades[0] - grades[1] - grades[2]).round().long()            # 1..4
    flat = curve_probs_raw.reshape(-1)
    slots = []
    for slot in range(1, SLOTS + 1):
        below = torch.full_like(real, slot) < g                              # the probe of grade `slot` sits at r - 4 + slot
        row = torch.where(g == slot, real, torch.where(below, real - 4 + slot, real - 5 + slot))
        slots.append(flat.index_select(0, row))
    rect = pava4_torch(torch.stack(slots, dim=1))
    return real, rect.gather(1, (g - 1).unsqueeze(1)).squeeze(1)


def rectify_curve(curve_probs_raw, global_labels):
    """curve_probs_raw [B, T] with every probed real row replaced by its grade's rectified value (out of place)."""
    rows, values = rectify_slots(curve_probs_raw, global_labels)
    if rows.numel() == 0:
        return curve_probs_raw
    return curve_probs_raw.reshape(-1).index_copy(0, rows, values.to(curve_probs_raw.dtype)).view_as(curve_probs_raw)
