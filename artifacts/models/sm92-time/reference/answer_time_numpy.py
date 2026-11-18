"""NumPy reference of the answer-time head (package sm72e25-time): the 26 explicit causal features, the head, and the
summaries the fork shows (median answer time per grade, P(fast)). Standalone: NumPy only.

The 448 frozen inputs come from the released small model's trunk at the query row (time_head.json "inputs.frozen");
the published vectors store them (fp32_time_frozen) so this file can recompute the head from them.
"""
import heapq
import math
import struct
from pathlib import Path

import numpy as np

BINS = 16
EXPLICIT_DIM = 26
LN8 = math.log(8.0)
BIN_EDGES_MS = [500.0 * 120.0 ** (k / 14) for k in range(15)]
LOG_EDGES = np.log(np.asarray(BIN_EDGES_MS) / 1000.0)
FAST_BINS = 7   # bins 0..6: under 3.891 s


class RunningMedian:
    """Exact running median (two heaps); median() of the values pushed so far (None if empty)."""

    def __init__(self):
        self.low, self.high = [], []

    def push(self, x):
        if not self.low or x <= -self.low[0]:
            heapq.heappush(self.low, -x)
        else:
            heapq.heappush(self.high, x)
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if not self.low:
            return None
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2


def explicit_features(card, rating, duration, elapsed):
    """[n, 26] float32: row i = the features for predicting review i from reviews 0..i-1 and review i's card id and
    elapsed days only."""
    card = np.asarray(card, np.int64); rating = np.asarray(rating, np.int64)
    duration = np.asarray(duration, np.int64); elapsed = np.asarray(elapsed, np.int64)
    n = len(duration)
    out = np.zeros((n, EXPLICIT_DIM), np.float64)
    logd = np.log(np.maximum(duration, 1) / 1000.0)
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    counts = [0, 0, 0, 0, 0]
    cards = {}
    prev_user, ema = None, LN8
    for i in range(n):
        c = int(card[i])
        st = cards.get(c)
        if st is None:
            st = [[], 0, 0]
            cards[c] = st
        last3, cnt, last_grade = st
        row = out[i]
        for k in range(3):
            if k < len(last3):
                row[k] = last3[k] - LN8
            else:
                row[3 + k] = 1.0
        row[6] = math.log1p(cnt) / 5.0
        if 1 <= last_grade <= 4:
            row[6 + last_grade] = 1.0
        em = every.median()
        fb = em if em is not None else LN8
        for g in (1, 2, 3, 4):
            m = per[g].median()
            row[10 + g] = (m if m is not None else fb) - LN8
            row[14 + g] = math.log1p(counts[g]) / 10.0
        row[19] = fb - LN8
        if prev_user is None:
            row[21] = 1.0
        else:
            row[20] = prev_user - LN8
        row[22] = ema - LN8
        e = int(elapsed[i])
        row[23] = math.log1p(max(e, 0)) / 3.0
        row[24] = 1.0 if e < 0 else 0.0
        row[25] = math.log1p(i) / 10.0
        r = int(rating[i])
        st[1] = cnt + 1
        st[2] = r
        if duration[i] > 0:
            x = float(logd[i])
            per.get(r, every).push(x)
            every.push(x)
            if r in per:
                counts[r] += 1
            last3.insert(0, x)
            del last3[3:]
            prev_user = x
            ema = 0.9 * ema + 0.1 * x
    return out.astype(np.float32)


def read_weights(path):
    """ARWKVWEIGHTS1 / ARWKVWEIGHTS8PC1 -> {name: float32 array} (int8 rows dequantized as code * scale)."""
    data = Path(path).read_bytes()
    pos = 0

    def take(n):
        nonlocal pos
        out = data[pos:pos + n]
        if len(out) != n:
            raise ValueError("truncated")
        pos += n
        return out

    int8 = data.startswith(b"ARWKVWEIGHTS8PC1")
    take(len(b"ARWKVWEIGHTS8PC1") if int8 else len(b"ARWKVWEIGHTS1"))
    count, = struct.unpack("<I", take(4))
    tensors = {}
    for _ in range(count):
        n, = struct.unpack("<H", take(2))
        name = take(n).decode("utf-8")
        kind = struct.unpack("<B", take(1))[0] if int8 else 0
        rank, = struct.unpack("<B", take(1))
        shape = struct.unpack(f"<{rank}I", take(4 * rank)) if rank else ()
        size = int(np.prod(shape)) if shape else 1
        if kind == 1:
            scale = np.frombuffer(take(4 * shape[0]), dtype="<f4")
            codes = np.frombuffer(take(size), dtype=np.int8).reshape(shape)
            tensors[name] = (codes.astype(np.float32) * scale[:, None]).astype(np.float32)
        else:
            tensors[name] = np.frombuffer(take(4 * size), dtype="<f4").reshape(shape).astype(np.float32)
    if pos != len(data):
        raise ValueError("trailing bytes")
    return tensors


def head_logits(weights, inputs):
    """[n, 474] -> [n, 64] float32 logits."""
    x = np.asarray(inputs, np.float32)
    h = x @ weights["time_mlps.0.hidden.weight"].T + weights["time_mlps.0.hidden.bias"]
    h = np.maximum(h, 0)
    return (h @ weights["time_mlps.0.out.weight"].T + weights["time_mlps.0.out.bias"]).astype(np.float32)


def grade_probabilities(logits):
    """[n, 64] -> [n, 4, 16] softmax per grade."""
    z = np.asarray(logits, np.float64).reshape(-1, 4, BINS)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def median_log_s(prob):
    """Median of [N, 16] bin distributions in ln seconds (linear in ln time inside the crossing bin)."""
    prob = np.asarray(prob, np.float64)
    lo = np.r_[math.log(0.25), LOG_EDGES]
    hi = np.r_[LOG_EDGES, LOG_EDGES[-1]]
    cdf = np.cumsum(prob, axis=1)
    b = np.minimum((cdf < 0.5).sum(axis=1), BINS - 1)
    before = np.where(b > 0, np.take_along_axis(cdf, np.maximum(b - 1, 0)[:, None], 1)[:, 0], 0.0)
    pb = np.take_along_axis(prob, b[:, None], 1)[:, 0]
    frac = np.clip((0.5 - before) / np.maximum(pb, 1e-12), 0, 1)
    return lo[b] + frac * (hi[b] - lo[b])


def summary(logits, button_probabilities=None):
    """Per row: median seconds per grade [n, 4], P(under 3.891 s) per grade [n, 4], and with the instant head's
    button probabilities [n, 4] the grade-mixed median seconds [n]."""
    p = grade_probabilities(logits)
    n = p.shape[0]
    med = np.exp(median_log_s(p.reshape(-1, BINS))).reshape(n, 4)
    fast = p[..., :FAST_BINS].sum(-1)
    mixed = None
    if button_probabilities is not None:
        w = np.asarray(button_probabilities, np.float64)
        mixed = np.exp(median_log_s((p * w[..., None]).sum(1)))
    return med, fast, mixed
