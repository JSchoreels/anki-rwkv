"""The small model in the review simulator (SM41): numpy port of srs-benchmark's finished 34-parameter FSRS-7
(`models/fsrs_v7.py`: dual stability, the mixture forgetting curve, `--secs` intervals in fractional days).

Numpy only, so both the anki-rwkv venv (`scripts/sm41_engine.py`) and the srs-benchmark venv
(`scripts/sm41_fsrs7_fit.py`, the pre-split refits and their proof) import the same code.
"""
import math

import numpy as np

FSRS7_DEFAULT = [0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0, 1.3826, 0.7024, 0.5999, 0.8146,
                 0.6398, 1.0, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567, 0.0801, 0.2421,
                 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048]


class FSRS7:
    """Finished dual-stability FSRS-7 (34 parameters); state [..., 3] = long-term S, short-term S, D (zeros = new)."""
    S_MIN, S_MAX, D_MIN, D_MAX = 0.0001, 36500.0, 1.0, 10.0      # --secs: s_min 1e-4 (srs-benchmark config.py)

    def __init__(self, w):
        self.w = np.asarray(w, np.float64)

    def init_d(self, rating):
        return self.w[4] - np.exp(self.w[5] * (rating - 1)) + 1

    def r1(self, t, ss):
        w = self.w
        t = np.maximum(t, 0.0)
        decay1 = -np.clip(w[23] * ss ** (w[33] - 0.3), 0.01, 0.95)
        factor1 = np.exp(np.minimum(np.log(w[25]) / decay1, 60.0)) - 1.0
        return (t / ss * factor1 + 1.0) ** decay1

    def recall(self, t, s, ss, d):
        w = self.w
        t = np.maximum(t, 0.0)
        r1 = self.r1(t, ss)
        decay2 = -np.clip(w[24], 0.01, 0.95)
        factor2 = w[26] ** (1.0 / decay2) - 1.0
        r2 = (t / s * factor2 * np.exp((d - 5.0) * (w[32] - 0.3)) + 1.0) ** decay2
        w1 = w[27] * ss ** (-w[29])
        w2 = w[28] * s ** w[30] * np.exp((d - 5.0) * (w[31] - 0.5))
        return ((w1 * r1 + w2 * r2) / (w1 + w2)) * (1.0 - 2e-5) + 1e-5

    def _stab(self, last_s, last_d, r, rating, k):
        w = self.w
        hard = np.where(rating == 2, w[k + 6], 1.0)
        easy = np.where(rating == 4, w[k + 7], 1.0)
        fail = w[k + 3] * ((last_s + 1.0) ** w[k + 4] - 1.0) * np.exp((1.0 - r) * w[k + 5])
        pls = np.minimum(last_s, fail)
        sinc = np.exp(w[k] - 1.5) * (11.0 - last_d) * last_s ** (-w[k + 1]) * (np.exp((1.0 - r) * w[k + 2]) - 1.0) \
            * hard * easy + 1.0
        return np.where(rating > 1, np.maximum(pls, last_s * sinc), pls)

    def step(self, dt_days, rating, state):
        w = self.w
        rating = np.asarray(rating, np.float64)
        first = (state == 0).all(-1)
        idx = np.clip(rating.astype(np.int64), 1, 4) - 1
        i_s = w[idx]
        i_d = np.clip(self.init_d(rating), self.D_MIN, self.D_MAX)
        s = np.clip(state[..., 0], self.S_MIN, self.S_MAX)
        ss = np.clip(state[..., 1], self.S_MIN, self.S_MAX)
        d = np.clip(state[..., 2], self.D_MIN, self.D_MAX)
        r = self.recall(dt_days, s, ss, d)
        u_s = self._stab(s, d, r, rating, 7)
        u_ss = self._stab(ss, d, self.r1(dt_days, ss), rating, 15)
        u_ss = np.where(rating == 1, np.minimum(u_ss, 0.8 * u_s), u_ss)
        delta = -w[6] * (rating - 3)
        delta = np.where(rating == 1, delta * (r + 0.1), delta)
        u_d = d + delta * (10 - d) / 9
        u_d = np.clip(0.01 * self.init_d(4.0) + 0.99 * u_d, self.D_MIN, self.D_MAX)
        out = np.stack([np.where(first, i_s, u_s), np.where(first, 0.8 * i_s, u_ss), np.where(first, i_d, u_d)], -1)
        out[..., :2] = np.clip(out[..., :2], self.S_MIN, self.S_MAX)
        out[..., 2] = np.clip(out[..., 2], self.D_MIN, self.D_MAX)
        return out

    def interval(self, state, r):
        """Days t where recall(t) = r (bisection in ln t over [1e-4, 1e6] days), per row."""
        s, ss, d = state[..., 0], state[..., 1], state[..., 2]
        r = np.broadcast_to(np.asarray(r, np.float64), s.shape)
        lo = np.full(s.shape, math.log(1e-4)); hi = np.full(s.shape, math.log(1e6))
        for _ in range(40):
            mid = (lo + hi) / 2
            ok = self.recall(np.exp(mid), s, ss, d) >= r
            lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
        return np.exp(lo)
