"""Pure NumPy reference forward of our Expertium models (iter36 width 32, w64 / w80 variants) from an exported
weights file + architecture.json (scripts/small_model_export.py). No PyTorch import.

It is the executable specification for a runtime port (the fork's Rust runtime, docs/SMALL_MODEL.md "From
checkpoint to Anki"): it mirrors expertium.model.srs_model.SrsRWKV / expertium.model.variant_model.VariantSrsRWKV
in evaluation mode (dropout off) on the parallel PreparedBatch layout of rwkv.prepare_batch (per-stream gathers,
time-shift selects, skip masks), with the float32 reference recurrence of expertium.model.rwkv_ops.single_timestep.
The one-review-at-a-time form of the same model is expertium/model/variant_rnn.py (checked against the parallel
model by scripts/check_expertium_catch_up.py).

    from scripts.small_model_export import read_weights
    model = NumpyModel(read_weights("weights.f32.bin"), json.load(open("architecture.json")))
    out = model.predict(start=..., sub_gather=..., sub_gather_lens=..., time_shift_selects=..., skips=...,
                        num_data=..., labels=...)   # -> p_curve, p_imm, p_rating
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x, threshold):
    return np.where(x > threshold, x, np.log1p(np.exp(np.minimum(x, threshold))))


def _lerp(start, end, weight):
    # torch.lerp's two-branch formula (exact match of the PyTorch op)
    return np.where(np.abs(weight) < 0.5, start + weight * (end - start), end - (end - start) * (1 - weight))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class NumpyModel:
    def __init__(self, weights, spec, dtype="float32"):
        self.dt = np.dtype(dtype)
        self.w = {k: np.asarray(v, dtype=self.dt) for k, v in weights.items()}
        self.spec = spec
        self.c = spec["constants"]
        self.d = spec["d_model"]

    # ---- primitives
    def linear(self, x, prefix):
        y = x @ self.w[prefix + ".weight"].T
        bias = self.w.get(prefix + ".bias")
        return y + bias if bias is not None else y

    def layer_norm(self, x, prefix):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + self.dt.type(self.c["layer_norm_eps"]))
        return y * self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    def group_norm(self, x, prefix, groups):
        n, channels = x.shape
        g = x.reshape(n, groups, channels // groups)
        mean = g.mean(axis=-1, keepdims=True)
        var = ((g - mean) ** 2).mean(axis=-1, keepdims=True)
        y = ((g - mean) / np.sqrt(var + self.dt.type(self.c["group_norm_eps"]))).reshape(n, channels)
        return y * self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    def normalize(self, x):
        norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
        return x / np.maximum(norm, self.dt.type(self.c["normalize_eps"]))

    @staticmethod
    def shift(x, select):
        return np.take_along_axis(x, select[..., None].astype(np.int64), axis=1)

    # ---- RWKV-7 layer
    def time_mixer(self, p, layer_id, heads, x, v0, select, skip):
        B, T, C = x.shape
        H, K = heads, C // heads
        xl = self.layer_norm(x, p + ".layer_norm")
        xs = self.shift(xl, select)
        mix = _lerp(xl[None], xs[None], self.w[p + ".rkvdag_lerp"])
        r, k, v, d, a, g, ks, vs = mix
        r = self.linear(r, p + ".W_r")
        k = self.linear(k, p + ".W_k")
        k_scale = _sigmoid(self.linear(ks, p + ".k_scale_linear"))
        v_scale = _sigmoid(self.linear(vs, p + ".v_scale_linear"))
        if layer_id == 0:
            v = self.linear(v, p + ".W_v")
            v0 = v
        else:
            v_lerp = _sigmoid(self.linear(self.linear(v, p + ".v_lora_simple.A"), p + ".v_lora_simple.B_and_lamb"))
            v = _lerp(self.linear(v, p + ".W_v"), v0, v_lerp)
        a = _sigmoid(self.linear(self.linear(a, p + ".a_lora_simple.A"), p + ".a_lora_simple.B_and_lamb"))
        g = self.linear(_sigmoid(self.linear(g, p + ".lora_A_g")), p + ".lora_B_g")
        dd = self.linear(np.tanh(self.linear(d, p + ".d_lora_mlp.A")), p + ".d_lora_mlp.B_and_lamb")
        dd = -0.5 - _softplus(-dd, self.c["softplus_threshold"])
        w = np.exp(-np.exp(dd))

        k = k_scale[..., None] * self.normalize(k.reshape(B, T, H, K))
        r = r.reshape(B, T, H, K)
        v = v_scale[..., None] * self.normalize(v.reshape(B, T, H, K))
        w = w.reshape(B, T, H, K)
        a = a.reshape(B, T, H, K)
        kd = k
        k = k * a

        out = np.empty((B, T, H, K), dtype=self.dt)
        state = np.zeros((B, H, K, K), dtype=self.dt)
        akd = a * kd
        for t in range(T):
            s_kd = np.einsum("bhij,bhj->bhi", state, kd[:, t])
            nxt = state * w[:, t, :, None, :] - s_kd[..., None] * akd[:, t, :, None, :]
            nxt = nxt + v[:, t, :, :, None] * k[:, t, :, None, :]
            out[:, t] = np.einsum("bhij,bhj->bhi", nxt, r[:, t])
            keep = skip[:, t].astype(bool)[:, None, None, None]
            state = np.where(keep, state, nxt)

        out = self.group_norm(out.reshape(B * T, C), p + ".out_group_norm", H).reshape(B, T, C)
        bonus = ((r * self.w[p + ".bonus"] * k).sum(axis=-1, keepdims=True) * v).reshape(B, T, C)
        out = self.linear(g * (out + bonus), p + ".W_o")
        return x + out, v0

    def channel_mixer(self, p, x, select):
        xl = self.layer_norm(x, p + ".layer_norm")
        xs = self.shift(xl, select)
        k = self.linear(_lerp(xl, xs, self.w[p + ".lerp_k"]), p + ".W_k")
        return x + self.linear(np.square(np.maximum(k, 0)), p + ".W_v")

    def block(self, stream, layer, x, v0, select, skip):
        s = self.spec["streams"][stream]
        p = f"rwkv_modules.{stream}.blocks.{layer}"
        x, v0 = self.time_mixer(p + ".time_mixer", layer, s["n_heads"], x, v0, select, skip)
        if layer not in s["channel_mixer_stripped_layers"]:
            x = self.channel_mixer(p + ".channel_mixer", x, select)
        return x, v0

    # ---- streams
    def sequential(self, x, gathers, lens, selects, skips):
        for i, stream in enumerate(self.spec["streams"]):
            outputs = []
            for gather, length, select, skip in zip(gathers[i], lens[i], selects[i], skips[i]):
                h = x[np.maximum(gather.astype(np.int64), 0)].reshape(-1, length, self.d)
                sel, sk = select.reshape(-1, length), skip.reshape(-1, length)
                v0 = None
                for layer in range(stream["n_layers"]):
                    h, v0 = self.block(i, layer, h, v0, sel, sk)
                outputs.append(h.reshape(-1, self.d))
            x = np.concatenate(outputs)
        return x

    def interleaved(self, x, gathers, lens, selects, skips):
        """VariantSrsRWKV._interleaved_streams: gathers re-anchored to the canonical feature rows."""
        cur = np.arange(x.shape[0])
        canon, keeps = [], []
        for splits in gathers:
            stream_canon, stream_keep, parts = [], [], []
            for split in splits:
                index = split.astype(np.int64)
                c = np.where(index >= 0, cur[np.maximum(index, 0)], -1)
                stream_canon.append(c)
                stream_keep.append(np.nonzero(c >= 0)[0])
                parts.append(c)
            canon.append(stream_canon)
            keeps.append(stream_keep)
            cur = np.concatenate(parts)
        v0s = [[None] * len(s) for s in gathers]
        for i, layer in self.spec["layer_steps"]:
            new_x = x.copy()
            for s, c in enumerate(canon[i]):
                length = lens[i][s]
                h = x[np.maximum(c, 0)].reshape(-1, length, self.d)
                h, v0s[i][s] = self.block(i, layer, h, v0s[i][s], selects[i][s].reshape(-1, length),
                                          skips[i][s].reshape(-1, length))
                flat = h.reshape(-1, self.d)
                keep = keeps[i][s]
                new_x[c[keep]] = flat[keep]
            x = new_x
        return x

    # ---- heads
    def predict(self, start, sub_gather, sub_gather_lens, time_shift_selects, skips, num_data, labels):
        spec, c = self.spec, self.c
        start = np.array(start, dtype=self.dt)
        if spec["zero_features"]:
            start[..., spec["zero_features"]] = 0
        x = self.linear(start, "features2card.0")
        x = x * _sigmoid(x)
        x = self.layer_norm(x, "features2card.2")
        x = self.linear(x, "features2card.3")
        x = x * _sigmoid(x)
        run = self.interleaved if spec["layer_schedule"] == "interleaved" else self.sequential
        x = run(x, sub_gather, sub_gather_lens, time_shift_selects, skips).reshape(num_data, -1, self.d)

        x = self.layer_norm(x, "prehead_norm")
        hw = np.maximum(self.linear(x, "head_w.0"), 0)
        hw = self.linear(self.layer_norm(hw, "head_w.2"), "head_w.4")
        t = np.asarray(labels, dtype=self.dt)[..., 0][..., None]
        t1 = np.maximum(t, self.dt.type(c["min_elapsed_seconds"]))
        eps = self.dt.type(c["prob_clamp"])
        if spec["curve_head"] == "power_mixture":
            wl = hw @ self.w["gru_w_weight"].T + self.w["gru_w_bias"]
            s_raw = hw @ self.w["gru_s_weight"].T + self.w["gru_s_bias"]
            d_raw = hw @ self.w["gru_d_weight"].T + self.w["gru_d_bias"]
            weights = _softmax(wl)
            s = np.exp(np.clip(s_raw, -c["power_clamp"], c["power_clamp"]))
            d = np.exp(np.clip(d_raw, -c["power_clamp"], c["power_clamp"]))
            r = (weights * np.exp(-d * np.log1p(t1 / (self.dt.type(c["power_s_eps"]) + s)))).sum(axis=-1)
        else:
            weights = _softmax(self.linear(hw, "w_linear"))
            raw = np.exp(np.linspace(0, c["s_point_spread"], spec["num_curves"], dtype=np.float32)).astype(self.dt)
            s_space = c["s_min"] + (raw - 1) * self.dt.type(np.e ** (c["s_max"] - c["s_point_spread"]))
            r = (weights * self.dt.type(c["curve_base"]) ** (t1 / s_space)).sum(axis=-1)
        curve_raw = eps + (1 - 2 * eps) * r
        if spec["ahead_residual"]:
            ahead = self.linear(np.maximum(self.linear(x, "head_ahead_logits.0"), 0), "ahead_linear")
            raw = np.exp(np.linspace(0, c["point_spread"], spec["num_points"], dtype=np.float32)).astype(self.dt)
            points = c["point_min"] + (raw - 1) * self.dt.type(np.e ** (c["max_e"] - c["point_spread"]))
            tq = t1[..., 0]
            right = np.searchsorted(points, tq.ravel(), side="left").reshape(tq.shape)
            left = np.maximum(right - 1, 0)
            xl, xr = points[left], points[right]
            yl = np.take_along_axis(ahead, left[..., None], axis=-1)[..., 0]
            yr = np.take_along_axis(ahead, right[..., None], axis=-1)[..., 0]
            residual = eps + (1 - 2 * eps) * (yl + (yr - yl) * (tq - xl) / (xr - xl))
        else:
            residual = eps  # interp() of the all-zero ahead logits
        p_curve = _sigmoid(np.log(curve_raw / (1 - curve_raw)) + residual)
        p_rating = _softmax(self.linear(np.maximum(self.linear(x, "head_p.0"), 0), "p_linear"))
        p_imm = np.clip(1.0 - p_rating[..., 0], eps, 1 - eps)
        return {"p_curve": p_curve, "p_imm": p_imm, "p_rating": p_rating}
