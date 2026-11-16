"""One-review-at-a-time NumPy reference of the w64 model (the form a runtime port implements).

Needs only numpy. Reads the weights and architecture.json of this package. It follows
the one-review step (VariantSrsRWKVRnn.step) of the training code and gives the same numbers as the
parallel reference small_model_numpy.py (checked by testvectors/check_testvectors.py).

Use:
    model = RecurrentModel(tensors, spec)           # tensors from weights_io.read_weights
    store = model.new_store()                       # per-entity states, one dict per stream
    heads = model.step(features92, ids, skip, store)   # ids = {"card_id": int, "deck_id": ..., ...}
    p_imm, p_rating = model.instant(heads)
    p = model.p_curve(heads, elapsed_seconds)

`skip` = True for a query row: the heads are computed but no state is committed (the row only asks a question).
`state_bits=8` stores the card and note streams like the fork's int8 state (rows of 16 int8 + f32 scale for the
S matrix, float16 for the two shift vectors). That option follows the fork's rslib/src/rwkv/precision.rs and the
scorer; it is NOT covered by the golden test vectors (see INTEGRATION.md, "Known gaps").
"""
import numpy as np

STATE_STREAMS = (0, 2)  # card_id, note_id (indices into spec["streams"]); only these are compacted by state_bits


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x, threshold):
    return np.where(x > threshold, x, np.log1p(np.exp(np.minimum(x, threshold))))


def _lerp(start, end, weight):
    # torch.lerp's two-branch formula
    return np.where(np.abs(weight) < 0.5, start + weight * (end - start), end - (end - start) * (1 - weight))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _quantize_rows(s, bits):
    """S [H, K, K]: each row of K values -> int8 codes with an f32 scale, returned dequantized."""
    qmax = 127.0 if bits == 8 else 7.0
    scale = np.abs(s).max(axis=-1, keepdims=True) / np.float32(qmax)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    return (np.clip(np.round(s / scale), -qmax, qmax) * scale).astype(np.float32)


class RecurrentModel:
    def __init__(self, weights, spec, state_bits=0):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.spec = spec
        self.c = spec["constants"]
        self.d = spec["d_model"]
        self.state_bits = state_bits
        self.f32 = np.float32

    # ---- primitives
    def linear(self, x, prefix):
        y = x @ self.w[prefix + ".weight"].T
        b = self.w.get(prefix + ".bias")
        return y + b if b is not None else y

    def layer_norm(self, x, prefix):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + self.f32(self.c["layer_norm_eps"]))
        return y * self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    def group_norm(self, x, prefix, groups):
        g = x.reshape(groups, -1)
        mean = g.mean(axis=-1, keepdims=True)
        var = ((g - mean) ** 2).mean(axis=-1, keepdims=True)
        y = ((g - mean) / np.sqrt(var + self.f32(self.c["group_norm_eps"]))).reshape(-1)
        return y * self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    def normalize(self, x):  # x [H, K], L2 over K
        norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
        return x / np.maximum(norm, self.f32(self.c["normalize_eps"]))

    # ---- one RWKV-7 layer for one entity
    def time_mixer(self, p, layer, heads, x, v0, state):
        """x [C]. state = None or {"shift": [C], "S": [H, K, K]}. Returns (x_out, v0, new_state)."""
        C = x.shape[0]
        H, K = heads, C // heads
        xl = self.layer_norm(x, p + ".layer_norm")
        xs = xl if state is None else state["shift"]
        S = np.zeros((H, K, K), np.float32) if state is None else state["S"]
        lerp = self.w[p + ".rkvdag_lerp"].reshape(8, C)
        r, k, v, d, a, g, ks, vs = [_lerp(xl, xs, lerp[j]) for j in range(8)]
        r = self.linear(r, p + ".W_r")
        k = self.linear(k, p + ".W_k")
        k_scale = _sigmoid(self.linear(ks, p + ".k_scale_linear"))  # [H]
        v_scale = _sigmoid(self.linear(vs, p + ".v_scale_linear"))  # [H]
        if layer == 0:
            v = self.linear(v, p + ".W_v")
            v0 = v
        else:
            v_lerp = _sigmoid(self.linear(self.linear(v, p + ".v_lora_simple.A"), p + ".v_lora_simple.B_and_lamb"))
            v = _lerp(self.linear(v, p + ".W_v"), v0, v_lerp)
        a = _sigmoid(self.linear(self.linear(a, p + ".a_lora_simple.A"), p + ".a_lora_simple.B_and_lamb"))
        g = self.linear(_sigmoid(self.linear(g, p + ".lora_A_g")), p + ".lora_B_g")
        dd = self.linear(np.tanh(self.linear(d, p + ".d_lora_mlp.A")), p + ".d_lora_mlp.B_and_lamb")
        dd = -0.5 - _softplus(-dd, self.c["softplus_threshold"])
        w = np.exp(-np.exp(dd)).reshape(H, K)

        kk = k_scale[:, None] * self.normalize(k.reshape(H, K))      # k_deformed
        vv = v_scale[:, None] * self.normalize(v.reshape(H, K))
        rr = r.reshape(H, K)
        aa = a.reshape(H, K)
        kb = kk * aa                                                  # k used for the write
        # S' = S * diag(w) - (S kk) (aa * kk)^T + vv kb^T      (S [H, K(value), K(key)])
        s_kk = np.einsum("hij,hj->hi", S, kk)
        S_new = S * w[:, None, :] - s_kk[:, :, None] * (aa * kk)[:, None, :] + vv[:, :, None] * kb[:, None, :]
        out = np.einsum("hij,hj->hi", S_new, rr).reshape(C)
        out = self.group_norm(out, p + ".out_group_norm", H)
        bonus = ((rr * self.w[p + ".bonus"].reshape(H, K) * kb).sum(axis=-1, keepdims=True) * vv).reshape(C)
        out = self.linear(g * (out + bonus), p + ".W_o")
        return x + out, v0, {"shift": xl, "S": S_new}

    def channel_mixer(self, p, x, shift):
        xl = self.layer_norm(x, p + ".layer_norm")
        xs = xl if shift is None else shift
        k = self.linear(_lerp(xl, xs, self.w[p + ".lerp_k"].reshape(-1)), p + ".W_k")
        return x + self.linear(np.square(np.maximum(k, 0)), p + ".W_v"), xl

    # ---- state store
    def new_store(self):
        return [dict() for _ in self.spec["streams"]]

    def _compact(self, stream, layer_state):
        if not self.state_bits or stream not in STATE_STREAMS:
            return layer_state
        out = dict(layer_state)
        out["S"] = _quantize_rows(layer_state["S"], self.state_bits)
        out["shift"] = layer_state["shift"].astype(np.float16).astype(np.float32)
        if out.get("cshift") is not None:
            out["cshift"] = layer_state["cshift"].astype(np.float16).astype(np.float32)
        return out

    # ---- one review
    def step(self, features, ids, skip, store):
        """features [92] float32. ids {stream name: int}. Returns the head tensors; commits states unless skip."""
        spec, c = self.spec, self.c
        f = np.array(features, dtype=np.float32)
        for index in spec["zero_features"]:
            f[index] = 0.0
        x = self.linear(f, "features2card.0")
        x = x * _sigmoid(x)
        x = self.layer_norm(x, "features2card.2")
        x = self.linear(x, "features2card.3")
        x = x * _sigmoid(x)
        streams = spec["streams"]
        v0 = [None] * len(streams)
        pending = []
        for i, layer in spec["layer_steps"]:
            s = streams[i]
            entity = ids[s["name"]]
            old = store[i].get(entity, {}).get(layer)
            p = f"rwkv_modules.{i}.blocks.{layer}"
            x, v0[i], tstate = self.time_mixer(p + ".time_mixer", layer, s["n_heads"], x, v0[i], old)
            cstate = None
            if layer not in s["channel_mixer_stripped_layers"]:
                x, cshift = self.channel_mixer(p + ".channel_mixer", x, None if old is None else old.get("cshift"))
                tstate["cshift"] = cshift
            pending.append((i, entity, layer, tstate))
        if not skip:
            for i, entity, layer, tstate in pending:
                store[i].setdefault(entity, {})[layer] = self._compact(i, tstate)

        x = self.layer_norm(x, "prehead_norm")
        hw = np.maximum(self.linear(x, "head_w.0"), 0)
        hw = self.linear(self.layer_norm(hw, "head_w.2"), "head_w.4")
        return {
            "w_logits": hw @ self.w["gru_w_weight"].T + self.w["gru_w_bias"],
            "s_raw": hw @ self.w["gru_s_weight"].T + self.w["gru_s_bias"],
            "d_raw": hw @ self.w["gru_d_weight"].T + self.w["gru_d_bias"],
            "p_logits": self.linear(np.maximum(self.linear(x, "head_p.0"), 0), "p_linear"),
        }

    # ---- outputs
    def p_curve(self, heads, elapsed_seconds):
        """Recall probability after `elapsed_seconds` (float, >= 0) since this review. Power-curve mixture."""
        c = self.c
        weights = _softmax(heads["w_logits"])
        s = np.exp(np.clip(heads["s_raw"], -c["power_clamp"], c["power_clamp"]))
        d = np.exp(np.clip(heads["d_raw"], -c["power_clamp"], c["power_clamp"]))
        t = np.float32(max(float(elapsed_seconds), c["min_elapsed_seconds"]))
        r = (weights * np.exp(-d * np.log1p(t / (np.float32(c["power_s_eps"]) + s)))).sum()
        eps = np.float32(c["prob_clamp"])
        raw = eps + (1 - 2 * eps) * r
        # no ahead residual in this model: the residual logit is the constant eps
        return float(_sigmoid(np.log(raw / (1 - raw)) + eps))

    def instant(self, heads):
        """(p_imm, p_rating[4]): p_imm = 1 - P(again), clamped to [1e-5, 1 - 1e-5]."""
        p = _softmax(heads["p_logits"])
        eps = self.c["prob_clamp"]
        return float(np.clip(1.0 - p[0], eps, 1 - eps)), p
