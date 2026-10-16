"""SM18: low-overhead single-review CPU step (pure NumPy) of the recurrent Expertium / fork RWKV models.

`NumpyStep(model)` takes a loaded eager recurrent model (`expertium.variants.build(..., rnn=True)` or the fork's
`SrsRWKVRnn`) and builds a fused, allocation-light forward of ONE review (`review()` has the eager signature and
output layout: ahead logits, w, rating logits, five next states; the GRU variants also expose s and d via
`review_dict`). Fusions (all exact algebra, folded in float64, executed in float32):

  * every LayerNorm affine is folded into the Linear weights that consume it (the normalised-but-unscaled vector n
    is what the state keeps, so the token shift needs no extra op);
  * the time mixer's eight lerps + their Linears become ONE matmul on [n, n_shift - n]:
    W_i (n g + b + w_i g (n_shift - n)) = (W_i g) n + (W_i w_i g)(n_shift - n) + W_i b;
  * the four LoRA B matrices (value, a, decay, gate) are one block-diagonal matmul;
  * -0.5 - softplus(-x) then exp(-exp(.)) is exp(-exp(-0.5) * sigmoid(x)); the WKV update is
    S w + (v - S k)(a k)^T (k_deformed * a is the same product the eager code builds twice);
  * the head Linears are composed (head_w.4 into the GRU / w_linear heads) and the three head first layers share one matmul;
  * zero_features is zeroed in the input weight (no clone), stripped channel mixers are skipped.

State of one entity = list per layer of [n_time (B,C), S (B,H,K,K), n_channel (B,C) or None]; None = fresh entity.
"""
import math

import numpy as np
from scipy.special import expit

np.seterr(all="ignore")
F32 = np.float32


def _w(t):
    return t.detach().double().numpy()


def _ln_eps(module):
    return F32(module.eps)


try:
    from numpy._core._multiarray_umath import c_einsum as _einsum  # skips the python dispatcher
except ImportError:  # pragma: no cover
    _einsum = np.einsum


def _centre(c):
    """(c, c) matrix P with x @ P = x - mean(x): one op instead of reduce, scale, subtract."""
    return np.ascontiguousarray((np.eye(c) - 1.0 / c).astype(F32))


def _ln_n(x, ceps, centre):
    """LayerNorm without affine, scaled by 1/sqrt(c): (x - mean) * (sum((x - mean)^2) + c*eps)^-1/2.
    The sqrt(c) is folded into every weight that consumes it, so the state keeps this scaled vector."""
    xc = x @ centre
    return xc * ((_einsum("bc,bc->b", xc, xc) + ceps) ** -0.5)[:, None]


class _Layer:
    __slots__ = ("lid", "C", "H", "K", "eps_t", "w_t", "b_t", "mix_in", "mix_b", "n_lora", "lora_slices",
                 "has_v", "lora_w", "lora_b", "ks_start", "gn_w", "gn_b", "gn_eps", "bonus", "w_o", "cm",
                 "eps_c", "wc_in", "wc_out", "centre", "centre_c", "centre_g", "sig_end", "gn_ceps")


class NumpyStep:
    def __init__(self, model):
        self.model = model
        d = self.d = model.d_model
        self.centre = _centre(d)
        f2c = model.features2card
        w0, b0 = _w(f2c[0].weight).copy(), _w(f2c[0].bias)
        for index in getattr(model, "zero_features", ()) or ():
            w0[:, index] = 0.0
        self.f_w0 = np.ascontiguousarray(w0.T.astype(F32))
        self.f_b0 = b0.astype(F32)
        fc = f2c[2].normalized_shape[0]
        g, b = _w(f2c[2].weight) * math.sqrt(fc), _w(f2c[2].bias)
        w3 = _w(f2c[3].weight)
        self.f_ln_eps = F32(fc * f2c[2].eps)
        self.f_centre = _centre(fc)
        self.f_w3 = np.ascontiguousarray((w3 * g).T.astype(F32))
        self.f_b3 = (_w(f2c[3].bias) + w3 @ b).astype(F32)
        self.streams = getattr(model, "streams", None) or ["card_id", "deck_id", "note_id", "preset_id", "user_id"]
        mods = [m.blocks for m in model.rwkv_modules]
        self.layers = [[self._layer(i, lid, blk) for lid, blk in enumerate(blocks)] for i, blocks in enumerate(mods)]
        if getattr(model, "interleave_on", False):
            self.schedule = list(model.layer_steps)
        else:
            self.schedule = [(i, lid) for i, blocks in enumerate(mods) for lid in range(len(blocks))]
        self._heads(model)

    # ------------------------------------------------------------------ build
    def _layer(self, stream, lid, block):
        tm, cm = block.time_mixer, block.channel_mixer
        C, H, K = self.d, tm.H, tm.K
        L = _Layer()
        L.lid, L.C, L.H, L.K = lid, C, H, K
        L.centre = self.centre
        L.eps_t = F32(C * tm.layer_norm.eps)
        g, b = _w(tm.layer_norm.weight) * math.sqrt(C), _w(tm.layer_norm.bias)
        lerp = _w(tm.rkvdag_lerp).reshape(8, C)  # r, k, v, d, a, g, k_scale, v_scale
        L.has_v = lid > 0
        mats = [(_w(tm.W_r.weight), None, 0), (_w(tm.W_k.weight), None, 1), (_w(tm.W_v.weight), None, 2)]
        lora = []
        if L.has_v:
            mats.append((_w(tm.v_lora_simple.A.weight), None, 2))
        mats.append((_w(tm.a_lora_simple.A.weight), None, 4))
        mats.append((_w(tm.d_lora_mlp.A.weight), None, 3))
        mats.append((_w(tm.lora_A_g.weight), None, 5))
        mats.append((_w(tm.k_scale_linear.weight), _w(tm.k_scale_linear.bias), 6))
        mats.append((_w(tm.v_scale_linear.weight), _w(tm.v_scale_linear.bias), 7))
        w_in, w_ln, bias = [], [], []
        for m, mb, which in mats:
            w_in.append(m * g)
            w_ln.append(m * (lerp[which] * g))
            bias.append(m @ b + (mb if mb is not None else 0.0))
        L.mix_in = np.ascontiguousarray(np.concatenate([np.concatenate(w_in, 0), np.concatenate(w_ln, 0)], 1).T.astype(F32))
        L.mix_b = np.concatenate(bias).astype(F32)
        # offsets in the fused output
        off = 3 * C
        dims = {}
        names = (["v"] if L.has_v else []) + ["a", "d", "g"]
        for name in names:
            dl = {"v": tm.v_lora_simple.A, "a": tm.a_lora_simple.A, "d": tm.d_lora_mlp.A, "g": tm.lora_A_g}[name].weight.shape[0]
            dims[name] = (off, off + dl)
            off += dl
        L.ks_start = off
        # block-diagonal B matrices, outputs ordered [v, a, d, g] (v absent in layer 0), sigmoid over [v, a, d]
        Bs = {"v": (tm.v_lora_simple.B_and_lamb.weight, tm.v_lora_simple.B_and_lamb.bias) if L.has_v else None,
              "a": (tm.a_lora_simple.B_and_lamb.weight, tm.a_lora_simple.B_and_lamb.bias),
              "d": (tm.d_lora_mlp.B_and_lamb.weight, tm.d_lora_mlp.B_and_lamb.bias),
              "g": (tm.lora_B_g.weight, None)}
        total_in = sum(dims[n][1] - dims[n][0] for n in names)
        blk = np.zeros((total_in, C * len(names)))
        bb = np.zeros(C * len(names))
        pos = 0
        for j, name in enumerate(names):
            wt, bs = Bs[name]
            dl = wt.shape[1]
            blk[pos:pos + dl, j * C:(j + 1) * C] = _w(wt).T
            if bs is not None:
                bb[j * C:(j + 1) * C] = _w(bs)
            pos += dl
        L.lora_w = np.ascontiguousarray(blk.astype(F32))
        L.lora_b = bb.astype(F32)
        L.lora_slices = dims
        L.n_lora = len(names)
        L.sig_end = C * (len(names) - 1)  # sigmoid over everything but the gate
        L.gn_w = (_w(tm.out_group_norm.weight) * math.sqrt(K)).astype(F32).reshape(H, K)
        L.gn_b = _w(tm.out_group_norm.bias).astype(F32).reshape(H, K)
        L.gn_ceps = F32(K * tm.out_group_norm.eps)
        L.centre_g = np.ascontiguousarray(np.kron(np.eye(H), np.eye(K) - 1.0 / K).astype(F32))
        L.bonus = _w(tm.bonus).astype(F32).reshape(H, K)
        L.w_o = np.ascontiguousarray(_w(tm.W_o.weight).T.astype(F32))
        L.cm = not getattr(block, "cmix_stripped", False)
        if L.cm:
            L.eps_c = F32(C * cm.layer_norm.eps)
            gc, bc = _w(cm.layer_norm.weight) * math.sqrt(C), _w(cm.layer_norm.bias)
            wk, lk = _w(cm.W_k.weight), _w(cm.lerp_k).reshape(-1)
            bias_c = None
            L.wc_in = np.ascontiguousarray(np.concatenate([wk * gc, wk * (lk * gc)], 1).T.astype(F32))
            L.mix_b = L.mix_b  # time mixer bias unchanged
            L.cm = (L.wc_in, (wk @ bc).astype(F32), np.ascontiguousarray(_w(cm.W_v.weight).T.astype(F32)))
        return L

    def _heads(self, model):
        d = self.d
        g, b = _w(model.prehead_norm.weight) * math.sqrt(d), _w(model.prehead_norm.bias)
        self.h_eps = F32(d * model.prehead_norm.eps)
        gru = getattr(model, "gru_on", False)
        self.gru = gru
        w0, b0 = _w(model.head_w[0].weight), _w(model.head_w[0].bias)
        p0, pb0 = _w(model.head_p[0].weight), _w(model.head_p[0].bias)
        mats, biases = [w0, p0], [b0, pb0]
        self.no_ahead = getattr(model, "no_ahead_residual", False)
        self.n_w, self.n_p = w0.shape[0], p0.shape[0]
        if not self.no_ahead:
            a0, ab0 = _w(model.head_ahead_logits[0].weight), _w(model.head_ahead_logits[0].bias)
            mats.append(a0)
            biases.append(ab0)
            self.n_a = a0.shape[0]
        W = np.concatenate(mats, 0)
        self.h_w = np.ascontiguousarray((W * g).T.astype(F32))
        self.h_b = (W @ b + np.concatenate(biases)).astype(F32)
        # head_w: relu -> LN -> head_w.4 -> (gru w,s,d | w_linear), composed
        ln2 = model.head_w[2]
        d2 = ln2.normalized_shape[0]
        self.h_ln2_eps, self.h_ln2_centre = F32(d2 * ln2.eps), _centre(d2)
        g2, b2 = _w(ln2.weight) * math.sqrt(d2), _w(ln2.bias)
        w4, b4 = _w(model.head_w[4].weight), _w(model.head_w[4].bias)
        if gru:
            fin_w = np.concatenate([_w(model.gru_w_weight), _w(model.gru_s_weight), _w(model.gru_d_weight)], 0)
            fin_b = np.concatenate([_w(model.gru_w_bias), _w(model.gru_s_bias), _w(model.gru_d_bias)])
        else:
            fin_w, fin_b = _w(model.w_linear.weight), _w(model.w_linear.bias)
        self.n_curves = fin_w.shape[0] // (3 if gru else 1)
        comp_w = fin_w @ w4
        comp_b = fin_w @ b4 + fin_b
        self.hw2 = np.ascontiguousarray((comp_w * g2).T.astype(F32))
        self.hb2 = (comp_w @ b2 + comp_b).astype(F32)
        self.p_w = np.ascontiguousarray(_w(model.p_linear.weight).T.astype(F32))
        self.p_b = _w(model.p_linear.bias).astype(F32)
        if not self.no_ahead:
            self.a_w = np.ascontiguousarray(_w(model.ahead_linear.weight).T.astype(F32))
            self.a_b = _w(model.ahead_linear.bias).astype(F32)
        self.num_points = model.num_points
        self.zeros_ahead = {}

    # ------------------------------------------------------------------ step
    def _layer_step(self, L, x, v0, st):
        C, H, K = L.C, L.H, L.K
        B = x.shape[0]
        n = _ln_n(x, L.eps_t, L.centre)
        if st is None:
            ns = n
            S = np.zeros((B, H, K, K), F32)
        else:
            ns, S = st[0], st[1]
        big = np.concatenate((n, ns - n), 1) @ L.mix_in
        big += L.mix_b
        r = big[:, :C].reshape(B, H, K)
        v_lin = big[:, 2 * C:3 * C]
        # LoRA stage: activations in place on the A outputs, then one block-diagonal matmul
        sl = L.lora_slices
        da, db = sl["d"]
        ga, gb = sl["g"]
        np.tanh(big[:, da:db], out=big[:, da:db])
        expit(big[:, ga:gb], out=big[:, ga:gb])
        lo = big[:, 3 * C:L.ks_start] @ L.lora_w
        lo += L.lora_b
        expit(lo[:, :L.sig_end], out=lo[:, :L.sig_end])
        if L.has_v:
            vl, a, wsig, g = lo[:, :C], lo[:, C:2 * C], lo[:, 2 * C:3 * C], lo[:, 3 * C:]
            v = v_lin + vl * (v0 - v_lin)
        else:
            a, wsig, g = lo[:, :C], lo[:, C:2 * C], lo[:, 2 * C:]
            v = v_lin
            v0 = v
        w = np.exp(wsig * F32(-math.exp(-0.5))).reshape(B, H, 1, K)
        a = a.reshape(B, H, K)
        kv = np.concatenate((big[:, C:2 * C], v), 1).reshape(B, 2, H, K)
        nrm = np.maximum(np.sqrt(_einsum("bjhk,bjhk->bjh", kv, kv)), F32(1e-12))
        kv *= (expit(big[:, L.ks_start:]).reshape(B, 2, H) / nrm)[..., None]
        kd, vv = kv[:, 0], kv[:, 1]
        k2 = kd * a
        k2c = k2[:, :, None, :]
        sk = S @ kd[..., None]
        S2 = S * w + (vv[..., None] - sk) * k2c
        out = (S2 @ r[..., None])[..., 0]
        bonus = _einsum("bhk,hk,bhk->bh", r, L.bonus, k2)[..., None] * vv
        oc = (out.reshape(B, C) @ L.centre_g).reshape(B, H, K)
        oc *= (_einsum("bhk,bhk->bh", oc, oc) + L.gn_ceps)[..., None] ** -0.5
        oc *= L.gn_w
        oc += L.gn_b
        oc += bonus
        y = (oc.reshape(B, C) * g) @ L.w_o
        x = x + y
        nc = None
        if L.cm:
            wc_in, bc, wv = L.cm
            nc2 = _ln_n(x, L.eps_c, L.centre)
            pc = st[2] if st is not None else None
            if pc is None:
                pc = nc2
            k = np.concatenate((nc2, pc - nc2), 1) @ wc_in
            k += bc
            np.maximum(k, 0, out=k)
            k *= k
            x = x + k @ wv
            nc = nc2
        return x, v0, [n, S2, nc]

    def step_dict(self, feats, states):
        """feats (B,92) float32; states: list over self.streams of None | list per layer. -> (out dict, next states)."""
        f = feats @ self.f_w0
        f += self.f_b0
        x = f / (1.0 + np.exp(-f))
        x = _ln_n(x, self.f_ln_eps, self.f_centre) @ self.f_w3
        x += self.f_b3
        x = x / (1.0 + np.exp(-x))
        nxt = [list(s) if s is not None else [None] * len(self.layers[i]) for i, s in enumerate(states)]
        v0 = [None] * len(self.layers)
        layers = self.layers
        step = self._layer_step
        for i, lid in self.schedule:
            x, v0[i], nxt[i][lid] = step(layers[i][lid], x, v0[i], nxt[i][lid])
        B = x.shape[0]
        h = _ln_n(x, self.h_eps, self.centre) @ self.h_w
        h += self.h_b
        np.maximum(h, 0, out=h)
        nw, npp = self.n_w, self.n_p
        hw = _ln_n(h[:, :nw], self.h_ln2_eps, self.h_ln2_centre) @ self.hw2
        hw += self.hb2
        nc = self.n_curves
        logits = hw[:, :nc]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        out = {"w": e / e.sum(-1, keepdims=True)}
        if self.gru:
            out["s"], out["d"] = hw[:, nc:2 * nc], hw[:, 2 * nc:]
        p = h[:, nw:nw + npp] @ self.p_w
        p += self.p_b
        out["p_logits"] = p
        if self.no_ahead:
            z = self.zeros_ahead.get(B)
            if z is None:
                z = self.zeros_ahead[B] = np.zeros((B, self.num_points), F32)
            out["ahead"] = z
        else:
            a = h[:, nw + npp:] @ self.a_w
            a += self.a_b
            out["ahead"] = a
        return out, nxt

    def review(self, feats, card, note, deck, preset, user):
        """Eager `review` argument order and return layout (states in chain order card, note, deck, preset, user)."""
        states = [card, deck, note, preset, user]  # stream order card, deck, note, preset, user
        out, nxt = self.step_dict(feats, states)
        return (out["ahead"], out["w"], out["p_logits"], nxt[0], nxt[2], nxt[1], nxt[3], nxt[4])
