"""SM147: low-rank structure of the card / note WKV states (CPU reference path of scripts/small_model_heldout.py only).

Two probes on the stored float32 WKV matrix of every head of the card and note modules, applied at each write (each
non-skipped step of an entity's stream; n = the entity's update count so far, this write included, counted as the
tier codec of SM141 counts it):

* spectrum (variant `+lrsv`): float state, unchanged arithmetic; per write and head the singular values (float64
  svdvals of the float32 matrix) give the share of Frobenius energy (sum of squared singular values) kept by rank
  1 / 2 / 4 / 8. Accumulated per (model tag, stream, layer, head, answer-count bucket): writes, zero matrices, sums of
  the shares, and a histogram of the residual 1 - share on a log10 grid (quantiles). Aggregates only, no ids.
* rank-r truncation (variant `+lr1` / `+lr2` / `+lr4`): after each write the stored matrix of each head is replaced by
  its best rank-r approximation (SVD in float64, factors L = U_r sqrt(S_r), R = sqrt(S_r) V_r^T rounded to float32,
  stored = L @ R in float32); no quantization; shift vectors stay float32. The answer's own output used the float
  update (same placement as the int codecs).

Expertium models (Small, Tiny): through state_vq.Hooks (the CPU WKV reference already routes there). Fork model (Big):
its patched module (scripts/compress_quant.py) is TorchScript, so the probe needs PYTORCH_JIT=0 (eager modules); then
`reference_rwkv7_q` is wrapped and forward pre-hooks name the running time mixer. Default runs are unchanged.
"""
import math

import torch

EDGES = (1, 3, 5, 9, 16, 32, 128)        # answer-count buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+
BUCKETS = ("1-2", "3-4", "5-8", "9-15", "16-31", "32-127", "128+")
RANKS = (1, 2, 4, 8)
LOG_LO, LOG_STEP, LOG_BINS = -9.0, 0.1, 90   # residual histogram: bin 0 = residual <= 1e-9, then 0.1 decades to 1
STREAMS = {0: "card", 2: "note"}


class LowRankState:
    """Variant state of `+lr<r>` (rank r truncation) / `+lrsv` (float state, spectrum recorded). Not a tuple / int /
    str on purpose: the scorer's other state types are told apart by those types."""

    def __init__(self, rank, spectrum):
        self.rank, self.spectrum = int(rank), bool(spectrum)

    def __bool__(self):
        return True

    def __repr__(self):
        return f"LowRankState(rank={self.rank}, spectrum={self.spectrum})"


def parse(state):
    """'lr1' / 'lr2' / 'lr4' / 'lrsv' -> LowRankState, else None."""
    if state == "lrsv":
        return LowRankState(0, True)
    sm148 = parse_sm148(state)   # SM148: rank-r int8 factor codec (LowRankCodec)
    if sm148 is not None:
        return sm148
    if state.startswith("lr") and state[2:].isdigit() and int(state[2:]) >= 1:
        return LowRankState(int(state[2:]), False)
    return None


def truncate(A, rank):
    """A [R, H, K, K] float32 -> best rank-`rank` approximation per [K, K] matrix (float32 factors, float32 product).
    Non-finite matrices are left as they are (the scorer's non-finite handling sees them)."""
    if rank >= A.shape[-1]:
        return A
    out = A.clone()
    ok = torch.isfinite(A).flatten(2).all(-1)          # [R, H]
    if not ok.any():
        return out
    M = A[ok].double()                                   # [N, K, K]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    s = S[:, :rank].sqrt()
    left = (U[:, :, :rank] * s[:, None, :]).float()     # [N, K, r]
    right = (Vh[:, :rank, :] * s[:, :, None]).float()   # [N, r, K]
    out[ok] = left @ right
    return out


class Spectrum:
    """Per (tag, stream, layer) accumulators over writes: count / zero [H, B], share sums [H, B, 4],
    residual histograms [H, B, 4, LOG_BINS + 1] (int64)."""

    def __init__(self):
        self.acc = {}
        self.tag = None

    def reset(self):
        self.acc = {}

    def _cell(self, stream, layer, H):
        key = (self.tag, stream, layer)
        if key not in self.acc:
            nb, nr = len(EDGES), len(RANKS)
            self.acc[key] = {"count": torch.zeros(H, nb, dtype=torch.int64),
                             "zero": torch.zeros(H, nb, dtype=torch.int64),
                             "nonfinite": torch.zeros(H, nb, dtype=torch.int64),
                             "share_sum": torch.zeros(H, nb, nr, dtype=torch.float64),
                             "hist": torch.zeros(H, nb, nr, LOG_BINS + 1, dtype=torch.int64)}
        return self.acc[key]

    def record(self, stream, layer, A, n):
        """A [R, H, K, K] stored matrices of R writes, n [R] their update counts (>= 1)."""
        if A.shape[0] == 0:
            return
        R, H, K, _ = A.shape
        cell = self._cell(stream, layer, H)
        b = (torch.bucketize(n.to(torch.int64), torch.tensor(EDGES, dtype=torch.int64), right=True) - 1).clamp(0)
        b = b[:, None].expand(R, H)                                         # [R, H]
        h = torch.arange(H)[None, :].expand(R, H)
        finite = torch.isfinite(A).flatten(2).all(-1)
        sv = torch.zeros(R, H, K, dtype=torch.float64)
        if finite.any():
            sv[finite] = torch.linalg.svdvals(A[finite].double())
        e = sv * sv
        tot = e.sum(-1)
        valid = finite & (tot > 0)
        flat = lambda hh, bb: hh * len(EDGES) + bb  # noqa: E731
        nbk = len(EDGES)
        cell["count"].view(-1).index_add_(0, flat(h, b).reshape(-1), torch.ones(R * H, dtype=torch.int64))
        cell["nonfinite"].view(-1).index_add_(0, flat(h, b)[~finite], torch.ones(int((~finite).sum()),
                                                                               dtype=torch.int64))
        zero = finite & ~(tot > 0)
        cell["zero"].view(-1).index_add_(0, flat(h, b)[zero], torch.ones(int(zero.sum()), dtype=torch.int64))
        if not valid.any():
            return
        cum = torch.cumsum(e[valid], -1)                                    # [V, K]
        totv = tot[valid][:, None]
        idx = torch.tensor([min(r, K) - 1 for r in RANKS])
        share = (cum[:, idx] / totv).clamp(0, 1)                            # [V, 4]
        res = (1 - share).clamp(min=0)
        lg = torch.log10(res.clamp(min=1e-300))
        bins = torch.where(res <= 10 ** LOG_LO, torch.zeros_like(lg, dtype=torch.int64),
                           (1 + torch.floor((lg - LOG_LO) / LOG_STEP)).to(torch.int64).clamp(1, LOG_BINS))
        hv, bv = h[valid], b[valid]
        nr = len(RANKS)
        base = (hv * nbk + bv)[:, None] * nr + torch.arange(nr)[None, :]  # [V, 4]
        cell["share_sum"].view(-1).index_add_(0, base.reshape(-1), share.reshape(-1))
        cell["hist"].view(-1).index_add_(0, (base * (LOG_BINS + 1) + bins).reshape(-1),
                                         torch.ones(base.numel(), dtype=torch.int64))

    def export(self, tag):
        """JSON-ready aggregates of model `tag` (then dropped from the accumulators)."""
        out = []
        for key in sorted(k for k in self.acc if k[0] == tag):
            _, stream, layer = key
            c = self.acc.pop(key)
            out.append({"stream": STREAMS.get(stream, stream), "layer": layer,
                        "count": c["count"].tolist(), "zero": c["zero"].tolist(),
                        "nonfinite": c["nonfinite"].tolist(), "share_sum": c["share_sum"].tolist(),
                        "hist": _sparse(c["hist"])})
        return out


def _sparse(hist):
    """[H, B, 4, bins] -> {"h,b,r": {bin: count}} (non-zero bins only)."""
    out = {}
    H, B, NR, _ = hist.shape
    for hh in range(H):
        for bb in range(B):
            for rr in range(NR):
                row = hist[hh, bb, rr]
                nz = torch.nonzero(row).flatten().tolist()
                if nz:
                    out[f"{hh},{bb},{rr}"] = {str(i): int(row[i]) for i in nz}
    return out


def bin_edges():
    """Residual value at the upper edge of each histogram bin (bin 0: <= 1e-9)."""
    return [10 ** LOG_LO] + [10 ** (LOG_LO + LOG_STEP * i) for i in range(1, LOG_BINS + 1)]


SPECTRUM = Spectrum()


class ExpertiumQuant:
    """state_vq.Hooks.wkv callable (tiered: gets the rows' update counts)."""
    tiered = True

    def __init__(self, state, stream, layer):
        self.state, self.stream, self.layer = state, stream, layer

    def __call__(self, nxt, cur, rows, t, n):
        if self.state.spectrum:
            SPECTRUM.record(self.stream, self.layer, nxt, n)
        return truncate(nxt, self.state.rank) if self.state.rank else nxt


def configure_expertium(layout, state):
    """Hooks for a LowRankState on an expertium model installed with state_vq.install (layout from it)."""
    from scripts import state_vq
    state_vq.Hooks.wkv, state_vq.Hooks.shift = {}, {}
    for stream, info in layout.items():
        for layer in range(info["wkv_layers"]):
            state_vq.Hooks.wkv[(stream, layer)] = ExpertiumQuant(state, stream, layer)


# ------------------------------------------------------------------------------------------------ fork model

class _Fork:
    current = None      # (module index, layer) of the running fork time mixer
    state = None        # active LowRankState (None = the patched module's own path)
    original = None     # the patched module's reference_rwkv7_q


def _fork_wkv(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, bits: int):
    st, cur = _Fork.state, _Fork.current
    if isinstance(st, LowRankCodec) and cur is not None and cur[0] in STREAMS:   # SM148, any device
        import sys
        from rwkv.model.rwkv_ops import single_timestep
        mod = sys.modules["rwkv.model.rwkv_model"]
        rows_bits = st.rows_bits
        return lowrank_loop(st, r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, single_timestep,
                            lambda x: mod._quantize_rows(x, rows_bits), where=(cur[0], cur[1]))
    if st is None or isinstance(st, LowRankCodec) or cur is None or cur[0] not in STREAMS or r_BTHK.is_cuda:
        return _Fork.original(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, bits)
    if bits:
        raise RuntimeError("SM147 low-rank probe: the fork's int state bits must be 0")
    from rwkv.model.rwkv_ops import single_timestep
    out_dtype = k_BTHK.dtype   # the loop of rwkv.model.rwkv_ops.reference_rwkv7, plus the probe on written rows
    r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK = (
        x.float() for x in (r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK))
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32)
    counts = torch.cumsum((~skip_BT.bool()).long(), 1)
    for t in range(T):
        out_BTHK[:, t], next_state_BHKK = single_timestep(
            r_BTHK[:, t], k_BTHK[:, t], v_BTHK[:, t], w_BTHK[:, t], a_BTHK[:, t], k_deformed_BTHK[:, t], state_BHKK)
        rows = torch.nonzero(~skip_BT[:, t].bool()).flatten()
        if rows.numel():
            if st.spectrum:
                SPECTRUM.record(cur[0], cur[1], next_state_BHKK[rows], counts[rows, t])
            if st.rank:
                next_state_BHKK = next_state_BHKK.clone()
                next_state_BHKK[rows] = truncate(next_state_BHKK[rows], st.rank)
        state_BHKK = torch.where(skip_BT111[:, t], state_BHKK, next_state_BHKK)
    return out_BTHK.to(out_dtype)


def install_fork(model):
    """Wrap the patched module's reference_rwkv7_q and name the fork model's time mixers (needs PYTORCH_JIT=0)."""
    import sys
    if torch.jit._state._enabled:
        raise SystemExit("SM147 low-rank labels on a fork model need PYTORCH_JIT=0 (eager modules)")
    mod = sys.modules["rwkv.model.rwkv_model"]
    if mod.reference_rwkv7_q is not _fork_wkv:
        _Fork.original = mod.reference_rwkv7_q
        mod.reference_rwkv7_q = _fork_wkv
    for index, module in enumerate(model.rwkv_modules):
        for layer, block in enumerate(module.blocks):
            block.time_mixer.register_forward_pre_hook(
                lambda _m, _i, name=(index, layer): setattr(_Fork, "current", name))


def set_fork(state):
    _Fork.state = state


def selftest():
    """Truncation and spectrum arithmetic on random matrices (seeded)."""
    g = torch.Generator().manual_seed(0)
    A = torch.randn(5, 4, 16, 16, generator=g)
    for r in (1, 2, 4):
        Ar = truncate(A, r)
        assert all(int(torch.linalg.matrix_rank(Ar[i, j].double(), atol=1e-4)) == r for i in range(5) for j in range(4))
        sv = torch.linalg.svdvals(A.double())
        want = (sv[..., r:] ** 2).sum(-1).sqrt()
        got = (A.double() - Ar.double()).flatten(2).norm(dim=-1)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (r, (got - want).abs().max())
    assert torch.equal(truncate(A, 16), A)
    low = torch.randn(3, 2, 16, 1, generator=g) @ torch.randn(3, 2, 1, 16, generator=g)
    sp = Spectrum()
    sp.tag = "t"
    sp.record(0, 0, low, torch.tensor([1, 4, 200]))
    sp.record(0, 0, torch.zeros(1, 2, 16, 16), torch.tensor([2]))
    c = sp.acc[("t", 0, 0)]
    assert c["count"].sum() == 8 and c["zero"].sum() == 2
    assert torch.allclose(c["share_sum"][:, [0, 1, 6]], torch.ones(2, 3, 4, dtype=torch.float64))
    assert c["hist"][..., 0].sum() == 6 * 4   # rank-1 matrix: residual ~0 for every rank
    nan = A[:1].clone()
    nan[0, 0, 0, 0] = math.nan
    assert torch.isnan(truncate(nan, 1)[0, 0]).any() and torch.isfinite(truncate(nan, 1)[0, 1:]).all()
    print("state_lowrank selftest OK")


# ------------------------------------------------------------------------------------------------ SM148

# SM148: post-training rank-r state with int8 factors (artifacts/small-model/sm148/design.md). Variant
# `lr<spec><q>x8`: spec = kind0[.N1.kind1[.N2.kind2]]; kind = a rank r >= 1 or R (REF's int8 rows of the full matrix);
# kind_i for answer counts N_i <= n < N_{i+1} (n = the entity's update count, this write included); R only last.
# q = q8 (int8 factors, symmetric, one float16 scale per factor vector) or f (float32 factors). Shift vectors and the
# R tier use REF's (`w8+s8hx8`) int8 formats (bits filled in by the scorer from REF's IntState); proof only: `x0` =
# float32 shift vectors (as SM147's lr<r> labels), so lr<r>fx0 is SM147's truncation through the factored solve.
# SM152: q6 = int6 factor vectors (codes in [-31, 31], one float16 scale per vector, the q8 arithmetic with qmax 31);
# x4 = int4 shift vectors (codes in [-7, 7], one float16 scale per vector: REF's shift format at width 4, as `+s8hx4`).
# q8 and x8 are unchanged (bit for bit: the same expressions with qmax 127 and REF's shift bits).
_SM148 = None
FACTOR_QMAX = {"q8": 127, "q6": 31, "f": 0}


class LowRankCodec:
    def __init__(self, spec, quant):
        parts = spec.split(".")
        if len(parts) % 2 == 0:
            raise ValueError(f"SM148 spec {spec!r}: kind[.N.kind]...")
        kinds = [k if k == "R" else int(k) for k in parts[0::2]]
        edges = [int(x) for x in parts[1::2]]
        if any(k != "R" and k < 1 for k in kinds) or "R" in kinds[:-1]:
            raise ValueError(f"SM148 spec {spec!r}: ranks >= 1, R only as the last tier")
        if any(e < 2 for e in edges) or edges != sorted(set(edges)):
            raise ValueError(f"SM148 spec {spec!r}: thresholds increasing, >= 2")
        if quant not in FACTOR_QMAX:
            raise ValueError(f"SM148 factor format {quant!r}: q8, q6 or f")
        self.spec, self.quant = spec, quant
        self.qmax = FACTOR_QMAX[quant]
        self.shift_width = 8   # SM152: 8 = REF's int8 shift vectors, 4 = int4 (same flags), 0 = float (float_shift)
        self.kinds, self.edges = kinds, edges
        self.rmax = max([k for k in kinds if k != "R"], default=0)
        self.has_rows = "R" in kinds
        self.rows_bits = self.shift_bits = None
        self.float_shift = False
        self.weight = None     # SM157: output-aware truncation, "r" / "rk" / "g" (label suffix w<kind>), None = plain
        self.grow = None       # SM159: energy-triggered rank growth, tau (label suffix g<max>e<tau x 1e4>), None = off
        self.grow_code = None  # SM159: the label's integer tau code (tau = code / 1e4)
        self._dev = {}

    def __bool__(self):
        return True

    def __repr__(self):
        return (f"LowRankCodec({self.spec!r}, {self.quant!r})" + (f" w{self.weight}" if self.weight else "")
                + (f" g{self.rmax}e{self.grow_code}" if self.grow is not None else ""))

    def set_ref(self, rows_bits, shift_bits):
        """REF's formats (the scorer's parse_variant): the R tier's rows bits; the shift vectors in REF's format, at
        int4 (codes in [-7, 7], REF's flags: float16 per-vector scale) when shift_width = 4 (SM152)."""
        self.rows_bits = rows_bits
        self.shift_bits = shift_bits if self.shift_width != 4 else (4 | (int(shift_bits) & ~255))

    def fields(self):
        out = {"spec": self.spec, "quant": self.quant, "kinds": [str(k) for k in self.kinds], "edges": self.edges,
               "shift_bits": 0 if self.float_shift else self.shift_bits}
        if self.shift_width == 4:   # SM152 (q8 / x8 rows keep SM148's descriptor)
            out["shift_width"] = 4
        if self.weight:   # SM157 (plain labels keep SM148's descriptor)
            out["weight"] = self.weight
            out["weight_stats_sha256"] = _W.sha256
            out["weight_ridge"] = RIDGE
            if self.weight == "g":
                out["weight_decay_max"] = DECAY_MAX
        if self.grow is not None:   # SM159 (other labels keep their descriptor)
            out["grow_base"] = self.kinds[0]
            out["grow_max"] = self.rmax
            out["grow_tau"] = self.grow
        return out

    def _tables(self, device):
        if device not in self._dev:
            self._dev[device] = (torch.tensor(self.edges, dtype=torch.int64, device=device),
                                 torch.tensor([-1 if k == "R" else k for k in self.kinds], dtype=torch.int64,
                                              device=device))
        return self._dev[device]

    def kind_of(self, n):
        """[N] int64 answer counts -> [N] rank of the row's tier (-1 = int8 rows)."""
        edges, kinds = self._tables(n.device)
        tier = torch.bucketize(n, edges, right=True) if len(self.edges) else torch.zeros_like(n)
        return kinds[tier]

    def encode(self, nxt, left_prev, v, n, rows_quant, wts=None):
        """nxt [N, H, K, K] float32 updated states (written rows), left_prev [N, H, K, rmax] their stored left factors
        (zero columns beyond the stored rank), v [N, H, K] this write's v, n [N] update counts.
        -> stored states [N, H, K, K] float32 and left factors [N, H, K, rmax] float32.
        SM157 wts = (M^(1/2), M^(-1/2)) [H, K, K] float32: the rank-r fit of S M^(1/2) (same column space as S, so the
        factored solve's basis span(P) still holds), right factors quantized in the weighted space, then x M^(-1/2).
        wts None = SM148 / SM152 (the expressions below unchanged)."""
        kind = self.kind_of(n)
        stored = nxt
        left_new = torch.zeros_like(left_prev)
        if self.rmax and nxt.is_cuda and FUSED_CUDA and wts is None:   # one fused kernel (scripts/state_lowrank_cuda.encode)
            from scripts import state_lowrank_cuda
            r_eff = torch.minimum(n, kind.clamp(min=0))
            stored, left_new = state_lowrank_cuda.encode(nxt, left_prev, v, r_eff, self.qmax)
        elif self.rmax:
            r_eff = torch.minimum(n, kind.clamp(min=0))                      # exact rank n for young states
            P = torch.cat([left_prev, v.unsqueeze(-1)], -1)
            finite = torch.isfinite(nxt).flatten(2).all(-1) & torch.isfinite(P).flatten(2).all(-1)   # [N, H]
            fin4 = finite[..., None, None]
            # a non-finite matrix is stored as it is (the scorer's non-finite handling sees it), as in SM147
            if wts is None:
                lf, rf = lowrank_factors(torch.where(fin4, nxt, torch.zeros_like(nxt)),
                                         torch.where(fin4, P, torch.zeros_like(P)), self.rmax)
            else:   # SM157: weighted fit (S M^(1/2)); rf in the weighted space
                lf, rf = lowrank_factors(torch.where(fin4, nxt, torch.zeros_like(nxt)) @ wts[0],
                                         torch.where(fin4, P, torch.zeros_like(P)), self.rmax)
            comp = (torch.arange(self.rmax, device=nxt.device)[None, :] < r_eff[:, None]).to(lf.dtype)   # [N, r]
            lf = lf * comp[:, None, None, :]
            rf = rf * comp[:, None, :, None]
            if self.qmax:
                lf = quant_vectors(lf, -2, self.qmax)                         # per column (left factor vector)
                rf = quant_vectors(rf, -1, self.qmax)                         # per row (right factor vector)
            if wts is not None:   # SM157: decode the right factors, x M^(-1/2)
                rf = rf @ wts[1]
            stored = torch.where(fin4, lf @ rf, nxt)
            left_new = torch.where(fin4, lf, torch.zeros_like(lf))
        if self.has_rows:
            rowsel = (kind < 0)
            if self.rmax == 0:
                return rows_quant(nxt), left_new
            if bool(rowsel.any()):
                stored = stored.clone()
                stored[rowsel] = rows_quant(nxt[rowsel])
                left_new[rowsel] = 0
        return stored, left_new


    def encode_energy(self, nxt, left_prev, v, n, cur):
        """SM159 energy-triggered rank growth (artifacts/small-model/sm159/design.md). nxt, left_prev, v, n as encode;
        cur [N, H] int64 the rank cap of each written (entity, head) before this write. Per head: the share of the
        Frobenius energy of the exact update `nxt` that a truncation at the cap would discard, 1 - (sum of the cap's
        largest squared singular values) / (sum of all), from the factored solve's eigenvalues; when it exceeds tau and
        the cap is below rmax, the cap grows by one (never shrinks). The head is then stored at rank min(n, cap) with
        the codec's factor format, as encode. -> stored, left factors, the new caps (non-finite heads: cap unchanged)."""
        P = torch.cat([left_prev, v.unsqueeze(-1)], -1)
        finite = torch.isfinite(nxt).flatten(2).all(-1) & torch.isfinite(P).flatten(2).all(-1)   # [N, H]
        fin4 = finite[..., None, None]
        lf, rf, s2 = lowrank_factors(torch.where(fin4, nxt, torch.zeros_like(nxt)),
                                     torch.where(fin4, P, torch.zeros_like(P)), self.rmax, with_s2=True)
        share = discarded_share(s2, cur)
        cur_new = torch.where(finite & (share > self.grow) & (cur < self.rmax), cur + 1, cur)
        r_eff = torch.minimum(n[:, None], cur_new)                                  # [N, H]
        comp = (torch.arange(self.rmax, device=nxt.device)[None, None, :] < r_eff[..., None]).to(lf.dtype)
        lf = lf * comp[:, :, None, :]
        rf = rf * comp[..., None]
        if self.qmax:
            lf = quant_vectors(lf, -2, self.qmax)
            rf = quant_vectors(rf, -1, self.qmax)
        stored = torch.where(fin4, lf @ rf, nxt)
        left_new = torch.where(fin4, lf, torch.zeros_like(lf))
        return stored, left_new, cur_new


def discarded_share(s2, cap):
    """SM159: s2 [N, H, m] eigenvalues of B B^T (ascending, the squared singular values of the update), cap [N, H] ->
    float32 [N, H] share of the energy outside the cap's largest components (0 for a zero matrix)."""
    sd = s2.flip(-1).clamp(min=0).float()
    total = sd.sum(-1)
    kept = sd.cumsum(-1).gather(-1, (cap.clamp(1, sd.shape[-1]) - 1).unsqueeze(-1)).squeeze(-1)
    return torch.where(total > 0, (1 - kept / torch.where(total > 0, total, torch.ones_like(total))).clamp(min=0),
                       torch.zeros_like(total))


def lowrank_factors(S, P, r, with_s2=False):
    """Best rank-r approximation of S [N, H, K, K] (float32) whose column space lies in span(P) (P [N, H, K, m],
    m >= r): orthonormal basis Q of span(P) (eigh of the column-normalised Gram, float64; directions with a Gram
    eigenvalue below 1e-12 x the largest dropped), B = Q^T S, eigh of B B^T (float64) -> left = Q U_r sqrt(sigma),
    right = sqrt(sigma) V_r^T as float32 ([N, H, K, r], [N, H, r, K]), components by decreasing singular value.
    Exact (up to rounding) because S = Q Q^T S; it equals the SVD truncation of S."""
    Sd, Pd = S.to(SOLVE_DTYPE), P.to(SOLVE_DTYPE)
    norms = Pd.norm(dim=-2, keepdim=True)
    Pn = Pd / torch.where(norms > 0, norms, torch.ones_like(norms))
    lam, W = _eigh(Pn.mT @ Pn)                                                # ascending
    top = lam[..., -1:].clamp(min=0)
    keep = (lam > top * KEEP_TOL[SOLVE_DTYPE]) & (lam > 0)
    inv = torch.where(keep, lam.clamp(min=1e-300).rsqrt(), torch.zeros_like(lam))
    Q = Pn @ (W * inv.unsqueeze(-2))                                           # [N, H, K, m] (zero columns dropped)
    B = Q.mT @ Sd                                                              # [N, H, m, K]
    s2, U = _eigh(B @ B.mT)
    U = U.flip(-1)[..., :r]
    left = Q @ U                                                               # unit columns
    right = U.mT @ B                                                           # rows sigma_i v_i^T
    sigma = right.norm(dim=-1)                                                 # [N, H, r]
    g = sigma.sqrt()
    lf = left * g.unsqueeze(-2)
    rf = right / torch.where(g > 0, g, torch.ones_like(g)).unsqueeze(-1)
    if with_s2:   # SM159: the squared singular values of S (ascending), for the energy trigger
        return lf.float(), rf.float(), s2
    return lf.float(), rf.float()


SOLVE_DTYPE = torch.float32   # the factored solve (lowrank_factors), both devices (design.md: float64 is 2.4x slower on the GPU)
JACOBI_CUDA = True
FUSED_CUDA = True    # CUDA: the fused encode kernel; False = the torch path above on the device (checks)
KEEP_TOL = {torch.float32: 1e-5, torch.float64: 1e-12}   # Gram eigenvalues kept: > tol x the largest (noise floor)
EIGH_CHUNK = 16384   # cusolver's batched syev refuses batches of about 32k matrices and more (measured 11-23)


def _eigh(A):
    """CPU: torch.linalg.eigh (LAPACK). CUDA float32 with m <= 8: the cyclic Jacobi kernel (scripts/state_lowrank_cuda,
    no host sync; JACOBI_CUDA = False for cusolver); else torch.linalg.eigh in chunks of EIGH_CHUNK matrices."""
    if not A.is_cuda:
        return torch.linalg.eigh(A)
    if JACOBI_CUDA and A.dtype == torch.float32 and A.shape[-1] <= 8:
        from scripts import state_lowrank_cuda
        return state_lowrank_cuda.eigh(A)
    shape = A.shape
    flat = A.reshape(-1, shape[-2], shape[-1])
    if flat.shape[0] <= EIGH_CHUNK:
        return torch.linalg.eigh(A)
    parts = [torch.linalg.eigh(flat[i:i + EIGH_CHUNK]) for i in range(0, flat.shape[0], EIGH_CHUNK)]
    lam = torch.cat([p[0] for p in parts]).reshape(*shape[:-1])
    vec = torch.cat([p[1] for p in parts]).reshape(shape)
    return lam, vec


def quant_vectors(x, dim, qmax=127):
    """Symmetric int per vector along `dim` with a float16 scale (max |x| / qmax, as REF's row scales): dequantised.
    qmax 127 = int8 (SM148), 31 = int6 (SM152)."""
    from scripts.small_model_wkv import round_scale_f16
    q = float(qmax)
    scale = round_scale_f16(x.abs().amax(dim, keepdim=True) / q)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(x / scale), -q, q) * scale


def lowrank_loop(codec, r, k, v, w, a, kd, skip, step, rows_quant, where=None):
    """The scorer's WKV step loop (expertium state_vq._reference_wkv_q / the fork's reference_rwkv7_q placement: the
    answer's output uses the float update, the stored state is the codec's reconstruction), any device."""
    # Inputs are cast to float32 per step and the output is written in the input dtype (no full-size float copies:
    # the largest users' [B, T, H, K] tensors are several GB on the GPU); the values equal x.float() / out.to(dtype).
    B, T, H, K = r.shape
    dev = r.device
    out = torch.empty(B, T, H, K, dtype=k.dtype, device=dev)
    state = torch.zeros(B, H, K, K, dtype=torch.float32, device=dev)
    left = torch.zeros(B, H, K, codec.rmax, dtype=torch.float32, device=dev)
    skip = skip.bool()
    counts = torch.cumsum((~skip).long(), 1)
    wts = weight_mats(SPECTRUM.tag, where, codec.weight, dev) if codec.weight else None   # SM157
    collect = _W.stats is not None   # SM157 fit (sm157_fit.py): key-side statistics of the written rows
    if collect and where is None:
        raise RuntimeError("SM157 statistics: the WKV call has no (stream, layer)")
    grow = codec.grow is not None   # SM159: per (entity, head) rank caps, from the base rank
    cap = torch.full((B, H), codec.kinds[0] if grow else 0, dtype=torch.int64, device=dev)
    for t in range(T):
        vt = v[:, t].float()
        out[:, t], nxt = step(r[:, t].float(), k[:, t].float(), vt, w[:, t].float(), a[:, t].float(),
                              kd[:, t].float(), state)
        rows = torch.nonzero(~skip[:, t]).flatten()
        if rows.numel():
            if collect:
                record_stats(where, r[rows, t], kd[rows, t], w[rows, t])
            if grow:
                s_new, l_new, c_new = codec.encode_energy(nxt[rows], left[rows], vt[rows], counts[rows, t], cap[rows])
                cap[rows] = c_new
            else:
                s_new, l_new = codec.encode(nxt[rows], left[rows], vt[rows], counts[rows, t], rows_quant, wts)
            state = state.clone()
            state[rows] = s_new
            left[rows] = l_new
    if RANK_READOUT.on and where is not None and codec.rmax and not codec.has_rows and T:   # SM159 byte read-out
        n_end = counts[:, -1]
        r_end = (torch.minimum(n_end[:, None], cap) if grow
                 else torch.minimum(n_end, codec.kind_of(n_end))[:, None].expand(B, H))
        RANK_READOUT.record(where, n_end, r_end, K)
    return out


class ExpertiumLoop:
    """state_vq.Hooks.wkv entry for an SM148 codec: state_vq._reference_wkv_q hands the whole loop to `.loop`.
    SM157: `where` = (stream, layer) of the CPU hook; None (GPU scoring slots) = _W.current (name_layers)."""

    def __init__(self, codec, where=None):
        self.codec = codec
        self.where = where
        from scripts.state_vq import int_row_quant
        self.rows_quant = int_row_quant(codec.rows_bits) if codec.has_rows else None

    def loop(self, r, k, v, w, a, kd, skip, state0=None):
        if state0 is not None:
            raise RuntimeError("SM148 low-rank codec: no learned initial state")
        from expertium.model.rwkv_ops import single_timestep
        return lowrank_loop(self.codec, r, k, v, w, a, kd, skip, single_timestep, self.rows_quant,
                            where=self.where if self.where is not None else _W.current)

    def __call__(self, *a, **kw):
        raise RuntimeError("SM148: ExpertiumLoop runs through .loop")


class LowRankScoreCodec(ExpertiumLoop):
    """state_vq.SCORE entry (GPU / CUDA eager path): WKV through `.loop` (state_vq.score_wkv), shift as REF."""
    score_kernel = None

    def __init__(self, codec):
        super().__init__(codec)
        from scripts.state_vq import int_row_quant
        self._shift = int_row_quant(codec.shift_bits) if codec.shift_bits and not codec.float_shift else None

    def shift(self, x, role):
        return self._shift(x) if self._shift is not None else x


def configure_expertium_sm148(layout, codec):
    """CPU Hooks: REF's shift codec (state_vq.configure), then the WKV hooks replaced by the SM148 loop."""
    from scripts import state_vq
    state_vq.configure(layout, None, codec.rows_bits, 0 if codec.float_shift else codec.shift_bits)
    for stream, info in layout.items():
        for layer in range(info["wkv_layers"]):
            state_vq.Hooks.wkv[(stream, layer)] = ExpertiumLoop(codec, (stream, layer))


def parse_sm148(state):
    import re
    m = re.fullmatch(r"lr([0-9R][0-9R.]*?)(q8|q6|f)x(8|4|0)(?:g([2-8])e([0-9]{1,5}))?(w(?:rk|r|g))?", state)
    if not m:
        return None
    codec = LowRankCodec(m.group(1), m.group(2))
    codec.float_shift = m.group(3) == "0"   # proof only: float32 shift vectors (SM147's lr<r> placement)
    codec.shift_width = int(m.group(3))      # SM152: x4 = int4 shift vectors
    if m.group(6):                           # SM157: output-aware truncation (WEIGHT_KINDS)
        if codec.rmax == 0:
            raise ValueError(f"SM157 weighted truncation {state!r}: needs a rank tier")
        codec.weight = m.group(6)[1:]
    if m.group(4):                           # SM159: energy-triggered rank growth g<max>e<tau x 1e4>
        if len(codec.kinds) != 1 or codec.kinds[0] == "R" or codec.weight:
            raise ValueError(f"SM159 rank growth {state!r}: one base rank, no tiers, no R, no weights")
        if not codec.kinds[0] < int(m.group(4)):
            raise ValueError(f"SM159 rank growth {state!r}: the maximum rank must exceed the base rank")
        codec.rmax = int(m.group(4))
        codec.grow_code = int(m.group(5))
        codec.grow = codec.grow_code / 1e4
    return codec


def selftest_sm148(device="cpu"):
    """SM148 identities on synthetic states (seeded): factored truncation == float64 SVD truncation; a state with n <= r
    answers is kept exactly (float factors); int8 factor round trip; tier selection."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for K in (16, 32):
        N, H, r = 64, 4, 4
        L = torch.randn(N, H, K, r, generator=g)
        R = torch.randn(N, H, r, K, generator=g)
        G = torch.randn(N, H, K, K, generator=g) * 0.3
        vv = torch.randn(N, H, K, generator=g)
        kk = torch.randn(N, H, K, generator=g)
        S = (L @ R @ G + vv.unsqueeze(-1) @ kk.unsqueeze(-2)).to(device)
        P = torch.cat([L.to(device), vv.to(device).unsqueeze(-1)], -1)
        worst = 0.0
        for rr in (1, 2, 4):
            lf, rf = lowrank_factors(S, P, rr)
            U, Sv, Vh = torch.linalg.svd(S.double().cpu())
            ref = (U[..., :rr] * Sv[..., None, :rr]) @ Vh[..., :rr, :]
            err = ((lf @ rf).double().cpu() - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(dim=-1)
            worst = max(worst, float(err.max()))
            assert torch.allclose(Sv[..., :rr], rf.double().cpu().norm(dim=-1) * lf.double().cpu().norm(dim=-2),
                                  rtol=1e-5)
        assert worst < 5e-5, worst
        out[f"K{K}_trunc_rel_err_max"] = worst
        # young states: rank n <= r exact (float factors), through the loop's codec
        codec = LowRankCodec("4", "f")
        lp = torch.zeros(N, H, K, 4, device=device)
        state = torch.zeros(N, H, K, K, device=device)
        ex = 0.0
        for n in range(1, 5):
            v1 = torch.randn(N, H, K, generator=g).to(device)
            k1 = torch.randn(N, H, K, generator=g).to(device)
            Gm = (torch.eye(K) * 0.9 + 0.05 * torch.randn(N, H, K, K, generator=g)).to(device)
            nxt = state @ Gm + v1.unsqueeze(-1) @ k1.unsqueeze(-2)
            state, lp = codec.encode(nxt, lp, v1, torch.full((N,), n, dtype=torch.int64, device=device), None)
            ex = max(ex, float(((state - nxt).flatten(2).norm(dim=-1) / nxt.flatten(2).norm(dim=-1)).max()))
            assert int((lp.abs().amax(-2) > 0).sum(-1).max()) <= n
        assert ex < 5e-5, ex
        out[f"K{K}_young_exact_rel_err_max"] = ex
    # int8 factor round trip: codes in [-127, 127], scale float16, max error <= scale / 2
    x = torch.randn(32, 4, 16, 4, generator=g).to(device)
    from scripts.small_model_wkv import round_scale_f16
    q = quant_vectors(x, -2)
    s = round_scale_f16(x.abs().amax(-2, keepdim=True) / 127.0)
    codes = q / s
    assert torch.equal(codes, torch.round(codes)) and float(codes.abs().max()) <= 127
    assert float(((q - x).abs() / s).max()) <= 0.5 + 1e-6
    assert torch.equal(quant_vectors(-x, -2), -q)
    assert torch.equal(quant_vectors(q, -2), q)   # idempotent (re-encoding a decoded factor is exact)
    out["int8_roundtrip"] = True
    # SM152: int6 factor round trip (codes in [-31, 31], max error <= scale / 2); qmax 127 is the int8 path
    q6 = quant_vectors(x, -2, 31)
    s6 = round_scale_f16(x.abs().amax(-2, keepdim=True) / 31.0)
    c6 = q6 / s6
    assert torch.equal(c6, torch.round(c6)) and float(c6.abs().max()) <= 31
    assert float(((q6 - x).abs() / s6).max()) <= 0.5 + 1e-6
    assert torch.equal(quant_vectors(q6, -2, 31), q6) and torch.equal(quant_vectors(x, -2, 127), q)
    assert not torch.equal(q6, q)
    out["int6_roundtrip"] = True
    # tier selection
    c = LowRankCodec("2.16.4.128.R", "q8")
    assert c.kind_of(torch.tensor([1, 15, 16, 127, 128, 900])).tolist() == [2, 2, 4, 4, -1, -1]
    assert c.rmax == 4 and c.has_rows and parse_sm148("lr2.16.4.128.Rq8x8").spec == "2.16.4.128.R"
    assert parse_sm148("lrRx8") is None and parse_sm148("lrRq8x8").kinds == ["R"]
    c6 = parse_sm148("lr2.16.4q6x4")   # SM152 grammar
    assert c6.qmax == 31 and c6.shift_width == 4 and not c6.float_shift and parse_sm148("lr2.16.4q8x8").qmax == 127
    c6.set_ref(264, 264)
    assert c6.shift_bits == 260 and c6.rows_bits == 264 and c6.fields()["shift_width"] == 4
    c8 = parse_sm148("lr2.16.4q8x8")
    c8.set_ref(264, 264)
    assert c8.shift_bits == 264 and "shift_width" not in c8.fields()
    assert parse_sm148("lr2q6x6") is None and parse_sm148("lr2q4x8") is None
    out["sm152_grammar"] = True
    out["tiers"] = True
    print("state_lowrank SM148 selftest OK", out)
    return out


# ------------------------------------------------------------------------------------------------ SM157

# SM157: output-aware (weighted) truncation (artifacts/small-model/sm157/design.md). Label suffix w<kind> on an SM148
# codec label, e.g. lr2q8x8wr, lr2.16.4q8x8wg. The rank-r fit minimises ||(S - S_hat) M^(1/2)||_F (the expected read
# error E||(S - S_hat) x||^2 for x with second moment M) instead of ||S - S_hat||_F: truncate S M^(1/2), multiply the
# right factors back by M^(-1/2). M per (model, stream, layer, head), K x K, from key-side statistics of the written rows
# on training users (sm157_fit.py -> the JSON named by SM157_STATS):
#   r   M = R = E[r r^T]                                   (the read S r)
#   rk  M = R + beta E[kd kd^T], beta = tr R / tr E[kd kd^T] (the removal term S kd as well, equal trace)
#   g   M = R + D M D, D = diag(E[w]) (the head's mean per-channel decay, clamped to DECAY_MAX): the discounted
#       observability Gramian, M_ij = R_ij / (1 - d_i d_j) (the Stein equation with a diagonal D, solved elementwise)
# Then M / (tr M / K) + RIDGE x I, rounded to float16 (as shipped), M^(1/2) and M^(-1/2) from its eigh in float64.
WEIGHT_KINDS = ("r", "rk", "g")
RIDGE = 1e-3
DECAY_MAX = 1 - 1 / 64   # g: mean decays clamped (a channel with d -> 1 would get an unbounded weight 1 / (1 - d^2))


class _W:
    stats = None      # sm157_fit.py: {(tag, stream, layer): accumulators} while collecting, else None
    file = None       # the loaded SM157_STATS document
    sha256 = None
    mats = {}         # (tag, stream, layer, kind, device) -> (M^(1/2), M^(-1/2)) float32 [H, K, K]
    current = None    # (stream, layer) of the running expertium time mixer on the GPU scoring path (name_layers)


def record_stats(where, r, kd, w):
    """Accumulate the written rows' r r^T, kd kd^T (float64 [H, K, K]) and w, log w (float64 [H, K]) of (tag, where)."""
    key = (SPECTRUM.tag, int(where[0]), int(where[1]))
    r, kd, w = r.double().cpu(), kd.double().cpu(), w.double().cpu()
    H, K = r.shape[1], r.shape[2]
    c = _W.stats.get(key)
    if c is None:
        c = _W.stats[key] = {"count": 0, "rr": torch.zeros(H, K, K, dtype=torch.float64),
                             "kk": torch.zeros(H, K, K, dtype=torch.float64), "w": torch.zeros(H, K, dtype=torch.float64),
                             "logw": torch.zeros(H, K, dtype=torch.float64)}
    c["count"] += int(r.shape[0])
    c["rr"] += torch.einsum("nhi,nhj->hij", r, r)
    c["kk"] += torch.einsum("nhi,nhj->hij", kd, kd)
    c["w"] += w.sum(0)
    c["logw"] += torch.log(w.clamp(min=1e-30)).sum(0)


def export_stats():
    """{tag: {"stream,layer": {count, rr, kk, w, logw}}} (JSON-ready sums, float64 as Python floats)."""
    out = {}
    for (tag, stream, layer), c in sorted(_W.stats.items()):
        out.setdefault(tag, {})[f"{stream},{layer}"] = {"count": c["count"], "rr": c["rr"].tolist(),
                                                        "kk": c["kk"].tolist(), "w": c["w"].tolist(),
                                                        "logw": c["logw"].tolist()}
    return out


def gramian(R, d):
    """Solution of M = R + D M D for D = diag(d) ([..., K, K], [..., K], 0 <= d < 1): M_ij = R_ij / (1 - d_i d_j)."""
    return R / (1 - d.unsqueeze(-1) * d.unsqueeze(-2))


def weight_matrix(stat, kind):
    """float64 [H, K, K] shipped M (normalised, ridge, float16-rounded) from one (stream, layer) statistics entry."""
    n = float(stat["count"])
    R = torch.tensor(stat["rr"], dtype=torch.float64) / n
    if kind == "r":
        M = R
    elif kind == "rk":
        Kd = torch.tensor(stat["kk"], dtype=torch.float64) / n
        tr = lambda x: x.diagonal(dim1=-2, dim2=-1).sum(-1)   # noqa: E731
        M = R + (tr(R) / tr(Kd))[:, None, None] * Kd
    elif kind == "g":
        d = (torch.tensor(stat["w"], dtype=torch.float64) / n).clamp(0, DECAY_MAX)
        M = gramian(R, d)
    else:
        raise ValueError(f"SM157 weight kind {kind!r}: one of {WEIGHT_KINDS}")
    M = (M + M.mT) / 2
    K = M.shape[-1]
    M = M / (M.diagonal(dim1=-2, dim2=-1).sum(-1) / K)[:, None, None] + RIDGE * torch.eye(K, dtype=torch.float64)
    return M.half().double()


def sqrt_pair(M):
    """(M^(1/2), M^(-1/2)) float32 of symmetric positive definite M (float64 eigh)."""
    lam, V = torch.linalg.eigh((M + M.mT) / 2)
    if float(lam.min()) <= 0:
        raise ValueError(f"SM157: M not positive definite (min eigenvalue {float(lam.min())})")
    s = lam.sqrt()
    return ((V * s.unsqueeze(-2)) @ V.mT).float(), ((V / s.unsqueeze(-2)) @ V.mT).float()


def load_stats(path=None):
    import hashlib
    import json
    import os
    path = path or os.environ.get("SM157_STATS")
    if not path:
        raise SystemExit("SM157 weighted labels (w<kind>) need SM157_STATS = the statistics JSON of sm157_fit.py")
    raw = open(path, "rb").read()
    _W.file, _W.sha256, _W.mats = json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), {}
    return _W.file


def weight_mats(tag, where, kind, device):
    """(M^(1/2), M^(-1/2)) on `device` for model `tag`, (stream, layer) `where`, weight kind (cached)."""
    if where is None:
        raise RuntimeError("SM157 weighted truncation: the WKV call has no (stream, layer)")
    key = (tag, int(where[0]), int(where[1]), kind, str(device))
    if key not in _W.mats:
        if _W.file is None:
            load_stats()
        try:
            stat = _W.file["models"][tag][f"{int(where[0])},{int(where[1])}"]
        except KeyError:
            raise SystemExit(f"SM157: no statistics for model {tag!r}, stream / layer {tuple(where)}") from None
        _W.mats[key] = tuple(x.to(device) for x in sqrt_pair(weight_matrix(stat, kind)))
    return _W.mats[key]


def name_layers(layout):
    """Expertium GPU scoring slots (state_vq.install_scoring layout): pre-hooks name the running time mixer in
    _W.current, once per module."""
    for stream, info in layout.items():
        for layer, module in enumerate(info["time"]):
            if not getattr(module, "_sm157_named", False):
                module.register_forward_pre_hook(lambda _m, _i, name=(stream, layer): setattr(_W, "current", name))
                module._sm157_named = True


def selftest_sm157(device="cpu"):
    """SM157 identities (seeded): the weighted factored fit == the weighted optimum (float64 SVD of S M^(1/2), times
    M^(-1/2)); its weighted error <= plain truncation's, its plain error >= plain truncation's; young states exact with
    weights; M = I gives the plain fit; the Gramian solves M = R + D M D; grammar and descriptors."""
    g = torch.Generator().manual_seed(157)
    out = {}
    for K in (16, 32):
        N, H, r = 48, 4, 4
        L = torch.randn(N, H, K, r, generator=g)
        R = torch.randn(N, H, r, K, generator=g)
        G = torch.randn(N, H, K, K, generator=g) * 0.3
        vv = torch.randn(N, H, K, generator=g)
        kk = torch.randn(N, H, K, generator=g)
        S = L @ R @ G + vv.unsqueeze(-1) @ kk.unsqueeze(-2)
        P = torch.cat([L, vv.unsqueeze(-1)], -1)
        X = torch.randn(H, 4 * K, K, generator=g) * torch.linspace(0.05, 2.0, K)
        Mw = (X.mT @ X).double() / (4 * K)
        Mw = Mw / (Mw.diagonal(dim1=-2, dim2=-1).sum(-1) / K)[:, None, None] + RIDGE * torch.eye(K, dtype=torch.float64)
        Mh, Mih = sqrt_pair(Mw)
        worst = 0.0
        for rr in (1, 2, 4):
            lf, rf = lowrank_factors(S.to(device) @ Mh.to(device), P.to(device), rr)
            got = (lf @ (rf @ Mih.to(device))).double().cpu()
            U, Sv, Vh = torch.linalg.svd(S.double() @ Mh.double())
            ref = ((U[..., :rr] * Sv[..., None, :rr]) @ Vh[..., :rr, :]) @ Mih.double()
            err = (got - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(dim=-1)
            worst = max(worst, float(err.max()))
            U0, S0, V0 = torch.linalg.svd(S.double())
            plain = (U0[..., :rr] * S0[..., None, :rr]) @ V0[..., :rr, :]
            werr = lambda A: ((S.double() - A) @ Mh.double()).flatten(2).norm(dim=-1)   # noqa: E731
            ferr = lambda A: (S.double() - A).flatten(2).norm(dim=-1)                   # noqa: E731
            assert bool((werr(ref) <= werr(plain) * (1 + 1e-6) + 1e-9).all()), rr
            assert bool((ferr(ref) >= ferr(plain) * (1 - 1e-6) - 1e-9).all()), rr
        assert worst < 1e-3, worst
        out[f"K{K}_weighted_fit_rel_err_max"] = worst
        # M = I: the weighted fit is the plain fit
        I = torch.eye(K)
        lf, rf = lowrank_factors(S.to(device) @ I.to(device), P.to(device), 2)
        lp, rp = lowrank_factors(S.to(device), P.to(device), 2)
        out[f"K{K}_identity_weight_max_abs"] = float((lf @ rf - lp @ rp).abs().max())
        assert out[f"K{K}_identity_weight_max_abs"] < 1e-4
        # young states exact through the codec with weights (float factors)
        codec = LowRankCodec("4", "f")
        wts = (Mh.to(device), Mih.to(device))
        lp4 = torch.zeros(N, H, K, 4, device=device)
        state = torch.zeros(N, H, K, K, device=device)
        ex = 0.0
        for n in range(1, 5):
            v1 = torch.randn(N, H, K, generator=g).to(device)
            k1 = torch.randn(N, H, K, generator=g).to(device)
            Gm = (torch.eye(K) * 0.9 + 0.05 * torch.randn(N, H, K, K, generator=g)).to(device)
            nxt = state @ Gm + v1.unsqueeze(-1) @ k1.unsqueeze(-2)
            state, lp4 = codec.encode(nxt, lp4, v1, torch.full((N,), n, dtype=torch.int64, device=device), None, wts)
            ex = max(ex, float(((state - nxt).flatten(2).norm(dim=-1) / nxt.flatten(2).norm(dim=-1)).max()))
        assert ex < 5e-4, ex
        out[f"K{K}_young_exact_weighted_rel_err_max"] = ex
    # Gramian
    Rm = torch.randn(3, 8, 8, generator=g, dtype=torch.float64)
    Rm = Rm @ Rm.mT
    d = torch.rand(3, 8, generator=g, dtype=torch.float64) * 0.98
    Mg = gramian(Rm, d)
    D = torch.diag_embed(d)
    out["gramian_residual"] = float((Mg - (Rm + D @ Mg @ D)).abs().max() / Rm.abs().max())
    assert out["gramian_residual"] < 1e-10
    # grammar: plain labels unchanged, weighted labels parsed
    for lab in ("lr2q8x8", "lr2.16.4q8x8", "lr2.16.4q6x4", "lr4fx0", "lrRq8x8"):
        c = parse_sm148(lab)
        c.set_ref(264, 264)
        assert c.weight is None and "weight" not in c.fields(), lab
    for lab, kind in (("lr2q8x8wr", "r"), ("lr2q8x8wrk", "rk"), ("lr2.16.4q8x8wg", "g")):
        c = parse_sm148(lab)
        assert c.weight == kind and c.spec in ("2", "2.16.4") and c.qmax == 127, lab
    assert parse_sm148("lr2q8x8wx") is None and parse_sm148("lr2q8x8w") is None and parse("lr2q8x8wr").weight == "r"
    try:
        parse_sm148("lrRq8x8wr")
        raise AssertionError("lrRq8x8wr accepted")
    except ValueError:
        pass
    out["sm157_grammar"] = True
    print("state_lowrank SM157 selftest OK", out)
    return out


# ------------------------------------------------------------------------------------------------ SM159

# SM159: energy-triggered rank growth (artifacts/small-model/sm159/design.md). Label suffix g<max>e<code> after the
# shift width of a single-rank SM148 label, e.g. lr2q8x8g4e150: every (entity, layer, head) starts with a rank cap of
# 2; at each write, when truncation at the cap would discard more than tau = code / 1e4 of the exact update's
# Frobenius energy, the cap grows by one (at most <max>, never shrinks); the head is stored at rank min(n, cap).
# RANK_READOUT: per scorer row (one user, one label), the end-of-history rank of every answered entity's heads (aggregates
# only), for the byte read-out of any rank-only SM148 codec (plain tiers too: rank min(n, tier rank)).

class RankStats:
    """The scorer runs each stream's layers once per group of entities of similar length (srs_model.forward_batch; on
    the expertium CPU path layer by layer over all groups, on the fork path group by group), so a (stream, layer) is
    recorded once per group, in the same group order for every layer: the g-th call of a layer is group g. Sums add
    up over groups, entities are counted at layer 0, and an entity's maximum rank over its layers is kept per group."""

    def __init__(self):
        self.on = True
        self.reset()

    def reset(self):
        self.streams = {}

    def record(self, where, n_end, r_end, K):
        """where (stream, layer); n_end [B] answer counts at the end of the history; r_end [B, H] stored ranks."""
        stream, layer = int(where[0]), int(where[1])
        name = STREAMS.get(stream, str(stream))
        m = (n_end >= 1).cpu()
        n = n_end.cpu()[m]
        r = r_end.cpu()[m].long()
        old = n >= 16
        H = int(r_end.shape[1])
        st = self.streams.setdefault(name, {"entities": 0, "entities_n1": 0, "entities_ge16": 0, "H": H, "K": int(K),
                                            "layers": [], "sum_rank": 0, "rank_hist": [0] * 9,
                                            "rank_hist_ge16": [0] * 9, "_groups": [], "_calls": {}})
        if (st["H"], st["K"]) != (H, int(K)):
            raise RuntimeError(f"SM159 ranks: stream {name} changed heads / head size")
        g = st["_calls"].get(layer, 0)
        st["_calls"][layer] = g + 1
        mx = r.amax(-1) if r.numel() else torch.zeros(0, dtype=torch.long)
        if layer == 0:   # a new group of entities
            if g != len(st["_groups"]):
                raise RuntimeError(f"SM159 ranks: stream {name} group order broken")
            st["entities"] += int(m.sum())
            st["entities_n1"] += int((n == 1).sum())
            st["entities_ge16"] += int(old.sum())
            st["_groups"].append([mx, old])
        else:
            if g >= len(st["_groups"]) or st["_groups"][g][0].shape[0] != mx.shape[0]:
                raise RuntimeError(f"SM159 ranks: stream {name} layer {layer} call {g} matches no layer-0 group")
            st["_groups"][g][0] = torch.maximum(st["_groups"][g][0], mx)
        if layer not in st["layers"]:
            st["layers"].append(layer)
        st["sum_rank"] += int(r.sum())
        for key, sel in (("rank_hist", r), ("rank_hist_ge16", r[old])):
            h = torch.bincount(sel.flatten(), minlength=9).tolist()[:9]
            st[key] = [a + b for a, b in zip(st[key], h)]

    def export(self):
        out = {}
        for name, st in sorted(self.streams.items()):
            calls = set(st["_calls"].values())
            if len(calls) != 1:
                raise RuntimeError(f"SM159 ranks: stream {name} layers have different group counts {st['_calls']}")
            d = {k: v for k, v in st.items() if not k.startswith("_")}
            d["layers"] = sorted(st["layers"])
            d["groups"] = calls.pop()
            mx = torch.cat([x for x, _ in st["_groups"]]) if st["_groups"] else torch.zeros(0, dtype=torch.long)
            old = torch.cat([o for _, o in st["_groups"]]) if st["_groups"] else torch.zeros(0, dtype=torch.bool)
            d["entity_max_hist"] = torch.bincount(mx, minlength=9).tolist()[:9]
            d["entity_max_hist_ge16"] = torch.bincount(mx[old], minlength=9).tolist()[:9]
            out[name] = d
        return out


RANK_READOUT = RankStats()


def selftest_sm159(device="cpu"):
    """SM159 identities (seeded): the discarded share equals the float64 SVD tail share; caps start at the base rank,
    grow by at most one per write, never shrink, stay <= max; a write that grows the cap stores the update exactly with
    float factors; tau above 1 never grows and equals the plain rank-2 codec; tau 0 grows every head that has a tail;
    grammar and descriptors; the rank read-out."""
    g = torch.Generator().manual_seed(159)
    out = {}
    for K in (16, 32):
        N, H = 40, 4
        A = torch.randn(N, H, K, 3, generator=g) @ torch.randn(N, H, 3, K, generator=g)
        A[0, 0] = 0
        P = torch.cat([torch.linalg.qr(A)[0][..., :3], torch.randn(N, H, K, 1, generator=g)], -1)   # span ⊇ col(A)
        lf, rf, s2 = lowrank_factors(A.to(device), P.to(device), 3, with_s2=True)
        sv = torch.linalg.svdvals(A.double())
        worst = 0.0
        for c in (1, 2, 3):
            share = discarded_share(s2, torch.full((N, H), c, dtype=torch.int64, device=device)).double().cpu()
            ref = 1 - (sv[..., :c] ** 2).sum(-1) / (sv ** 2).sum(-1).clamp(min=1e-300)
            ref[0, 0] = 0
            worst = max(worst, float((share - ref).abs().max()))
        assert worst < 1e-4, worst
        out[f"K{K}_share_abs_err_max"] = worst
        # the step loop of the codec on a synthetic recurrence: cap rules and exact growth (float factors)
        cf = parse_sm148("lr2fx8g4e100")
        cap = torch.full((N, H), 2, dtype=torch.int64, device=device)
        left = torch.zeros(N, H, K, 4, device=device)
        state = torch.zeros(N, H, K, K, device=device)
        grown_exact = 0.0
        for n in range(1, 13):
            v1 = torch.randn(N, H, K, generator=g).to(device)
            k1 = torch.randn(N, H, K, generator=g).to(device)
            Gm = (torch.eye(K) * 0.95 + 0.05 * torch.randn(N, H, K, K, generator=g)).to(device)
            nxt = state @ Gm + v1.unsqueeze(-1) @ k1.unsqueeze(-2)
            nn = torch.full((N,), n, dtype=torch.int64, device=device)
            state, left, c2 = cf.encode_energy(nxt, left, v1, nn, cap)
            d = c2 - cap
            assert int(d.min()) >= 0 and int(d.max()) <= 1 and int(c2.max()) <= 4 and int(c2.min()) >= 2
            if bool((d > 0).any()):
                e = (state - nxt).flatten(2).norm(dim=-1) / nxt.flatten(2).norm(dim=-1)
                grown_exact = max(grown_exact, float(e[d > 0].max()))
            rk = (left.abs().amax(-2) > 0).sum(-1)
            assert bool((rk <= torch.minimum(nn[:, None], c2)).all())
            cap = c2
        assert grown_exact < 5e-4, grown_exact
        assert int(cap.max()) == 4
        out[f"K{K}_grown_write_exact_rel_err_max"] = grown_exact
        # tau > 1: never grows and equals the plain rank-2 codec: float factors to float rounding; int8 factors bit for
        # bit on most heads (a float-noise difference can flip an int8 code, discrete-codec tolerance)
        for q, lim in (("f", 1e-4), ("q8", None)):
            never, plain = parse_sm148(f"lr2{q}x8g4e10001"), parse_sm148(f"lr2{q}x8")
            cap = torch.full((N, H), 2, dtype=torch.int64, device=device)
            l1 = torch.zeros(N, H, K, 4, device=device)
            l2 = torch.zeros(N, H, K, 2, device=device)
            s1 = s2_ = torch.zeros(N, H, K, K, device=device)
            for n in range(1, 9):
                v1 = torch.randn(N, H, K, generator=g).to(device)
                k1 = torch.randn(N, H, K, generator=g).to(device)
                Gm = (torch.eye(K) * 0.95 + 0.05 * torch.randn(N, H, K, K, generator=g)).to(device)
                nn = torch.full((N,), n, dtype=torch.int64, device=device)
                s1, l1, cap = never.encode_energy(s1 @ Gm + v1.unsqueeze(-1) @ k1.unsqueeze(-2), l1, v1, nn, cap)
                s2_, l2 = plain.encode(s2_ @ Gm + v1.unsqueeze(-1) @ k1.unsqueeze(-2), l2, v1, nn, None)
                assert int(cap.max()) == 2
            e = (s1 - s2_).flatten(2).norm(dim=-1) / s2_.flatten(2).norm(dim=-1)
            if lim is not None:
                assert float(e.max()) < lim, float(e.max())
                out[f"K{K}_tau_above_1_vs_lr2fx8_rel_diff_max"] = float(e.max())
            else:
                same = float((e == 0).float().mean())
                assert same >= 0.8, same
                out[f"K{K}_tau_above_1_vs_lr2q8x8_identical_head_share"] = same
        always = parse_sm148("lr2fx8g4e0")
        cap = torch.full((N, H), 2, dtype=torch.int64, device=device)
        l1 = torch.zeros(N, H, K, 4, device=device)
        s1 = torch.zeros(N, H, K, K, device=device)
        for n in range(1, 6):
            v1 = torch.randn(N, H, K, generator=g).to(device)
            k1 = torch.randn(N, H, K, generator=g).to(device)
            nn = torch.full((N,), n, dtype=torch.int64, device=device)
            s1, l1, cap = always.encode_energy(s1 * 0.9 + v1.unsqueeze(-1) @ k1.unsqueeze(-2), l1, v1, nn, cap)
        assert int(cap.min()) == 4, cap.min()   # every head grows to the maximum
    # grammar
    c = parse_sm148("lr2q8x8g4e150")
    c.set_ref(264, 264)
    assert c.grow == 0.015 and c.rmax == 4 and c.kinds == [2] and c.qmax == 127 and c.weight is None
    f = c.fields()
    assert f["grow_tau"] == 0.015 and f["grow_max"] == 4 and f["grow_base"] == 2 and f["spec"] == "2"
    c6 = parse_sm148("lr2q6x8g4e75")
    assert c6.qmax == 31 and c6.grow == 0.0075 and parse_sm148("lr2q8x8g3e20").rmax == 3
    for lab in ("lr2q8x8", "lr2.16.4q8x8", "lr2.16.4q6x8", "lr2q8x8wrk", "lrRq8x8"):
        cc_ = parse_sm148(lab)
        cc_.set_ref(264, 264)
        assert cc_.grow is None and "grow_tau" not in cc_.fields(), lab
    for bad in ("lr2.16.4q8x8g4e10", "lr4q8x8g4e10", "lr2q8x8g4e10wr", "lrRq8x8g4e10", "lr2q8x8g2e10"):
        try:
            parse_sm148(bad)
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass
    assert parse_sm148("lr2q8x8g4e") is None and parse_sm148("lr2q8x8g9e10") is None
    assert parse_sm148("lr2q8x8e10") is None and parse_sm148("lr2q8x8g4e123456") is None
    out["sm159_grammar"] = True
    # rank read-out
    rs = RankStats()
    n_end = torch.tensor([0, 1, 3, 20, 40])
    rs.record((0, 0), n_end, torch.tensor([[2, 2], [1, 1], [2, 3], [4, 2], [2, 2]]), 16)
    rs.record((0, 0), torch.tensor([2, 30]), torch.tensor([[2, 2], [2, 2]]), 16)   # a second group, layer-major
    rs.record((0, 1), n_end, torch.tensor([[2, 2], [1, 1], [2, 2], [2, 2], [3, 2]]), 16)
    rs.record((0, 1), torch.tensor([2, 30]), torch.tensor([[2, 2], [2, 4]]), 16)
    e = rs.export()["card"]
    assert e["entities"] == 6 and e["entities_ge16"] == 3 and e["entities_n1"] == 1
    assert e["entity_max_hist"][1:5] == [1, 1, 2, 2] and e["entity_max_hist_ge16"][1:5] == [0, 0, 1, 2]
    assert e["rank_hist"][1:5] == [4, 16, 2, 2]
    rs.reset()
    rs.record((0, 0), n_end, torch.tensor([[2, 2], [1, 1], [2, 3], [4, 2], [2, 2]]), 16)
    rs.record((0, 1), n_end, torch.tensor([[2, 2], [1, 1], [2, 2], [2, 2], [3, 2]]), 16)
    e = rs.export()["card"]
    assert e["entities"] == 4 and e["entities_ge16"] == 2 and e["entities_n1"] == 1 and e["layers"] == [0, 1] and e["H"] == 2
    assert e["sum_rank"] == (1 + 1 + 2 + 3 + 4 + 2 + 2 + 2) + (1 + 1 + 2 + 2 + 2 + 2 + 3 + 2)
    assert e["entity_max_hist"][1:5] == [1, 0, 2, 1] and e["entity_max_hist_ge16"][1:5] == [0, 0, 1, 1]
    assert e["rank_hist"][1:5] == [4, 9, 2, 1] and e["rank_hist_ge16"][1:5] == [0, 6, 1, 1]
    out["sm159_rank_readout"] = True
    print("state_lowrank SM159 selftest OK", out)
    return out


if __name__ == "__main__":
    selftest()
    selftest_sm148()
    selftest_sm157()
    selftest_sm159()
