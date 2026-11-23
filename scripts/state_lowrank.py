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
                            lambda x: mod._quantize_rows(x, rows_bits))
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
_SM148 = None


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
        self.spec, self.quant = spec, quant
        self.kinds, self.edges = kinds, edges
        self.rmax = max([k for k in kinds if k != "R"], default=0)
        self.has_rows = "R" in kinds
        self.rows_bits = self.shift_bits = None
        self.float_shift = False
        self._dev = {}

    def __bool__(self):
        return True

    def __repr__(self):
        return f"LowRankCodec({self.spec!r}, {self.quant!r})"

    def fields(self):
        return {"spec": self.spec, "quant": self.quant, "kinds": [str(k) for k in self.kinds], "edges": self.edges,
                "shift_bits": 0 if self.float_shift else self.shift_bits}

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

    def encode(self, nxt, left_prev, v, n, rows_quant):
        """nxt [N, H, K, K] float32 updated states (written rows), left_prev [N, H, K, rmax] their stored left factors
        (zero columns beyond the stored rank), v [N, H, K] this write's v, n [N] update counts.
        -> stored states [N, H, K, K] float32 and left factors [N, H, K, rmax] float32."""
        kind = self.kind_of(n)
        stored = nxt
        left_new = torch.zeros_like(left_prev)
        if self.rmax and nxt.is_cuda and FUSED_CUDA:   # one fused kernel (scripts/state_lowrank_cuda.encode)
            from scripts import state_lowrank_cuda
            r_eff = torch.minimum(n, kind.clamp(min=0))
            stored, left_new = state_lowrank_cuda.encode(nxt, left_prev, v, r_eff, self.quant == "q8")
        elif self.rmax:
            r_eff = torch.minimum(n, kind.clamp(min=0))                      # exact rank n for young states
            P = torch.cat([left_prev, v.unsqueeze(-1)], -1)
            finite = torch.isfinite(nxt).flatten(2).all(-1) & torch.isfinite(P).flatten(2).all(-1)   # [N, H]
            fin4 = finite[..., None, None]
            # a non-finite matrix is stored as it is (the scorer's non-finite handling sees it), as in SM147
            lf, rf = lowrank_factors(torch.where(fin4, nxt, torch.zeros_like(nxt)),
                                     torch.where(fin4, P, torch.zeros_like(P)), self.rmax)
            comp = (torch.arange(self.rmax, device=nxt.device)[None, :] < r_eff[:, None]).to(lf.dtype)   # [N, r]
            lf = lf * comp[:, None, None, :]
            rf = rf * comp[:, None, :, None]
            if self.quant == "q8":
                lf = quant_vectors(lf, -2)                                    # per column (left factor vector)
                rf = quant_vectors(rf, -1)                                    # per row (right factor vector)
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


def lowrank_factors(S, P, r):
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


def quant_vectors(x, dim):
    """Symmetric int8 per vector along `dim` with a float16 scale (max |x| / 127, as REF's row scales): dequantised."""
    from scripts.small_model_wkv import round_scale_f16
    scale = round_scale_f16(x.abs().amax(dim, keepdim=True) / 127.0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale


def lowrank_loop(codec, r, k, v, w, a, kd, skip, step, rows_quant):
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
    for t in range(T):
        vt = v[:, t].float()
        out[:, t], nxt = step(r[:, t].float(), k[:, t].float(), vt, w[:, t].float(), a[:, t].float(),
                              kd[:, t].float(), state)
        rows = torch.nonzero(~skip[:, t]).flatten()
        if rows.numel():
            s_new, l_new = codec.encode(nxt[rows], left[rows], vt[rows], counts[rows, t], rows_quant)
            state = state.clone()
            state[rows] = s_new
            left[rows] = l_new
    return out


class ExpertiumLoop:
    """state_vq.Hooks.wkv entry for an SM148 codec: state_vq._reference_wkv_q hands the whole loop to `.loop`."""

    def __init__(self, codec):
        self.codec = codec
        from scripts.state_vq import int_row_quant
        self.rows_quant = int_row_quant(codec.rows_bits) if codec.has_rows else None

    def loop(self, r, k, v, w, a, kd, skip, state0=None):
        if state0 is not None:
            raise RuntimeError("SM148 low-rank codec: no learned initial state")
        from expertium.model.rwkv_ops import single_timestep
        return lowrank_loop(self.codec, r, k, v, w, a, kd, skip, single_timestep, self.rows_quant)

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
            state_vq.Hooks.wkv[(stream, layer)] = ExpertiumLoop(codec)


def parse_sm148(state):
    import re
    m = re.fullmatch(r"lr([0-9R][0-9R.]*?)(q8|f)x(8|0)", state)
    if not m:
        return None
    codec = LowRankCodec(m.group(1), m.group(2))
    codec.float_shift = m.group(3) == "0"   # proof only: float32 shift vectors (SM147's lr<r> placement)
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
    # tier selection
    c = LowRankCodec("2.16.4.128.R", "q8")
    assert c.kind_of(torch.tensor([1, 15, 16, 127, 128, 900])).tolist() == [2, 2, 4, 4, -1, -1]
    assert c.rmax == 4 and c.has_rows and parse_sm148("lr2.16.4.128.Rq8x8").spec == "2.16.4.128.R"
    assert parse_sm148("lrRx8") is None and parse_sm148("lrRq8x8").kinds == ["R"]
    out["tiers"] = True
    print("state_lowrank SM148 selftest OK", out)
    return out


if __name__ == "__main__":
    selftest()
    selftest_sm148()
