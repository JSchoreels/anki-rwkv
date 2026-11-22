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
    if st is None or cur is None or cur[0] not in STREAMS or r_BTHK.is_cuda:
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


if __name__ == "__main__":
    selftest()
