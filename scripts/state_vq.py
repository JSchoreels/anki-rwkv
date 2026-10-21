"""SM32: Expertium-style vector quantization of the student's card/note recurrent state (post-training, CPU).

Reimplements (in torch, CPU) the deploy-time state quantizer of Expertium/rwkv-anki-autoresearch@1f2e6b0
(rust/rwkv-infer/src/model.rs: `compress_wkv_state`, `PqCodebook::encode_decode_joint_warm`, the shift `PqCodebook`
path, `quant_norm`). Per card/note entity, after every review (same placement as the int8 `+s8` emulation: the
answer's output uses the float update, the STORED state is replaced by its reconstruction before the next review of
that entity reads it):

WKV matrix of each head (K x K, K = 16):
  1. rank-r truncation through the eigendecomposition of the max-abs normalised Gram A A^T (his fast path); per
     component j: unit left vector u, unit right vector v = A^T u / sigma, both factors scaled by sqrt(sigma);
  2. sign canonicalisation (u's largest-magnitude entry made positive, v flipped with it);
  3. JOINT code: concat(u, v) (2K = 32 dims) coded by residual VQ, `stages` stages of 2^index_bits centroids each
     (stage s codes what stages < s left; the reconstruction is the sum of the picks);
  4. one norm per component: sqrt(sigma) (|u| = |v| = sqrt(sigma), so his two norm quantizers store the same value),
     `norm_bits` bits uniform in log2 over a fixed corpus range; the reconstruction is q(sqrt sigma)^2 * u_hat v_hat^T.
Token-shift vectors (the previous review's layer-normed input of each time mixer and of each kept channel mixer):
  unit direction split into `shift_m` sub-vectors, each coded by `shift_stages` residual stages of 2^shift_bits
  centroids (his product quantization; he uses one stage), norm at `shift_norm_bits` bits in log2.
Deck / preset / user state, weights and arithmetic stay float (as his deployed engine). Codebooks are global (shared
by every card and note entity, so they do not count per entity): one WKV catalog per rank component shared by the
card and note streams and all their layers/heads (his single joint catalog), one shift catalog per role
(time / channel mixer) and sub-vector position, shared by streams and layers (his two shift roles).

Bits per entity = sum over WKV layers x heads x components (stages * index_bits + norm_bits)
                + sum over shift vectors (shift_m * shift_bits * shift_stages + shift_norm_bits).
His deployed d=64 recipe (LEDGER row 16, `s64hqr4b`): rank 1, 4 stages of b10, 1-bit norms, shift m2b12 + 1-bit norm
-> card 2*4*(40+1) + 3*25 = 403 bits, note 1*4*41 + 2*25 = 214 bits. One stage: 163 / 94 bits.

Deviations from his scheme (documented in docs/SMALL_MODEL.md SM32): post-training k-means codebooks (his deployed
catalogs are learned in QAT), rank-2 components get their own catalog (his joint catalog is shared by components),
norm ranges fitted on the corpus (MSE-optimal uniform log2 grid), no warm-start search (a speed trick, pick-identical
in his engine), exact nearest-centroid search by matmul distances.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
import math

import numpy as np
import torch

K = 16


# ------------------------------------------------------------------------------------------------ recipes

@dataclass(frozen=True)
class StreamRecipe:
    """One stream's (card or note) state codec. rank 0 = WKV left float; stages[j] 0 = component j's directions kept
    exact; norm_bits None = exact norms; shift_m 0 = shift vectors left float; shift_stages 0 = exact shift direction.
    Any exact part makes the budget infinite (bits reported as None)."""
    rank: int = 1
    stages: tuple = (4,)            # residual WKV stages per rank component (len = rank)
    index_bits: int = 10
    norm_bits: object = 1           # None = exact float norm, 0 = fixed midpoint (0 bits)
    shift_m: int = 2
    shift_bits: int = 12
    shift_stages: int = 1
    shift_norm_bits: object = 1
    factor: str = "eigh"            # rank-1 direction: "eigh" (linalg.eigh) | "sq" (Gram squaring, factorize_sq; S09)

    def exact_parts(self):
        return (self.rank == 0 or 0 in self.stages or self.norm_bits is None or self.shift_m == 0
                or self.shift_stages == 0 or self.shift_norm_bits is None)

    def wkv_bits(self):
        nb = self.norm_bits or 0
        return sum(s * self.index_bits + nb for s in self.stages)

    def shift_vector_bits(self):
        return self.shift_m * self.shift_bits * self.shift_stages + (self.shift_norm_bits or 0)


def _r(**kw):
    return StreamRecipe(**kw)


_E1 = _r(stages=(1,))
_E4 = _r(stages=(4,))
RECIPES = {
    # pipeline checks / diagnostics (no finite budget)
    "vqexact": (_r(rank=K, stages=(0,) * K, norm_bits=None, shift_stages=0, shift_norm_bits=None),) * 2,
    "vqr1exact": (_r(stages=(0,), norm_bits=None, shift_m=0),) * 2,          # rank-1 truncation only
    "vqr2exact": (_r(rank=2, stages=(0, 0), norm_bits=None, shift_m=0),) * 2,
    "vqE4wkv": (_r(stages=(4,), shift_m=0),) * 2,                            # his WKV codec, float shift
    "vqE4shift": (_r(rank=0, stages=()),) * 2,                               # his shift codec, float WKV
    # Expertium's points (rank 1, b10 stages, 1-bit norms, shift m2b12 + 1-bit norm)
    "vq163x94": (_E1, _E1),
    "vq403x214": (_E4, _E4),
    # allocation screen (tune on training users)
    "vqE4n4": (_r(stages=(4,), norm_bits=4, shift_norm_bits=4),) * 2,
    "vqE4n4s2": (_r(stages=(4,), norm_bits=4, shift_stages=2, shift_norm_bits=4),) * 2,
    "vqR2n4": (_r(rank=2, stages=(4, 2), norm_bits=4, shift_norm_bits=4),) * 2,
    "vqE8n4": (_r(stages=(8,), norm_bits=4, shift_norm_bits=4),) * 2,
    "vqS3n6": (_r(stages=(3,), norm_bits=6, shift_norm_bits=6),) * 2,
    "vqS7n6s2": (_r(stages=(7,), norm_bits=6, shift_stages=2, shift_norm_bits=6),) * 2,
    "vqS8n6": (_r(stages=(8,), norm_bits=6, shift_norm_bits=6),) * 2,
    "vqR62n6": (_r(rank=2, stages=(6, 2), norm_bits=6, shift_norm_bits=6),) * 2,
    "vqR84n6m4": (_r(rank=2, stages=(8, 4), norm_bits=6, shift_m=4, shift_stages=3, shift_norm_bits=6),) * 2,
}
# SM32 budget points (Expertium's recommendation 400/400 -> 800/400 -> 800/800), picked on the training-user screen:
# 6-bit norms everywhere (1-bit norms were most of the PTQ loss), then WKV stages, shift stages, rank 2 + shift m4.
_C400 = _r(stages=(3,), norm_bits=6, shift_norm_bits=6)                                   # card 378 bits
_C800 = _r(stages=(7,), norm_bits=6, shift_stages=2, shift_norm_bits=6)                   # card 770 bits
_N400 = _r(stages=(7,), norm_bits=6, shift_stages=2, shift_norm_bits=6)                   # note 412 bits
_N800 = _r(rank=2, stages=(8, 4), norm_bits=6, shift_m=4, shift_stages=3, shift_norm_bits=6)  # note 828 bits
RECIPES.update({"vq378x412": (_C400, _N400), "vq770x412": (_C800, _N400), "vq770x828": (_C800, _N800),
                "vq1506x828": (_N800, _N800)})   # upper reference: rank 2 (8 + 4 stages), shift m4 x 3 stages
# S09: the same codecs with the rank-1 direction from Gram squaring instead of linalg.eigh (same bits and codebooks;
# the fused CUDA QAT step computes exactly this, so training, scoring and deployment can share one factorization).
RECIPES.update({name + "p": tuple(replace(rec, factor="sq") for rec in RECIPES[name])
                for name in ("vq378x412", "vq770x412")})


# SM46 (2026-10-20): the large budget asked by the outside researcher, ~6,000 bits per card and ~3,000 per note (int8
# card state is 26,624 bits). Picked with the model-free held-out proxy (artifacts/small-model/sm46/alloc_proxy.py) and
# a training-user screen (state_vq_fit.py tune, 8 users, model-side loss vs w8+s8): rank 6 per 16 x 16 head (rank 6
# keeps 99.8% of the head energy on SM36's states, rank 4 98.9%; rank-6 truncation alone costs nothing measurable),
# stages spread so the components' coding errors are about equal, 10-bit norms on a full-range log2 grid
# (fit_norm_range, bits >= 10), token shift m4 x 8 stages (12-bit codes) + 10-bit norm = 394 bits per vector.
# The screen had the two splits of the budget level (vq5902x3028, ~1,200 card bits on the shift: +0.000058 curve /
# +0.000141 instant; vq5934x2996, ~900 on the shift and more WKV stages: +0.000048 / +0.000152); instant is the binding
# head, so the shift-heavier one is the SM46 recipe. Codebooks: artifacts/small-model/sm46/books/codebooks-sm46.npz
# (k-means on SM36's states of 45 training users).
_S46 = dict(norm_bits=10, shift_m=4, shift_stages=8, shift_norm_bits=10)
RECIPES.update({
    # card: 8 heads x 590 (rank 6, 53 stages) + 3 x 394 = 5,902; note: 4 x 560 (50 stages) + 2 x 394 = 3,028
    "vq5902x3028": (_r(rank=6, stages=(15, 13, 10, 7, 5, 3), **_S46), _r(rank=6, stages=(15, 12, 9, 7, 4, 3), **_S46)),
    # screen: more WKV stages, shift m4 x 6 (298 bits): card 8 x 630 + 3 x 298 = 5,934, note 4 x 600 + 2 x 298 = 2,996
    "vq5934x2996": (_r(rank=6, stages=(16, 14, 11, 8, 5, 3), **{**_S46, "shift_stages": 6}),
                    _r(rank=6, stages=(16, 13, 10, 7, 5, 3), **{**_S46, "shift_stages": 6})),
    # screen: rank-6 truncation only (float shift, exact norms)
    "vqr6exact": (_r(rank=6, stages=(0,) * 6, norm_bits=None, shift_m=0),) * 2,
})
RECIPES["vqS46sh"] = RECIPES["vq5902x3028"]   # its screen name (state_vq_fit.py tune rows)


def recipe(name):
    if name not in RECIPES:
        raise SystemExit(f"unknown VQ recipe {name}; known: {sorted(RECIPES)}")
    return RECIPES[name]


# ------------------------------------------------------------------------------------------------ quantizers

def nearest(x, centroids, cn2=None, chunk=4096):
    """Index of the nearest centroid (squared L2) for every row of x [N, D]; centroids [M, D]. Rows are independent;
    `chunk` only bounds the [chunk, M] distance matrix (one chunk when N <= chunk, the GPU training path's setting)."""
    if cn2 is None:
        cn2 = (centroids * centroids).sum(-1)
    if x.shape[0] <= chunk:
        return (cn2[None, :] - 2.0 * (x @ centroids.T)).argmin(-1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        out[i:i + chunk] = (cn2[None, :] - 2.0 * (xb @ centroids.T)).argmin(-1)
    return out


def rvq(x, books, norms2=None, chunk=4096):
    """Residual VQ reconstruction of x [N, D] with stage codebooks books[s] [M, D]."""
    acc = torch.zeros_like(x)
    for s, cb in enumerate(books):
        idx = nearest(x - acc, cb, None if norms2 is None else norms2[s], chunk)
        acc = acc + cb[idx]
    return acc


def rvq_picks(x, books, norms2=None, chunk=4096):
    """rvq with the picks: (acc [N, D], picks [N, S] long). Same picks and sums as rvq."""
    acc = torch.zeros_like(x)
    picks = []
    for s, cb in enumerate(books):
        idx = nearest(x - acc, cb, None if norms2 is None else norms2[s], chunk)
        acc = acc + cb[idx]
        picks.append(idx)
    return acc, (torch.stack(picks, -1) if picks else torch.zeros(x.shape[0], 0, dtype=torch.long, device=x.device))


@dataclass
class NormQuant:
    bits: object = None
    lo: float = 0.0   # log2
    hi: float = 0.0

    def __call__(self, n):
        if self.bits is None:
            return n
        if self.bits == 0:
            return torch.full_like(n, 2.0 ** ((self.lo + self.hi) / 2))
        levels = float(2 ** self.bits - 1)
        t = (torch.log2(n) - self.lo) / (self.hi - self.lo)
        q = torch.clamp(torch.round(t * levels), 0, levels)
        return torch.exp2(self.lo + q / levels * (self.hi - self.lo))


def fit_norm_range(log2n, bits, weights=None, grid=41):
    """lo/hi of a uniform `bits`-bit log2 grid minimizing the weighted squared log2 error on the sample (grid search
    over quantiles 1..99%). Weights: the WKV component's energy share (sigma), so near-degenerate second components
    (sqrt(sigma) down to 2^-17) do not stretch the range; uniform for shift norms. His ranges are corpus-derived
    globals too (WKV sqrt(sigma) log2 [-2.5, -0.6], shift [2.4, 2.7])."""
    x = np.asarray(log2n, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    pick = np.linspace(0, len(x) - 1, min(len(x), 20000)).astype(int)
    order = np.argsort(x)
    x, w = x[order][pick], w[order][pick]
    w = w / w.sum()
    if bits == 0:
        mid = float((w * x).sum())
        return mid, mid
    if bits >= 10:
        # SM46: at 10+ bits the grid is fine enough (20 / 1023 ~ 0.02 in log2) to span the whole sample: hi = the
        # largest norm, lo 20 octaves below (no clipping of large norms; smaller ones are clamped up). The quantile
        # search below clips the top 1% of norms, which dominated the norm error at 10 bits (SM46 proxy, rank-6 heads
        # with exact directions: relative error 2.60e-3 -> 2.07e-3; exact norms 2.06e-3).
        hi = float(np.max(x))
        return max(float(np.min(x)), hi - 20.0), hi
    levels = 2 ** bits - 1
    cdf = np.cumsum(w)
    cand = np.interp(np.linspace(0.01, 0.99, grid), cdf, x)
    best = (math.inf, float(cand[0]), float(cand[-1]))
    for i in range(grid):
        for j in range(i + 1, grid):
            lo, hi = cand[i], cand[j]
            if hi <= lo:
                continue
            q = np.clip(np.round((x - lo) / (hi - lo) * levels), 0, levels)
            err = float((w * (lo + q / levels * (hi - lo) - x) ** 2).sum())
            if err < best[0]:
                best = (err, float(lo), float(hi))
    return best[1], best[2]


# ------------------------------------------------------------------------------------------------ codebooks

@dataclass
class Codebooks:
    wkv: list = field(default_factory=list)       # wkv[j] = [stage tensors [2^b, 2K]] for rank component j
    shift: dict = field(default_factory=dict)     # shift[(role, m)] = [[stage tensors [2^b, C/m]] per position]
    wkv_norm: dict = field(default_factory=dict)  # (component, bits) -> (lo, hi)
    shift_norm: dict = field(default_factory=dict)  # (role, bits) -> (lo, hi)

    def save(self, path):
        arrays = {}
        for j, stages in enumerate(self.wkv):
            for s, cb in enumerate(stages):
                arrays[f"wkv/{j}/{s}"] = cb.numpy()
        for (role, m), positions in self.shift.items():
            for p, stages in enumerate(positions):
                for s, cb in enumerate(stages):
                    arrays[f"shift/{role}/{m}/{p}/{s}"] = cb.numpy()
        for (j, bits), (lo, hi) in self.wkv_norm.items():
            arrays[f"wkvnorm/{j}/{bits}"] = np.array([lo, hi])
        for (role, bits), (lo, hi) in self.shift_norm.items():
            arrays[f"shiftnorm/{role}/{bits}"] = np.array([lo, hi])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        books = cls()
        wkv, shift = {}, {}
        for key in z.files:
            parts = key.split("/")
            if parts[0] == "wkv":
                wkv.setdefault(int(parts[1]), {})[int(parts[2])] = torch.from_numpy(z[key]).float()
            elif parts[0] == "shift":
                shift.setdefault((parts[1], int(parts[2])), {}).setdefault(int(parts[3]), {})[int(parts[4])] = \
                    torch.from_numpy(z[key]).float()
            elif parts[0] == "wkvnorm":
                books.wkv_norm[(int(parts[1]), int(parts[2]))] = tuple(float(v) for v in z[key])
            elif parts[0] == "shiftnorm":
                books.shift_norm[(parts[1], int(parts[2]))] = tuple(float(v) for v in z[key])
        books.wkv = [[wkv[j][s] for s in sorted(wkv[j])] for j in sorted(wkv)]
        books.shift = {key: [[pos[p][s] for s in sorted(pos[p])] for p in sorted(pos)] for key, pos in shift.items()}
        return books


# ------------------------------------------------------------------------------------------------ state codecs

def factorize(A, rank, dtype=torch.float64):
    """Top-`rank` factors of A [N, K, K] (`dtype` inside, float64 for the scorer): unit u, unit v, sigma, valid mask,
    each [N, rank, (K)].
    Same construction as model.rs (Gram of the max-abs normalised matrix, v = A^T u / sigma), plus his sign canon."""
    A64 = A.to(dtype)
    amax = A64.abs().amax(dim=(1, 2))
    scale = torch.where(torch.isfinite(amax) & (amax > 1e-30), amax, torch.ones_like(amax))
    an = A64 / scale[:, None, None]
    evals, evecs = torch.linalg.eigh(an @ an.transpose(1, 2))       # ascending
    order = torch.arange(K - 1, K - 1 - rank, -1, device=A.device)
    ev = evals[:, order]                                             # [N, r]
    u = evecs[:, :, order].transpose(1, 2)                           # [N, r, K]
    v_un = torch.einsum("nij,nri->nrj", A64, u)                      # (A^T u) per component
    sigma = v_un.norm(dim=-1)
    valid = torch.isfinite(ev) & (ev > 0) & torch.isfinite(sigma) & (sigma > 1e-20)
    v = v_un / torch.where(valid, sigma, torch.ones_like(sigma))[..., None]
    lead = u.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(torch.gather(u, -1, lead) >= 0, 1.0, -1.0).to(u.dtype)
    return u * sign, v * sign, sigma, valid


SQ_SQUARINGS = 12   # P <- P @ P twelve times = G^4096: direction error ~ (lambda2 / lambda1)^4096
SQ_RENORM = 4       # trace renormalisation before every 4th squaring (p1 >= 1/16 after it, p1^16 stays above 1e-20)


def factorize_sq(A, dtype=torch.float64, squarings=SQ_SQUARINGS):
    """Rank-1 `factorize` without an eigensolver (S09): top eigenvector of the same normalised Gram G = An An^T by
    repeated squaring. P = G / tr(G); `squarings` times P <- P @ P (trace renormalised before every SQ_RENORM-th);
    u = G p / |G p| with p the column of P with the largest diagonal (it has the largest u1 share); then v, sigma, the
    valid mask and the sign canon exactly as `factorize`. Only matrix products, reductions and elementwise ops: no
    cuSOLVER call or host sync on CUDA, and the fused CUDA QAT step (expertium/model/state_vq_cuda.py) runs the same
    sequence per head. Equals `factorize` up to the (lambda2 / lambda1)^4096 residue (near-degenerate tops) and rounding."""
    A64 = A.to(dtype)
    N = A64.shape[0]
    amax = A64.abs().amax(dim=(1, 2))
    scale = torch.where(torch.isfinite(amax) & (amax > 1e-30), amax, torch.ones_like(amax))
    an = A64 / scale[:, None, None]
    G = an @ an.transpose(1, 2)
    P = G
    for i in range(0, squarings, SQ_RENORM):
        tr = P.diagonal(dim1=1, dim2=2).sum(-1)
        P = P / torch.where(tr > 0, tr, torch.ones_like(tr))[:, None, None]
        P = torch.linalg.matrix_power(P, 2 ** min(SQ_RENORM, squarings - i))   # = that many squarings P <- P @ P
    j = P.diagonal(dim1=1, dim2=2).argmax(-1)
    col = P[torch.arange(N, device=A.device), :, j]                 # P[n, :, j_n]
    p = (G @ col.unsqueeze(-1)).squeeze(-1)
    pn = p.norm(dim=-1, keepdim=True)
    u = p / torch.where(pn > 0, pn, torch.ones_like(pn))
    ev = (u * (G @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)            # Rayleigh quotient (eigh's top eigenvalue)
    v_un = (A64.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)      # A^T u
    sigma = v_un.norm(dim=-1)
    valid = torch.isfinite(ev) & (ev > 0) & torch.isfinite(sigma) & (sigma > 1e-20)
    v = v_un / torch.where(valid, sigma, torch.ones_like(sigma))[..., None]
    lead = u.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(torch.gather(u, -1, lead) >= 0, 1.0, -1.0).to(u.dtype)
    return (u * sign)[:, None], (v * sign)[:, None], sigma[:, None], valid[:, None]


class StateCodec:
    """Encode-decode of one stream's stored state (WKV matrices and shift vectors) for a StreamRecipe.
    wkv_chunk / shift_chunk: rows per nearest-centroid distance matrix (the CPU scorer keeps 4096; the GPU training
    path uses one chunk per WKV step and 8192-row chunks for the shift vectors of a whole split)."""

    def __init__(self, rec, books, wkv_chunk=4096, shift_chunk=4096, factor_dtype=torch.float64):
        self.rec = rec
        self.factor_dtype = factor_dtype
        self.wkv_chunk, self.shift_chunk = wkv_chunk, shift_chunk
        self.wkv_books, self.wkv_n2, self.wkv_nq = [], [], []
        for j in range(rec.rank):
            stages = books.wkv[j][:rec.stages[j]] if rec.stages[j] else []
            if len(stages) < rec.stages[j]:
                raise SystemExit(f"codebook has {len(stages)} WKV stages for component {j}, recipe needs {rec.stages[j]}")
            self.wkv_books.append(stages)
            self.wkv_n2.append([(cb * cb).sum(-1) for cb in stages])
            self.wkv_nq.append(NormQuant(rec.norm_bits, *books.wkv_norm[(j, rec.norm_bits)])
                               if rec.norm_bits is not None else NormQuant())
        self.shift_books, self.shift_n2, self.shift_nq = {}, {}, {}
        if rec.shift_m:
            for role in ("t", "c"):
                if rec.shift_stages:
                    positions = books.shift[(role, rec.shift_m)]
                    self.shift_books[role] = [stages[:rec.shift_stages] for stages in positions]
                    if any(len(st) < rec.shift_stages for st in self.shift_books[role]):
                        raise SystemExit(f"shift codebook {role}/m{rec.shift_m} has too few stages")
                    self.shift_n2[role] = [[(cb * cb).sum(-1) for cb in st] for st in self.shift_books[role]]
                self.shift_nq[role] = (NormQuant(rec.shift_norm_bits, *books.shift_norm[(role, rec.shift_norm_bits)])
                                       if rec.shift_norm_bits is not None else NormQuant())

    learned = None   # SM32c: the LearnedBooks whose current values this codec codes with (sync() before each use)

    def sync(self):
        """SM32c: take the learned catalogs' current values (detached) as this codec's books; no-op when frozen."""
        lb = self.learned
        if lb is None:
            return
        with torch.no_grad():
            if self.rec.rank:
                stages = [lb.wkv[s].detach() for s in range(self.rec.stages[0])]
                self.wkv_books[0] = stages
                self.wkv_n2[0] = [(cb * cb).sum(-1) for cb in stages]
                kernel = getattr(self, "kernel", None)
                if kernel is not None and hasattr(kernel, "refresh"):
                    kernel.refresh(lb.wkv[:self.rec.stages[0]])
            if self.rec.shift_m and self.rec.shift_stages:
                for role in ("t", "c"):
                    param = lb.shift[role]
                    self.shift_books[role] = [[param[p, st].detach() for st in range(self.rec.shift_stages)]
                                              for p in range(self.rec.shift_m)]
                    self.shift_n2[role] = [[(cb * cb).sum(-1) for cb in pos] for pos in self.shift_books[role]]

    def encode_wkv(self, S):
        """SM32c torch path (rank 1): S [N, H, K, K] -> (reconstruction float32 [N, H, K, K], picks [N, H, stages]
        long, cscale [N, H] = q(n)^2, 0 where the factorization is invalid). Same value as wkv()."""
        if self.rec.rank != 1 or not self.wkv_books[0]:
            raise ValueError("encode_wkv needs a rank-1 recipe with coded directions")
        N, H = S.shape[:2]
        A = S.reshape(N * H, K, K)
        fd = self.factor_dtype
        if self.rec.factor == "sq":
            u, v, sigma, valid = factorize_sq(A, fd)
        else:
            u, v, sigma, valid = factorize(A, 1, fd)
        q = torch.cat([u[:, 0], v[:, 0]], dim=-1).float()
        acc, picks = rvq_picks(q, self.wkv_books[0], self.wkv_n2[0], self.wkv_chunk)
        acc = acc.to(fd)
        n = torch.sqrt(torch.where(valid[:, 0], sigma[:, 0], torch.ones_like(sigma[:, 0])))
        nq = self.wkv_nq[0](n.float()).to(fd) if self.rec.norm_bits is not None else n
        c = torch.where(valid[:, 0], nq * nq, torch.zeros_like(nq))
        out = c[:, None, None] * acc[:, :K, None] * acc[:, None, K:]
        return (out.float().reshape(N, H, K, K), picks.reshape(N, H, -1),
                c.float().reshape(N, H))

    def shift_learned(self, x, role):
        """SM32c: shift() whose reconstruction is differentiable in the learned shift catalog (picks frozen,
        embedding-style gradients into the picked rows); x [N, C] detached. Same value as shift()."""
        lb = self.learned
        if lb is None or not self.rec.shift_m or not self.rec.shift_stages:
            return self.shift(x, role)
        with torch.no_grad():
            n = x.norm(dim=-1)
            valid = torch.isfinite(n) & (n > 1e-20)
            n1 = torch.where(valid, n, torch.ones_like(n))
            unit = x / n1[:, None]
            m = self.rec.shift_m
            sub = x.shape[1] // m
            picks = [rvq_picks(unit[:, p * sub:(p + 1) * sub].contiguous(), self.shift_books[role][p],
                               self.shift_n2[role][p], self.shift_chunk)[1] for p in range(m)]
            scale = self.shift_nq[role](n1)
        param = lb.shift[role]
        parts = []
        for p in range(m):
            acc = param[p, 0].index_select(0, picks[p][:, 0])
            for st in range(1, self.rec.shift_stages):
                acc = acc + param[p, st].index_select(0, picks[p][:, st])
            parts.append(acc)
        rec = torch.cat(parts, dim=-1) * scale[:, None]
        return torch.where(valid[:, None], rec, x)

    def to(self, device):
        """Move the codebooks (and their squared norms) to `device`; returns self."""
        move = lambda tensors: [t.to(device) for t in tensors]  # noqa: E731
        self.wkv_books = [move(stages) for stages in self.wkv_books]
        self.wkv_n2 = [move(stages) for stages in self.wkv_n2]
        self.shift_books = {role: [move(st) for st in pos] for role, pos in self.shift_books.items()}
        self.shift_n2 = {role: [move(st) for st in pos] for role, pos in self.shift_n2.items()}
        return self

    def wkv(self, S):
        """S [N, H, K, K] -> reconstruction (float32)."""
        if self.rec.rank == 0:
            return S
        N, H = S.shape[:2]
        A = S.reshape(N * H, K, K)
        fd = self.factor_dtype
        if self.rec.factor == "sq":
            if self.rec.rank != 1:
                raise ValueError("factor 'sq' (Gram squaring) is rank 1 only")
            u, v, sigma, valid = factorize_sq(A, fd)
        else:
            u, v, sigma, valid = factorize(A, self.rec.rank, fd)
        out = torch.zeros(N * H, K, K, dtype=fd, device=S.device)
        for j in range(self.rec.rank):
            uj, vj = u[:, j], v[:, j]
            if self.wkv_books[j]:
                q = torch.cat([uj, vj], dim=-1).float()
                rec = rvq(q, self.wkv_books[j], self.wkv_n2[j], self.wkv_chunk).to(fd)
                uj, vj = rec[:, :K], rec[:, K:]
            n = torch.sqrt(torch.where(valid[:, j], sigma[:, j], torch.ones_like(sigma[:, j])))
            nq = self.wkv_nq[j](n.float()).to(fd) if self.rec.norm_bits is not None else n
            term = (nq * nq)[:, None, None] * uj[:, :, None] * vj[:, None, :]
            out = out + torch.where(valid[:, j, None, None], term, torch.zeros_like(term))
        return out.float().reshape(N, H, K, K)

    def shift(self, x, role):
        """x [N, C] stored shift vectors -> reconstruction."""
        if not self.rec.shift_m or (not self.rec.shift_stages and self.rec.shift_norm_bits is None):
            return x
        n = x.norm(dim=-1)
        valid = torch.isfinite(n) & (n > 1e-20)
        n1 = torch.where(valid, n, torch.ones_like(n))
        unit = x / n1[:, None]
        if self.rec.shift_stages:
            m = self.rec.shift_m
            sub = x.shape[1] // m
            unit = torch.cat([rvq(unit[:, p * sub:(p + 1) * sub].contiguous(), self.shift_books[role][p],
                                  self.shift_n2[role][p], self.shift_chunk) for p in range(m)], dim=-1)
        rec = unit * self.shift_nq[role](n1)[:, None]
        return torch.where(valid[:, None], rec, x)


def int_row_quant(bits):
    """The fork's int8/int4 per-row WKV state rounding (`+s8`/`+s4`), as scripts/small_model_wkv.reference.
    SM47: `bits | small_model_wkv.F16_SCALE` rounds the row scale to float16 first (the kernel's f16_round_scale).
    Also the per-vector int8 token-shift rounding (rows = the stored 64-float shift vectors)."""
    from scripts.small_model_wkv import F16_SCALE, round_scale_f16
    qmax = 127.0 if (bits & 255) == 8 else 7.0
    f16 = bool(bits & F16_SCALE)

    def quant(nxt):
        scale = nxt.abs().amax(-1, keepdim=True) / qmax
        if f16:
            scale = round_scale_f16(scale)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return torch.clamp(torch.round(nxt / scale), -qmax, qmax) * scale
    return quant


# ------------------------------------------------------------------------------------------------ model hooks

class Hooks:
    """Routes the CPU WKV reference and the token-shift gathers of an eager expertium model through per-stream state
    codecs. `wkv[(stream, layer)]` / `shift[(stream, layer, role)]` map to callables (or are absent = float state)."""
    current = None   # (stream, layer, role) of the mixer currently running
    skip = None      # skip_BT of the running block's time mixer (the channel mixer runs on the same rows)
    wkv = {}
    shift = {}
    record_shift = None  # corpus collection: fn(x_BTC[~skip] = the vectors that get stored, cur)


def _reference_wkv_q(r, k, v, w, a, kd, skip):
    from expertium.model.rwkv_ops import single_timestep
    cur = Hooks.current
    quant = Hooks.wkv.get(cur[:2]) if cur is not None and cur[2] == "t" else None
    out_dtype = k.dtype
    r, k, v, w, a, kd = (x.float() for x in (r, k, v, w, a, kd))
    B, T, H, Kh = r.shape
    out = torch.empty(B, T, H, Kh, dtype=torch.float32)
    state = torch.zeros(B, H, Kh, Kh, dtype=torch.float32)
    for t in range(T):
        out[:, t], nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        keep = skip[:, t]
        if quant is None:
            state = torch.where(keep[:, None, None, None], state, nxt)
            continue
        rows = torch.nonzero(~keep).flatten()
        if rows.numel():
            # The answer's output used the float update; the stored state is the codec's reconstruction.
            state = state.clone()
            state[rows] = quant(nxt[rows], cur, rows, t)
    return out.to(out_dtype)


class _TorchProxy:
    """Stands in for `torch` inside expertium.model.rwkv_model so the token-shift gather can be intercepted."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def gather(self, input, dim, index, **kwargs):
        out = self._real.gather(input, dim, index, **kwargs)
        cur = Hooks.current
        if dim != 1 or input.dim() != 3 or cur is None:
            return out
        if Hooks.record_shift is not None and cur[:2] in Hooks.wkv and Hooks.skip is not None:
            Hooks.record_shift(input[~Hooks.skip.bool()], cur)
        fn = Hooks.shift.get(cur)
        if fn is None:
            return out
        T = index.shape[1]
        stored = index[:, :, 0] != torch.arange(T)[None, :]   # t = 0 reads its own input, not a stored vector
        if stored.any():
            out = out.clone()
            out[stored] = fn(out[stored], cur)
        return out


def install(model, streams=(0, 2)):
    """Patch an eager expertium model (built by small_model_heldout.build_expertium): pre-hooks name the running
    mixer, the CPU WKV reference and the shift gather consult Hooks. Returns the stream layout used for bit counts:
    {stream: {"wkv_layers": n, "heads": H, "shift_vectors": [(layer, role), ...]}}."""
    from expertium.model import rwkv_model as erm
    erm.reference_rwkv7 = _reference_wkv_q
    if not isinstance(erm.torch, _TorchProxy):
        erm.torch = _TorchProxy(torch)
    layout = {}
    for index, module in enumerate(model.rwkv_modules):
        info = layout.setdefault(index, {"wkv_layers": 0, "heads": 0, "shift_vectors": []})
        for layer, block in enumerate(module.blocks):
            tm = block.time_mixer
            tm.register_forward_pre_hook(_naming_hook((index, layer, "t"), True), with_kwargs=True)
            info["wkv_layers"] += 1
            info["heads"] = tm.H
            info["shift_vectors"].append((layer, "t"))
            cm = block.channel_mixer
            if isinstance(cm, erm.RWKV7ChannelMixer):
                cm.register_forward_pre_hook(_naming_hook((index, layer, "c")))
                info["shift_vectors"].append((layer, "c"))
    return {s: layout[s] for s in streams}


def _naming_hook(name, time_mixer=False):
    if time_mixer:
        def hook(_module, args, kwargs):
            Hooks.current = name
            Hooks.skip = kwargs["skip_BT"] if "skip_BT" in kwargs else args[3]
        return hook

    def hook(_module, _inputs):
        Hooks.current = name
    return hook


def configure(layout, codecs=None, int_bits=0, shift_bits=0):
    """Set the active quantizers. codecs: {stream: StateCodec} (VQ) or None; int_bits 8/4 = `+s8`/`+s4` on CPU
    (| F16_SCALE: float16 row scales); shift_bits (SM47, with int_bits): the card/note stored token-shift vectors
    through int_row_quant(shift_bits) (8 | F16_SCALE = int8 codes, float16 per-vector scale)."""
    Hooks.wkv, Hooks.shift = {}, {}
    for stream, info in layout.items():
        for layer in range(info["wkv_layers"]):
            if codecs is not None:
                codec = codecs[stream]
                Hooks.wkv[(stream, layer)] = lambda s, cur, rows, t, c=codec: c.wkv(s)
            elif int_bits:
                # SM45: int_bits may be a (card, note) tuple of per-stream bits
                bits = dict(zip((0, 2), int_bits)).get(stream, 0) if isinstance(int_bits, tuple) else int_bits
                if not bits:
                    continue
                q = int_row_quant(bits)
                Hooks.wkv[(stream, layer)] = lambda s, cur, rows, t, q=q: q(s)
        if codecs is not None:
            for layer, role in info["shift_vectors"]:
                Hooks.shift[(stream, layer, role)] = lambda x, cur, c=codecs[stream], r=role: c.shift(x, r)
        elif shift_bits:
            q = int_row_quant(shift_bits)
            for layer, role in info["shift_vectors"]:
                Hooks.shift[(stream, layer, role)] = lambda x, cur, q=q: q(x)


def entity_bits(rec, info):
    if rec.exact_parts():
        return None
    return (info["wkv_layers"] * info["heads"] * rec.wkv_bits()
            + len(info["shift_vectors"]) * rec.shift_vector_bits())


def describe(name, layout, recs=None):
    recs = recs or recipe(name)
    out = {}
    for (stream, info), rec in zip(layout.items(), recs):
        out[{0: "card", 2: "note"}.get(stream, str(stream))] = {
            "bits": entity_bits(rec, info), "wkv_layers": info["wkv_layers"], "heads": info["heads"],
            "shift_vectors": len(info["shift_vectors"]), "rank": rec.rank, "stages": list(rec.stages),
            "index_bits": rec.index_bits, "norm_bits": rec.norm_bits, "shift": f"m{rec.shift_m}b{rec.shift_bits}"
            f"x{rec.shift_stages}", "shift_norm_bits": rec.shift_norm_bits}
        if rec.factor != "eigh":   # absent for the eigh recipes, so their scorer rows are unchanged
            out[{0: "card", 2: "note"}.get(stream, str(stream))]["factor"] = rec.factor
    return out


# ------------------------------------------------------------------------------------------------ QAT (SM32b)

DEFAULT_CODEBOOKS = Path(__file__).resolve().parents[1] / "artifacts" / "small-model" / "state-vq" / "codebooks.npz"
STATE_STREAMS = (0, 2)  # card, note (rwkv_modules order card, deck, note, preset, user)


def file_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


IMPLS = ("torch", "hybrid", "cuda")


class LearnedBooks(torch.nn.Module):
    """SM32c: the codec catalogs as trainable parameters (the outside research's learnable QAT catalogs: gradient
    into the picked centroids, picks frozen per step). One joint (u, v) WKV catalog `wkv` [S, M, 32] shared by the card
    and note streams (each uses its first `stages` stages), and per shift role a product catalog `shift[role]`
    [m, S_shift, M_shift, sub] shared by the streams. Norm grids stay fixed. Initialised from the k-means file."""

    def __init__(self, books, recs, device):
        super().__init__()
        ranks = {rec.rank for rec in recs}
        if ranks != {1}:
            raise ValueError("learned state_vq catalogs are rank 1 only")
        S = max(rec.stages[0] for rec in recs)
        self.wkv = torch.nn.Parameter(torch.stack([cb.float() for cb in books.wkv[0][:S]]).to(device))
        ms = {rec.shift_m for rec in recs}
        if len(ms) != 1:
            raise ValueError("learned shift catalogs need one shift_m for both streams")
        self.shift_m = ms.pop()
        S_shift = max(rec.shift_stages for rec in recs)
        self.shift = torch.nn.ParameterDict({
            role: torch.nn.Parameter(torch.stack([torch.stack([cb.float() for cb in pos[:S_shift]])
                                                  for pos in books.shift[(role, self.shift_m)]]).to(device))
            for role in ("t", "c")})
        self.source = books

    def export(self, path):
        """Write the current catalogs as a Codebooks .npz (the scorer's --vq-codebooks format): the learned stages
        replace the first stages of the source file, every other entry (more stages, other m, norm ranges) is kept."""
        src = self.source
        out = Codebooks(wkv=[list(stages) for stages in src.wkv], shift={k: [list(st) for st in v] for k, v in src.shift.items()},
                        wkv_norm=dict(src.wkv_norm), shift_norm=dict(src.shift_norm))
        with torch.no_grad():
            for s in range(self.wkv.shape[0]):
                out.wkv[0][s] = self.wkv[s].detach().float().cpu()
            for role, param in self.shift.items():
                for p in range(param.shape[0]):
                    for st in range(param.shape[1]):
                        out.shift[(role, self.shift_m)][p][st] = param[p, st].detach().float().cpu()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp.npz")
        out.save(tmp)
        tmp.replace(path)
        return file_sha256(path)


LEARNED = None   # SM32c: the LearnedBooks of the running training (install_training with "learn": true), else None


def training_spec(value):
    """The `state_vq` training option: a recipe name ("vq378x412") or {"recipe": ..., "codebooks": <.npz>,
    "sha256": <hex of the .npz, checked when given>, "factor_dtype": "float64" (default, = the CPU scorer) | "float32"
    (faster factorization; picks can differ from the scorer's on near-ties), "impl": "torch" (default: per-step
    PyTorch loop, explicit backward) | "hybrid" (torch forward, fused WKV backward kernel on CUDA) | "cuda" (fused
    forward and backward kernels on CUDA, scripts/state_vq_cuda.py; needs a factor "sq" recipe such as vq378x412p)}.
    On a CPU device every impl runs the torch path. Codebooks default to this checkout's SM32 k-means file."""
    spec = {"recipe": value} if isinstance(value, str) else dict(value)
    unknown = set(spec) - {"recipe", "codebooks", "sha256", "factor_dtype", "impl", "learn"}
    if unknown or "recipe" not in spec:
        raise ValueError(f"state_vq option needs a recipe; unknown keys {sorted(unknown)}")
    recs = recipe(spec["recipe"])
    if any(rec.exact_parts() for rec in recs):
        raise ValueError(f"state_vq recipe {spec['recipe']} has exact (unbudgeted) parts; QAT needs a finite budget")
    path = Path(spec.get("codebooks") or DEFAULT_CODEBOOKS)
    digest = file_sha256(path)
    if spec.get("sha256") and spec["sha256"] != digest:
        raise ValueError(f"state_vq codebooks {path} have sha256 {digest}, the option pins {spec['sha256']}")
    if spec.get("factor_dtype", "float64") not in ("float64", "float32"):
        raise ValueError(f"state_vq factor_dtype {spec['factor_dtype']!r}: float64 or float32")
    impl = spec.get("impl", "torch")
    if impl not in IMPLS:
        raise ValueError(f"state_vq impl {impl!r}: one of {IMPLS}")
    if impl == "cuda" and any(rec.factor != "sq" or rec.rank != 1 for rec in recs):
        raise ValueError(f"state_vq impl 'cuda' computes the Gram-squaring codec: use a factor 'sq' recipe "
                         f"({spec['recipe']}p), not {spec['recipe']}")
    out = {"recipe": spec["recipe"], "codebooks": str(path), "sha256": digest,
           "factor_dtype": spec.get("factor_dtype", "float64")}
    if impl != "torch":   # the key is absent for the default, so earlier option dicts describe themselves unchanged
        out["impl"] = impl
    if spec.get("learn"):   # SM32c; absent = frozen catalogs (SM32b dicts describe themselves unchanged)
        if any(rec.rank != 1 for rec in recs):
            raise ValueError("state_vq learn: rank-1 recipes only")
        out["learn"] = True
    return out


def training_layout(model):
    """{stream: {"wkv_layers", "heads", "shift_vectors", "time": [mixers], "channel": [mixers]}} for the card/note
    streams of an expertium model (scripted or eager); channel mixers without a shift (stripped) are left out."""
    layout = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) != 5 or parts[0] != "rwkv_modules" or parts[2] != "blocks" or int(parts[1]) not in STATE_STREAMS:
            continue
        if not hasattr(module, "state_vq_slot"):
            continue
        stream, layer = int(parts[1]), int(parts[3])
        info = layout.setdefault(stream, {"wkv_layers": 0, "heads": 0, "shift_vectors": [], "time": [], "channel": []})
        if parts[4] == "time_mixer":
            info["wkv_layers"] += 1
            info["heads"] = int(module.H)
            info["shift_vectors"].append((layer, "t"))
            info["time"].append(module)
        elif parts[4] == "channel_mixer":
            info["shift_vectors"].append((layer, "c"))
            info["channel"].append(module)
    for info in layout.values():
        info["shift_vectors"].sort(key=lambda item: (item[0], item[1] != "t"))
    return {s: layout[s] for s in STATE_STREAMS}


def install_training(model, value, device, wkv_chunk=1 << 20, shift_chunk=8192):
    """Route the card/note mixers of `model` through the QAT hooks (expertium/model/state_vq_hooks.py) with the
    recipe's codecs on `device`. Returns the description written into the training result."""
    global LEARNED
    from expertium.model import state_vq_hooks as hooks
    spec = training_spec(value)
    books = Codebooks.load(spec["codebooks"])
    layout = training_layout(model)
    hooks.CODECS.clear()
    impl = spec.get("impl", "torch") if torch.device(device).type == "cuda" else "torch"
    LEARNED = LearnedBooks(books, recipe(spec["recipe"]), device) if spec.get("learn") else None
    for (stream, info), rec in zip(layout.items(), recipe(spec["recipe"])):
        codec = StateCodec(rec, books, wkv_chunk, shift_chunk, getattr(torch, spec["factor_dtype"])).to(device)
        attach_kernel(codec, impl, device)
        codec.learned = LEARNED
        codec.sync()
        hooks.CODECS[stream] = codec
        for module in info["time"] + info["channel"]:
            if getattr(module, "fused_shift", False):
                raise ValueError("state_vq cannot run with fused_shift (the fused gather+lerp has no shift hook)")
            module.state_vq_slot = stream
    plain = {s: {k: v for k, v in info.items() if k in ("wkv_layers", "heads", "shift_vectors")}
             for s, info in layout.items()}
    return {**spec, "impl_on_device": impl, "streams": describe(spec["recipe"], plain),
            "codebook_handling": ("learned (SM32c: gradient into the picked centroids, picks frozen per step; AdamW "
                                  "group at the schedule LR, weight decay 0; norm grids fixed)" if LEARNED is not None
                                  else "frozen (k-means, no gradient)"),
            "mixers": {str(s): len(info["time"]) + len(info["channel"]) for s, info in layout.items()}}


def attach_kernel(codec, impl, device):
    """impl "cuda" / "hybrid": give `codec` the fused-kernel tensors (scripts/state_vq_cuda.KernelCodec) that
    expertium/model/state_vq_hooks.py dispatches on; "torch" leaves it a plain codec. Returns the codec."""
    codec.impl = impl
    if impl in ("cuda", "hybrid"):
        from scripts import state_vq_cuda
        # hybrid: the torch forward (any recipe) with the fused backward only
        codec.kernel = (state_vq_cuda.KernelCodec(codec, torch.device(device), codec.factor_dtype) if impl == "cuda"
                        else state_vq_cuda.BackwardKernel())
    return codec


def uninstall_training(model):
    global LEARNED
    LEARNED = None
    from expertium.model import state_vq_hooks as hooks
    for info in training_layout(model).values():
        for module in info["time"] + info["channel"]:
            module.state_vq_slot = -1
    hooks.CODECS.clear()


# ------------------------------------------------------------------------------------------------ GPU scoring (SM44)
#
# The held-out scorer's `+vq<recipe>` variants on CUDA (scripts/small_model_heldout.py). Same placement as the CPU
# Hooks path above: per card/note review the answer uses the float update, the STORED state is the codec's
# reconstruction before that entity's next review; a review that reads its entity's previous review through the token
# shift reads the coded vector; deck / preset / user state stays float (the scorer's long WKV kernel). The eager model's
# own state-VQ slot (expertium/model/rwkv_model.py: `state_vq_slot >= 0` -> `vq_wkv` / `vq_shift`) routes the card/note
# mixers here: install_scoring() points those two module globals of rwkv_model at score_wkv / score_shift (forward-only:
# no straight-through Function, no per-step state record, so whole histories fit).
#   WKV, rank-1 stream (every card recipe, the 412-bit note): the UNCHANGED fused kernel scripts/kernels/
#     state_vq_wkv.cu (scripts/state_vq_cuda.KernelCodec), i.e. the Gram-squaring factorization (`...p`, factorize_sq)
#     in `factor_dtype` (float64 by default, the CPU scorer's type). On the SM32 state corpus it stores bitwise the
#     same states as the deployed eigh codec in float64 (docs/SMALL_MODEL.md SM32b, Speed). The kernel writes the state
#     before every step, which the scorer does not need; rows of a split are independent sequences, so a split is run
#     in row blocks of at most SCORE_KERNEL_POSITIONS (rows x steps) to bound that buffer.
#   WKV, rank-2 stream (the 828-bit note): no fused kernel; the per-step torch loop of _reference_wkv_q on the device
#     with the recipe's own (eigh, float64) codec.
#   Token shift: StateCodec.shift on the device (any recipe).
# On a CPU tensor (the CPU-first check, scripts/check_state_vq_score_gpu.py) the kernel streams run the same code as
# the GPU through SCORE_EMULATE (scripts/state_vq_cuda_emu.forward) when set, else the torch loop with the same "sq"
# codec (the kernel's arithmetic in torch).

class _ThreadSlots:
    """stream -> codec, one table per thread (SM48b: --lanes N runs a model copy per lane thread, and a lane scoring an
    int variant must not see or clear another lane's codecs). Supports what this module uses: [], clear, update."""

    def __init__(self):
        self._local = threading.local()

    @property
    def _table(self):
        table = getattr(self._local, "table", None)
        if table is None:
            table = self._local.table = {}
        return table

    def __getitem__(self, slot):
        return self._table[slot]

    def clear(self):
        self._table.clear()

    def update(self, codecs):
        self._table.update(codecs)


SCORE = _ThreadSlots()            # stream -> StateCodec with .score_kernel (KernelCodec or None), per thread
SCORE_EMULATE = None              # check only: fn(kernel_codec, r, k, v, w, a, kd, skip) -> out, for CPU tensors
SCORE_KERNEL_POSITIONS = 1 << 17  # rows x steps per kernel launch (the kernel's state record: 4 KB per position)
SCORE_TORCH_CHUNK = 1 << 15       # SM46: CUDA torch-loop streams: rows per nearest-centroid distance matrix (128 MB);
                                  # rows are independent, so only the launch count changes (the CPU keeps 4096)


def score_codecs(name, books, device, factor_dtype=torch.float64, kernel=True):
    """{stream: StateCodec} for GPU scoring of recipe `name`. Rank-1 streams with coded directions and quantized norms
    get the fused kernel (their recipe with factor "sq"); others keep the recipe's codec (torch loop)."""
    device = torch.device(device)
    out = {}
    for stream, rec in zip(STATE_STREAMS, recipe(name)):
        fused = kernel and rec.rank == 1 and bool(rec.stages[0]) and rec.norm_bits is not None
        if fused:
            rec = replace(rec, factor="sq")
        chunk = SCORE_TORCH_CHUNK if device.type == "cuda" and not fused else 4096
        codec = StateCodec(rec, books, wkv_chunk=chunk, shift_chunk=chunk, factor_dtype=factor_dtype).to(device)
        codec.score_kernel = None
        if fused:
            from scripts import state_vq_cuda
            codec.score_kernel = state_vq_cuda.KernelCodec(codec, device, factor_dtype)
        out[stream] = codec
    return out


class IntScoreCodec:
    """SM47 GPU scoring of the int state variants through the state-VQ slots: WKV = scripts/small_model_wkv.wkv_long
    with `wkv_bits` (the int kernel on CUDA, its float32 reference on CPU; | F16_SCALE = float16 row scales), token
    shift = int_row_quant(shift_bits) per stored vector (0 = float)."""
    score_kernel = None

    def __init__(self, wkv_bits, shift_bits=0):
        self.wkv_bits = wkv_bits
        self.shift_bits = shift_bits
        self._shift = int_row_quant(shift_bits) if shift_bits else None

    def shift(self, x, role):
        return self._shift(x) if self._shift is not None else x


def int_score_codecs(wkv_bits, shift_bits=0):
    """{stream: IntScoreCodec}; wkv_bits an int (both streams) or a (card, note) tuple."""
    per = dict(zip(STATE_STREAMS, wkv_bits)) if isinstance(wkv_bits, tuple) else dict.fromkeys(STATE_STREAMS, wkv_bits)
    return {s: IntScoreCodec(per[s], shift_bits) for s in STATE_STREAMS}


def score_path(codecs):
    """Row metadata: which computation each stream's WKV codec ran."""
    names = {0: "card", 2: "note"}
    if all(isinstance(c, IntScoreCodec) for c in codecs.values()):
        return {names[s]: f"int kernel bits {c.wkv_bits}, shift bits {c.shift_bits}" for s, c in codecs.items()}
    return {names[s]: (f"fused kernel, Gram squaring {str(c.factor_dtype).replace('torch.', '')}"
                       if c.score_kernel is not None else
                       f"torch step loop, {c.rec.factor} {str(c.factor_dtype).replace('torch.', '')}")
            for s, c in codecs.items()}


def _score_kernel(kc, r, k, v, w, a, kd, skip):
    B, T = skip.shape
    if not r.is_cuda:
        return SCORE_EMULATE(kc, r, k, v, w, a, kd, skip)
    rows = max(1, SCORE_KERNEL_POSITIONS // max(T, 1))
    if B <= rows:
        return kc.forward(r, k, v, w, a, kd, skip)[0]
    out = torch.empty_like(r)
    for i in range(0, B, rows):
        part = slice(i, i + rows)
        out[part] = kc.forward(*(x[part].contiguous() for x in (r, k, v, w, a, kd)), skip[part].contiguous())[0]
    return out


def score_wkv(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, slot):
    """Forward-only card/note WKV with the stored state coded after every non-skipped step (scoring)."""
    codec = SCORE[slot]
    out_dtype = k_BTHK.dtype
    if isinstance(codec, IntScoreCodec):   # SM47: the int kernel (scripts/small_model_wkv), any device
        from scripts.small_model_wkv import wkv_long
        return wkv_long(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, codec.wkv_bits).to(out_dtype)
    r, k, v, w, a, kd = (x.float().contiguous() for x in (r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK))
    skip = skip_BT.bool().contiguous()
    kc = codec.score_kernel
    if kc is not None and (r.is_cuda or SCORE_EMULATE is not None):
        return _score_kernel(kc, r, k, v, w, a, kd, skip).to(out_dtype)
    from expertium.model.rwkv_ops import single_timestep
    B, T, H, Kh = r.shape
    out = torch.empty(B, T, H, Kh, dtype=torch.float32, device=r.device)
    state = torch.zeros(B, H, Kh, Kh, dtype=torch.float32, device=r.device)
    for t in range(T):   # _reference_wkv_q on the device
        out[:, t], nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        rows = torch.nonzero(~skip[:, t]).flatten()
        if rows.numel():
            state = state.clone()
            state[rows] = codec.wkv(nxt[rows])
    return out.to(out_dtype)


def score_shift(x_shift_BTC, time_shift_select_BT, slot, role):
    """Token shift for scoring: vectors read from the entity's previous review (select != t) are coded (_TorchProxy)."""
    T = x_shift_BTC.shape[1]
    stored = time_shift_select_BT.long() != torch.arange(T, device=x_shift_BTC.device)[None, :]
    if not bool(stored.any()):
        return x_shift_BTC
    out = x_shift_BTC.clone()
    out[stored] = SCORE[slot].shift(x_shift_BTC[stored].float(), "t" if role == 0 else "c").to(out.dtype)
    return out


def install_scoring(model):
    """Point the eager model's state-VQ slot functions at the scoring path; returns the card/note layout (with the
    mixer modules) for set_scoring and describe."""
    from expertium.model import rwkv_model as erm
    erm.vq_wkv, erm.vq_shift = score_wkv, score_shift
    layout = training_layout(model)
    for info in layout.values():
        for module in info["time"] + info["channel"]:
            if getattr(module, "fused_shift", False):
                raise ValueError("GPU +vq scoring cannot run with fused_shift (no shift hook)")
    return layout


def set_scoring(layout, codecs=None):
    """Turn the card/note slots on with `codecs` ({stream: StateCodec} from score_codecs) or off (None)."""
    SCORE.clear()
    if codecs:
        SCORE.update(codecs)
    for stream, info in layout.items():
        for module in info["time"] + info["channel"]:
            module.state_vq_slot = stream if codecs else -1
