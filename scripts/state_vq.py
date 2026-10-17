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


def recipe(name):
    if name not in RECIPES:
        raise SystemExit(f"unknown VQ recipe {name}; known: {sorted(RECIPES)}")
    return RECIPES[name]


# ------------------------------------------------------------------------------------------------ quantizers

def nearest(x, centroids, cn2=None, chunk=4096):
    """Index of the nearest centroid (squared L2) for every row of x [N, D]; centroids [M, D]."""
    if cn2 is None:
        cn2 = (centroids * centroids).sum(-1)
    out = torch.empty(x.shape[0], dtype=torch.long)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        out[i:i + chunk] = (cn2[None, :] - 2.0 * (xb @ centroids.T)).argmin(-1)
    return out


def rvq(x, books, norms2=None):
    """Residual VQ reconstruction of x [N, D] with stage codebooks books[s] [M, D]."""
    acc = torch.zeros_like(x)
    for s, cb in enumerate(books):
        idx = nearest(x - acc, cb, None if norms2 is None else norms2[s])
        acc = acc + cb[idx]
    return acc


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

def factorize(A, rank):
    """Top-`rank` factors of A [N, K, K] (float64 inside): unit u, unit v, sigma, valid mask, each [N, rank, (K)].
    Same construction as model.rs (Gram of the max-abs normalised matrix, v = A^T u / sigma), plus his sign canon."""
    A64 = A.double()
    amax = A64.abs().amax(dim=(1, 2))
    scale = torch.where(torch.isfinite(amax) & (amax > 1e-30), amax, torch.ones_like(amax))
    an = A64 / scale[:, None, None]
    evals, evecs = torch.linalg.eigh(an @ an.transpose(1, 2))       # ascending
    order = torch.arange(K - 1, K - 1 - rank, -1)
    ev = evals[:, order]                                             # [N, r]
    u = evecs[:, :, order].transpose(1, 2)                           # [N, r, K]
    v_un = torch.einsum("nij,nri->nrj", A64, u)                      # (A^T u) per component
    sigma = v_un.norm(dim=-1)
    valid = torch.isfinite(ev) & (ev > 0) & torch.isfinite(sigma) & (sigma > 1e-20)
    v = v_un / torch.where(valid, sigma, torch.ones_like(sigma))[..., None]
    lead = u.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(torch.gather(u, -1, lead) >= 0, 1.0, -1.0).to(u.dtype)
    return u * sign, v * sign, sigma, valid


class StateCodec:
    """Encode-decode of one stream's stored state (WKV matrices and shift vectors) for a StreamRecipe."""

    def __init__(self, rec, books):
        self.rec = rec
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

    def wkv(self, S):
        """S [N, H, K, K] -> reconstruction (float32)."""
        if self.rec.rank == 0:
            return S
        N, H = S.shape[:2]
        A = S.reshape(N * H, K, K)
        u, v, sigma, valid = factorize(A, self.rec.rank)
        out = torch.zeros(N * H, K, K, dtype=torch.float64)
        for j in range(self.rec.rank):
            uj, vj = u[:, j], v[:, j]
            if self.wkv_books[j]:
                q = torch.cat([uj, vj], dim=-1).float()
                rec = rvq(q, self.wkv_books[j], self.wkv_n2[j]).double()
                uj, vj = rec[:, :K], rec[:, K:]
            n = torch.sqrt(torch.where(valid[:, j], sigma[:, j], torch.ones_like(sigma[:, j])))
            nq = self.wkv_nq[j](n.float()).double() if self.rec.norm_bits is not None else n
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
                                  self.shift_n2[role][p]) for p in range(m)], dim=-1)
        rec = unit * self.shift_nq[role](n1)[:, None]
        return torch.where(valid[:, None], rec, x)


def int_row_quant(bits):
    """The fork's int8/int4 per-row WKV state rounding (`+s8`/`+s4`), as scripts/small_model_wkv.reference."""
    qmax = 127.0 if bits == 8 else 7.0

    def quant(nxt):
        scale = nxt.abs().amax(-1, keepdim=True) / qmax
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


def configure(layout, codecs=None, int_bits=0):
    """Set the active quantizers. codecs: {stream: StateCodec} (VQ) or None; int_bits 8/4 = `+s8`/`+s4` on CPU."""
    Hooks.wkv, Hooks.shift = {}, {}
    for stream, info in layout.items():
        for layer in range(info["wkv_layers"]):
            if codecs is not None:
                codec = codecs[stream]
                Hooks.wkv[(stream, layer)] = lambda s, cur, rows, t, c=codec: c.wkv(s)
            elif int_bits:
                q = int_row_quant(int_bits)
                Hooks.wkv[(stream, layer)] = lambda s, cur, rows, t, q=q: q(s)
        if codecs is not None:
            for layer, role in info["shift_vectors"]:
                Hooks.shift[(stream, layer, role)] = lambda x, cur, c=codecs[stream], r=role: c.shift(x, r)


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
    return out
