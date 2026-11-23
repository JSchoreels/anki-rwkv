"""SM155: bounded-memory evaluation for scripts/small_model_heldout.py --device cpu --engine stream.

The stock CPU scorer evaluates a user's whole history in one forward pass: every stream module runs on [B, T] blocks
that together hold about twice the user's review count, so its peak memory grows with the reviews (about 3.2 GB at
102,545 reviews, 5.6 GB at 150,500; the long-card state drift check (SM136)). This engine replays the same history in
time chunks of `chunk_rows` canonical rows (the rows of features2card, i.e. the review order) and carries, between
chunks, only the per-entity recurrent state of every stream and layer: the WKV matrix and the two token-shift vectors
(time mixer, channel mixer). Memory is then set by the number of entities (cards, notes, decks, presets, the user)
and the chunk size, not by the review count.

Nothing of the model or the state codecs is re-implemented. Per chunk, for every stream (fork models: the stream
module, all its layers in one call; interleaved expertium models: one block per layer step, in the model's own
`layer_steps` order), the model's own modules run on [B, T] blocks of that chunk's rows, grouped by entity. Two module
globals are swapped for the duration of an evaluation, both in the module the model already routes through:

* the WKV reference (`reference_rwkv7_q` of the patched fork module, `reference_rwkv7` of expertium.model.rwkv_model,
  which the scorer already points at scripts/state_vq._reference_wkv_q): `CarryWKV` runs the same step loop
  (single_timestep, the answer's output from the float update, the stored state = the active codec's reconstruction)
  from each entity's carried state and update count, with the label's own codec callables (compress_quant's
  _quantize_rows / _quantize_tier, state_vq.Hooks.wkv entries, state_lowrank.LowRankCodec.encode, SM147 truncate);
* `torch` (the token-shift gather): `CarryTorch.gather` puts an entity's carried shift vector where the shift source
  lies in an earlier chunk, records the vectors later chunks read, and (expertium) applies state_vq.Hooks.shift to the
  stored positions exactly like state_vq._TorchProxy (the fork model rounds its shift vectors itself).

The heads and the loss are the model's own: per chunk the instance's forward_batch(_variant) is replaced by a function
returning the head outputs of that chunk's rows (head_and_out(_variant)), then model.get_loss runs on the chunk's
labels; the per-row statistics are concatenated and the totals recombined as heldout_fast_metrics.evaluate_with_arrays
does. Predictions per review come out of the same arithmetic as the stock path; differences can only come from
row-position-dependent rounding inside CPU kernels (BLAS blocking for very small output widths, vectorised tails), so the
exactness rule is checked, not assumed (artifacts/small-model/sm155/design.md).

Entity state storage (exact, verified at every write, with a float fallback per entity): card and note WKV states of an
integer-row format are kept as int8 codes + one float32 scale per row (codes x scale == stored state, bit for bit);
low-rank formats (state_lowrank.LowRankCodec) keep the factor pair (left @ right == stored state, bit for bit) plus the
left factors the codec reads back; anything that fails the check, and every float-state format, is kept as float32.
CPU only, PYTORCH_JIT=0 for fork models (eager modules), one sample per batch (full-history scoring).
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

STATE_STREAMS = (0, 2)   # card, note
ALIGN = 64               # chunk_rows multiple of this (vectorised-kernel blocks start at the same row offsets)
LAST = {}                # statistics of the last evaluate() call (the scorer adds them to the row)


# ------------------------------------------------------------------------------------------------ batch layout

SPLIT_ROWS = (1024, 2048, 4096, 8192)   # 32768 / w for the trunk's elementwise widths w = 32, 16, 8, 4: above it
#                                         ATen splits that op over 2 threads (at::internal::GRAIN_SIZE = 32768 elements)
EDGE = 32                                 # rows around a block's end and its thread split treated as special


class Layout:
    """Per stream, per canonical row: entity (sequence) id, position in the entity's sequence, shift source position
    (prepare()'s time_shift_select: the last earlier non-skipped position, else 0), skip flag and the row's flat
    position in its stock [B, T] split block; per row the block's size M = B * T."""

    def __init__(self, batch):
        if batch.num_data != 1:
            raise ValueError("stream engine: one sample per batch (full-history scoring)")
        n = int(batch.start.shape[0])
        self.n = n
        cur = torch.arange(n, dtype=torch.long)
        self.streams = []
        for i, splits in enumerate(batch.sub_gather):
            ent = np.full(n, -1, dtype=np.int64)
            pos = np.zeros(n, dtype=np.int64)
            src = np.zeros(n, dtype=np.int64)
            skip = np.zeros(n, dtype=bool)
            flat = np.zeros(n, dtype=np.int64)
            block = np.zeros(n, dtype=np.int64)
            parts, offset = [], 0
            for s, split in enumerate(splits):
                index = split.long()
                canonical = torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1))
                parts.append(canonical)
                length = batch.sub_gather_lens[i][s]
                can = canonical.view(-1, length).numpy()
                sel = batch.time_shift_selects[i][s].view(-1, length).numpy()
                skp = batch.skips[i][s].view(-1, length).numpy().astype(bool)
                real = can >= 0
                b, t = np.nonzero(real)
                rows = can[real]
                if (ent[rows] >= 0).any():
                    raise ValueError(f"stream {i}: a canonical row appears twice")
                ent[rows], pos[rows], src[rows], skip[rows] = offset + b, t, sel[real], skp[real]
                flat[rows] = b * length + t
                block[rows] = can.size
                offset += can.shape[0]
            cur = torch.cat(parts)
            if (ent < 0).any():
                raise ValueError(f"stream {i}: canonical rows missing")
            order = np.lexsort((pos, ent))
            same = ent[order][1:] == ent[order][:-1]
            if not (np.diff(order)[same] > 0).all() or not (np.diff(pos[order])[same] == 1).all():
                raise ValueError(f"stream {i}: an entity's rows are not in canonical (time) order")
            self.streams.append({"ent": ent, "pos": pos, "src": src, "skip": skip, "flat": flat, "block": block,
                                 "entities": offset})

    @staticmethod
    def _replica_size(M, need):
        """Smallest M' <= M with M' = M (mod 64), M' >= need and M' on the same side of every SPLIT_ROWS threshold
        as M (so every elementwise op splits over the threads, or not, as in the stock block)."""
        side = [M > x for x in SPLIT_ROWS]
        floor = max([need, 1] + [x + 1 for x, on in zip(SPLIT_ROWS, side) if on])
        if floor >= M:
            return M
        Mp = M - ((M - floor) // 64) * 64
        while [Mp > x for x in SPLIT_ROWS] != side or Mp < need:
            Mp += 64
        return min(Mp, M)

    def groups(self, i, a, b):
        """[B, T] blocks of stream i for canonical rows [a, b).

        Row placement reproduces the arithmetic the stock pass's CPU kernels give each row. Two kernel facts decide it
        (artifacts/small-model/sm155/design.md): MKL's GEMMs with 3 to 8 outputs (k / v scale, value LoRA, rating
        head) pair rows, so a row's result depends on its position parity in the block and on an odd last row; ATen's
        vectorised elementwise kernels (sigmoid, tanh on 4- to 16-wide tensors) run the last elements of a block, and
        of each thread's half when a block has more than 32768 elements, through the scalar path. So:
        * normal rows (farther than EDGE rows from their stock block's end and from its middle) go into blocks of
          width T = a power of two >= 2, each entity starting at offset o in {0, 1} so its rows keep their stock
          parity, B padded with dummy all-skip sequences until B * T is a multiple of 16 (no tail, no split remainder);
        * an entity whose in-chunk rows come within EDGE rows of its stock block's end or middle gets its own [1, M']
          replica block: M' = M (mod 64), the same side of every thread-split threshold, and its rows at the same
          distance from the end (or from the middle) as in the stock block, so every kernel treats them alike."""
        st = self.streams[i]
        rows = np.arange(a, b)
        ent, pos = st["ent"][rows], st["pos"][rows]
        order = np.lexsort((pos, ent))
        rows, ent, pos = rows[order], ent[order], pos[order]
        uniq, first, counts = np.unique(ent, return_index=True, return_counts=True)
        t0 = pos[first]
        f0 = st["flat"][rows[first]]
        f1 = f0 + counts
        M = st["block"][rows[first]]
        near_end = f1 > M - EDGE
        near_mid = (M > SPLIT_ROWS[0]) & (f1 > M // 2 - EDGE) & (f0 < M // 2 + EDGE)
        special = near_end | near_mid
        out = []
        normal = np.flatnonzero(~special)
        lead = f0 % 2
        width = np.maximum(2, 1 << np.ceil(np.log2(counts + lead)).astype(np.int64))
        for T in np.unique(width[normal]):
            T = int(T)
            members = normal[width[normal] == T]
            mult = max(1, 16 // T)
            B = -(-len(members) // mult) * mult
            out.append(self._group(st, a, rows, first, counts, uniq, t0, members, B, T, lead[members]))
        for m in np.flatnonzero(special):
            Mb, lo, hi = int(M[m]), int(f0[m]), int(f1[m])
            if near_end[m] and near_mid[m]:
                Mp, shift = Mb, 0
            elif near_end[m]:   # same distance from the end; rows clear of the replica's own middle
                Mp = self._replica_size(Mb, 2 * (Mb - lo) + 2 * EDGE)
                shift = Mb - Mp
            else:               # same distance from the middle; rows clear of the replica's end
                Mp = self._replica_size(Mb, 2 * max(Mb // 2 - lo, hi - Mb // 2) + 2 * EDGE + 2)
                shift = Mb // 2 - Mp // 2
            if Mp != Mb and (lo - shift < 0 or hi - shift > Mp):
                Mp, shift = Mb, 0
            out.append(self._group(st, a, rows, first, counts, uniq, t0, np.array([m]), 1, Mp,
                                   np.array([lo - shift]), replica=True))
        return out

    @staticmethod
    def _group(st, a, rows, first, counts, uniq, t0, members, B, T, o, replica=False):
        nb = len(members)
        c = counts[members]
        seq = np.repeat(np.arange(nb), c)
        j = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        r = rows[np.repeat(first[members], c) + j]
        col = j + o[seq]
        g = Group()
        g.replica = replica
        local = np.full((B, T), -1, dtype=np.int64)
        local[seq, col] = r - a
        skip = np.ones((B, T), dtype=bool)
        skip[seq, col] = st["skip"][r]
        src, p = st["src"][r], st["pos"][r]
        tstart = t0[members][seq]
        sel = np.zeros((B, T), dtype=np.int64)
        sel[seq, col] = np.where(p == 0, col, np.where(src >= tstart, src - tstart + o[seq], col))
        carried = np.zeros((B, T), dtype=bool)
        carried[seq, col] = (p > 0) & (src < tstart)
        ents = np.full(B, -1, dtype=np.int64)
        ents[:nb] = uniq[members]
        offs = np.zeros(B, dtype=np.int64)
        offs[:nb] = o
        nonskip = ~skip
        last = np.where(nonskip.any(1), T - 1 - np.argmax(nonskip[:, ::-1], 1), -1)
        g.T, g.B = T, B
        g.gather = torch.from_numpy(np.maximum(local, 0).reshape(-1))
        flat = local.reshape(-1)
        g.positions = torch.from_numpy(np.flatnonzero(flat >= 0))
        g.targets = torch.from_numpy(flat[flat >= 0])
        g.steps = np.flatnonzero((local >= 0).any(0)).tolist()   # WKV time steps with a real row
        g.sel = torch.from_numpy(sel)
        g.skip = torch.from_numpy(skip)
        g.carried = torch.from_numpy(carried)
        cb, _ = np.nonzero(carried)
        g.carried_ent = torch.from_numpy(ents[cb])
        g.any_carried = bool(carried.any())
        g.ents = ents
        written = (ents >= 0) & (last >= 0)
        g.w_seq = torch.from_numpy(np.flatnonzero(written))
        g.w_ent = torch.from_numpy(ents[written])
        g.w_pos = torch.from_numpy(last[written])
        first_only = (ents >= 0) & (last < 0) & np.concatenate([t0[members] == 0, np.zeros(B - nb, dtype=bool)])
        g.f_seq = torch.from_numpy(np.flatnonzero(first_only))
        g.f_ent = torch.from_numpy(ents[first_only])
        g.f_pos = torch.from_numpy(offs[first_only])
        return g


class Group:
    pass


# ------------------------------------------------------------------------------------------------ entity state tables

def _pack_rows(X):
    """X [R, K] float32 rows -> (codes int8 [R, K], scale float32 [R], ok [R]) with codes * scale == X bit for bit.
    Tried per row: max |x| / q for q = 127, 7, 3, 1 (every int-row codec of the scorer), one float32 ulp either side
    (float32 row scales), then the exact common divisor of the row (a float16 subnormal scale, a per-head scale)."""
    R = X.shape[0]
    codes = torch.zeros(X.shape, dtype=torch.int8)
    scale_out = torch.zeros(R, dtype=torch.float32)
    ok = torch.zeros(R, dtype=torch.bool)
    todo = torch.nonzero(torch.isfinite(X).all(1)).flatten()
    for q in (127.0, 7.0, 3.0, 1.0):
        for nudge in (0, 1, -1):
            if not todo.numel():
                return codes, scale_out, ok
            Y = X[todo]
            scale = Y.abs().amax(-1, keepdim=True) / q
            if nudge:
                scale = torch.nextafter(scale, torch.full_like(scale, math.inf if nudge > 0 else 0.0))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            c = torch.round(Y / scale)
            good = (c.abs() <= 127).all(1) & ((c * scale) == Y).all(1)
            if good.any():
                idx = todo[good]
                codes[idx] = c[good].to(torch.int8)
                scale_out[idx] = scale[good].squeeze(-1)
                ok[idx] = True
                todo = todo[~good]
    if not todo.numel():
        return codes, scale_out, ok
    # exact common divisor: x = mantissa * 2^e (24-bit integer mantissa), integers on the row's finest grid, gcd
    Y = X[todo]
    mant, ex = torch.frexp(Y)
    ints = (mant * 16777216.0).to(torch.int64)
    e = (ex - 24).to(torch.int64)
    nz = ints != 0
    emin = torch.where(nz, e, torch.full_like(e, 1 << 20)).amin(1, keepdim=True)
    emin = torch.where(emin == (1 << 20), torch.zeros_like(emin), emin)
    span = torch.where(nz, e - emin, torch.zeros_like(e))
    fits = (span <= 38).all(1)
    ints = torch.where(nz & (span <= 38), ints * (2 ** span.clamp(max=38)), torch.zeros_like(ints))
    g = ints[:, 0].abs()
    for k in range(1, ints.shape[1]):
        g = torch.gcd(g, ints[:, k])
    g = torch.where(g > 0, g, torch.ones_like(g))
    c = ints // g[:, None]
    scale = (g.double() * torch.pow(2.0, emin.squeeze(1).double())).float()
    good = fits & (g < (1 << 24)) & (c.abs() <= 127).all(1) & ((c.float() * scale[:, None]) == Y).all(1)
    if good.any():
        idx = todo[good]
        codes[idx] = c[good].to(torch.int8)
        scale_out[idx] = scale[good]
        ok[idx] = True
    return codes, scale_out, ok


def _pack_int8(S):
    """S [N, H, K, K] float32 -> (codes int8, row scales float32 [N, H, K], ok [N]): codes * scale == S bit for bit
    (every row of the entity packed)."""
    N, H, K, _ = S.shape
    codes, scales, ok = _pack_rows(S.reshape(-1, K))
    return codes.view(S.shape), scales.view(N, H, K), ok.view(N, H * K).all(1)


class WkvTable:
    """Carried WKV state of every entity of one (stream, layer): kind 0 = never written (zeros), 1 = low-rank factors,
    2 = int8 codes + row scales, 3 = float32. Also the update counts and (low-rank) the codec's stored left factors."""

    def __init__(self, entities, H, K, mode, rmax=0):
        self.E, self.H, self.K, self.mode, self.rmax = entities, H, K, mode, rmax
        self.kind = torch.zeros(entities, dtype=torch.uint8)
        self.count = torch.zeros(entities, dtype=torch.int64)
        self.dense = torch.zeros(entities, H, K, K) if mode == "float" else None
        self.codes = self.scales = None
        self.float = {}
        if mode == "lowrank":
            self.left = torch.zeros(entities, H, K, rmax)
            self.right = torch.zeros(entities, H, rmax, K)
        self.fallback = 0

    def nbytes(self):
        n = self.kind.numel() + self.count.numel() * 8 + sum(v.numel() * 4 for v in self.float.values())
        for t in (self.dense, self.codes, self.scales, getattr(self, "left", None), getattr(self, "right", None)):
            if t is not None:
                n += t.numel() * t.element_size()
        return n

    def load(self, ents):
        """ents [B] (-1 = dummy) -> state [B, H, K, K], counts [B], left [B, H, K, rmax]."""
        B = len(ents)
        state = torch.zeros(B, self.H, self.K, self.K)
        count = torch.zeros(B, dtype=torch.int64)
        left = torch.zeros(B, self.H, self.K, self.rmax) if self.mode == "lowrank" else None
        real = np.flatnonzero(ents >= 0)
        if not len(real):
            return state, count, left
        e = torch.from_numpy(ents[real])
        rs = torch.from_numpy(real)
        count[rs] = self.count[e]
        if self.mode == "float":
            state[rs] = self.dense[e]
            return state, count, left
        if left is not None:
            left[rs] = self.left[e]
        kind = self.kind[e]
        k1 = kind == 1
        if k1.any():
            state[rs[k1]] = self.left[e[k1]] @ self.right[e[k1]]
        k2 = kind == 2
        if k2.any():
            state[rs[k2]] = self.codes[e[k2]].float() * self.scales[e[k2]].unsqueeze(-1)
        for b, ent in zip(rs[kind == 3].tolist(), e[kind == 3].tolist()):
            state[b] = self.float[ent]
        return state, count, left

    def save(self, ents, seqs, state, count, left=None, right=None):
        """Store the final state of sequences `seqs` (entities `ents`, tensors indexed by sequence)."""
        if not len(seqs):
            return
        S = state[seqs]
        self.count[ents] = count[seqs]
        if self.mode == "float":
            self.dense[ents] = S
            return
        done = torch.zeros(len(seqs), dtype=torch.bool)
        if self.mode == "lowrank":
            L, R = left[seqs], right[seqs]
            self.left[ents] = L
            self.right[ents] = R
            prod_ok = (torch.isfinite(S).flatten(1).all(1)) & ((L @ R) == S).flatten(1).all(1)
            self.kind[ents[prod_ok]] = 1
            done |= prod_ok
        rest = torch.nonzero(~done).flatten()
        if rest.numel():
            codes, scales, ok = _pack_int8(S[rest])
            if ok.any():
                if self.codes is None:
                    self.codes = torch.zeros(self.E, self.H, self.K, self.K, dtype=torch.int8)
                    self.scales = torch.zeros(self.E, self.H, self.K)
                e2 = ents[rest[ok]]
                self.codes[e2] = codes[ok]
                self.scales[e2] = scales[ok]
                self.kind[e2] = 2
            for b in rest[~ok].tolist():
                ent = int(ents[b])
                self.float[ent] = S[b].clone()
                self.kind[ent] = 3
                self.fallback += 1


class ShiftTable:
    """Carried token-shift vector (the layer-normed input of the shift source row) of every entity of one
    (stream, layer, role)."""

    def __init__(self, entities, C):
        self.vec = torch.zeros(entities, C)
        self.has = torch.zeros(entities, dtype=torch.bool)

    def nbytes(self):
        return self.vec.numel() * 4 + self.has.numel()


# ------------------------------------------------------------------------------------------------ carry hooks

class Carry:
    """State of the running evaluation (one at a time, process-global like state_vq.Hooks)."""
    active = False
    kind = None          # "fork" | "expertium"
    current = None       # (stream, layer, role) of the running mixer (forward pre-hooks)
    group = None         # Group of the running block call
    wkv = {}             # (stream, layer) -> WkvTable
    shift = {}           # (stream, layer, role) -> ShiftTable
    entities = []        # per stream entity counts
    fork_state = None    # state_lowrank._Fork.state when the scorer installed the fork wrapper, else None
    fork_module = None
    original_wkv = None
    step = None          # single_timestep of the model's family


def _name_hook(name):
    def hook(_module, _inputs):
        Carry.current = name
    return hook


def install_names(model):
    """Forward pre-hooks naming every time and channel mixer (stream, layer, role); once per model object."""
    if getattr(model, "_sm155_named", False):
        return
    for i, module in enumerate(model.rwkv_modules):
        for layer, block in enumerate(module.blocks):
            block.time_mixer.register_forward_pre_hook(_name_hook((i, layer, "t")))
            block.channel_mixer.register_forward_pre_hook(_name_hook((i, layer, "c")))
    model._sm155_named = True


class CarryTorch:
    """`torch` stand-in for the model module during a stream evaluation: the token-shift gather with carried vectors."""

    def __init__(self, real, hooks):
        self._real = real
        self._hooks = hooks   # state_vq.Hooks for expertium models (their shift codec), None for the fork

    def __getattr__(self, name):
        return getattr(self._real, name)

    def gather(self, input, dim, index, **kwargs):
        out = self._real.gather(input, dim, index, **kwargs)
        g, cur = Carry.group, Carry.current
        if not Carry.active or g is None or cur is None or dim != 1 or input.dim() != 3:
            return out
        tab = Carry.shift.get(cur)
        if tab is None:
            tab = Carry.shift[cur] = ShiftTable(Carry.entities[cur[0]], input.shape[-1])
        if g.any_carried:
            if not bool(tab.has[g.carried_ent].all()):
                raise RuntimeError(f"stream engine: missing carried shift vector at {cur}")
            out[g.carried] = tab.vec[g.carried_ent]
        if len(g.w_seq):
            tab.vec[g.w_ent] = input[g.w_seq, g.w_pos]
            tab.has[g.w_ent] = True
        if len(g.f_seq):   # entity seen only on skipped rows so far: later rows shift from its position 0
            fresh = ~tab.has[g.f_ent]
            if fresh.any():
                tab.vec[g.f_ent[fresh]] = input[g.f_seq[fresh], g.f_pos[fresh]]
                tab.has[g.f_ent[fresh]] = True
        hooks = self._hooks
        if hooks is not None:   # state_vq._TorchProxy.gather's codec step, the carried positions counted as stored
            fn = hooks.shift.get(hooks.current) if hooks.current is not None else None
            if fn is not None:
                T = index.shape[1]
                stored = (index[:, :, 0] != torch.arange(T)[None, :]) | g.carried
                if stored.any():
                    out = out.clone()
                    out[stored] = fn(out[stored], hooks.current)
        return out


def _store_rule(stream, layer, bits):
    """(table mode, loop mode, fn, extra) of the active label's WKV codec for (stream, layer)."""
    if Carry.kind == "fork":
        from scripts import state_lowrank as sl
        mod = Carry.fork_module
        st = Carry.fork_state
        if isinstance(st, sl.LowRankCodec) and stream in sl.STREAMS:
            rows_bits = st.rows_bits
            return "lowrank", "lowrank", st, (lambda x: mod._quantize_rows(x, rows_bits))
        if isinstance(st, sl.LowRankState) and stream in sl.STREAMS:
            if bits:
                raise RuntimeError("SM147 low-rank probe: the fork's int state bits must be 0")

            def fn(x, rows, t, n, st=st, name=(stream, layer)):
                if st.spectrum:
                    sl.SPECTRUM.record(name[0], name[1], x, n)
                return sl.truncate(x, st.rank) if st.rank else x
            return "float", "rows", fn, None
        if not bits:
            return "float", "float", None, None
        if bits & 1024:
            return "int", "all", (lambda x, n: mod._quantize_tier(x, bits, n)), None
        return "int", "all", (lambda x, n: mod._quantize_rows(x, bits)), None
    from scripts import state_vq
    cur = state_vq.Hooks.current
    quant = state_vq.Hooks.wkv.get(cur[:2]) if cur is not None and cur[2] == "t" else None
    if quant is None:
        return "float", "float", None, None
    if getattr(quant, "loop", None) is not None:   # SM148 ExpertiumLoop
        return "lowrank", "lowrank", quant.codec, quant.rows_quant
    if getattr(quant, "tiered", False):
        return ("int" if isinstance(quant, state_vq.TierQuant) else "float"), "rows", \
            (lambda x, rows, t, n, q=quant, c=cur: q(x, c, rows, t, n)), None
    # state_vq.configure's lambdas: int rows (default q = int_row_quant(...)) or VQ (default c = StateCodec)
    vq = any(isinstance(d, state_vq.StateCodec) for d in (getattr(quant, "__defaults__", None) or ()))
    return ("float" if vq else "int"), "rows", (lambda x, rows, t, n, q=quant, c=cur: q(x, c, rows, t)), None


def carry_wkv(r, k, v, w, a, kd, skip, bits=0, state0=None):
    """The scorer's WKV step loop (fork reference_rwkv7_q / rwkv_ops.reference_rwkv7, expertium
    state_vq._reference_wkv_q, state_lowrank.lowrank_loop / _fork_wkv) from each sequence's carried entity state."""
    if not Carry.active or Carry.group is None:
        return Carry.original_wkv(r, k, v, w, a, kd, skip, bits) if Carry.kind == "fork" \
            else Carry.original_wkv(r, k, v, w, a, kd, skip, state0=state0)
    if state0 is not None:
        raise RuntimeError("stream engine: no learned initial state (SM77)")
    g = Carry.group
    stream, layer, _ = Carry.current
    tmode, mode, fn, extra = _store_rule(stream, layer, int(bits))
    key = (stream, layer)
    B, T, H, K = r.shape
    tab = Carry.wkv.get(key)
    if tab is None:
        if stream not in STATE_STREAMS:
            tmode = "float"
        tab = Carry.wkv[key] = WkvTable(Carry.entities[stream], H, K, tmode, fn.rmax if mode == "lowrank" else 0)
    elif (tab.mode == "lowrank") != (mode == "lowrank"):
        raise RuntimeError("stream engine: the state format changed within one evaluation")
    state, counts, left = tab.load(g.ents)
    out_dtype = k.dtype
    out = torch.zeros(B, T, H, K, dtype=torch.float32)
    skip = skip.bool()
    skip4 = skip[:, :, None, None, None]
    right = torch.zeros(B, H, tab.rmax, K) if mode == "lowrank" else None
    step = Carry.step
    # update counts (non-skipped steps so far, this one included), from the entity's carried count
    cnt = counts[:, None] + torch.cumsum((~skip).long(), 1)
    for t in g.steps:   # steps with only padding rows leave every state as it is
        vt = v[:, t].float()
        out[:, t], nxt = step(r[:, t].float(), k[:, t].float(), vt, w[:, t].float(), a[:, t].float(),
                              kd[:, t].float(), state)
        if mode == "float":
            state = torch.where(skip4[:, t], state, nxt)
        elif mode == "all":
            state = torch.where(skip4[:, t], state, fn(nxt, cnt[:, t]))
        else:
            rows = torch.nonzero(~skip[:, t]).flatten()
            if not rows.numel():
                continue
            state = state.clone()
            if mode == "rows":
                state[rows] = fn(nxt[rows], rows, t, cnt[rows, t])
            else:
                s_new, l_new, r_new = fn.encode(nxt[rows], left[rows], vt[rows], cnt[rows, t], extra,
                                                return_right=True)
                state[rows] = s_new
                left[rows] = l_new
                right[rows] = r_new
    tab.save(g.w_ent, g.w_seq, state, cnt[:, -1], left, right)
    return out.to(out_dtype)


# ------------------------------------------------------------------------------------------------ evaluation

def _install(model, kind):
    install_names(model)
    Carry.kind = kind
    Carry.wkv, Carry.shift = {}, {}
    if kind == "fork":
        if torch.jit._state._enabled:
            raise SystemExit("stream engine on a fork model needs PYTORCH_JIT=0 (eager modules)")
        from scripts import state_lowrank as sl
        mod = sys.modules["rwkv.model.rwkv_model"]
        from rwkv.model.rwkv_ops import single_timestep
        Carry.fork_module = mod
        Carry.fork_state = sl._Fork.state if mod.reference_rwkv7_q is sl._fork_wkv else None
        saved = {"reference_rwkv7_q": mod.reference_rwkv7_q, "torch": mod.torch}
        Carry.original_wkv = mod.reference_rwkv7_q
        mod.reference_rwkv7_q = carry_wkv
        mod.torch = CarryTorch(torch, None)
        Carry.step = single_timestep
        return mod, saved
    from expertium.model import rwkv_model as erm
    from expertium.model.rwkv_ops import single_timestep
    from scripts import state_vq
    if not isinstance(erm.torch, state_vq._TorchProxy) or erm.reference_rwkv7 is not state_vq._reference_wkv_q:
        raise SystemExit("stream engine: expertium models need the CPU Hooks path (state_vq.install)")
    saved = {"reference_rwkv7": erm.reference_rwkv7, "torch": erm.torch}
    Carry.original_wkv = erm.reference_rwkv7
    erm.reference_rwkv7 = carry_wkv
    erm.torch = CarryTorch(torch, state_vq.Hooks)
    Carry.step = single_timestep
    if any(type(m).__name__ == "Sm77Layer" for m in model.modules()):
        raise SystemExit("stream engine: SM77 layers (learned initial state, key extras) are not supported")
    return erm, saved


def _restore(module, saved):
    for name, value in saved.items():
        setattr(module, name, value)
    Carry.active = False
    Carry.group = Carry.current = None


def _trunk(model, layout, start, a, b):
    """Chunk rows [a, b) through every stream: [b - a, d] canonical rows."""
    x = start[a:b].clone()
    if getattr(model, "zero_features_on", False):   # forward_batch_variant's RWKV_ZERO_FEATURES step
        x = x.index_fill(-1, torch.tensor(model.zero_features, dtype=torch.long), 0.0)
    x = model.features2card(x)
    d = x.shape[-1]
    groups = {}

    def run(i, call):
        nonlocal x
        if i not in groups:
            groups[i] = layout.groups(i, a, b)
        new = x
        for gi, g in enumerate(groups[i]):
            Carry.group = g
            x_in = x.index_select(0, g.gather).view(g.B, g.T, d)
            out = call(gi, g, x_in)
            new = new.index_copy(0, g.targets, out.reshape(-1, d).index_select(0, g.positions))
        Carry.group = None
        x = new

    if getattr(model, "interleave_on", False):
        v0 = {}
        for i, layer in model.layer_steps:
            block = model.rwkv_modules[i].blocks[layer]

            def call(gi, g, x_in, i=i, layer=layer, block=block):
                v0_in = torch.empty_like(x_in) if layer == 0 else v0[(i, gi)]
                out, v0[(i, gi)] = block.forward(x_in, v0_in, g.sel, g.skip)
                return out
            run(i, call)
    else:
        for i, submodule in enumerate(model.rwkv_modules):
            run(i, lambda gi, g, x_in, m=submodule: m(x_in, time_shift_select_BT=g.sel, skip_BT=g.skip))
    return x


class _ChunkBatch:
    def __init__(self, start, labels, review_th):
        self.start, self.labels, self.label_review_th = start, labels, review_th
        self.sub_gather = self.sub_gather_lens = self.time_shift_selects = self.skips = None
        self.num_data = 1


def evaluate(model, kind, batch, chunk_rows=16384, layout=None):
    """heldout_fast_metrics.evaluate_with_arrays(model, [{"batch": batch}]) with the stream engine:
    (totals, ArrayPredictions). LAST gets the engine statistics."""
    from scripts.heldout_fast_metrics import ArrayPredictions, collect_arrays  # noqa: F401
    if chunk_rows % ALIGN:
        raise ValueError(f"chunk_rows must be a multiple of {ALIGN}")
    started = time.perf_counter()
    layout = layout or Layout(batch)
    Carry.entities = [s["entities"] for s in layout.streams]
    module, saved = _install(model, kind)
    variant = hasattr(model, "forward_batch_variant")
    head = model.head_and_out_variant if variant else model.head_and_out
    attr = "forward_batch_variant" if variant else "forward_batch"
    model.eval()
    totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0,
              "curve_raw_log_loss": 0.0, "instant_log_loss": 0.0,
              "instant_rating_cross_entropy": 0.0}
    fields = {k: [] for k in ("has_label", "is_query", "label_review_th", "label_rating", "p_curve", "p_imm")}
    peak_tables = 0
    Carry.active = True
    try:
        with torch.no_grad():
            starts = list(range(0, layout.n, chunk_rows))
            if len(starts) > 1 and layout.n - starts[-1] < ALIGN:   # no short last chunk (small-GEMM paths)
                starts.pop()
            for a, b in zip(starts, starts[1:] + [layout.n]):
                x = _trunk(model, layout, batch.start, a, b)
                outs = head(x.view(1, b - a, -1))
                setattr(model, attr, lambda *args, outs=outs, **kw: outs)
                try:
                    labels = batch.labels[:, a:b]
                    stats = model.get_loss(_ChunkBatch(batch.start[a:b], labels, batch.label_review_th[:, a:b]))
                finally:
                    delattr(model, attr)
                if stats is None or not torch.isfinite(stats.average_loss):
                    raise RuntimeError("Non-finite evaluation loss")
                mask = stats.is_query.bool() & stats.has_label.bool()
                losses = F.binary_cross_entropy(stats.p_imm.clamp(1e-5, 1 - 1e-5), labels[:, :, 2], reduction="none")
                totals["curve_labels"] += stats.ahead_n
                totals["instant_labels"] += stats.imm_n
                totals["curve_log_loss"] += stats.ahead_avg.item() * stats.ahead_n
                totals["curve_raw_log_loss"] += stats.ahead_raw_avg.item() * stats.ahead_n
                totals["instant_rating_cross_entropy"] += stats.imm_avg.item() * stats.imm_n
                totals["instant_log_loss"] += losses[mask].sum().item()
                for key in fields:
                    fields[key].append(getattr(stats, key))
                peak_tables = max(peak_tables, sum(t.nbytes() for t in Carry.wkv.values())
                                  + sum(t.nbytes() for t in Carry.shift.values()))
    finally:
        _restore(module, saved)
    joined = type("Stats", (), {k: torch.cat(v, dim=1) for k, v in fields.items()})
    predictions = collect_arrays(joined)
    assert totals["curve_labels"] > 0 and totals["instant_labels"] > 0
    for key in ["curve_log_loss", "curve_raw_log_loss"]:
        totals[key] /= totals["curve_labels"]
    for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
        totals[key] /= totals["instant_labels"]
    kinds = {}
    for (s, layer), t in Carry.wkv.items():
        if s in STATE_STREAMS:
            counts = torch.bincount(t.kind.long(), minlength=4).tolist()
            name = {0: "card", 2: "note"}[s]
            agg = kinds.setdefault(name, {"entities": t.E, "table_mode": t.mode, "kinds": [0, 0, 0, 0],
                                          "float_fallback_writes": 0})
            agg["kinds"] = [x + y for x, y in zip(agg["kinds"], counts)]
            agg["float_fallback_writes"] += t.fallback
    LAST.clear()
    LAST.update({"chunk_rows": chunk_rows, "chunks": len(starts),
                 "entities": Carry.entities, "state_table_bytes": peak_tables, "state_tables": kinds,
                 "seconds": round(time.perf_counter() - started, 3)})
    Carry.wkv, Carry.shift = {}, {}
    return totals, predictions
