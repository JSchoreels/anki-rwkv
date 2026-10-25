"""SM21 v1 and v2 in the R01 trainer: training with carried recurrent state (docs/SMALL_MODEL.md, "Design 2026-10-16:
training with carried state (SM21)", "Recommended design (v1): fresh carry at update granularity", "v2: strided lanes").

Opt-in track option (absent = today's batches, losses and gradients bit for bit; nothing in this file runs):

    "state_carry": true | {"precision": "deploy" | "fp32", "carry": true | false, "lanes": "update" | "strided",
                           "table_gpu_mib": N}

"table_gpu_mib" (SM72, absent = the table always lives on the GPU and commits as below): a user whose table would
take more than N MiB keeps it in host memory (lookups gather the update's rows on the CPU and copy them over, commits
copy the touched rows back); every user's commits write only the touched slots (index_copy_, no table-sized
temporaries), each (stream, layer) commits right after its own forward (its slot is read once per update, before
that forward, so the order of commits does not matter), and the CUDA cache is emptied at each new user. Stored and
read values are the same as without the key, so losses and gradients are unchanged; only memory and speed differ.

"carry": false is the paired control: the same per-user inputs (consistent encodings, user-level day features) and the
same code path, but every window starts from zero state (the table is never read or written).

"lanes": "update" (default, v1): update u holds the 16 consecutive windows 16u .. 16u + 15 of the user (today's layout).
"lanes": "strided" (v2): the user's N windows (chunk_reviews each, the last one possibly short) are cut into L lanes
(L = batch_chunks, 16) of consecutive windows, and update u holds window u of every lane, so lane j's window at update u
continues lane j's window at update u - 1 and reads its end state fresh (no stale state inside a lane). Ragged rule:
U = ceil(N / L) updates (the same count as today), N = L (U - 1) + m with 1 <= m <= L; the first L - m lanes hold U - 1
windows and the last m lanes hold U, so updates 0 .. U - 2 hold L full windows and the last update holds m windows
(lanes L - m .. L - 1) whose last one is the user's final, possibly short, window: every update has today's window
count and review count, hence the same LR schedule. The batch slots of an update are in lane order (slot b = lane
L - n + b for an update of n windows). Users with N <= L have U = 1 (every window is a lane start = v1 = zero carry).
Lane starts (the first window of every lane = all of update 0) start from zero state, and so does every entity whose
first review in the lane comes later but which has history in earlier lanes. Alternative, not implemented: seed each
lane from stale states (option 3: lane j - 1's end states from the previous epoch, <= 140 GB int8 per epoch; the CPU
check seeds lane starts with exact full-history states to show the mechanism). The table holds one state per (lane,
stream, layer, entity) for the (lane, entity) pairs that occur (lane universe, sum over lanes of the entities of each
lane, <= L x the user's entities). item.offset of update u is u x span (as today), so seeds and dropout are unchanged.

* Worker side (CPU preparation, `scripts/gpu_preprocessing_stage.prepare_job` via `gpu_r01_prefetch`):
  - `user_segment_features`: day_offset / day_offset_first / day_of_week relative to the user's training section
    (add_segment_features on the whole section, as the full-history scorer and the prototype's `consistent_start`
    see it) instead of per 2,048-review window; computed once per user in the main process before slicing;
  - int64 entity ids in the samples (`RWKVSample.ids` is int32 today; the loader's placeholder ids for missing notes,
    ID_PLACEHOLDER + card id, overflow it, so distinct notes would collide in the carry table);
  - `consistent_id_table` / `consistent_baseline`: one id encoding per (user, stream, entity) and one day-baseline
    draw per user (a counter-based hash of the user seed = training_seed + user), so every window of the user feeds
    the same input rows a full-history pass would;
  - `carry_entities`: per batch stream, the int64 entity of every canonical row (-1 = padding), on the TrialBatch;
  - `entity_universe`: per stream, the user's sorted entity ids (main process, attached to the user's first item).
* GPU side (`StateCarry`, created by `scripts.expertium_options.CatchUp`):
  - `CarryTable`: per user, per (stream, layer), the latest state of every entity of the user: WKV state S (int8
    rows + fp16 row scales for card and note = the deployment s8; fp16 for deck, preset and user; "fp32" stores
    everything in float32, for checks), the time-mixer shift vector and the channel-mixer shift vector (fp16), and
    a has-state flag. Indexed by searchsorted into the user's entity universe; no host sync, deterministic ops only.
    Cleared (re-allocated) at the user's first update.
  - All 16 windows of an update read the table as of the update start. After the forward (detached, no gradient)
    each entity's final state from its latest window of the update is written back.
  - The student forward: `VariantSrsRWKV._interleaved_streams` (Python, jit.ignore) hands over to `StateCarry.streams`
    while `StateCarry.active(model)` holds, for that model object only (the teacher is never carried). The loss,
    KD mixing and backward are the model's own `_get_loss_variant`.
  - `carry_layer`: VariantRWKV7Layer.forward with the same kernel switches as the stock module (exact shift gather,
    exact group norm, exact glue, row-tiled exact recurrence) and three changes: the token shift at rows with no
    earlier non-skipped row of the sequence takes the carried shift vector; the WKV starts from the carried S
    (`rwkv.model.wkv_state.wkv_state`, S0 detached); it returns the final (shift, S, channel shift) per row.
"""
from contextlib import contextmanager

import numpy as np
import torch

from expertium.variants import BATCH_STREAMS

INT8_STREAMS = ("card_id", "note_id")  # deployment s8; deck/preset/user carry fp16
_MASK = (1 << 64) - 1


# ------------------------------------------------------------------ options
def parse(spec):
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError(f"state_carry must be true or a dict, got {spec!r}")
    unknown = set(spec) - {"precision", "carry", "lanes", "table_gpu_mib"}
    if unknown:
        raise ValueError(f"Unknown state_carry keys {sorted(unknown)}")
    precision = spec.get("precision", "deploy")
    if precision not in ("deploy", "fp32"):
        raise ValueError("state_carry precision must be 'deploy' (int8 card/note, fp16 others) or 'fp32'")
    if not isinstance(spec.get("carry", True), bool):
        raise ValueError("state_carry carry must be true or false")
    lanes = spec.get("lanes", "update")
    if lanes not in ("update", "strided"):
        raise ValueError("state_carry lanes must be 'update' (v1) or 'strided' (v2)")
    out = {"precision": precision, "carry": spec.get("carry", True)}
    if lanes == "strided":  # v1 specs keep their exact parsed form (describe() of v1 runs is unchanged)
        out["lanes"] = "strided"
    if "table_gpu_mib" in spec:  # SM72; absent keeps the parsed form of every earlier run
        budget = spec["table_gpu_mib"]
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("state_carry table_gpu_mib must be an integer >= 0")
        out["table_gpu_mib"] = budget
    return out


def strided(spec):
    return bool(spec) and parse(spec).get("lanes") == "strided"


def strided_layout(windows, lanes):
    """v2: per update, the (lane, window) pairs it holds, in lane order (see the module docstring for the rule)."""
    if windows < 1 or lanes < 1:
        raise ValueError("strided_layout needs at least one window and one lane")
    updates = -(-windows // lanes)
    full = windows - lanes * (updates - 1)  # m, 1 <= m <= lanes
    lengths = [updates - 1] * (lanes - full) + [updates] * full
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int).tolist()
    return [[(j, starts[j] + u) for j in range(lanes) if u < lengths[j]] for u in range(updates)]


def validate_options(options):
    """The preparation path the carry supports: prefetch + columnar fast samples + batched id encodings."""
    if not options.get("state_carry"):
        return
    parse(options["state_carry"])
    needed = [key for key in ("prefetch_users", "fast_samples", "columnar_samples", "batched_id_encodings")
              if not options.get(key)]
    clash = [key for key in ("pack_users", "preprocessing_cache", "worker_frames", "batch_segment_features",
                             "fused_shift", "native_wkv", "native_group_norm", "native_glue", "native_index",
                             "dynamic_graphs", "concurrent_splits", "batched_split_gather", "fused_padded_gather",
                             "wave_aware_backward") if options.get(key)]
    if needed or clash or options.get("recurrence_load_pipeline", 0) in (7, 8):
        raise ValueError(f"state_carry needs {needed} and does not combine with {clash}")


# ------------------------------------------------------------------ worker side (CPU)
def user_segment_features(section):
    """add_segment_features over the user's whole training section (day offsets relative to the section start,
    each card's first day = its first review in the section); the per-window recomputation is then skipped."""
    from rwkv.data_processing import add_segment_features
    return add_segment_features(section.copy())


def entity_universe(section):
    """Per batch stream: the sorted int64 entity ids of the user's training section."""
    return {stream: np.unique(section[stream].to_numpy().astype(np.int64)) for stream in BATCH_STREAMS}


def lane_universe(section, layout, chunk):
    """v2: per batch stream, (sorted entity ids of the section, sorted keys lane x E + entity code of every (lane,
    entity) pair that occurs). `layout` = strided_layout(...) of the section's chunk-review windows."""
    lane_of_row = np.empty(len(section), dtype=np.int64)
    for update in layout:
        for lane, window in update:
            lane_of_row[window * chunk:(window + 1) * chunk] = lane
    out = {}
    for stream in BATCH_STREAMS:
        values = section[stream].to_numpy().astype(np.int64)
        entities = np.unique(values)
        keys = np.unique(lane_of_row * len(entities) + np.searchsorted(entities, values))
        out[stream] = (entities, keys)
    return out


def _mix(x):
    """splitmix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _key(user_seed, *parts):
    with np.errstate(over="ignore"):
        key = _mix(np.uint64(user_seed & _MASK) + np.uint64(0x9E3779B97F4A7C15))
        for part in parts:
            key = _mix(key ^ (np.asarray(part).astype(np.int64).view(np.uint64) + np.uint64(0x9E3779B97F4A7C15)))
        return key


def consistent_id_table(ids, dimensions, dtype, user_seed, stream):
    """Like gpu_id_encodings.batched_id_table (table over the sorted unique ids + inverse indices), but the
    encoding of an id is a function of (user seed, stream, int64 id) only: the same in every window of the user.
    Values are uniform on {0, ..., ID_SPLIT - 1} - (ID_SPLIT - 1) / 2, as the per-window draws."""
    from rwkv.config import ID_SPLIT
    values = ids.numpy().astype(np.int64)
    unique, inverse = np.unique(values, return_inverse=True)
    stream_index = BATCH_STREAMS.index(stream)
    keys = _key(user_seed, np.full(len(unique), stream_index), unique)
    with np.errstate(over="ignore"):
        draws = _mix(keys[:, None] + np.arange(1, dimensions + 1, dtype=np.uint64)[None, :]
                     * np.uint64(0xD1B54A32D192ED03)) % np.uint64(ID_SPLIT)
    table = torch.from_numpy(draws.astype(np.int64)).to(dtype) - ((ID_SPLIT - 1) / 2)
    return table, torch.from_numpy(inverse.reshape(-1))


def consistent_baseline(user_seed, period):
    """One day-encoding baseline per (user, period), uniform on [0, period) like the per-window draw."""
    value = int(_key(user_seed, np.array([1 << 40]), np.array([period]))[0]) % period
    return torch.tensor([value], dtype=torch.long)


def carry_entities(samples):
    """Per batch stream, the int64 entity id of every canonical row (sample i, position t) = i * global_T + t."""
    global_T = max(sample.card_features.size(0) for sample in samples)
    out = []
    for stream in BATCH_STREAMS:
        ids = torch.full((len(samples) * global_T,), -1, dtype=torch.long)
        for index, sample in enumerate(samples):
            values = sample.ids[stream]
            if values.dtype != torch.long:
                raise ValueError("state_carry samples need int64 ids")
            ids[index * global_T:index * global_T + values.numel()] = values
        out.append(ids)
    return out


# ------------------------------------------------------------------ GPU side: the per-user table
def quantize_rows(S):
    """Per state row absmax int8 with an fp16 scale (scripts/state_carry.quantize_rows, bits=8)."""
    scale = S.abs().amax(-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(S / scale), -127.0, 127.0).to(torch.int8), scale.to(torch.float16)


def rank1_pack(S):
    """SM32c: a VQ-coded state is exactly rank 1 (q(n)^2 u v^T) or zero. Pack S [n, H, K, K] as [n, H, 2, K] float32:
    the column and the row through its largest |entry| (column scaled by 1 / that entry); unpack = outer product.
    Exact up to float rounding for rank-1 states, zero stays zero; 8x smaller than float32 rows, half the int8 rows."""
    S = S.float()
    n, H, K, _ = S.shape
    flat = S.reshape(n, H, K * K)
    j = flat.abs().argmax(-1)
    pivot = torch.gather(flat, -1, j.unsqueeze(-1)).squeeze(-1)                       # S[i*, j*]
    row = torch.gather(S, 2, (j // K)[..., None, None].expand(n, H, 1, K)).squeeze(2)   # S[i*, :]
    col = torch.gather(S, 3, (j % K)[..., None, None].expand(n, H, K, 1)).squeeze(3)    # S[:, j*]
    safe = torch.where(pivot != 0, pivot, torch.ones_like(pivot))
    col = torch.where((pivot != 0).unsqueeze(-1), col / safe.unsqueeze(-1), torch.zeros_like(col))
    return torch.stack([col, row], dim=2)


def rank1_unpack(P):
    return P[:, :, 0, :, None] * P[:, :, 1, None, :]


class CarryTable:
    def __init__(self, precision="deploy", gpu_mib=None):
        self.precision = precision
        # SM72: None = today's table (always on the GPU, table-wide where-merge on commit); an integer = the
        # budgeted table (host memory above gpu_mib MiB, touched-slot commits); see the module docstring
        self.gpu_mib = gpu_mib
        self.host = False
        # SM32c: streams whose WKV state is VQ-coded (state_vq): their stored S is the codec's reconstruction, kept in
        # float32 (the deployed code reconstructs exactly; int8 rows on top would add a rounding the phone never does)
        self.fp32_streams = set()
        self.universe = None
        self.keys = None
        self.size = None
        self.slots = {}

    def reset(self, universe, device):
        """New user: drop every state; `universe` = entity_universe(section) of that user (v1: one slot per entity)
        or lane_universe(...) (v2: one slot per (lane, entity) pair that occurs)."""
        self.slots = {}
        self.host = False
        laned = isinstance(universe[BATCH_STREAMS[0]], tuple)
        entities = [universe[stream][0] if laned else universe[stream] for stream in BATCH_STREAMS]
        self.universe = [torch.as_tensor(values, dtype=torch.long).to(device) for values in entities]
        self.keys = ([torch.as_tensor(universe[stream][1], dtype=torch.long).to(device) for stream in BATCH_STREAMS]
                     if laned else None)
        self.size = [int((self.keys or self.universe)[stream].numel()) for stream in range(len(BATCH_STREAMS))]

    def codes(self, stream, entity, lane=None):
        code = torch.searchsorted(self.universe[stream], entity)
        if self.keys is None:
            return code
        return torch.searchsorted(self.keys[stream], lane * self.universe[stream].numel() + code)

    def place(self, nbytes):
        """SM72 budgeted table: the new user's table goes to host memory when it would exceed the budget."""
        self.host = nbytes > self.gpu_mib * 2 ** 20
        return self.host

    def lookup(self, stream, layer, codes, dtype):
        """(has [n], tm_shift [n, C], S [n, H, K, K] float32, cm_shift [n, C] or None), or None before any commit."""
        slot = self.slots.get((stream, layer))
        if slot is None:
            return None
        if self.host:  # the same rows, gathered on the CPU and copied to the update's device
            device, rows = codes.device, codes.cpu()

            def pick(name):
                return slot[name].index_select(0, rows).to(device)
            if slot.get("rank1"):
                S = rank1_unpack(pick("S"))
            elif slot["scale"] is not None:
                S = pick("S").float() * pick("scale").float()
            else:
                S = pick("S").float()
            cm = pick("cm").to(dtype) if slot["cm"] is not None else None
            return pick("has"), pick("tm").to(dtype), S.contiguous(), cm
        if slot.get("rank1"):
            S = rank1_unpack(slot["S"].index_select(0, codes))
        elif slot["scale"] is not None:
            S = slot["S"].index_select(0, codes).float() * slot["scale"].index_select(0, codes).float()
        else:
            S = slot["S"].index_select(0, codes).float()
        cm = slot["cm"].index_select(0, codes).to(dtype) if slot["cm"] is not None else None
        return slot["has"].index_select(0, codes), slot["tm"].index_select(0, codes).to(dtype), S.contiguous(), cm

    @torch.no_grad()
    def commit(self, stream, layer, records, windows_per_update):
        """records: (codes [n], windows [n], (tm [n, C], S [n, H, K, K], cm [n, C] or None)) per split. Every entity's
        final state from its latest window replaces its slot; untouched slots keep their state."""
        codes = torch.cat([r[0] for r in records])
        windows = torch.cat([r[1] for r in records])
        tm = torch.cat([r[2][0] for r in records])
        S = torch.cat([r[2][1] for r in records])
        cm = torch.cat([r[2][2] for r in records]) if records[0][2][2] is not None else None
        E = self.size[stream]
        order = torch.argsort(codes * windows_per_update + windows, stable=True)
        sorted_codes = codes.index_select(0, order)
        slots = torch.arange(E, device=codes.device)
        last = torch.searchsorted(sorted_codes, slots, right=True) - 1
        touched = (last >= 0) & (sorted_codes.index_select(0, last.clamp(min=0)) == slots)
        source = order.index_select(0, last.clamp(min=0))
        int8 = (self.precision == "deploy" and BATCH_STREAMS[stream] in INT8_STREAMS
                and stream not in self.fp32_streams)
        if self.precision == "fp32":
            store, half = S.float(), torch.float32
            scale = None
        elif stream in self.fp32_streams:
            store, scale, half = rank1_pack(S), None, torch.float16
        elif int8:
            store, scale = quantize_rows(S.float())
            half = torch.float16
        else:
            store, scale, half = S.to(torch.float16), None, torch.float16
        slot = self.slots.get((stream, layer))
        if slot is None:
            where = "cpu" if self.host else codes.device
            slot = self.slots[(stream, layer)] = {
                "has": torch.zeros(E, dtype=torch.bool, device=where),
                "tm": torch.zeros((E, *tm.shape[1:]), dtype=half, device=where),
                "S": torch.zeros((E, *store.shape[1:]), dtype=store.dtype, device=where),
                "scale": None if scale is None else torch.ones((E, *scale.shape[1:]), dtype=scale.dtype, device=where),
                "cm": None if cm is None else torch.zeros((E, *cm.shape[1:]), dtype=half, device=where),
                "rank1": stream in self.fp32_streams and self.precision != "fp32"}

        if self.gpu_mib is not None:  # SM72: write the touched slots only (the same values as the where-merge below)
            hit = torch.nonzero(touched).squeeze(-1)
            rows = source.index_select(0, hit)
            target_rows = hit.cpu() if self.host else hit
            slot["has"].index_fill_(0, target_rows, True)
            for name, values in (("tm", tm), ("S", store), ("scale", scale), ("cm", cm)):
                if values is not None:
                    target = slot[name]
                    target.index_copy_(0, target_rows, values.index_select(0, rows).to(target.dtype).to(target.device))
            return

        def update(name, values):
            target = slot[name]
            gathered = values.index_select(0, source).to(target.dtype)
            mask = touched.view(-1, *([1] * (target.dim() - 1)))
            target.copy_(torch.where(mask, gathered, target))

        slot["has"] |= touched
        update("tm", tm)
        update("S", store)
        if scale is not None:
            update("scale", scale)
        if cm is not None:
            update("cm", cm)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for slot in self.slots.values() for t in slot.values()
                   if isinstance(t, torch.Tensor))


# ------------------------------------------------------------------ one layer with carried state
def _shift(x, select, exact):
    if exact and x.is_cuda:
        from rwkv.model.exact_index import ExactShiftGather
        return ExactShiftGather.apply(x, select)
    return torch.gather(x, dim=1, index=select.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def carry_layer(block, in_BTC, v0_BTC, select, skip, init):
    """VariantRWKV7Layer.forward from `init` = (has [B], tm_shift [B, C], S [B, H, K, K], cm_shift [B, C] or None)
    or None (zero state). Returns (x, v0, final) with final = (tm_shift [B, C], S [B, H, K, K], cm_shift or None),
    detached. With init None (or nothing carried) the arithmetic is the stock module's except the recurrence kernel
    (the S0 variant of the same kernel; CPU: the same per-step reference).
    SM32c: a mixer with state_vq_slot >= 0 codes what the entity stores (expertium/model/state_vq_hooks.py): the
    token-shift vectors read from an earlier review (in this window, or carried: the carried vector is coded on read
    with the current catalog) and the WKV state after every non-skipped step (from the carried, already coded S0); the
    returned final S is the last stored (coded) state."""
    from rwkv.model.wkv_state import wkv_state
    tm = block.time_mixer
    vq_t = getattr(tm, "state_vq_slot", -1)
    if (tm.fused_shift or tm.native_shift or tm.native_wkv or tm.native_group_norm or tm.native_glue
            or tm.state_qmax != float("inf") or tm.state_lowrank_rank > 0):
        raise NotImplementedError("state_carry: fused/native kernels and state-QAT streams are not covered")
    B, T, C = in_BTC.shape
    H, K = tm.H, tm.K
    cuda = in_BTC.is_cuda
    nonskip = ~skip
    t_index = torch.arange(T, device=in_BTC.device).expand(B, T)
    first = torch.where(nonskip, t_index, torch.full_like(t_index, T)).amin(dim=1)
    last = torch.where(nonskip, t_index, torch.full_like(t_index, -1)).amax(dim=1).clamp(min=0)
    rows = torch.arange(B, device=in_BTC.device)
    head = None if init is None else (t_index <= first.unsqueeze(1)) & init[0].unsqueeze(1)

    x_BTC = tm.layer_norm(in_BTC)
    x_shift = _shift(x_BTC, select, tm.exact_shift)
    if head is not None:
        x_shift = torch.where(head.unsqueeze(-1), init[1].unsqueeze(1).to(x_shift.dtype), x_shift)
    stored = None
    if vq_t >= 0:
        from expertium.model.state_vq_hooks import vq_shift_mask
        stored = select.long() != t_index
        if head is not None:
            stored = stored | head
        x_shift = vq_shift_mask(x_shift, stored, vq_t, 0)
    rkvdag = torch.lerp(x_BTC.unsqueeze(0), x_shift.unsqueeze(0), tm.rkvdag_lerp)
    r_BTC, k_BTC, v_BTC, d_BTC, a_BTC, g_BTC, k_scale_BTC, v_scale_BTC = rkvdag.unbind(dim=0)
    r_BTC = tm.W_r(r_BTC)
    k_BTC = tm.W_k(k_BTC)
    k_scale = torch.nn.functional.sigmoid(tm.k_scale_linear(k_scale_BTC))
    v_scale = torch.nn.functional.sigmoid(tm.v_scale_linear(v_scale_BTC))
    if tm.layer_id == 0:
        v_BTC = tm.W_v(v_BTC)
        v0_BTC = v_BTC
    else:
        v_lerp = torch.nn.functional.sigmoid(tm.v_lora_simple(v_BTC))
        v_BTC = torch.lerp(tm.W_v(v_BTC), v0_BTC, v_lerp)
    a_BTC = torch.nn.functional.sigmoid(tm.a_lora_simple(a_BTC))
    g_BTC = tm.lora_B_g(torch.nn.functional.sigmoid(tm.lora_A_g(g_BTC)))
    _d = -0.5 - torch.nn.functional.softplus(-tm.d_lora_mlp(d_BTC))
    w_BTC = torch.exp(-torch.exp(_d.float()))
    k_BTHK = k_scale.unsqueeze(-1) * torch.nn.functional.normalize(k_BTC.view(B, T, H, K), dim=-1, p=2.0)
    r_BTHK = r_BTC.view(B, T, H, K)
    v_BTHK = v_scale.unsqueeze(-1) * torch.nn.functional.normalize(v_BTC.view(B, T, H, K), dim=-1, p=2.0)
    w_BTHK = w_BTC.view(B, T, H, K)
    a_BTHK = a_BTC.view(B, T, H, K)
    kd_BTHK = k_BTHK
    k_BTHK = k_BTHK * a_BTHK
    if init is not None:
        state0 = init[2]
    elif cuda:
        state0 = torch.zeros(B, H, K, K, dtype=torch.float32, device=in_BTC.device)
    else:
        state0 = None
    if vq_t >= 0:
        from expertium.model.state_vq_hooks import vq_wkv_state
        out_BTHK, state_T = vq_wkv_state(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, kd_BTHK, skip, state0, vq_t)
    else:
        out_BTHK, state_T = wkv_state(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, kd_BTHK, skip, state0)
    if tm.exact_group_norm and cuda:
        from rwkv.model.exact_group_norm import ExactGroupNorm
        out_BTC = ExactGroupNorm.apply(out_BTHK.view(B * T, C), tm.out_group_norm.weight, tm.out_group_norm.bias,
                                       tm.out_group_norm.num_groups, tm.out_group_norm.eps).view(B, T, C)
    else:
        out_BTC = tm.out_group_norm(out_BTHK.view(B * T, C)).view(B, T, C)
    if tm.exact_bonus_products and cuda and K == 32:
        from rwkv.model.exact_glue import ExactBonusProducts
        bonus_products = ExactBonusProducts.apply(r_BTHK, tm.bonus, k_BTHK)
    else:
        bonus_products = r_BTHK * tm.bonus * k_BTHK
    bonus_BTC = (bonus_products.sum(dim=-1, keepdim=True) * v_BTHK).view(B, T, C)
    if tm.exact_gate_glue and cuda:
        from rwkv.model.exact_glue import ExactGateGlue
        gated = ExactGateGlue.apply(out_BTC, bonus_BTC, g_BTC, tm.exact_gate_glue == 2)
    else:
        gated = g_BTC * (out_BTC + bonus_BTC)
    x_out = in_BTC + tm.dropout(tm.W_o(gated))
    with torch.no_grad():
        final_tm = x_BTC[rows, last]
    if block.cmix_stripped:
        return x_out, v0_BTC, (final_tm, state_T.detach(), None)
    cm = block.channel_mixer
    if cm.native_shift or cm.native_glue:
        raise NotImplementedError("state_carry: native channel-mixer kernels are not covered")
    xc = cm.layer_norm(x_out)
    xc_shift = _shift(xc, select, cm.exact_shift)
    if head is not None and init[3] is not None:
        xc_shift = torch.where(head.unsqueeze(-1), init[3].unsqueeze(1).to(xc_shift.dtype), xc_shift)
    vq_c = getattr(cm, "state_vq_slot", -1)
    if vq_c >= 0:
        from expertium.model.state_vq_hooks import vq_shift_mask
        stored_c = select.long() != t_index
        if head is not None and init[3] is not None:
            stored_c = stored_c | head
        xc_shift = vq_shift_mask(xc_shift, stored_c, vq_c, 1)
    kc = cm.W_k(torch.lerp(xc, xc_shift, cm.lerp_k))
    if cm.exact_channel_glue and cuda:
        from rwkv.model.exact_glue import ExactChannelGlue
        hidden = ExactChannelGlue.apply(kc)
    else:
        hidden = torch.square(torch.nn.functional.relu(kc))
    y = x_out + cm.dropout(cm.W_v(hidden))
    with torch.no_grad():
        final_cm = xc[rows, last]
    return block.dropout(y), v0_BTC, (final_tm, state_T.detach(), final_cm)


# ------------------------------------------------------------------ the trainer adapter
def _identity(model):
    parameter = next(model.parameters(), None)
    if parameter is None:
        # SM72: the scripted wrapper lists root-level parameters only, and the fork architecture has none (w64's GRU
        # head has some); its first parameter, features2card.0.weight, is reachable through the C++ module.
        parameter = model._c.getattr("features2card").getattr("0").getattr("weight")
    return parameter.data_ptr(), parameter.device, parameter.dtype, parameter.shape


class StateCarry:
    """Per run: the carry table and the hand-over from VariantSrsRWKV._interleaved_streams."""

    def __init__(self, spec, options=None):
        self.spec = parse(spec)
        if options is not None:
            validate_options(options)
        self.table = CarryTable(self.spec["precision"], self.spec.get("table_gpu_mib"))
        self.place_pending = False
        self.user = None
        self.model = None
        self.model_key = None
        self.batch = None
        self.entities = None
        self.global_T = None
        self.windows = None
        self.lane_offset = 0
        self.stats = {"users": 0, "updates": 0, "streams_calls": 0, "peak_table_bytes": 0}
        if self.spec.get("table_gpu_mib") is not None:
            self.stats["host_table_users"] = 0

    def attach(self, item, batch, user):
        """Before the update's loss: the item's entity rows, and the table reset at a new user's first update."""
        if user != self.user:
            if item.offset != 0 or getattr(item, "carry_universe", None) is None:
                raise RuntimeError(f"state_carry: user {user} must start at offset 0 with its entity universe")
            device = batch.labels.device
            if isinstance(item.carry_universe[BATCH_STREAMS[0]], tuple) != (self.spec.get("lanes") == "strided"):
                raise RuntimeError("state_carry: the item's entity universe does not match the lanes option")
            self.table.reset(item.carry_universe, device)
            if self.table.gpu_mib is not None:  # SM72: drop the previous user's cached blocks; place at the first forward
                torch.cuda.empty_cache()
                self.place_pending = True
            self.user = user
            self.stats["users"] += 1
        device = batch.labels.device
        self.entities = [ids.to(device, non_blocking=True) for ids in item.carry]
        self.global_T = int(batch.labels.shape[1])
        self.windows = int(batch.labels.shape[0])
        if self.spec.get("lanes") == "strided":
            # v2: slots are in lane order and hold consecutive lanes (L - n .. L - 1 in the last update), so the lane
            # of slot b is b + lane_offset: no lane tensor and no transfer.
            lanes = list(getattr(item, "carry_lanes", None) or ())
            if len(lanes) != self.windows or lanes != list(range(lanes[0], lanes[0] + self.windows)):
                raise RuntimeError(f"state_carry lanes: update of user {user} needs consecutive lanes, got {lanes}")
            self.lane_offset = lanes[0]
        self.batch = batch
        self.stats["updates"] += 1

    @contextmanager
    def active(self, model):
        from expertium.model import variant_model
        if self.entities is None:
            raise RuntimeError("state_carry: attach(item, batch, user) before the loss")
        previous = variant_model.CARRY_HOOK[0]
        variant_model.CARRY_HOOK[0] = self
        self.model = model
        self.model_key = _identity(model)
        try:
            yield self
        finally:
            variant_model.CARRY_HOOK[0] = previous
            self.model = None
            self.entities = None
            self.stats["peak_table_bytes"] = max(self.stats["peak_table_bytes"], self.table.nbytes())

    def owns(self, model):
        # Under TorchScript the ignored _interleaved_streams receives a RecursiveScriptModule wrapper, not the
        # object passed to active(): identify the model by its parameter storage (the teacher has its own).
        return model is self.model or (self.model is not None and _identity(model) == self.model_key)

    def streams(self, model, x, batch_sub_gather, batch_sub_gather_lens, batch_time_shift_selects, batch_skips):
        """VariantSrsRWKV._interleaved_streams (same gathers, scatters and exact-index switches) with carry_layer."""
        self.stats["streams_calls"] += 1
        # Resolve submodules on the object passed to active(): under TorchScript `model` is the RecursiveScriptModule
        # wrapper the ignored method receives, whose children are raw torch._C.ScriptModules (not callable).
        model = self.model
        cur = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        gathers, positions, targets = [], [], []
        for splits in batch_sub_gather:
            stream_gathers, stream_positions, stream_targets, parts = [], [], [], []
            for split in splits:
                index = split.long()
                canonical = torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1))
                keep = torch.nonzero(canonical >= 0).squeeze(-1)
                stream_gathers.append(canonical)
                stream_positions.append(keep)
                stream_targets.append(canonical.index_select(0, keep))
                parts.append(canonical)
            gathers.append(stream_gathers)
            positions.append(stream_positions)
            targets.append(stream_targets)
            cur = torch.cat(parts)
        fused_gather, fused_sizes, fused_positions, fused_targets = [], [], [], []
        for i in range(len(gathers)):
            sizes = [canonical.numel() for canonical in gathers[i]]
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            fused_gather.append(torch.cat(gathers[i]).clamp(min=0))
            fused_sizes.append(sizes)
            fused_positions.append(torch.cat([keep + offset for keep, offset in zip(positions[i], offsets)]))
            fused_targets.append(torch.cat(targets[i]))
        v0s = [[None] * len(splits) for splits in batch_sub_gather]
        exact = model.exact_index and x.is_cuda
        if exact:
            from rwkv.model.exact_index import ExactIndexCopy, OrderedIndexSelect, row_order
            rows = x.size(0)
            gather_orders, position_orders = [], []
            for i in range(len(gathers)):
                source = torch.cat(gathers[i]) if model.skip_padding_sources else fused_gather[i]
                gather_orders.append(row_order(source, rows))
                position_orders.append(row_order(fused_positions[i], fused_gather[i].numel()))
        # entity and window of every split row (its first canonical row; rows are right-padded)
        codes, windows = [], []
        for i in range(len(gathers)):
            stream_codes, stream_windows = [], []
            for s, canonical in enumerate(gathers[i]):
                head_rows = canonical.view(-1, batch_sub_gather_lens[i][s])[:, 0]
                window = torch.div(head_rows, self.global_T, rounding_mode="floor")
                lane = window + self.lane_offset if self.table.keys is not None else None  # v1: no lanes
                stream_codes.append(self.table.codes(i, self.entities[i].index_select(0, head_rows), lane))
                stream_windows.append(window)
            codes.append(stream_codes)
            windows.append(stream_windows)
        records = {}
        # SM32c: VQ-coded streams keep their stored S in float32 (CarryTable.fp32_streams)
        self.table.fp32_streams = {i for i, layer in model.layer_steps
                                   if getattr(getattr(getattr(model.rwkv_modules, str(i)).blocks, str(layer)).time_mixer,
                                              "state_vq_slot", -1) >= 0}
        budgeted = self.table.gpu_mib is not None
        if budgeted and self.place_pending:
            self.place_pending = False
            if self.table.place(self._table_bytes(model)):
                self.stats["host_table_users"] += 1
        if budgeted and len(set(model.layer_steps)) != len(model.layer_steps):
            raise RuntimeError("state_carry table_gpu_mib: early commits need each (stream, layer) once per update")
        for i, layer in model.layer_steps:
            block = getattr(getattr(model.rwkv_modules, str(i)).blocks, str(layer))
            if exact:
                inputs = OrderedIndexSelect.apply(x, fused_gather[i], *gather_orders[i]).split(fused_sizes[i])
            else:
                inputs = x.index_select(0, fused_gather[i]).split(fused_sizes[i])
            outputs = []
            for s, x_split in enumerate(inputs):
                length = batch_sub_gather_lens[i][s]
                x_in = x_split.view(-1, length, model.d_model)
                v0_in = torch.empty_like(x_in) if layer == 0 else v0s[i][s]
                init = self.table.lookup(i, layer, codes[i][s], x.dtype) if self.spec["carry"] else None
                x_out, v0s[i][s], final = carry_layer(block, x_in, v0_in, batch_time_shift_selects[i][s].view(-1, length),
                                                      batch_skips[i][s].view(-1, length), init)
                records.setdefault((i, layer), []).append((codes[i][s], windows[i][s], final))
                outputs.append(x_out.reshape(-1, model.d_model))
            if budgeted and self.spec["carry"]:  # SM72: this slot was read for every split above; free the finals now
                self.table.commit(i, layer, records.pop((i, layer)), self.windows)
            flat = torch.cat(outputs)
            if exact:
                kept = OrderedIndexSelect.apply(flat, fused_positions[i], *position_orders[i])
                x = ExactIndexCopy.apply(x, fused_targets[i], kept)
            else:
                x = x.index_copy(0, fused_targets[i], flat.index_select(0, fused_positions[i]))
        # every window read the table as of the update start; now each entity's latest window writes back
        # (v2: each (lane, entity) occurs in one window of the update, so every touched pair writes back)
        for (i, layer), stream_records in records.items() if self.spec["carry"] else ():
            self.table.commit(i, layer, stream_records, self.windows)
        return x

    def _table_bytes(self, model):
        """SM72: bytes the user's table will take (every (stream, layer) slot of model.layer_steps, as commit stores it)."""
        total = 0
        for i, layer in model.layer_steps:
            tm = getattr(getattr(model.rwkv_modules, str(i)).blocks, str(layer)).time_mixer
            H, K, C = int(tm.H), int(tm.K), int(model.d_model)
            if self.table.precision == "fp32":
                per = H * K * K * 4 + 2 * C * 4
            elif BATCH_STREAMS[i] in INT8_STREAMS:
                per = H * K * K + H * K * 2 + 2 * C * 2
            else:
                per = H * K * K * 2 + 2 * C * 2
            total += self.table.size[i] * (per + 1)
        return total

    def describe(self):
        # stats is the live dict: result.json snapshots carry the current counters
        if self.spec.get("lanes") == "strided":
            return {**self.spec, "mode": "strided lanes (v2): update u holds window u of every lane; lane starts from "
                    "zero", "stats": self.stats}
        return {**self.spec, "mode": "fresh carry at update granularity (v1)", "stats": self.stats}
