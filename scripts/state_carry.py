"""SM21 prototype (CPU reference): carry recurrent state into training windows (docs/SMALL_MODEL.md, "Design
2026-10-16: training with carried state (SM21)").

Nothing here is wired into the R01 trainer yet; the default training path is untouched. The pieces:

* `carry_forward(model, batch, ids, windows, source)`: the w64 (`VariantSrsRWKV`, interleaved streams) forward of a
  PreparedBatch in which every entity row (a card, note, deck, preset or user sequence inside a window) starts from a
  carried state instead of zero: per (stream, layer) the WKV-7 state S [H, K, K], the time-mixer token-shift vector and
  the channel-mixer token-shift vector (the deployment RNN state, expertium/model/variant_rnn.py). It returns the
  head outputs of `forward_batch_variant` and the final state of every row. CPU eager arithmetic, op for op the
  stock CPU path (`reference_rwkv7` per-step WKV), so with no carried state it equals the stock forward bit for bit.
  GPU kernels with initial/final state: rwkv/model/wkv_state.py; trainer integration: scripts/state_carry_train.py.
* State sources (`source.lookup` before the forward, `source.commit` after it):
  - `FreshCarry`: the latest state per (stream, entity) in memory, written by the training pass itself (no extra
    forward, no disk). All rows of one batch read the state as of the batch start; later windows win on commit.
  - `StaleCache`: option 3 proper: the end-of-window states of the previous pass, per window, quantised (int8 WKV
    rows with one scale per row, fp16 shift vectors) and looked up as "the entity's state at the end of its last
    window before this one". `nbytes()` reports the storage.
* `consistent_start(...)`: windows whose input rows are the rows of the user's full-history sample (one draw of the
  id/day encodings per user, user-relative day offsets), so that carried states and new inputs agree, as at
  deployment. Today's per-window encodings are re-drawn per sample.
"""
import bisect

import torch

from expertium.variants import BATCH_STREAMS
from rwkv.model.wkv_state import wkv_with_state  # noqa: F401  (moved to the model package for the GPU path)


# ---------------------------------------------------------------- bookkeeping
def canonical_ids(samples, global_T, frame):
    """Per batch stream: entity id (int64) of every canonical row (sample i, position t) = i * global_T + t; -1 for
    padding. Ids come from the user's frame by review_th: `RWKVSample.ids` is int32 and the data loader's placeholder
    ids for missing notes/decks/presets (ID_PLACEHOLDER + card id) overflow it, so distinct notes would collide."""
    import numpy as np
    review_th = frame["review_th"].to_numpy()
    out = []
    for stream in BATCH_STREAMS:
        table = np.full(int(review_th.max()) + 1, -1, dtype=np.int64)
        table[review_th] = frame[stream].to_numpy().astype(np.int64)
        ids = torch.full((len(samples) * global_T,), -1, dtype=torch.long)
        for i, sample in enumerate(samples):
            values = torch.from_numpy(table[sample.review_ths.long().numpy()])
            ids[i * global_T:i * global_T + values.numel()] = values
        out.append(ids)
    return out


def canonical_windows(samples, global_T, first_window=0):
    """Window number of every canonical row (sample order = window order of the user)."""
    return torch.arange(len(samples), dtype=torch.long).repeat_interleave(global_T) + first_window


def _canonical_gathers(x_rows, batch_sub_gather):
    """`VariantSrsRWKV._interleaved_streams` re-anchoring (non-exact branch): per stream, per split, canonical rows."""
    cur = torch.arange(x_rows, dtype=torch.long)
    gathers = []
    for splits in batch_sub_gather:
        stream_gathers, parts = [], []
        for split in splits:
            index = split.long()
            canonical = torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1))
            stream_gathers.append(canonical)
            parts.append(canonical)
        gathers.append(stream_gathers)
        cur = torch.cat(parts)
    return gathers


# ---------------------------------------------------------------- one layer with carried state
def _shift(x, select, head, shift0):
    """Stock token shift (gather of the last non-skipped row), with the carried vector where nothing precedes."""
    shifted = torch.gather(x, dim=1, index=select.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    if shift0 is not None:
        shifted = torch.where(head.unsqueeze(-1), shift0.unsqueeze(1).to(shifted.dtype), shifted)
    return shifted


def carry_layer(block, in_BTC, v0_BTC, select, skip, init):
    """VariantRWKV7Layer.forward (CPU eager path) from `init` = (has [B], tm_shift [B, C], S [B, H, K, K], cm_shift
    [B, C] or None), or None for zero state. Returns (x, v0, final) with final = (tm_shift, S, cm_shift)."""
    tm = block.time_mixer
    B, T, C = in_BTC.shape
    H, K = tm.H, tm.K
    nonskip = ~skip
    before = torch.cumsum(nonskip.long(), dim=1) - nonskip.long()  # non-skipped rows strictly before t
    has = None if init is None else init[0]
    head = (before == 0) & has.unsqueeze(1) if has is not None and bool(has.any()) else None
    t_index = torch.arange(T).expand(B, T)
    last = torch.where(nonskip, t_index, torch.full_like(t_index, -1)).amax(dim=1).clamp(min=0)
    rows = torch.arange(B)

    x_BTC = tm.layer_norm(in_BTC)
    x_shift = _shift(x_BTC, select, head, None if head is None else init[1])
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
    if tm.state_qmax != float("inf") or tm.state_lowrank_rank > 0:
        raise NotImplementedError("state-QAT streams are not covered by the carry prototype")
    state0 = None
    if has is not None and bool(has.any()):
        state0 = torch.where(has.view(B, 1, 1, 1), init[2].float(), torch.zeros(B, H, K, K))
    out_BTHK, state_T = wkv_with_state(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, kd_BTHK, skip, state0)
    out_BTC = tm.out_group_norm(out_BTHK.view(B * T, C)).view(B, T, C)
    bonus = ((r_BTHK * tm.bonus * k_BTHK).sum(dim=-1, keepdim=True) * v_BTHK).view(B, T, C)
    x_out = in_BTC + tm.dropout(tm.W_o(g_BTC * (out_BTC + bonus)))
    final_tm_shift = x_BTC[rows, last].detach()
    if block.cmix_stripped:
        return x_out, v0_BTC, (final_tm_shift, state_T.detach(), None)
    cm = block.channel_mixer
    xc = cm.layer_norm(x_out)
    xc_shift = _shift(xc, select, head, None if head is None or init[3] is None else init[3])
    kc = cm.W_k(torch.lerp(xc, xc_shift, cm.lerp_k))
    y = x_out + cm.dropout(cm.W_v(torch.square(torch.nn.functional.relu(kc))))
    return block.dropout(y), v0_BTC, (final_tm_shift, state_T.detach(), xc[rows, last].detach())


# ---------------------------------------------------------------- the batch forward
def carry_forward(model, batch, ids, windows, source=None):
    """`forward_batch_variant` (interleaved w64) with carried state. ids/windows: canonical_ids / canonical_windows.
    Returns (head outputs, finals) where finals = list of (stream, layer, entity [n], window [n], final tuple)."""
    if not model.interleave_on:
        raise NotImplementedError("prototype covers the interleaved variants (w64)")
    start = batch.start
    if model.zero_features_on:
        start = start.index_fill(-1, torch.tensor(model.zero_features, dtype=torch.long), 0.0)
    x = model.features2card(start)
    gathers = _canonical_gathers(x.size(0), batch.sub_gather)
    fused_gather, fused_sizes, fused_positions, fused_targets = [], [], [], []
    for stream_gathers in gathers:
        sizes = [g.numel() for g in stream_gathers]
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)
        fused_gather.append(torch.cat(stream_gathers).clamp(min=0))
        fused_sizes.append(sizes)
        keeps = [torch.nonzero(g >= 0).squeeze(-1) for g in stream_gathers]
        fused_positions.append(torch.cat([keep + off for keep, off in zip(keeps, offsets)]))
        fused_targets.append(torch.cat([g.index_select(0, keep) for g, keep in zip(stream_gathers, keeps)]))
    v0s = [[None] * len(splits) for splits in batch.sub_gather]
    finals = []
    for i, layer in model.layer_steps:
        block = getattr(getattr(model.rwkv_modules, str(i)).blocks, str(layer))
        inputs = x.index_select(0, fused_gather[i]).split(fused_sizes[i])
        outputs = []
        for s, x_split in enumerate(inputs):
            length = batch.sub_gather_lens[i][s]
            x_in = x_split.view(-1, length, model.d_model)
            first = gathers[i][s].view(-1, length)[:, 0]  # rows are right-padded: column 0 is a real row
            entity, window = ids[i].index_select(0, first), windows.index_select(0, first)
            init = source.lookup(i, layer, entity, window) if source is not None else None
            v0_in = torch.empty_like(x_in) if layer == 0 else v0s[i][s]
            x_out, v0s[i][s], final = carry_layer(block, x_in, v0_in, batch.time_shift_selects[i][s].view(-1, length),
                                                  batch.skips[i][s].view(-1, length), init)
            finals.append((i, layer, entity, window, final))
            outputs.append(x_out.reshape(-1, model.d_model))
        flat = torch.cat(outputs)
        x = x.index_copy(0, fused_targets[i], flat.index_select(0, fused_positions[i]))
    x = x.view(batch.num_data, -1, model.d_model)
    return model.head_and_out_variant(x), finals


# ---------------------------------------------------------------- state sources
def quantize_rows(S, bits=8):
    """Per state row (last axis) absmax int8, as the fork's s8 runtime and small_model_wkv.reference."""
    qmax = 127.0 if bits == 8 else 7.0
    scale = S.abs().amax(-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(S / scale), -qmax, qmax).to(torch.int8), scale.to(torch.float16)


def dequantize_rows(q, scale):
    return q.float() * scale.float()


class FreshCarry:
    """Latest state per (stream, layer, entity), in memory (fp32, or int8 rows with `bits`=8)."""

    def __init__(self, bits=0):
        self.bits = bits
        self.table = {}

    def lookup(self, i, layer, entity, window):
        rows = [self.table.get((i, layer, int(e))) for e in entity.tolist()]
        return _stack(rows, self.bits)

    def commit(self, finals):
        # Rows of one entity may sit in different splits; apply in window order so the latest window wins.
        rows = [(int(window[row]), i, layer, int(entity[row]), final, row)
                for i, layer, entity, window, final in finals for row in range(entity.numel())]
        for _, i, layer, e, final, row in sorted(rows, key=lambda item: item[0]):
            self.table[(i, layer, e)] = _pack(final, row, self.bits)

    def nbytes(self):
        return sum(_size(item) for item in self.table.values())


class StaleCache:
    """Option 3: end-of-window states of the previous pass, int8, per (stream, layer, entity) sorted by window.
    lookup(window k) returns the state at the end of the entity's last recorded window j < k, else zero."""

    def __init__(self, bits=8):
        self.bits = bits
        self.previous, self.current = {}, {}

    def lookup(self, i, layer, entity, window):
        rows = []
        for e, w in zip(entity.tolist(), window.tolist()):
            history = self.previous.get((i, layer, e))
            if not history:
                rows.append(None)
                continue
            position = bisect.bisect_left(history[0], w) - 1
            rows.append(history[1][position] if position >= 0 else None)
        return _stack(rows, self.bits)

    def commit(self, finals):
        for i, layer, entity, window, final in finals:
            for row in range(entity.numel()):
                windows, states = self.current.setdefault((i, layer, int(entity[row])), ([], []))
                position = bisect.bisect_left(windows, int(window[row]))
                windows.insert(position, int(window[row]))
                states.insert(position, _pack(final, row, self.bits))

    def end_pass(self):
        self.previous, self.current = self.current, {}

    def nbytes(self):
        return sum(_size(item) for _, states in self.previous.values() for item in states)


def _pack(final, row, bits):
    tm_shift, S, cm_shift = final
    s = S[row]
    packed_S = quantize_rows(s, bits) if bits else s.clone()
    half = (lambda t: None if t is None else t[row].to(torch.float16)) if bits else \
        (lambda t: None if t is None else t[row].clone())
    return half(tm_shift), packed_S, half(cm_shift)


def _size(item):
    tm_shift, S, cm_shift = item
    total = sum(t.numel() * t.element_size() for t in (tm_shift, cm_shift) if t is not None)
    return total + sum(t.numel() * t.element_size() for t in (S if isinstance(S, tuple) else (S,)))


def _stack(rows, bits):
    if all(r is None for r in rows):
        return None
    ref = next(r for r in rows if r is not None)
    has = torch.tensor([r is not None for r in rows])
    ref_S = dequantize_rows(*ref[1]) if bits else ref[1]

    def take(index, like):
        out = []
        for r in rows:
            if r is None:
                out.append(torch.zeros_like(like))
            else:
                value = r[index]
                out.append(dequantize_rows(*value).float() if index == 1 and bits else value.float())
        return torch.stack(out)
    tm_shift = take(0, ref[0].float())
    S = take(1, ref_S.float())
    cm_shift = take(2, ref[2].float()) if ref[2] is not None else None
    return has, tm_shift, S, cm_shift


# ---------------------------------------------------------------- consistent inputs across windows
def consistent_start(window_batch, window_samples, full_batch, full_sample):
    """Replace each window sample's input rows by the same rows of the full-history sample (one encoding draw per
    user, user-relative day features). Rows are matched by (review_th, query flag); a window is a contiguous slice of
    the full sample's rows (same queries: `is_first_review` comes from the full frame)."""
    global_T = window_batch.start.size(0) // window_batch.num_data
    full_T = full_batch.start.size(0) // full_batch.num_data
    full_keys = (full_sample.review_ths.long() * 2 + full_sample.skips.long()).tolist()
    position = {key: index for index, key in enumerate(full_keys)}
    start = window_batch.start.clone()
    for i, sample in enumerate(window_samples):
        keys = (sample.review_ths.long() * 2 + sample.skips.long()).tolist()
        rows = torch.tensor([position[k] for k in keys], dtype=torch.long)
        if rows.numel() and not bool((rows[1:] - rows[:-1] == 1).all()):
            raise ValueError("window rows are not a contiguous slice of the full-history rows")
        start[i * global_T:i * global_T + rows.numel()] = full_batch.start[:full_T].index_select(0, rows)
    window_batch.start = start
    return window_batch
