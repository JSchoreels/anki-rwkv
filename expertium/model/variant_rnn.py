"""Recurrent (one review at a time) inference for expertium.variants (docs/EXPERTIUM_CATCH_UP.md).

Loads a VariantSrsRWKV state dict strictly. Module i carries the state of stream variant.base.modules[i][0]; the
variants list streams in the batch chain order of rwkv.prepare_batch (card, deck, note, preset, user), so this is
the routing the parallel model trains with (the vendored SrsRWKVRnn hard-codes the same order). The GRU head, no
ahead residual, zero_features, stripped mixers and the interleaved schedule follow the variant (upstream
rwkv/model/srs_model_rnn.py at a9fcf466 implements the same switches). scripts/check_expertium_catch_up.py compares
these predictions with the parallel model row by row.
"""
import copy
import math

import torch

from expertium.model.rwkv_model import LoraSimple
from expertium.model.rwkv_rnn_model import RWKV7RNN, RWKV7RNNLayer
from expertium.model.srs_model_rnn import SrsRWKVRnn
from expertium.model.variant_model import interleave_schedule, is_excluded, layer_steps


class StrippedRNNChannelMixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(1)
        self.lerp_k = torch.nn.Parameter(torch.zeros(1, 1, 1))
        self.W_k = torch.nn.Linear(1, 1, bias=False)
        self.W_v = torch.nn.Linear(1, 1, bias=False)


class VariantRWKV7RNNLayer(RWKV7RNNLayer):
    def __init__(self, config, layer_id, strip_cmix=False, strip_v_lora=False):
        super().__init__(config, layer_id)
        self.cmix_stripped = bool(strip_cmix)
        if strip_cmix:
            self.channel_mixer = StrippedRNNChannelMixer()
        if strip_v_lora:
            self.time_mixer.v_lora_simple = LoraSimple(name="v", d_model=1, d_lora=1, layer_id=layer_id)

    def forward(self, in_BC, v0_BC, state):
        if state is None:
            state = None, None
        time_state, channel_state = state
        x_BC, v0_BC, time_state = self.time_mixer(in_BC=in_BC, v0_BC=v0_BC, state=time_state)
        if not self.cmix_stripped:
            x_BC, channel_state = self.channel_mixer(x_BC, state=channel_state)
        return x_BC, v0_BC, (time_state, channel_state)


class VariantRWKV7RNN(RWKV7RNN):
    def __init__(self, config, stream_name, strip_cmix=(), strip_l0_vlora=False):
        torch.nn.Module.__init__(self)
        stripped = {int(entry.partition(":")[2]) for entry in strip_cmix if entry.partition(":")[0] == stream_name}
        self.blocks = torch.nn.ModuleList([
            VariantRWKV7RNNLayer(config, layer_id, strip_cmix=layer_id in stripped,
                                 strip_v_lora=strip_l0_vlora and layer_id == 0)
            for layer_id in range(config.n_layers)])


class VariantSrsRWKVRnn(SrsRWKVRnn):
    def __init__(self, variant):
        torch.nn.Module.__init__(self)
        config = variant.base
        self.variant = variant
        self.streams = [name for name, _ in config.modules]
        self.gru_n = int(variant.gru_head)
        self.gru_on = self.gru_n > 0
        self.no_ahead_residual = bool(variant.no_ahead_residual or self.gru_on)
        self.zero_features = list(variant.zero_features)
        self.interleave_on = bool(variant.interleave)
        self.layer_steps = layer_steps(interleave_schedule([c.n_layers for _, c in config.modules]))
        self.card_features_dim = 92
        self.d_model = config.d_model
        self.features_fc_dim = config.features_fc_mult * config.d_model
        self.ahead_head_dim = config.head_fc_mult * self.d_model
        self.p_head_dim = config.head_fc_mult * self.d_model
        self.w_head_dim = config.head_fc_mult * self.d_model
        self.num_curves = self.gru_n if self.gru_on else config.num_curves
        self.features2card = torch.nn.Sequential(
            torch.nn.Linear(self.card_features_dim, self.features_fc_dim), torch.nn.SiLU(),
            torch.nn.LayerNorm(self.features_fc_dim), torch.nn.Linear(self.features_fc_dim, self.d_model),
            torch.nn.SiLU())
        self.rwkv_modules = torch.nn.ModuleList([
            VariantRWKV7RNN(c, name, variant.strip_cmix, variant.strip_l0_vlora) for name, c in config.modules])
        self.prehead_norm = torch.nn.LayerNorm(self.d_model)
        self.prehead_dropout = torch.nn.Dropout(p=config.dropout)
        if self.gru_on:
            self.head_ahead_logits = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
        else:
            self.head_ahead_logits = torch.nn.Sequential(torch.nn.Linear(self.d_model, self.ahead_head_dim),
                                                         torch.nn.ReLU())
        self.head_w = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model), torch.nn.ReLU(), torch.nn.LayerNorm(self.d_model),
            torch.nn.Dropout(p=0.1), torch.nn.Linear(self.d_model, self.w_head_dim))
        self.head_p = torch.nn.Sequential(torch.nn.Linear(self.d_model, self.p_head_dim), torch.nn.ReLU())
        self.max_e = 21
        self.point_spread = 18.5
        self.num_points = config.num_points
        if self.gru_on:
            n = self.gru_n
            self.ahead_linear = torch.nn.Linear(1, 1)
            self.w_linear = torch.nn.Linear(1, 1)
            self.gru_w_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
            self.gru_w_bias = torch.nn.Parameter(torch.zeros(n))
            self.gru_s_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
            self.gru_s_bias = torch.nn.Parameter(torch.linspace(math.log(3600.0), math.log(31536000.0), n))
            self.gru_d_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
            self.gru_d_bias = torch.nn.Parameter(torch.full((n,), math.log(0.5)))
        else:
            self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)
            self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)
        self.s_point_spread = 18.5
        self.s_max = 22
        self.p_linear = torch.nn.Linear(self.p_head_dim, 4)

    def gru_forgetting_curve(self, w, s_raw, d_raw, label_elapsed_seconds):
        s = torch.exp(torch.clamp(s_raw, min=-25.0, max=25.0))
        d = torch.exp(torch.clamp(d_raw, min=-25.0, max=25.0))
        t = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(w * torch.exp(-d * torch.log1p(t / (1e-7 + s))), dim=-1)

    def step(self, card_features, states):
        """One review. `states` maps data stream name -> that entity's per-layer state dict (or None).

        Returns (head outputs dict, next states dict); the caller commits next states only for non-skipped rows.
        """
        assert card_features.dim() == 2
        if self.zero_features:
            card_features = card_features.clone()
            card_features[:, self.zero_features] = 0
        x = self.features2card(card_features)
        next_states = {}
        for i, stream in enumerate(self.streams):
            previous = states.get(stream)
            next_states[stream] = dict(previous) if previous is not None else {}
        if self.interleave_on:
            v0 = [torch.empty_like(x) for _ in self.streams]
            for i, layer in self.layer_steps:
                stream = self.streams[i]
                x, v0[i], next_states[stream][layer] = self.rwkv_modules[i].blocks[layer](
                    in_BC=x, v0_BC=v0[i], state=next_states[stream].get(layer))
        else:
            for i, stream in enumerate(self.streams):
                v0 = torch.empty_like(x)
                for layer, block in enumerate(self.rwkv_modules[i].blocks):
                    x, v0, next_states[stream][layer] = block(in_BC=x, v0_BC=v0, state=next_states[stream].get(layer))
        h = self.prehead_dropout(self.prehead_norm(x))
        x_w = self.head_w(h).float()
        out = {}
        if self.gru_on:
            logits = torch.nn.functional.linear(x_w, self.gru_w_weight, self.gru_w_bias)
            out["s"] = torch.nn.functional.linear(x_w, self.gru_s_weight, self.gru_s_bias)
            out["d"] = torch.nn.functional.linear(x_w, self.gru_d_weight, self.gru_d_bias)
        else:
            logits = self.w_linear(x_w)
        out["w"] = torch.nn.functional.softmax(logits, dim=-1)
        if self.no_ahead_residual:
            out["ahead"] = torch.zeros(x.size(0), self.num_points, dtype=torch.float32, device=x.device)
        else:
            out["ahead"] = self.ahead_linear(self.head_ahead_logits(h).float())
        out["p_logits"] = self.p_linear(self.head_p(h).float())
        return out, next_states

    def curve_probability(self, out, label_elapsed_seconds):
        """Recall probability at `label_elapsed_seconds` (shape (B, 1)), as the parallel model's p_curve."""
        if self.gru_on:
            raw = self.gru_forgetting_curve(out["w"], out["s"], out["d"], label_elapsed_seconds)
        else:
            raw = self.forgetting_curve(out["w"], label_elapsed_seconds)
        return torch.sigmoid(torch.log(raw / (1 - raw)) + self.interp(out["ahead"], label_elapsed_seconds))

    @torch.inference_mode()
    def predict_rows(self, features, ids, skips, label_elapsed_seconds):
        """Sequential predictions for rows in time order: features (N, 92); ids {stream: (N,) ids}; skips (N,) bool.

        Returns (p_curve (N,), p_imm (N,)) like SrsRWKVIterStatistics.p_curve / p_imm.
        """
        entity_states = {stream: {} for stream in self.streams}
        curve, instant = [], []
        for row in range(features.size(0)):
            keys = {stream: int(ids[stream][row]) for stream in self.streams}
            states = {stream: entity_states[stream].get(keys[stream]) for stream in self.streams}
            out, next_states = self.step(features[row:row + 1], states)
            if not bool(skips[row]):
                for stream in self.streams:
                    entity_states[stream][keys[stream]] = next_states[stream]
            curve.append(self.curve_probability(out, label_elapsed_seconds[row].view(1, 1)).view(()))
            p = torch.softmax(out["p_logits"], dim=-1)
            instant.append(torch.clamp(1.0 - p[0, 0], min=1e-5, max=1.0 - 1e-5))
        return torch.stack(curve), torch.stack(instant)

    def review(self, card_features, card_state, note_state, deck_state, preset_state, global_state):
        """SrsRWKVRnn.review's signature and return layout (ahead logits, w, rating logits, card, note, deck, preset,
        user states), with states routed by data stream. GRU variants also need `step` for the S and d outputs."""
        states = {"card_id": card_state, "note_id": note_state, "deck_id": deck_state, "preset_id": preset_state,
                  "user_id": global_state}
        out, nxt = self.step(card_features, copy.copy(states))
        return (out["ahead"], out["w"], out["p_logits"], nxt["card_id"], nxt["note_id"], nxt["deck_id"],
                nxt["preset_id"], nxt["user_id"])

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                target_dtype = torch.float32 if is_excluded(name) else dtype
                param.data.copy_(master_params[name].to(target_dtype))
