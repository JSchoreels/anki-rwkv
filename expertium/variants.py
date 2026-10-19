"""Opt-in model variants from Expertium's 5,000-user era (docs/EXPERTIUM_CATCH_UP.md).

The default model stays the vendored iter36 class (`expertium.model.srs_model.SrsRWKV` with
`expertium.architecture.DEFAULT_ANKI_RWKV_CONFIG`); nothing here is imported on the default path.
A variant is selected by ONE key, the R01 track option `"architecture"`:

    "architecture": "w80"                       # the iter-53 champion: d=80, GRU N=3, interleave, ...
    "architecture": "w80-sequential"            # same without the round-robin layer schedule
    "architecture": "w64"                       # s64h, his shipped d=64 trunk (w80 recipe, 4 heads, mults 2)
    "architecture": "iter36"                    # the default model through the variant class (KD-capable)
    "architecture": "w64-d2p1u1"                # w64 with the trailing deck/preset/user layers dropped (SM15 depth cut;
                                                # "-nocmix" also strips the card:0, deck:0, note:0 channel mixers)
    "architecture": {"base": "iter36", "zero_features": [22]}   # any preset plus field overrides

Each field below mirrors one upstream environment switch, named in its comment. Upstream sources are pinned
in expertium/upstream.json ("catch_up").
"""
from dataclasses import dataclass, field, fields, replace
import re

from expertium.architecture import AnkiRWKVConfig, DEFAULT_ANKI_RWKV_CONFIG
from expertium.model.rwkv_model import RWKV7Config


@dataclass
class VariantConfig:
    name: str
    base: AnkiRWKVConfig
    gru_head: int = 0                 # RWKV_GRU_HEAD=N: N predicted (w, S, d) power curves replace the 64/128-basis mixture
    no_ahead_residual: bool = False   # RWKV_NO_AHEAD_RESIDUAL=1 (forced on by gru_head)
    zero_features: tuple = ()         # RWKV_ZERO_FEATURES="22": input dims zeroed in training AND evaluation
    interleave: bool = False          # RWKV_INTERLEAVE=1: round-robin layer schedule (front-loaded, iter 41)
    strip_l0_vlora: bool = False      # RWKV_STRIP_L0_VLORA=1: layer 0's dead value-residual LoRA becomes a 1x1 dummy
    strip_cmix: tuple = ()            # RWKV_STRIP_CMIX="<stream>:<layer>,...": channel mixers removed
    extra: dict = field(default_factory=dict)

    def stream_names(self):
        return [name for name, _ in self.base.modules]


# ---- width 80: scratchpad/track2_a18/architecture_d80_lora4_cnd.py at a9fcf466, run with RWKV_DROPOUT_SCALE=0.5 ----
_HEAD_DIM, _N_HEADS, _DROPOUT_SCALE = 16, 5, 0.5
_DROPOUT, _DROPOUT_LONG, _DROPOUT_LAYER = 0.02 * _DROPOUT_SCALE, 0.05 * _DROPOUT_SCALE, 0.01 * _DROPOUT_SCALE


def _w80_stream(n_layers, dropout, heads=_N_HEADS):
    return RWKV7Config(d_model=_HEAD_DIM * heads, n_heads=heads, n_layers=n_layers, layer_offset=0,
                       total_layers=n_layers, channel_mixer_factor=1.0, decay_lora=4, a_lora=4, v0_mix_amt_lora=2,
                       gate_lora=4, dropout=dropout, dropout_layer=_DROPOUT_LAYER)


def w80_config(heads=_N_HEADS, mult=4):
    # Each stream keeps the champion's depth, mixer and dropout by NAME. The order is this repository's batch chain:
    # rwkv.prepare_batch.prepare gathers streams in rwkv.architecture's tuple order (card, deck, note, preset, user)
    # for every model, so module i always processes that stream. Upstream iter 41 moved note before deck by
    # changing prepare(); iter 43 measured the order change as free (docs/EXPERTIUM_CATCH_UP.md).
    modules = [("card_id", _w80_stream(2, _DROPOUT, heads)), ("deck_id", _w80_stream(4, _DROPOUT_LONG, heads)),
               ("note_id", _w80_stream(1, _DROPOUT, heads)), ("preset_id", _w80_stream(3, _DROPOUT_LONG, heads)),
               ("user_id", _w80_stream(3, _DROPOUT_LONG, heads))]
    return AnkiRWKVConfig(d_model=_HEAD_DIM * heads, modules=modules, dropout=_DROPOUT, num_curves=128, num_points=128,
                          head_fc_mult=mult, features_fc_mult=mult)


def w64_config():
    # s64h, his shipped d=64 trunk (adopted 2026-09-22/23): the w80 recipe at 4 heads x 16 with the head and
    # input-encoder multipliers halved (4 -> 2). His architecture file (scratchpad/shrink/architecture_s64h.py) is not
    # published; this reconstruction reproduces his parameter counts exactly at his 103-input layout: 332,922 for s64h
    # and, with multipliers 4/4, 372,986 for his `s64` arm (docs/EXPERTIUM_CATCH_UP.md).
    return w80_config(heads=4, mult=2)


# rwkv.prepare_batch.prepare builds gathers in this order (rwkv.architecture.DEFAULT_ANKI_RWKV_CONFIG's tuples).
BATCH_STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]
# iter 53 runner (scratchpad/iter53_muonlora/run_iter53.cmd): RWKV_STRIP_CMIX value, verbatim.
W80_STRIP_CMIX = ("user_id:0", "user_id:1", "user_id:2", "preset_id:0", "preset_id:1", "preset_id:2",
                  "deck_id:1", "deck_id:2", "card_id:1")
# 558,212 upstream minus pava_theta's 3 floats (PAVA is not ported; see docs/EXPERTIUM_CATCH_UP.md).
W80_PARAMETERS = 558_209
W64_PARAMETERS = 331_511  # 332,922 upstream at 103 inputs, minus 11 x 128 input weights and PAVA's 3
ITER36_PARAMETERS = 192_800


# SM15 (docs/SMALL_MODEL_PRUNING.md 2.4): depth cuts of the context streams. `w64-d{D}p{P}u{U}[-nocmix]` keeps D deck,
# P preset and U user layers (the LEADING ones: interleave_schedule runs layer r of every stream in round r, so the
# survivors are an exact subsequence of w64's execution order); card (2) and note (1) stay intact. Width, heads,
# mults, LoRAs, GRU head, interleave and zero_features are w64's. strip_cmix keeps only entries of surviving layers.
PRUNED_W64 = re.compile(r"^w64-d([1-4])p([1-3])u([1-3])(-nocmix)?$")
PRUNED_STRIP_CMIX = ("card_id:0", "deck_id:0", "note_id:0")  # the channel mixers w64 still has below the last deck layer
# asserted by the self-test / CPU counts: w64-d2p2u2 244,232; d2p1u2 and d2p2u1 224,507; d2p1u1 204,782; d2p1u1-nocmix 179,645
PRUNED_PARAMETERS = {"w64-d2p2u2": 244_232, "w64-d2p1u2": 224_507, "w64-d2p2u1": 224_507, "w64-d2p1u1": 204_782,
                     "w64-d2p1u1-nocmix": 179_645}


def pruned_w64(name, decks, presets, users, nocmix):
    parent = preset("w64")
    depths = {"card_id": 2, "deck_id": decks, "note_id": 1, "preset_id": presets, "user_id": users}
    modules = [(stream, replace(config, n_layers=depths[stream], total_layers=depths[stream]))
               for stream, config in parent.base.modules]
    strip = [entry for entry in parent.strip_cmix if int(entry.partition(":")[2]) < depths[entry.partition(":")[0]]]
    if nocmix:
        strip += [entry for entry in PRUNED_STRIP_CMIX if entry not in strip]
    return replace(parent, name=name, base=replace(parent.base, modules=modules), strip_cmix=tuple(strip))


def preset(name):
    pruned = PRUNED_W64.match(name) if isinstance(name, str) else None
    if pruned:
        return pruned_w64(name, int(pruned[1]), int(pruned[2]), int(pruned[3]), bool(pruned[4]))
    if name == "iter36":
        return VariantConfig(name="iter36", base=DEFAULT_ANKI_RWKV_CONFIG)
    if name in ("w80", "w80-sequential"):
        return VariantConfig(name=name, base=w80_config(), gru_head=3, no_ahead_residual=True, zero_features=(22,),
                             interleave=name == "w80", strip_l0_vlora=True, strip_cmix=W80_STRIP_CMIX)
    if name in ("w64", "w64-sequential"):
        return VariantConfig(name=name, base=w64_config(), gru_head=3, no_ahead_residual=True, zero_features=(22,),
                             interleave=name == "w64", strip_l0_vlora=True, strip_cmix=W80_STRIP_CMIX)
    raise ValueError(f"Unknown Expertium architecture preset {name!r}; expected iter36, w64, w64-sequential, w80, "
                     "w80-sequential or w64-d{1-4}p{1-3}u{1-3}[-nocmix]")


def variant(spec):
    """Resolve the `"architecture"` option: a preset name or {"base": <preset>, <field>: <value>, ...}."""
    if isinstance(spec, VariantConfig):
        return spec
    if isinstance(spec, str):
        config = preset(spec)
    elif isinstance(spec, dict):
        overrides = dict(spec)
        config = preset(overrides.pop("base", "iter36"))
        allowed = {f.name for f in fields(VariantConfig)} - {"name", "base", "extra"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unknown architecture fields {sorted(unknown)}; allowed: {sorted(allowed)}")
        for key in ("zero_features", "strip_cmix"):
            if key in overrides:
                overrides[key] = tuple(overrides[key])
        config = replace(config, name=config.name + "+" + ",".join(f"{k}={v}" for k, v in sorted(overrides.items())),
                         **overrides)
    else:
        raise TypeError("architecture must be a preset name or a dict")
    if config.gru_head:
        config = replace(config, no_ahead_residual=True)
    names = config.stream_names()
    if names != BATCH_STREAMS:
        raise ValueError(f"Streams must follow the batch chain order {BATCH_STREAMS}, got {names}")
    for entry in config.strip_cmix:
        stream, _, layer = entry.partition(":")
        depth = dict(config.base.modules)[stream].n_layers if stream in names else -1
        if not layer.isdigit() or not 0 <= int(layer) < depth:
            raise ValueError(f"strip_cmix entry {entry!r} names no layer of {names}")
    for index in config.zero_features:
        if not 0 <= int(index) < 92:
            raise ValueError(f"zero_features index {index} outside the 92 input dims")
    return config


def is_default(spec):
    return spec is None


def build(spec, rnn=False):
    """Construct the parallel (or recurrent) model for an architecture spec."""
    config = variant(spec)
    if rnn:
        from expertium.model.variant_rnn import VariantSrsRWKVRnn
        return VariantSrsRWKVRnn(config)
    from expertium.model.variant_model import VariantSrsRWKV
    return VariantSrsRWKV(config)
