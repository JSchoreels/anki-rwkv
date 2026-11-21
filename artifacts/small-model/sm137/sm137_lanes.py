#!/usr/bin/env python
"""Trained smaller state for Tiny (SM137): the card / note state lane cut of the six-layer Tiny (SM130), CPU only.
Design: artifacts/small-model/sm137/design.md.

The cut: in the card and note streams' time mixers (one layer each), every head keeps its first V value lanes (WKV state
rows) and first K key lanes (state columns). The model zeroes the other k lanes after W_k and v lanes after W_v
(RWKV7TimeMixer.set_state_lanes, architecture `<preset>-v{V}k{K}`), so those state rows / columns stay exactly 0 at
every step and the stored state is a V x K block per head. Before the cut, the lanes of every head are permuted so that
the lanes with the most stored-state energy come first (an exact symmetry of the model up to float summation order).

  users                 write users-lanes.json (energy users) and users-screen.json (zero-shot screen users): SM116's
                        direction users (validation ids >= 5000, outside the 732 set) with 15,000-40,000 reviews, sorted
                        by review count; even positions -> energy, odd positions -> screen
  stats --chain C       per head, the energy (sum of squares) of every state row and column of every STORED card / note
                        WKV state (the CPU scorer's Hooks, before the int8 rounding), over the energy users; also counts
                        values outside the kept block for a lane model (--arch/--checkpoint) -> work/stats-*.json
  mask --chain C --value-lanes V --key-lanes K --stats F --out DIR
                        DIR/master.pth = the chain's start (SM130 io epoch 41 / th epoch 32, sha256 pinned) with the card
                        and note lanes permuted by energy and the dropped W_k / W_v rows set to 0; DIR/optimizer.pth = its
                        AdamW state permuted the same way, both moments of the zeroed rows set to 0; DIR/extend.json
  verify DIR_OR_PTH --value-lanes V --key-lanes K
                        the dropped W_k / W_v rows of a (trained) master are still exactly 0 (exit 1 otherwise)
  bytes                 bytes per card / note of the release-candidate format (int8 rows, float16 row scales, int8
                        shifts) per lane shape, and the lane codec round trip (random + real states)
  cpu-check --out F     the CPU proofs before any GPU ticket (L1-L6 below; minutes, < 3 GB RAM)

usage: python sm137_lanes.py users | stats --chain io [--arch A --checkpoint P --tag T] | mask ... | verify ... |
       bytes [--states work/states-io-v16k8-cut.pt] | cpu-check --out check-cpu-sm137.json
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # the sm137 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
WORK = HERE / "work"
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
ARCH = "w64-c1-d2p1u1"
H, K = 4, 16
STREAMS = {0: "card", 2: "note"}             # rwkv_modules index -> stored-state stream
SM116_DIRECTIONS = W / "sm116/artifacts/small-model/sm116/users-directions.json"
BASELINE_COMMIT = "c131b3ae7f"               # sm135 tip = this branch's code before the SM137 source change
CHAINS = {   # the start of the first lane epoch (read only, sha256 pinned as in sm135_epoch.py)
    "io": {"case": W / "sm130/artifacts/small-model/sm130-io-e41-st/expertium" / TRACK,
           "master_sha": "af16139e28bbd322f42790bfed25d5be2c62a92351a9045b14d3e6b66e7c96ff",
           "optimizer_sha": "c7d00f0ef2e25694b5470b960f59ee647c1efa282af211b1a9b213a3a42ed48f",
           "name": "the six-layer Tiny (SM130)'s Instant-Only epoch 41"},
    "th": {"case": W / "sm130/artifacts/small-model/sm130-th-e32-st/expertium" / TRACK,
           "master_sha": "635f1a12893b4d87c7c946e6311103de3aa519417535919fb4c5e13314ea6a46",
           "optimizer_sha": "33fe84765252fade25805958cd62d5abb9b53ebd81c3749f070c2d11b4b013d0",
           "name": "the six-layer Tiny (SM130)'s two-head epoch 32"},
}
# tensors of a time mixer indexed by its key lanes (rows unless noted) and by its value lanes
KEY_ROWS = ("W_k.weight", "W_r.weight", "a_lora_simple.B_and_lamb.weight", "a_lora_simple.B_and_lamb.bias",
            "d_lora_mlp.B_and_lamb.weight", "d_lora_mlp.B_and_lamb.bias")
KEY_BONUS = "bonus"                          # (1, 1, H, K): key lanes on the last two dims
VALUE_ROWS = ("W_v.weight", "lora_B_g.weight", "out_group_norm.weight", "out_group_norm.bias")
VALUE_COLS = ("W_o.weight",)
# release-candidate entity layout (fork ModuleState::compact, SM136 sm136_bytes.py w8+s8hx8): 14-byte magic + precision
# byte + u32 layer count; per layer three u32 lengths, the time-shift block (f16 scale + 64 int8), the matrix block (per
# head per kept row: f16 scale + kept-column int8 codes), the channel-shift block (f16 scale + 64 int8)
HEADER, LAYER_LENGTHS, SHIFT_BLOCK = 14 + 1 + 4, 12, 2 + 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def lane_arch(v, k):
    return ARCH if (v, k) == (K, K) else f"{ARCH}-v{v}k{k}"


def entity_bytes(v, k, layers=1, channel_mixers=1):
    """Bytes of one stored card or note (one layer, channel mixer kept: Tiny's card and note) at the RC format."""
    return HEADER + layers * (LAYER_LENGTHS + SHIFT_BLOCK + H * v * (2 + k)) + channel_mixers * SHIFT_BLOCK


# ----------------------------------------------------------------------------------------------- users

def cmd_users(_args):
    from scripts import compress_common as cc
    from scripts.eval_pools import pool_users
    counts = cc.run_info()["review_counts"]
    doc = json.loads(SM116_DIRECTIONS.read_text())
    users = doc["users_used"] if isinstance(doc.get("users_used"), list) else doc["users"]
    users = sorted((u for u in users if 15_000 <= counts[str(u)] <= 40_000), key=lambda u: (counts[str(u)], u))
    if set(users) & set(pool_users(732)):
        raise SystemExit("a direction user is in the 732 set")
    for name, part, what in (("users-lanes.json", users[0::2], "lane energy (the permutation before the cut)"),
                             ("users-screen.json", users[1::2], "zero-shot screen (information only, no decision)")):
        atomic_json(HERE / name, {"note": f"Trained smaller state for Tiny (SM137): {what} users. The training-free state "
                                          "check (SM116)'s direction users (validation users with id >= 5000, outside the "
                                          "732 set) with 15,000-40,000 reviews, sorted by review count, every second one. "
                                          "Public benchmark user ids only.",
                                  "users": part, "reviews": sum(counts[str(u)] for u in part)})
        print(f"{name}: {len(part)} users, {sum(counts[str(u)] for u in part):,} reviews")
    return 0


# ----------------------------------------------------------------------------------------------- state energy

def _wrap_configure(on_wkv):
    """scripts.state_vq.configure patched so every card / note WKV hook sees the float stored state first (SM116)."""
    from scripts import state_vq
    original = state_vq.configure

    def configure(layout, codecs=None, int_bits=0, shift_bits=0):
        original(layout, codecs, int_bits, shift_bits)
        hooks = state_vq.Hooks
        for stream, info in layout.items():
            for layer in range(info["wkv_layers"]):
                stock = hooks.wkv.get((stream, layer))
                hooks.wkv[(stream, layer)] = (lambda s, cur, rows, t, st=stock, key=(stream, layer):
                                              (lambda y: y if st is None else st(y, cur, rows, t))(on_wkv(s, key)))
    state_vq.configure = configure


def cmd_stats(args):
    """In-process CPU scorer run with an energy hook; writes work/stats-<tag>.json."""
    import atexit
    import torch
    c = CHAINS[args.chain]
    ckpt = Path(args.checkpoint) if args.checkpoint else c["case"] / "master.pth"
    arch = args.arch or ARCH
    v_keep, k_keep = (args.value_lanes, args.key_lanes)
    tag = args.tag or f"{args.chain}-{'src' if not args.checkpoint else Path(ckpt).parent.name}"
    out = WORK / f"stats-{tag}.json"
    WORK.mkdir(parents=True, exist_ok=True)
    acc = {}
    keep_states = [] if args.keep_states else None

    def on_wkv(S, key):
        S = S.detach()
        e = acc.setdefault(key, {"n": 0, "rows": torch.zeros(S.shape[1], S.shape[2], dtype=torch.float64),
                                 "cols": torch.zeros(S.shape[1], S.shape[3], dtype=torch.float64),
                                 "outside_nonzero": 0, "outside_max": 0.0})
        sq = S.double() ** 2
        e["n"] += S.shape[0]
        e["rows"] += sq.sum(dim=(0, 3))
        e["cols"] += sq.sum(dim=(0, 2))
        if v_keep < S.shape[2] or k_keep < S.shape[3]:
            mask = torch.ones(S.shape[2], S.shape[3], dtype=torch.bool)
            mask[:v_keep, :k_keep] = False
            outside = S[:, :, mask]
            e["outside_nonzero"] += int((outside != 0).sum())
            e["outside_max"] = max(e["outside_max"], float(outside.abs().max()) if outside.numel() else 0.0)
        if keep_states is not None and len(keep_states) < 64:
            keep_states.append(S[: max(1, 64 // max(1, len(acc)))].clone())
        return S

    def save():
        if not acc:
            return
        doc = {"chain": args.chain, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "arch": arch,
               "users_file": str(args.users_file), "kept_block": [v_keep, k_keep], "streams": {}}
        for (stream, layer), e in sorted(acc.items()):
            doc["streams"][f"{STREAMS[stream]}.L{layer}"] = {
                "stored_states": e["n"], "row_energy": e["rows"].tolist(), "col_energy": e["cols"].tolist(),
                "outside_nonzero": e["outside_nonzero"], "outside_max_abs": e["outside_max"]}
        atomic_json(out, doc)
        if keep_states:
            torch.save(torch.cat(keep_states)[:64].float().clone(), WORK / f"states-{tag}.pt")
        print(f"[sm137-lanes] stats of {len(acc)} streams -> {out}", flush=True)

    atexit.register(save)
    torch.set_grad_enabled(False)
    _wrap_configure(on_wkv)
    from scripts import small_model_heldout
    sys.argv = ["small_model_heldout.py", "--device", "cpu", "--pool", "1500", "--users-file", str(args.users_file),
                "--model", f"{tag}=expertium:{arch}:{ckpt}", "--only", f"{tag}/w8+s8hx8", "--inputs", "e2s-query",
                "--split-existing", "--deleted-cards", "neighbour", "--time-limit", "100000",
                "--out", str(WORK / f"stats-{tag}-rows.jsonl")]
    small_model_heldout.main()
    return 0


# ----------------------------------------------------------------------------------------------- the cut

def permutations(stats, v_keep, k_keep):
    """Per stream prefix: (value permutation, key permutation) over the 64 channels, kept-first by energy per head."""
    import torch
    perms = {}
    for index, name in STREAMS.items():
        e = stats["streams"][f"{name}.L0"]
        rows, cols = torch.tensor(e["row_energy"]), torch.tensor(e["col_energy"])
        pv, pk, kept = [], [], {"value_energy_kept": [], "key_energy_kept": []}
        for h in range(H):
            ov = sorted(range(K), key=lambda i: (-float(rows[h, i]), i))
            ok = sorted(range(K), key=lambda j: (-float(cols[h, j]), j))
            pv += [h * K + i for i in ov]
            pk += [h * K + j for j in ok]
            kept["value_energy_kept"].append(float(rows[h, ov[:v_keep]].sum() / rows[h].sum()))
            kept["key_energy_kept"].append(float(cols[h, ok[:k_keep]].sum() / cols[h].sum()))
        perms[f"rwkv_modules.{index}.blocks.0.time_mixer."] = (pv, pk, kept)
    return perms


def apply_lanes(sd, perms, v_keep, k_keep, moments=False):
    """Permute (and zero the dropped rows of) one state dict or one AdamW moment dict keyed by parameter name."""
    import torch
    out = dict(sd)
    for prefix, (pv, pk, _) in perms.items():
        iv, ik = torch.tensor(pv), torch.tensor(pk)
        lane = torch.arange(H * K) % K
        for name in KEY_ROWS:
            if prefix + name in out:
                out[prefix + name] = out[prefix + name][ik].clone()
        if prefix + KEY_BONUS in out:
            b = out[prefix + KEY_BONUS]
            out[prefix + KEY_BONUS] = b.reshape(-1)[ik].reshape(b.shape).clone()
        for name in VALUE_ROWS:
            if prefix + name in out:
                out[prefix + name] = out[prefix + name][iv].clone()
        for name in VALUE_COLS:
            if prefix + name in out:
                out[prefix + name] = out[prefix + name][:, iv].clone()
        for name, drop in (("W_k.weight", lane >= k_keep), ("W_v.weight", lane >= v_keep)):
            if prefix + name in out:
                t = out[prefix + name].clone()
                t[drop] = 0
                out[prefix + name] = t
    return out


def cmd_mask(args):
    import torch
    from scripts import sm77_warm as warm
    from expertium.variants import build
    c = CHAINS[args.chain]
    v_keep, k_keep = args.value_lanes, args.key_lanes
    master, optim = c["case"] / "master.pth", c["case"] / "optimizer.pth"
    if not args.unpinned and (sha256(master) != c["master_sha"] or sha256(optim) != c["optimizer_sha"]):
        raise SystemExit(f"{c['name']}: master / optimizer sha256 are not the pinned ones")
    stats = json.loads(Path(args.stats).read_text())
    if stats.get("checkpoint_sha256") != sha256(master):
        raise SystemExit("the stats were not collected on this start checkpoint")
    perms = permutations(stats, v_keep, k_keep)
    old = torch.load(master, map_location="cpu", weights_only=True)
    new = apply_lanes(old, perms, v_keep, k_keep)
    model = build(lane_arch(v_keep, k_keep)).float()
    model.load_state_dict(new, strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(new, out / "master.pth")
    # AdamW state: same layout as the source (the architecture adds no parameter); moments permuted like the weights
    groups, optimizer = warm.adamw_names(build(ARCH).float())
    state = torch.load(optim, map_location="cpu", weights_only=True)
    if [len(g) for g in groups] != [len(g["params"]) for g in state["param_groups"]]:
        raise SystemExit("the source optimizer.pth does not have the constant-LR AdamW group layout")
    names = [n for g in groups for n in g]
    new_state = {}
    for key in ("exp_avg", "exp_avg_sq"):
        moments = {names[i]: s[key] for i, s in state["state"].items() if key in s}
        moved = apply_lanes(moments, perms, v_keep, k_keep, moments=True)
        for i, s in state["state"].items():
            new_state.setdefault(i, dict(s))[key] = moved[names[i]]
    remapped = {"state": new_state, "param_groups": state["param_groups"]}
    optimizer.load_state_dict(remapped)
    check = optimizer.state_dict()
    reload_ok = all(torch.equal(check["state"][i][k2], new_state[i][k2]) for i in new_state for k2 in new_state[i])
    torch.save(remapped, out / "optimizer.pth")
    lane = torch.arange(H * K) % K
    zero_ok = all(bool((new[p + n][lane >= keep] == 0).all()) and
                  all(bool((new_state[names.index(p + n)][m][lane >= keep] == 0).all()) for m in ("exp_avg", "exp_avg_sq"))
                  for p in perms for n, keep in (("W_k.weight", k_keep), ("W_v.weight", v_keep)))
    rec = {"init": "sm137 lane cut", "chain": args.chain, "architecture": lane_arch(v_keep, k_keep), "source_architecture": ARCH,
           "value_lanes": v_keep, "key_lanes": k_keep, "source_master": str(master), "source_master_sha256": sha256(master),
           "source_optimizer": str(optim), "source_optimizer_sha256": sha256(optim), "stats": str(args.stats),
           "stats_users_file": stats.get("users_file"),
           "permutations": {p: {"value": pv, "key": pk, **kept} for p, (pv, pk, kept) in perms.items()},
           "master_sha256": sha256(out / "master.pth"), "optimizer_sha256": sha256(out / "optimizer.pth"),
           "optimizer_states_equal_after_load": reload_ok, "dropped_rows_and_moments_zero": zero_ok,
           "parameters": sum(p.numel() for p in model.parameters()),
           "bytes_per_card": entity_bytes(v_keep, k_keep), "bytes_per_note": entity_bytes(v_keep, k_keep)}
    if not (reload_ok and zero_ok):
        raise SystemExit(f"mask failed: reload {reload_ok}, zero rows {zero_ok}")
    atomic_json(out / "extend.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "permutations"}))
    return 0


def dropped_rows_zero(sd, v_keep, k_keep):
    import torch
    lane = torch.arange(H * K) % K
    bad = []
    for index in STREAMS:
        p = f"rwkv_modules.{index}.blocks.0.time_mixer."
        for n, keep in (("W_k.weight", k_keep), ("W_v.weight", v_keep)):
            t = sd[p + n][lane >= keep]
            if not bool((t == 0).all()):
                bad.append(f"{p + n}: max |dropped row| {float(t.abs().max()):.3g}")
    return bad


def cmd_verify(args):
    import torch
    path = Path(args.path)
    path = path / "expertium" / TRACK / "master.pth" if path.is_dir() and (path / "expertium").is_dir() else \
        (path / "master.pth" if path.is_dir() else path)
    bad = dropped_rows_zero(torch.load(path, map_location="cpu", weights_only=True), args.value_lanes, args.key_lanes)
    print(f"[sm137-lanes] verify {path}: " + ("dropped W_k / W_v rows exactly 0" if not bad else "; ".join(bad)))
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------------- lane codec

def _quant_rows(x):
    """The scorer's int8 row rounding with float16 scales (scripts.state_vq.int_row_quant, bits 8 | F16_SCALE)."""
    import torch
    from scripts.small_model_wkv import round_scale_f16
    scale = round_scale_f16(x.abs().amax(-1, keepdim=True) / 127)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round(x / scale), -127, 127)
    return codes, scale


def enc_matrix(s, v_keep, k_keep):
    """s = [H, K, K] float state (rows = value lanes); writes the kept V x K block per head: f16 scale + int8 codes."""
    import numpy as np
    codes, scale = _quant_rows(s[:, :v_keep, :k_keep])
    out = bytearray()
    for h in range(H):
        for r in range(v_keep):
            out += np.float16(scale[h, r, 0].item()).tobytes() + codes[h, r].numpy().astype(np.int8).tobytes()
    return bytes(out), codes * scale


def dec_matrix(buf, v_keep, k_keep):
    import numpy as np
    import torch
    out, pos = torch.zeros(H, K, K), 0
    for h in range(H):
        for r in range(v_keep):
            scale = float(np.frombuffer(buf[pos:pos + 2], dtype=np.float16)[0])
            codes = np.frombuffer(buf[pos + 2:pos + 2 + k_keep], dtype=np.int8).astype(np.float32)
            out[h, r, :k_keep] = torch.from_numpy(codes) * torch.tensor(scale)
            pos += 2 + k_keep
    return out, pos


def codec_roundtrip(states, v_keep, k_keep):
    """(ok, bytes) for states [N, H, K, K] that are exactly 0 outside the kept block: decode(encode(x)) == the scorer's
    quantization of the full state (zeros stay zeros), bitwise; matrix bytes = H x V x (2 + K)."""
    import torch
    ok, size = True, None
    for s in states:
        buf, _ = enc_matrix(s, v_keep, k_keep)
        dec, used = dec_matrix(buf, v_keep, k_keep)
        codes, scale = _quant_rows(s)          # the full 16 x 16 rounding the scorer applies (zero rows -> scale 1)
        ok &= used == len(buf) and torch.equal(dec, codes * scale)
        size = len(buf)
    return ok, size


def synthetic_states(v_keep, k_keep, n=40, seed=137):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, H, K, K, generator=g) * torch.logspace(-6, 3, n).view(n, 1, 1, 1)
    x[0] = 0
    x[1, :, 0] = 1e-30                         # subnormal / f16-underflow rows
    x[2, :, 1] = 1e9                           # f16 clamp
    x[:, :, v_keep:, :] = 0
    x[:, :, :, k_keep:] = 0
    return x


def cmd_bytes(args):
    import torch
    table = []
    for v, k in ((16, 16), (16, 12), (12, 16), (16, 8), (12, 12), (8, 16), (12, 8), (8, 8)):
        ok, size = codec_roundtrip(synthetic_states(v, k), v, k)
        table.append({"lanes": f"v{v}k{k}", "values_kept_pct": round(100 * v * k / (K * K), 1), "matrix_bytes": size,
                      "bytes_per_card": entity_bytes(v, k), "bytes_per_note": entity_bytes(v, k),
                      "card_plus_note": 2 * entity_bytes(v, k),
                      "saving_pct": round(100 * (1 - entity_bytes(v, k) / entity_bytes(K, K)), 1),
                      "synthetic_roundtrip_bitwise": ok})
    real = None
    if args.states and Path(args.states).exists():
        s = torch.load(args.states, map_location="cpu", weights_only=True)
        ok, size = codec_roundtrip(s, args.value_lanes, args.key_lanes)
        real = {"states": str(args.states), "count": int(s.shape[0]), "lanes": f"v{args.value_lanes}k{args.key_lanes}",
                "roundtrip_bitwise": ok, "matrix_bytes": size}
    doc = {"layout": "fork ModuleState::compact with the release-candidate blocks (w8+s8hx8: int8 rows, float16 row scales, "
                     "int8 shift vectors), one layer with its channel mixer per card and per note (Tiny); the matrix "
                     "block keeps the first V rows x first K columns of every head",
           "arms": table, "real_states": real}
    if args.out:
        atomic_json(args.out, doc)
    for row in table:
        print(f"{row['lanes']}: {row['bytes_per_card']} B per card / note ({row['saving_pct']}% less), round trip "
              f"{row['synthetic_roundtrip_bitwise']}")
    if real:
        print(f"real states {real['count']}: round trip {real['roundtrip_bitwise']}")
    return 0 if all(r["synthetic_roundtrip_bitwise"] for r in table) and (real is None or real["roundtrip_bitwise"]) else 1


# ----------------------------------------------------------------------------------------------- CPU proofs

def run_scorer(tree, models, users_file, out, dump, variants, pool="732", count=2):
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    cmd = [sys.executable, "-X", "utf8", str(Path(tree) / "scripts/small_model_heldout.py"), "--device", "cpu",
           "--pool", pool, "--users-file", str(users_file), "--count", str(count), "--variants", variants,
           "--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour", "--out", str(out),
           "--dump", str(dump), "--time-limit", "3000"]
    for tag, spec in models.items():
        cmd += ["--model", f"{tag}={spec}"]
    result = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"scorer failed in {tree}: {result.stderr[-3000:]}")
    return [json.loads(line) for line in Path(out).read_text().splitlines()]


def cmd_cpu_check(args):
    """L1 default path unchanged (trainer probe, eager + TorchScript, bitwise vs the sm135 code);
    L2 lane architecture trainer path: finite, the dropped W_k / W_v rows get exactly zero gradient (eager + TorchScript);
    L3 fork equivalence: lane architecture == plain six-layer architecture on the cut checkpoint, scorer metrics and
       predictions bitwise (fp32, w8+s8, w8+s8hx8): the fork needs no math change, the zero rows do the cut;
    L4 the lane permutation alone (V = K = 16) == the source model within 1e-5 per user and head;
    L5 stored states: no nonzero value outside the kept block (CPU scorer hooks), and the lane codec round trip;
    L6 the cut checkpoints: AdamW state reloads, dropped rows and their moments are 0."""
    import math
    import torch
    from scripts import sm77_warm as warm
    started = time.perf_counter()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    fails = []

    def check(name, ok, **detail):
        report["checks"][name] = dict(detail, ok=bool(ok))
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name, json.dumps(detail)[:400], flush=True)

    v_keep, k_keep = args.value_lanes, args.key_lanes
    cut = Path(args.cut)
    perm_only = Path(args.perm_only)
    chain = json.loads((cut / "extend.json").read_text())["chain"]
    src = CHAINS[chain]["case"] / "master.pth"
    protocol = json.loads((warm.RUN72S / "epoch-21" / "protocol.json").read_text())
    rows = sorted((r for r in protocol["rows"] if r["train_range"][1] >= 300), key=lambda r: r["train_range"][1])[:2]
    config = json.loads((warm.RUN72S / "run.json").read_text())
    options = {k2: v for k2, v in config["options"].items() if k2 not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2, architecture=ARCH)
    model = __import__("expertium.variants", fromlist=["build"]).build(ARCH)
    all_names = [n for n, _ in model.named_parameters()]
    dropped = [f"rwkv_modules.{i}.blocks.0.time_mixer.{n}" for i in STREAMS for n in ("W_k.weight", "W_v.weight")]
    with tempfile.TemporaryDirectory(dir=os.environ.get("SM137_TMP") or None) as tmp:
        tmp = Path(tmp)
        # L1: the sm135 code (the source before this branch's change): git archive of expertium/ rwkv/ scripts/ at
        # BASELINE_COMMIT into the temporary folder (a full checkout hits Windows path-length limits)
        base = tmp / "base"
        base.mkdir()
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", BASELINE_COMMIT, "expertium", "rwkv",
                                  "scripts"], check=True, capture_output=True).stdout
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(base, filter="data")
        (base / "artifacts").mkdir()
        try:
            for jit in (False, True):
                spec = {"options": options, "rows": rows, "span": 256, "arch": ARCH, "shared": all_names, "checkpoint": str(src)}
                a = warm.run_probe(ROOT, spec, str(tmp), jit=jit)
                b = warm.run_probe(base, spec, str(tmp), jit=jit)
                same = [(x["loss_hex"], x["p_digest"], x["grad_digest"]) for x in a] == \
                       [(x["loss_hex"], x["p_digest"], x["grad_digest"]) for x in b]
                check(f"L1_default_unchanged_{'jit' if jit else 'eager'}", same and len(a) == len(b) > 0,
                      users=[x["user"] for x in a], loss=[x["loss"] for x in a])
        finally:
            pass
        # L2: the lane architecture's trainer path from the cut checkpoint
        for jit in (False, True):
            dump = tmp / f"grads-{'jit' if jit else 'eager'}"
            spec = {"options": dict(options, architecture=lane_arch(v_keep, k_keep)), "rows": rows, "span": 256,
                    "arch": lane_arch(v_keep, k_keep), "shared": dropped, "checkpoint": str(cut / "master.pth"),
                    "dump": str(dump)}
            out = warm.run_probe(ROOT, spec, str(tmp), jit=jit)
            lane = torch.arange(H * K) % K
            zero = True
            for i in range(len(out)):
                g = torch.load(f"{dump}-{i}.pt", weights_only=True)
                for n in dropped:
                    keep = k_keep if "W_k" in n else v_keep
                    zero &= g[n] is not None and bool((g[n][lane >= keep] == 0).all()) and bool((g[n][lane < keep] != 0).any())
            check(f"L2_lane_trainer_path_{'jit' if jit else 'eager'}", zero and all(math.isfinite(x["loss"]) for x in out),
                  loss=[x["loss"] for x in out], dropped_rows_zero_grad=zero)
        # L3 / L4: scorer on 2 small users of the 732 set (proof only)
        users = tmp / "users.json"
        from scripts import compress_common as cc
        from scripts.eval_pools import pool_users
        counts = cc.run_info()["review_counts"]
        pick = sorted((u for u in pool_users(732) if 2_000 <= counts[str(u)] <= 6_000), key=lambda u: (counts[str(u)], u))[:2]
        users.write_text(json.dumps({"users": pick}))
        variants = "fp32,w8+s8,w8+s8hx8"
        ra = run_scorer(ROOT, {"lane": f"expertium:{lane_arch(v_keep, k_keep)}:{cut / 'master.pth'}"}, users,
                        tmp / "lane.jsonl", tmp / "dump-lane", variants)
        rb = run_scorer(ROOT, {"plain": f"expertium:{ARCH}:{cut / 'master.pth'}"}, users, tmp / "plain.jsonl",
                        tmp / "dump-plain", variants)
        va = {(r["user_id"], r["label"].split("/", 1)[1]): warm.metric_view(r) for r in ra}
        vb = {(r["user_id"], r["label"].split("/", 1)[1]): warm.metric_view(r) for r in rb}
        same_dump, n_dump = warm.dumps_equal(tmp / "dump-lane", tmp / "dump-plain", "lane", "plain")
        check("L3_fork_equivalence_bitwise", va == vb and same_dump and len(va) == 2 * 3, users=pick, rows=len(va),
              dumps=n_dump)
        rp = run_scorer(ROOT, {"perm": f"expertium:{ARCH}:{perm_only / 'master.pth'}"}, users, tmp / "perm.jsonl",
                        tmp / "dump-perm", variants)
        rs = run_scorer(ROOT, {"src": f"expertium:{ARCH}:{src}"}, users, tmp / "src.jsonl", tmp / "dump-src", variants)
        worst = 0.0
        for x in rp:
            y = next(r for r in rs if r["user_id"] == x["user_id"] and r["label"].split("/", 1)[1] == x["label"].split("/", 1)[1])
            for head in ("ahead", "instant"):
                worst = max(worst, abs(x["benchmark"][head]["LogLoss"] - y["benchmark"][head]["LogLoss"]))
        check("L4_permutation_exact_within_1e-5", worst <= 1e-5 and len(rp) == 6, max_abs_logloss_diff=worst)
        zs = {(x["user_id"], x["label"].split("/", 1)[1], h): round(100 * (x["benchmark"][h]["LogLoss"] /
              next(r for r in rs if r["user_id"] == x["user_id"] and r["label"].split("/", 1)[1] == x["label"].split("/", 1)[1])
              ["benchmark"][h]["LogLoss"] - 1), 3) for x in ra for h in ("ahead", "instant")}
        report["zero_shot_proof_users_pct"] = {f"{u}/{v}/{h}": p for (u, v, h), p in zs.items()}
    # L5: stored-state zeros (the stats run of the cut model on the energy users) and the codec on its states
    st = json.loads(Path(args.cut_stats).read_text())
    outside = {k2: v["outside_nonzero"] for k2, v in st["streams"].items()}
    check("L5_stored_state_zero_outside_block", st["kept_block"] == [v_keep, k_keep] and all(v == 0 for v in outside.values())
          and st["checkpoint_sha256"] == sha256(cut / "master.pth"), outside_nonzero=outside,
          stored_states={k2: v["stored_states"] for k2, v in st["streams"].items()})
    states = Path(args.cut_stats).with_name(Path(args.cut_stats).name.replace("stats-", "states-").replace(".json", ".pt"))
    s = torch.load(states, weights_only=True)
    ok_real, size = codec_roundtrip(s, v_keep, k_keep)
    ok_syn, _ = codec_roundtrip(synthetic_states(v_keep, k_keep), v_keep, k_keep)
    check("L5_lane_codec_roundtrip", ok_real and ok_syn and size == H * v_keep * (2 + k_keep), real_states=int(s.shape[0]),
          matrix_bytes=size, bytes_per_entity=entity_bytes(v_keep, k_keep))
    # L6
    rec = json.loads((cut / "extend.json").read_text())
    check("L6_cut_checkpoint", rec["optimizer_states_equal_after_load"] and rec["dropped_rows_and_moments_zero"]
          and rec["master_sha256"] == sha256(cut / "master.pth") and rec["optimizer_sha256"] == sha256(cut / "optimizer.pth")
          and not dropped_rows_zero(torch.load(cut / "master.pth", weights_only=True), v_keep, k_keep)
          and rec["source_master_sha256"] == CHAINS[chain]["master_sha"], parameters=rec["parameters"])
    report.update(seconds=round(time.perf_counter() - started, 1), failed=fails, ok=not fails,
                  cut=str(cut), cut_master_sha256=sha256(cut / "master.pth"))
    atomic_json(args.out, report)
    print(f"[sm137-lanes] cpu-check {'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)} ({report['seconds']} s)")
    return 0 if not fails else 1


def main():
    p = ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("users")
    s = sub.add_parser("stats")
    s.add_argument("--chain", choices=CHAINS, required=True)
    s.add_argument("--users-file", default=str(HERE / "users-lanes.json"))
    s.add_argument("--arch", default="")
    s.add_argument("--checkpoint", default="")
    s.add_argument("--tag", default="")
    s.add_argument("--value-lanes", type=int, default=K)
    s.add_argument("--key-lanes", type=int, default=K)
    s.add_argument("--keep-states", action="store_true")
    m = sub.add_parser("mask")
    m.add_argument("--chain", choices=CHAINS, required=True)
    m.add_argument("--value-lanes", type=int, required=True)
    m.add_argument("--key-lanes", type=int, required=True)
    m.add_argument("--stats", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--unpinned", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("path")
    v.add_argument("--value-lanes", type=int, default=16)
    v.add_argument("--key-lanes", type=int, default=8)
    b = sub.add_parser("bytes")
    b.add_argument("--states", default="")
    b.add_argument("--value-lanes", type=int, default=16)
    b.add_argument("--key-lanes", type=int, default=8)
    b.add_argument("--out", default="")
    c = sub.add_parser("cpu-check")
    c.add_argument("--cut", default=str(SM / "sm137-io-v16k8-e41"))
    c.add_argument("--perm-only", default=str(WORK / "io-v16k16-perm"))
    c.add_argument("--cut-stats", default=str(WORK / "stats-io-v16k8-cut.json"))
    c.add_argument("--value-lanes", type=int, default=16)
    c.add_argument("--key-lanes", type=int, default=8)
    c.add_argument("--out", default=str(HERE / "check-cpu-sm137.json"))
    args = p.parse_args()
    return {"users": cmd_users, "stats": cmd_stats, "mask": cmd_mask, "verify": cmd_verify, "bytes": cmd_bytes,
            "cpu-check": cmd_cpu_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
