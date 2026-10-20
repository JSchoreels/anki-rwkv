"""Small-model track: full-history held-out scoring of the fork's 2.76M RWKV and our smaller checkpoints on the same
users, inputs and labels (docs/SMALL_MODEL.md).

Users (--pool 464, default): the accuracy-loop screen tier's evaluation users the fork never saw (artifacts/accuracy/tiers.json, user id
< 5000: 464 validation users of the 8k/2k run, so unseen by the full run, the screens and the fork). Ordered
round-robin over history-length deciles (--pool 732 / 2225: larger sets, scripts/eval_pools.py), so any prefix is stratified (a time limit truncates evenly).
Protocol: whole history as one sequence (the fork's replay; compress_heldout_b's protocol), upstream get_stats on the
run's label DB, curve (ahead) and instant heads, FSRS-6 benchmark test reviews.

Models (repeatable --model TAG=KIND:ARCH:PATH):
  KIND fork      : the fork's SrsRWKV (ARCH ignored), the compress_heldout_b path (state rounding on card/note).
  KIND expertium : our checkpoints. ARCH iter36 = the R01 width-32 model (expertium.model.srs_model.SrsRWKV),
                   anything else = expertium.variants.build(ARCH) (w64, w80, ...). Eager mode (RWKV_NO_JIT), WKV via
                   scripts/small_model_wkv.py (any length, head size 16/32).
Variants (--variants, applied to every model): fp32, w8 (int8 per-channel weights), +s8 (int8 card/note state rows,
like the fork's runtime int8), +vq<recipe> (expertium models: Expertium-style VQ of the card/note state,
scripts/state_vq.py recipes, codebooks from --vq-codebooks; SM32 on CPU, SM44 on CUDA: rank-1 streams on the fused
kernel scripts/kernels/state_vq_wkv.cu with the Gram-squaring factorization in --vq-factor-dtype, rank-2 streams on a
per-step torch loop; see scripts/state_vq.py "GPU scoring"). Example: fp32,w8+s8,w8+vq403x214.
+s4c8n (SM45, expertium): int4 card / int8 note WKV rows. SM47 suffixes on +s8 / +s4c8n (expertium): h = float16 row
scales instead of float32, x8 = the card/note stored token-shift vectors as int8 codes + one per-vector scale (float16
with h, else float32), e.g. w8+s8h, w8+s8x8, w8+s4c8nhx8; on CUDA they run through the state-VQ slots
(scripts/state_vq.IntScoreCodec: the int WKV kernel with the F16_SCALE flag + the shift hook).
On CPU the expertium models route their WKV recurrence through scripts/state_vq.py (before SM32, +s8 was silently
ignored on the CPU path: RWKV7TimeMixer calls reference_rwkv7, not the patched RWKV7_WKV, for CPU tensors).
Resumable JSONL, one row per (user, tag/variant).
--window W (diagnostic): the training protocol instead, i.e. the same history cut into independent W-review samples
(state reset at every sample start); labels get the suffix "@wW" so they never mix with full-history rows.

usage: small_model_heldout.py --model fork=fork::<pth> --model decay=expertium:iter36:<pth> --variants fp32,w8+s8
       --out artifacts/small-model/heldout.jsonl --time-limit 3000 [--device cpu --count 2]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")  # eager expertium modules so the WKV call can be redirected

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import compress_quant as cq  # noqa: E402

cq.install_state_quant_model()

from scripts import compress_common as cc  # noqa: E402
from scripts.compress_score_heldout import UserBatches, first  # noqa: E402

TIERS = ROOT / "artifacts" / "accuracy" / "tiers.json"
STATE_MODULES = (0, 2)  # card, note (rwkv_modules order card, deck, note, preset, user)


def heldout_users(counts, pool=464):
    if pool != 464:  # 732 / 2225: scripts/eval_pools.py
        from scripts.eval_pools import pool_users
        users = pool_users(pool)
    else:
        tiers = json.loads(TIERS.read_text())
        users = sorted(u for u in tiers["tiers"]["screen"]["eval_users_list"] if u < 5000) if "eval_users_list" in tiers["tiers"]["screen"] else None
        if users is None:
            users = sorted(u for u in _screen_eval_users(tiers) if u < 5000)
    users.sort(key=lambda u: (counts[str(u)], u))
    deciles = [users[len(users) * d // 10: len(users) * (d + 1) // 10] for d in range(10)]
    order = []
    for rank in range(max(len(g) for g in deciles)):
        order += [g[rank] for g in deciles if rank < len(g)]
    return order


def _screen_eval_users(tiers):
    screen = tiers["tiers"]["screen"]
    for key in ("evaluation_users", "eval_user_ids", "eval_users"):
        if isinstance(screen.get(key), list):
            return screen[key]
    raise SystemExit(f"no screen evaluation user list in {TIERS}: keys {sorted(screen)}")


class LongWKV:
    """Stand-in for expertium.model.rwkv_model.RWKV7_WKV: any-length forward, per-mixer state bits."""
    bits = 0

    @staticmethod
    def apply(r, k, v, w, a, kd, skip):
        from scripts.small_model_wkv import wkv_long
        return wkv_long(r, k, v, w, a, kd, skip, LongWKV.bits).to(k.dtype)


def build_expertium(arch, state):
    from expertium.model import rwkv_model as erm
    erm.RWKV7_WKV = LongWKV
    if arch == "iter36":
        from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from expertium.model.srs_model import SrsRWKV
        model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).float()
    else:
        from expertium.variants import build
        model = build(arch).float()
    model.load_state_dict(state, strict=True)
    mixers = []
    for index, module in enumerate(model.rwkv_modules):
        for sub in module.modules():
            if isinstance(sub, erm.RWKV7TimeMixer):
                sub._small_model_stream = index
                mixers.append(sub)
    if not mixers:
        raise SystemExit("no expertium time mixers found")
    for sub in mixers:
        sub.register_forward_pre_hook(_set_bits)
    return model.eval()


def _set_bits(module, _inputs):
    bits = module._small_model_bits
    if isinstance(bits, tuple):   # SM45: per-stream bits (card, note), e.g. +s4c8n
        bits = dict(zip(STATE_MODULES, bits)).get(module._small_model_stream, 0)
    LongWKV.bits = bits if module._small_model_stream in STATE_MODULES else 0


def set_expertium_state_bits(model, bits):
    from expertium.model import rwkv_model as erm
    for sub in model.modules():
        if isinstance(sub, erm.RWKV7TimeMixer):
            sub._small_model_bits = bits


def dump_predictions(folder, user, label, labels, predictions):
    import numpy as np
    review_ths, bins = labels
    kept = [i for i, th in enumerate(review_ths)
            if th in predictions["ahead"] and th in predictions["instant"] and th in predictions["rating"]]
    ths = [review_ths[i] for i in kept]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{user}_{label.replace('/', '_')}.npz", review_th=np.array(ths, dtype=np.int64),
                        bin=np.array([bins[i] for i in kept]),
                        y=np.clip(np.array([predictions["rating"][th] for th in ths]), 0, 1),
                        p_ahead=np.array([predictions["ahead"][th] for th in ths]),
                        p_instant=np.array([predictions["instant"][th] for th in ths]))


class WindowBatches(UserBatches):
    """The history as independent `window`-review samples (the R01 training protocol), one batch per user."""
    window = 128

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            from rwkv.data_processing import get_rwkv_data
            from rwkv.prepare_batch import prepare
            frame = get_rwkv_data(self.data, user)
            batch = prepare(cc.make_samples(frame, user, self.window), seed=2025 + user)
        except Exception as error:  # recorded, not fatal
            return user, None, repr(error), time.perf_counter() - started
        return user, batch, None, time.perf_counter() - started


class IntState(tuple):
    """SM47 int state variant: (wkv_bits, shift_bits, scale_bits). wkv_bits = int or (card, note) tuple, F16_SCALE
    flag set for float16 row scales; shift_bits = 0 (float shift vectors) or 8 (| F16_SCALE); scale_bits 16 / 32."""
    __slots__ = ()
    wkv_bits = property(lambda self: self[0])
    shift_bits = property(lambda self: self[1])
    scale_bits = property(lambda self: self[2])


def parse_variant(name):
    weights, _, state = name.partition("+")
    scheme = {"fp32": "fp32", "w8": "w8-pc", "w4": "w4-g32"}.get(weights)
    if state.startswith("vq") and scheme is not None:
        return scheme, state
    import re
    m = re.fullmatch(r"(s8|s4c8n)(h?)(x8)?", state)
    if scheme is not None and m and (m.group(2) or m.group(3)):   # SM47: f16 row scales (h), int8 shift vectors (x8)
        from scripts.small_model_wkv import F16_SCALE
        flag = F16_SCALE if m.group(2) else 0
        wkv = 8 | flag if m.group(1) == "s8" else (4 | flag, 8 | flag)
        return scheme, IntState((wkv, (8 | flag) if m.group(3) else 0, 16 if flag else 32))
    if scheme is not None and state == "s4c8n":   # SM45: int4 card state, int8 note state
        return scheme, (4, 8)
    if scheme is None or state not in ("", "s8", "s4"):
        raise SystemExit(f"unknown variant {name}")
    return scheme, {"": 0, "s8": 8, "s4": 4}[state]


def state_fields(state_bits):
    """Row fields describing the state precision (SM47 rows add scale / shift precision and bits per entity)."""
    if isinstance(state_bits, IntState):
        from scripts.state_vq_report import int_entity_bits
        wkv = state_bits.wkv_bits
        plain = [b & 255 for b in wkv] if isinstance(wkv, tuple) else wkv & 255
        card, note = int_entity_bits(plain, state_bits.scale_bits, state_bits.shift_bits & 255)
        return {"state_bits": plain, "state_scale_bits": state_bits.scale_bits,
                "state_shift_bits": state_bits.shift_bits & 255 or 32, "state_entity_bits": {"card": card, "note": note}}
    return {"state_bits": list(state_bits) if isinstance(state_bits, tuple) else state_bits or 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="TAG=KIND:ARCH:PATH (repeatable)")
    parser.add_argument("--variants", default="fp32")
    parser.add_argument("--only", action="append", default=[], help="TAG/VARIANT labels to score instead of the grid")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="0 = all users of the pool")
    parser.add_argument("--pool", type=int, choices=(464, 732, 2225), default=464, help="base user set: 464 screen "
                        "eval users (default), 732 validation users with id < 5000, 2225 = 732 + gate-unseen train users "
                        "(invalid for models trained on the final tier); --users-file still filters on top")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=3000, help="stop starting new users after N seconds")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", default="artifacts/small-model/heldout.jsonl")
    parser.add_argument("--max-reviews", type=int, default=0, help="CPU smoke: skip users above this many reviews")
    parser.add_argument("--users-file", default="", help="JSON with a 'users' list (e.g. artifacts/small-model/"
                        "gate-users.json): score only these users, in the usual stratified order")
    parser.add_argument("--window", type=int, default=0, help="diagnostic: score independent W-review samples")
    parser.add_argument("--vq-codebooks", default="artifacts/small-model/state-vq/codebooks.npz",
                        help="scripts/state_vq.py codebooks for +vq variants (scripts/state_vq_fit.py)")
    parser.add_argument("--vq-factor-dtype", choices=("float64", "float32"), default="float64",
                        help="CUDA +vq: factorization type of the fused kernel (float64 = the CPU scorer's)")
    parser.add_argument("--emulate-gpu-path", action="store_true", help="check only (SM47, --device cpu, int "
                        "variants with h/x8): run the CUDA dispatch (state-VQ slots, IntScoreCodec) on CPU tensors")
    parser.add_argument("--dump", default="", help="also write per-review benchmark predictions (review_th, bin, y, "
                        "p_ahead, p_instant) to DIR/<user>_<tag>_<variant>.npz, for calibration diagnostics")
    args = parser.parse_args()
    cpu = args.device == "cpu"
    if cpu:
        torch.set_num_threads(2)
        args.workers = 0
    else:
        from scripts import compress_wkv_cuda, small_model_wkv
        compress_wkv_cuda.register_cuda()
        errors = {**{f"fork/bits{b}": e for b, e in compress_wkv_cuda.self_test().items()}, **small_model_wkv.self_test()}
        print(f"[small-model] CUDA recurrences vs float32 reference, max abs error: {errors}", flush=True)
        if max(e for key, e in errors.items() if key.endswith("0")) > 1e-4:
            raise SystemExit("CUDA recurrence disagrees with the reference")
    import lmdb
    from scripts.compress_metrics import metrics, read_labels
    from scripts.experiment_provenance import process_provenance
    from scripts.full_run_metrics import evaluate_with_predictions

    device = torch.device(args.device)
    if not cpu:
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    run = cc.run_info()
    counts = run["review_counts"]
    users = heldout_users(counts, args.pool)
    if args.users_file:
        wanted = set(json.loads(Path(args.users_file).read_text())["users"])
        users = [u for u in users if u in wanted]
    if args.max_reviews:
        users = [u for u in users if counts[str(u)] <= args.max_reviews]
    users = users[args.start:args.start + args.count] if args.count else users[args.start:]
    specs = {}
    for item in args.model:
        tag, _, rest = item.partition("=")
        kind, arch, path = rest.split(":", 2)
        specs[tag] = (kind, arch, path)
    grid = [f"{tag}/{v}" for tag in specs for v in args.variants.split(",")]
    if args.only:
        grid = args.only
    if args.window:
        grid = [f"{label}@w{args.window}" for label in grid]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for row in map(json.loads, out.read_text().splitlines()):
            if "error" not in row:  # an OOM row is retried (e.g. with more memory or expandable segments)
                done.setdefault(row["user_id"], set()).add(row["label"])
    todo =[u for u in users if not set(grid) <= done.get(u, set())]

    models, states, layouts, codecs = {}, {}, {}, {}
    for label in grid:
        tag, variant = label.split("@")[0].split("/", 1)
        kind, arch, path = specs[tag]
        scheme, _ = parse_variant(variant)
        if tag not in models:
            base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
            if kind == "fork":
                model = cq.load_model(state=base, reference_wkv=cpu)
                for module in model.rwkv_modules:
                    for block in module.blocks:
                        block.time_mixer.use_reference_wkv = True
            else:
                model = build_expertium(arch, base)
                from scripts import state_vq
                # CPU: the Hooks path (reference WKV + shift gather); CUDA: the eager model's state-VQ slots (SM44)
                layouts[tag] = (state_vq.install(model) if cpu and not args.emulate_gpu_path
                                else state_vq.install_scoring(model))
            models[tag] = (kind, model.to(device), base, cq.linear_weight_names(base))
        kind, model, base, names = models[tag]
        if (tag, scheme) not in states:
            bits, group = cq.WEIGHT_SCHEMES[scheme]
            states[(tag, scheme)] = (cq.quantize_state_dict(base, names, bits, group),
                                     cq.weight_bytes(base, names, bits, group)["total"],
                                     sum(t.numel() for t in base.values()))
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    prov = process_provenance()
    print(f"[small-model] {len(users)} users, {len(todo)} to score, grid {grid}", flush=True)
    WindowBatches.window = args.window
    dataset = (WindowBatches if args.window else UserBatches)(todo, cc.data_path(run), counts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    started = time.perf_counter()
    scored = 0
    for index, (user, batch, error, prep_seconds) in enumerate(loader):
        if time.perf_counter() - started > args.time_limit:
            print(f"[small-model] time limit reached after {index} users", flush=True)
            break
        with env.begin() as txn:
            labels = read_labels(txn, user)
        rows = []
        if batch is None or labels is None:
            rows.append({"user_id": user, "label": "error", "error": error or "no labels"})
        else:
            gpu_batch = batch.to(device)
            for label in grid:
                if label in done.get(user, set()):
                    continue
                tag, variant = label.split("@")[0].split("/", 1)
                kind, model, _, _ = models[tag]
                scheme, state_bits = parse_variant(variant)
                state, nbytes, params = states[(tag, scheme)]
                model.load_state_dict(state)
                vq_info = vq_path = None
                if kind == "fork":
                    if isinstance(state_bits, tuple):
                        raise SystemExit("per-stream state bits are for expertium-kind models only")
                    cq.set_state_bits(model, state_bits)
                elif isinstance(state_bits, IntState):   # SM47
                    from scripts import state_vq
                    set_expertium_state_bits(model, 0)
                    if cpu and not args.emulate_gpu_path:
                        state_vq.configure(layouts[tag], None, state_bits.wkv_bits, state_bits.shift_bits)
                    else:
                        codecs_int = state_vq.int_score_codecs(state_bits.wkv_bits, state_bits.shift_bits)
                        state_vq.set_scoring(layouts[tag], codecs_int)
                        vq_path = state_vq.score_path(codecs_int)
                elif cpu and args.emulate_gpu_path:
                    raise SystemExit("--emulate-gpu-path is for the SM47 int variants (h / x8) only")
                elif cpu:
                    from scripts import state_vq
                    set_expertium_state_bits(model, 0)  # the GPU kernel's bits; CPU state goes through state_vq
                    if isinstance(state_bits, str):
                        if state_bits not in codecs:
                            path = ROOT / args.vq_codebooks
                            books = state_vq.Codebooks.load(path) if path.exists() else state_vq.Codebooks()
                            codecs["sha256"] = state_vq.file_sha256(path) if path.exists() else None
                            codecs[state_bits] = {s: state_vq.StateCodec(rec, books)
                                                  for s, rec in zip(layouts[tag], state_vq.recipe(state_bits))}
                            print(f"[small-model] {state_bits}: {state_vq.describe(state_bits, layouts[tag])}",
                                  flush=True)
                        state_vq.configure(layouts[tag], codecs[state_bits])
                        vq_info = state_vq.describe(state_bits, layouts[tag])
                    else:
                        state_vq.configure(layouts[tag], None, state_bits)
                else:
                    from scripts import state_vq
                    if isinstance(state_bits, str):   # SM44: GPU +vq
                        set_expertium_state_bits(model, 0)
                        if state_bits not in codecs:
                            path = ROOT / args.vq_codebooks
                            books = state_vq.Codebooks.load(path)
                            codecs["sha256"] = state_vq.file_sha256(path)
                            codecs[state_bits] = state_vq.score_codecs(state_bits, books, device,
                                                                       getattr(torch, args.vq_factor_dtype))
                            print(f"[small-model] {state_bits}: {state_vq.describe(state_bits, layouts[tag])}; "
                                  f"{state_vq.score_path(codecs[state_bits])}", flush=True)
                        state_vq.set_scoring(layouts[tag], codecs[state_bits])
                        vq_info = state_vq.describe(state_bits, layouts[tag])
                        vq_path = state_vq.score_path(codecs[state_bits])
                    else:
                        state_vq.set_scoring(layouts[tag], None)
                        set_expertium_state_bits(model, state_bits)
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        values, predictions = evaluate_with_predictions(model, [{"batch": gpu_batch}])
                except (torch.OutOfMemoryError, RuntimeError) as oom:
                    if not isinstance(oom, torch.OutOfMemoryError) and "out of memory" not in str(oom):
                        raise  # TorchScript wraps a CUDA OOM in a plain RuntimeError
                    torch.cuda.empty_cache()
                    rows.append({"user_id": user, "label": label, "error": f"OOM: {oom}"[:300]})
                    continue
                if not cpu:
                    torch.cuda.synchronize()
                rows.append({"user_id": user, "label": label, "tag": tag, "variant": variant, "kind": kind,
                             "arch": specs[tag][1], "checkpoint": specs[tag][2], "parameters": params,
                             "weight_bytes": nbytes, **state_fields(state_bits), "reviews": counts[str(user)],
                             **({"state_vq": vq_info, "vq_codebooks": args.vq_codebooks,
                                 "vq_codebooks_sha256": codecs.get("sha256")} if vq_info else {}),
                             **({"state_vq_path": vq_path} if vq_path else {}),
                             "prepare_seconds": round(prep_seconds, 2),
                             "model_seconds": round(time.perf_counter() - t0, 3), **values,
                             "benchmark": metrics(user, labels, predictions)})
                if args.dump:
                    dump_predictions(ROOT / args.dump, user, label, labels, predictions)
                scored += counts[str(user)]
            del gpu_batch
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                row["provenance_commit"] = prov["git_commit"]
                row["provenance_dirty"] = prov["git_dirty"]
                f.write(json.dumps(row) + "\n")
        elapsed = time.perf_counter() - started
        print(f"[small-model] {index + 1}/{len(todo)} user {user} ({counts[str(user)]} reviews) | {elapsed:.0f}s, "
              f"{scored / max(elapsed, 1e-9):,.0f} review-evals/s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
