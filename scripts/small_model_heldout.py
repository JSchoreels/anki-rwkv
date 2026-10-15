"""Small-model track: full-history held-out scoring of the fork's 2.76M RWKV and our smaller checkpoints on the same
users, inputs and labels (docs/SMALL_MODEL.md).

Users: the accuracy-loop screen tier's evaluation users the fork never saw (artifacts/accuracy/tiers.json, user id
< 5000: 464 validation users of the 8k/2k run, so unseen by the full run, the screens and the fork). Ordered
round-robin over history-length deciles, so any prefix is stratified (a time limit truncates evenly).
Protocol: whole history as one sequence (the fork's replay; compress_heldout_b's protocol), upstream get_stats on the
run's label DB, curve (ahead) and instant heads, FSRS-6 benchmark test reviews.

Models (repeatable --model TAG=KIND:ARCH:PATH):
  KIND fork      : the fork's SrsRWKV (ARCH ignored), the compress_heldout_b path (state rounding on card/note).
  KIND expertium : our checkpoints. ARCH iter36 = the R01 width-32 model (expertium.model.srs_model.SrsRWKV),
                   anything else = expertium.variants.build(ARCH) (w64, w80, ...). Eager mode (RWKV_NO_JIT), WKV via
                   scripts/small_model_wkv.py (any length, head size 16/32).
Variants (--variants, applied to every model): fp32, w8 (int8 per-channel weights), +s8 (int8 card/note state rows,
like the fork's runtime int8). Example: fp32,w8+s8.
Resumable JSONL, one row per (user, tag/variant).
--window W (diagnostic): the training protocol instead, i.e. the same history cut into independent W-review samples
(state reset at every sample start); labels get the suffix "@wW" so they never mix with full-history rows.
--batch-users N (S05, docs/GPU_OPTIMIZATIONS.md): N users per forward pass instead of one, grouped by history length
(length buckets, at most --batch-reviews padded reviews per group). Each user keeps its batch-1 inputs (per-user
encoding seed, checked with --input-sha), its own int8 state rows (the rounding is per state row) and the batch-1
loss path on its own outputs; only the forward's floating-point reduction order may differ, so its rows must pass
the AGENT.md evaluation tolerance (scripts/small_model_batch_compare.py). Absent or 1: the original one-user loop.

usage: small_model_heldout.py --model fork=fork::<pth> --model decay=expertium:iter36:<pth> --variants fp32,w8+s8
       --out artifacts/small-model/heldout.jsonl --time-limit 3000 [--device cpu --count 2] [--batch-users 8]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


def heldout_users(counts):
    tiers = json.loads(TIERS.read_text())
    users = sorted(u for u in tiers["tiers"]["screen"]["eval_users_list"] if u < 5000) if "eval_users_list" in \
        tiers["tiers"]["screen"] else None
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
    LongWKV.bits = module._small_model_bits if module._small_model_stream in STATE_MODULES else 0


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


class SeededSamples(list):
    """Samples of several users for one prepare() call. Iteration reseeds torch before each sample, so prepare()'s
    random ID and day-offset encodings of every user equal its batch-1 draws (prepare(..., seed=2025 + user) seeds once,
    then encodes its single sample first; no later step of prepare() draws random numbers)."""

    def __init__(self, samples, seeds):
        super().__init__(samples)
        self.seeds = list(seeds)

    def __iter__(self):
        for sample, seed in zip(list.__iter__(self), self.seeds):
            torch.manual_seed(seed)
            yield sample


class GroupBatches(torch.utils.data.Dataset):
    """S05: several users' whole histories as one prepared batch, one row per user (row i = users[i])."""

    def __init__(self, groups, data):
        self.groups, self.data = groups, data

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        from rwkv.data_processing import get_rwkv_data
        from rwkv.prepare_batch import prepare
        started = time.perf_counter()
        samples, users, errors = [], [], {}
        for user in self.groups[index]:
            try:
                frame = get_rwkv_data(self.data, user)
                found = cc.make_samples(frame, user, len(frame))
                if len(found) != 1:
                    raise ValueError(f"expected one full-history sample, got {len(found)}")
            except Exception as error:  # recorded, not fatal
                errors[user] = repr(error)
                continue
            samples += found
            users.append(user)
        batch = None
        if samples:
            try:
                batch = prepare(SeededSamples(samples, [2025 + u for u in users]))
            except Exception as error:
                errors.update({u: repr(error) for u in users})
                users = []
        lengths = [s.card_features.size(0) for s in samples] if batch is not None else []
        return users, lengths, batch, errors, time.perf_counter() - started


def length_groups(users, counts, size, cap):
    """Length buckets: consecutive users by review count, at most `size` per group and at most `cap` padded reviews
    (members x longest history); a user above the cap is scored alone. Groups run in the stratified order of their
    first member, so a time limit still cuts across history lengths."""
    position = {u: i for i, u in enumerate(users)}
    groups, current = [], []
    for user in sorted(users, key=lambda u: (counts[str(u)], u)):
        if current and (len(current) == size or (len(current) + 1) * counts[str(user)] > cap):
            groups.append(current)
            current = []
        current.append(user)
    if current:
        groups.append(current)
    return sorted(groups, key=lambda g: min(position[u] for u in g))


def input_sha(batch, row, length):
    """SHA-256 of one user's model inputs (encoded features, labels, label review indices) in a prepared batch."""
    features = batch.start.view(batch.num_data, -1, batch.start.size(-1))[row, :length]
    digest = hashlib.sha256()
    for tensor in (features, batch.labels[row, :length], batch.label_review_th[row, :length]):
        digest.update(tensor.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_rows(model, batch, rows):
    """S05: one forward pass over several users, then the batch-1 loss and prediction path per user.

    rows = [(row, length)]. Each user's forward outputs are cut to its own row and history length, so the loss head,
    masks and reductions see exactly the batch-1 shapes (batch-1 pads to the user's own length); the model's forward
    method is replaced by that slice for the per-user get_loss call. Returns evaluate_with_predictions per row."""
    from scripts.full_run_metrics import evaluate_with_predictions
    name = "forward_batch_variant" if hasattr(model, "forward_batch_variant") else "forward_batch"
    outputs = getattr(model, name)(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects,
                                   batch.skips, batch.num_data)
    results = []
    try:
        for row, length in rows:
            part = tuple(o[row:row + 1, :length].contiguous()
                         if torch.is_tensor(o) and o.dim() >= 2 and o.size(0) == batch.num_data else o
                         for o in outputs)
            setattr(model, name, lambda *_args, _part=part: _part)
            view = SimpleNamespace(start=None, sub_gather=None, sub_gather_lens=None, time_shift_selects=None,
                                   skips=None, num_data=1, labels=batch.labels[row:row + 1, :length],
                                   label_review_th=batch.label_review_th[row:row + 1, :length])
            results.append(evaluate_with_predictions(model, [{"batch": view}]))
    finally:
        model.__dict__.pop(name, None)
    return results


def parse_variant(name):
    weights, _, state = name.partition("+")
    scheme = {"fp32": "fp32", "w8": "w8-pc", "w4": "w4-g32"}.get(weights)
    if scheme is None or state not in ("", "s8", "s4"):
        raise SystemExit(f"unknown variant {name}")
    return scheme, {"": 0, "s8": 8, "s4": 4}[state]


def configure(models, states, label):
    """Load a label's (weight scheme, state bits) into its model; returns (tag, variant, kind, model, bits, bytes, n)."""
    tag, variant = label.split("@")[0].split("/", 1)
    kind, model, _, _ = models[tag]
    scheme, state_bits = parse_variant(variant)
    state, nbytes, params = states[(tag, scheme)]
    model.load_state_dict(state)
    if kind == "fork":
        cq.set_state_bits(model, state_bits)
    else:
        set_expertium_state_bits(model, state_bits)
    return tag, variant, kind, model, state_bits, nbytes, params


def score_batched(args, todo, grid, done, specs, models, states, counts, env, prov, out, run):
    """S05 loop: length-bucketed groups of --batch-users users, one forward per (group, label). Rows match the
    one-user loop's, plus batch_users / batch_model_seconds; model_seconds and prepare_seconds are the group's time
    shared by history length. A group that runs out of GPU memory is rescored one user at a time (batch_fallback)."""
    from scripts.compress_metrics import metrics, read_labels
    from scripts.full_run_metrics import evaluate_with_predictions
    device, cpu = torch.device(args.device), args.device == "cpu"
    groups = length_groups(todo, counts, args.batch_users, args.batch_reviews)
    print(f"[small-model] S05 batched scoring: {len(groups)} groups of <= {args.batch_users} users, "
          f"<= {args.batch_reviews:,} padded reviews each", flush=True)
    loader = torch.utils.data.DataLoader(GroupBatches(groups, cc.data_path(run)), batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    started = time.perf_counter()
    scored = users_done = 0
    for index, (users, lengths, batch, errors, prep_seconds) in enumerate(loader):
        if time.perf_counter() - started > args.time_limit:
            print(f"[small-model] time limit reached after {index} groups ({users_done} users)", flush=True)
            break
        rows = [{"user_id": u, "label": "error", "error": e} for u, e in errors.items()]
        with env.begin() as txn:
            labels = {u: read_labels(txn, u) for u in users}
        rows += [{"user_id": u, "label": "error", "error": "no labels"} for u in users if labels[u] is None]
        live = [(i, u) for i, u in enumerate(users) if labels[u] is not None]
        total = max(sum(lengths), 1)
        gpu_batch = batch.to(device) if live else None
        for label in grid:
            need = [(i, u) for i, u in live if label not in done.get(u, set())]
            if not need:
                continue
            tag, variant, kind, model, state_bits, nbytes, params = configure(models, states, label)
            common = {"label": label, "tag": tag, "variant": variant, "kind": kind, "arch": specs[tag][1],
                      "checkpoint": specs[tag][2], "parameters": params, "weight_bytes": nbytes,
                      "state_bits": state_bits or 32}
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    results = evaluate_rows(model, gpu_batch, [(i, lengths[i]) for i, _ in need])
                if not cpu:
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                need_total = sum(lengths[i] for i, _ in need)
                timed = [(values, predictions, seconds * lengths[i] / need_total, batch.num_data, None)
                         for (i, _), (values, predictions) in zip(need, results)]
            except torch.OutOfMemoryError as oom:
                torch.cuda.empty_cache()
                seconds, fallback = 0.0, f"OOM at {batch.num_data} users: {oom}"[:300]
                print(f"[small-model] group {index + 1}: {fallback[:120]}; rescoring one user at a time", flush=True)
                timed = []
                for _, user in need:
                    t1 = time.perf_counter()
                    try:
                        single = cc.full_history_batch(cc.data_path(run), user).to(device)
                        with torch.inference_mode():
                            values, predictions = evaluate_with_predictions(model, [{"batch": single}])
                        del single
                    except torch.OutOfMemoryError as oom_single:
                        torch.cuda.empty_cache()
                        timed.append((None, f"OOM: {oom_single}"[:300], 0.0, 1, fallback))
                        continue
                    if not cpu:
                        torch.cuda.synchronize()
                    timed.append((values, predictions, time.perf_counter() - t1, 1, fallback))
            for (i, user), (values, predictions, share, batch_users, fallback) in zip(need, timed):
                if values is None:
                    rows.append({"user_id": user, "label": label, "error": predictions})
                    continue
                row = {"user_id": user, **common, "reviews": counts[str(user)],
                       "prepare_seconds": round(prep_seconds * lengths[i] / total, 2),
                       "model_seconds": round(share, 3), **values,
                       "benchmark": metrics(user, labels[user], predictions),
                       "batch_users": batch_users, "batch_model_seconds": round(seconds, 3)}
                if fallback:
                    row["batch_fallback"] = fallback
                if args.input_sha:
                    row["input_sha256"] = input_sha(batch, i, lengths[i])
                rows.append(row)
                if args.dump:
                    dump_predictions(ROOT / args.dump, user, label, labels[user], predictions)
                scored += counts[str(user)]
        del gpu_batch
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                row["provenance_commit"] = prov["git_commit"]
                row["provenance_dirty"] = prov["git_dirty"]
                f.write(json.dumps(row) + "\n")
        users_done += len(groups[index])
        elapsed = time.perf_counter() - started
        print(f"[small-model] group {index + 1}/{len(groups)}: {len(users)} users, "
              f"{sum(counts[str(u)] for u in groups[index]):,} reviews (longest {max(lengths, default=0):,}) | "
              f"{users_done}/{len(todo)} users, {elapsed:.0f}s, {scored / max(elapsed, 1e-9):,.0f} review-evals/s",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="TAG=KIND:ARCH:PATH (repeatable)")
    parser.add_argument("--variants", default="fp32")
    parser.add_argument("--only", action="append", default=[], help="TAG/VARIANT labels to score instead of the grid")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="0 = all 464 users")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=3000, help="stop starting new users after N seconds")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", default="artifacts/small-model/heldout.jsonl")
    parser.add_argument("--max-reviews", type=int, default=0, help="CPU smoke: skip users above this many reviews")
    parser.add_argument("--users-file", default="", help="JSON with a 'users' list (e.g. artifacts/small-model/"
                        "gate-users.json): score only these users, in the usual stratified order")
    parser.add_argument("--window", type=int, default=0, help="diagnostic: score independent W-review samples")
    parser.add_argument("--dump", default="", help="also write per-review benchmark predictions (review_th, bin, y, "
                        "p_ahead, p_instant) to DIR/<user>_<tag>_<variant>.npz, for calibration diagnostics")
    parser.add_argument("--batch-users", type=int, default=1, help="S05: users per forward pass, length-bucketed "
                        "(1 = the original one-user loop)")
    parser.add_argument("--batch-reviews", type=int, default=1_000_000, help="S05: at most this many padded reviews "
                        "(users x longest review count) per group, about the largest held-out user (860,510), so a "
                        "group needs about the memory batch-1 already needs; a longer user is scored alone")
    parser.add_argument("--input-sha", action="store_true", help="record input_sha256 (encoded features + labels) "
                        "per row, to check that batched inputs equal batch-1 inputs")
    args = parser.parse_args()
    if args.batch_users > 1 and args.window:
        raise SystemExit("--batch-users needs full-history scoring (no --window)")
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
    run = cc.run_info()
    counts = run["review_counts"]
    users = heldout_users(counts)
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
            done.setdefault(row["user_id"], set()).add(row["label"])
    todo = [u for u in users if not set(grid) <= done.get(u, set())]

    models, states = {}, {}
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
    if args.batch_users > 1:
        if any(models[label.split("@")[0].split("/", 1)[0]][0] == "fork" for label in grid):
            raise SystemExit("--batch-users supports expertium models only: the fork's TorchScript _get_loss calls its "
                             "own forward_batch, so per-user slicing cannot be injected; score fork labels at batch 1")
        score_batched(args, todo, grid, done, specs, models, states, counts, env, prov, out, run)
        env.close()
        return
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
                if kind == "fork":
                    cq.set_state_bits(model, state_bits)
                else:
                    set_expertium_state_bits(model, state_bits)
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        values, predictions = evaluate_with_predictions(model, [{"batch": gpu_batch}])
                except torch.OutOfMemoryError as oom:
                    torch.cuda.empty_cache()
                    rows.append({"user_id": user, "label": label, "error": f"OOM: {oom}"[:300]})
                    continue
                if not cpu:
                    torch.cuda.synchronize()
                rows.append({"user_id": user, "label": label, "tag": tag, "variant": variant, "kind": kind,
                             "arch": specs[tag][1], "checkpoint": specs[tag][2], "parameters": params,
                             "weight_bytes": nbytes, "state_bits": state_bits or 32, "reviews": counts[str(user)],
                             "prepare_seconds": round(prep_seconds, 2),
                             "model_seconds": round(time.perf_counter() - t0, 3), **values,
                             "benchmark": metrics(user, labels, predictions)})
                if args.input_sha:
                    rows[-1]["input_sha256"] = input_sha(batch, 0, batch.labels.size(1))
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
