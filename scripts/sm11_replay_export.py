#!/usr/bin/env python
"""SM11: distillation targets from the fork's full-history replay (docs/SMALL_MODEL.md, "Design 2026-10-16", option 4).

The fork teacher runs once over each training user's whole history (one sequence, state carried from the first
review, as in the 464-user scorer `scripts/small_model_heldout.py` and in Anki). Per review it stores what the KD
loss reads (scripts/expertium_options.Teacher.targets): the curve probability of the ahead row whose label is this
review, and the 4 rating probabilities of this review's query row. `Teacher` with "kd_source": "replay" scatters them
into training batches by `batch.label_review_th` (scripts/expertium_options.ReplayTargets).

Layout: <out>/manifest.json (teacher sha256, dataset path/revision, precision, segment settings) and one file per
user, <out>/users/<user>.npz: `targets` float16 [reviews, 5], row = review_th - 1, columns = curve logit
(log(p / (1 - p)) of the clamped curve probability; float16 logits keep 1 - p accurate near 1) and the 4 rating
probabilities (again, hard, good, easy); NaN where the review has no such label (a card's first review). 10 B per
review. `meta` is the user's JSON record (segments, timing, device). Files are written to .tmp and renamed, so a
user is either complete or absent; a rerun skips complete users (resumable at user granularity).

Teacher precision: float32 weights and float32 state (the compress_quant fork path, state_bits 0; any length on
GPU through scripts/compress_wkv_cuda.py). The online window teacher of the trainer runs in bfloat16 by default.
Encodings: one id/day encoding draw per pass, seed 2025 + user + first review of the pass (the scorer's seed for a
single pass).

Users longer than --max-pass-reviews (default 600,000: the largest user the GPU scorer has replayed) run in passes:
the first covers reviews [0, M), each later one targets the next M - B reviews and starts B = --burn-in reviews
earlier (default 300,000), so every target still sees at least B reviews of history. A target row belongs to the
pass whose target range contains the row's own review. A GPU out-of-memory error halves M and B for that user and
retries. The 5,000 gate training users: 38 are longer than 600k (39.2M of 369.5M reviews; +16M replayed reviews).

  export: python -X utf8 scripts/sm11_replay_export.py export --out DIR --run <run.json> [--device cuda|cpu]
          [--time-limit S] [--users 1,2] [--max-users K]
  status: python -X utf8 scripts/sm11_replay_export.py status --out DIR --run <run.json>   (exit 0 = complete, 3 = not)
  verify: python -X utf8 scripts/sm11_replay_export.py verify --out DIR --run <run.json> [--sample 8]   (CPU recompute)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("RWKV_NO_JIT", "1")  # same process setting as the 464-user scorer (its measured rate)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import compress_quant as cq  # noqa: E402

cq.install_state_quant_model()

FORMAT = 1
KIND = "sm11-full-history-replay"
COLUMNS = ("curve_logit", "p_again", "p_hard", "p_good", "p_easy")
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
GATE_RUN = ROOT / "artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json"
CURVE_CLAMP = 1e-5  # Teacher.targets clamps the curve to [1e-5, 1 - 1e-5]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(teacher, teacher_sha, data, revision, max_pass, burn_in):
    return {"format": FORMAT, "kind": KIND, "teacher": str(teacher), "teacher_sha256": teacher_sha,
            "teacher_precision": "float32 weights, float32 state (compress_quant fork path, state_bits 0)",
            "dataset_path": str(data), "dataset_revision": revision,
            "layout": {"file": "users/<user>.npz", "array": "targets", "dtype": "float16", "columns": list(COLUMNS),
                       "row": "review_th - 1", "missing": "NaN", "curve_clamp": CURVE_CLAMP},
            "encoding_seed": "2025 + user + first review index of the pass",
            "max_pass_reviews": int(max_pass), "burn_in_reviews": int(burn_in)}


def plan_passes(reviews, max_pass, burn_in):
    """[(section_start, target_start, target_end)] in frame rows (0-based, end exclusive)."""
    if reviews <= max_pass:
        return [(0, 0, reviews)]
    if not 0 <= burn_in < max_pass:
        raise ValueError("burn-in must be smaller than the pass length")
    passes, start = [(0, 0, max_pass)], max_pass
    while start < reviews:
        end = min(reviews, start + max_pass - burn_in)
        passes.append((start - burn_in, start, end))
        start = end
    return passes


def expected_masks(frame):
    """Reviews that have a curve target (not the first row of their card in the history) and a rating target
    (a query row exists: not is_first_review), as boolean arrays indexed by review_th - 1."""
    card_first = ~frame["card_id"].duplicated().to_numpy()
    query = frame["is_first_review"].to_numpy() == 0
    return ~card_first, query


def prepare_pass(frame, user, section_start, end):
    from rwkv.data_processing import create_sample
    from rwkv.prepare_batch import prepare
    section = frame.iloc[section_start:end]
    sample = create_sample(user, section, [], torch.float32, torch.device("cpu"))
    batch = prepare([sample], seed=2025 + user + section_start)
    return batch, sample.review_ths.long()


def fill(table, stats, own_review_th, target_start, target_end):
    """Write one pass's labelled predictions into table ([reviews, 5], float16 on export, NaN = unset). Rows of this pass's
    target range only; each review's curve and rating are written exactly once."""
    rows = own_review_th.numel()
    has = stats.has_label.reshape(-1)[:rows].bool().cpu()
    query = stats.is_query.reshape(-1)[:rows].bool().cpu()
    label_th = stats.label_review_th.reshape(-1)[:rows].long().cpu()
    keep = has & (own_review_th > target_start) & (own_review_th <= target_end)
    ahead, ask = keep & ~query, keep & query
    curve = stats.p_curve.reshape(-1)[:rows].float().clamp(CURVE_CLAMP, 1 - CURVE_CLAMP).cpu()
    rating = stats.p_imm_all.reshape(-1, 4)[:rows].float().cpu()
    curve_index, rating_index = (label_th[ahead] - 1).numpy(), (label_th[ask] - 1).numpy()
    if len(np.unique(curve_index)) != len(curve_index) or len(np.unique(rating_index)) != len(rating_index):
        raise RuntimeError("a review received two targets in one pass")
    if not (np.isnan(table[curve_index, 0]).all() and np.isnan(table[rating_index, 1]).all()):
        raise RuntimeError("a review received targets from two passes")
    table[curve_index, 0] = torch.logit(curve[ahead]).numpy().astype(table.dtype)
    table[rating_index, 1:] = rating[ask].numpy().astype(table.dtype)
    return int(ahead.sum()), int(ask.sum())


class Exporter:
    def __init__(self, out, teacher, data, revision, device, max_pass, burn_in, threads):
        self.out, self.users_dir = Path(out), Path(out) / "users"
        self.data, self.device = Path(data), torch.device(device)
        self.max_pass, self.burn_in = int(max_pass), int(burn_in)
        torch.set_num_threads(threads)
        self.teacher_sha = sha256_file(teacher)
        self.identity = identity(teacher, self.teacher_sha, data, revision, max_pass, burn_in)
        self.users_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.out / "manifest.json"
        if manifest.exists():
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            if stored != self.identity:
                raise SystemExit(f"{manifest} was written with other settings:\n{json.dumps(stored, indent=1)}")
        else:
            temporary = manifest.with_name("manifest.json.tmp")
            temporary.write_text(json.dumps(self.identity, indent=1) + "\n", encoding="utf-8")
            os.replace(temporary, manifest)
        base = {k: v.float() for k, v in torch.load(teacher, map_location="cpu", weights_only=True).items()}
        if self.device.type == "cuda":
            from scripts import compress_wkv_cuda
            compress_wkv_cuda.register_cuda()
            errors = {f"fork/bits{b}": e for b, e in compress_wkv_cuda.self_test().items()}
            print(f"[sm11] CUDA recurrence vs float32 reference, max abs error: {errors}", flush=True)
            if errors["fork/bits0"] > 1e-4:
                raise SystemExit("CUDA recurrence disagrees with the reference")
        self.model = cq.set_state_bits(cq.load_model(state=base, reference_wkv=True), 0).to(self.device).eval()
        from scripts.experiment_provenance import process_provenance
        provenance = process_provenance()
        self.commit, self.dirty = provenance["git_commit"], provenance["git_dirty"]

    def path(self, user):
        return self.users_dir / f"{user}.npz"

    def forward(self, batch):
        with torch.inference_mode():
            stats = self.model.get_loss(batch.to(self.device))
        if stats is None:
            raise RuntimeError("teacher produced NaN outputs")
        return stats

    def write(self, user, frame, masks, passes, prepare_seconds, prepared=None):
        """Run the passes (prepared lazily, next one in a helper thread), check coverage, write atomically.
        frame may be None when the single pass comes prepared (a loader worker built it)."""
        want_curve, want_rating = masks
        reviews = len(want_curve)
        table = np.full((reviews, len(COLUMNS)), np.nan, dtype=np.float16)
        model_seconds, labelled = 0.0, [0, 0]
        with ThreadPoolExecutor(max_workers=1) as helper:
            pending = None
            for index, (section_start, target_start, target_end) in enumerate(passes):
                if index == 0 and prepared is not None:
                    batch, own = prepared
                    prepared = None
                else:
                    batch, own = (pending.result() if pending is not None else
                                  prepare_pass(frame, user, section_start, target_end))
                pending = (helper.submit(prepare_pass, frame, user, passes[index + 1][0], passes[index + 1][2])
                           if index + 1 < len(passes) else None)
                started = time.perf_counter()
                stats = self.forward(batch)
                curve_n, rating_n = fill(table, stats, own, target_start, target_end)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                model_seconds += time.perf_counter() - started
                labelled[0] += curve_n
                labelled[1] += rating_n
                del stats, batch
        got_curve, got_rating = ~np.isnan(table[:, 0]), ~np.isnan(table[:, 1])
        if not (np.array_equal(want_curve, got_curve) and np.array_equal(want_rating, got_rating)):
            raise RuntimeError(f"coverage mismatch: curve {int(got_curve.sum())}/{int(want_curve.sum())}, "
                               f"rating {int(got_rating.sum())}/{int(want_rating.sum())}")
        if not np.isfinite(table[got_curve, 0]).all() or not np.isfinite(table[got_rating, 1:]).all():
            raise RuntimeError("non-finite target")
        meta = {"user_id": user, "reviews": reviews, "curve_targets": int(got_curve.sum()),
                "rating_targets": int(got_rating.sum()), "passes": [list(p) for p in passes],
                "single_pass": len(passes) == 1, "device": self.device.type, "teacher_sha256": self.teacher_sha,
                "prepare_seconds": round(prepare_seconds, 3),
                "model_seconds": round(model_seconds, 3), "provenance_commit": self.commit,
                "provenance_dirty": self.dirty, "created_unix": round(time.time(), 1)}
        path = self.path(user)
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as f:
            np.savez(f, targets=table, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
        os.replace(temporary, path)
        self.log(meta)
        return meta

    def log(self, row):
        with (self.out / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def run_user(self, user, prepared=None, masks=None, prepare_seconds=0.0):
        """prepared/masks: a single full-history pass built by a loader worker; otherwise the frame is loaded here."""
        from rwkv.data_processing import get_rwkv_data
        frame = None
        max_pass, burn_in = self.max_pass, self.burn_in
        for attempt in range(4):
            if prepared is None and frame is None:
                started = time.perf_counter()
                frame = get_rwkv_data(self.data, user)
                masks = expected_masks(frame)
                prepare_seconds += time.perf_counter() - started
            passes = plan_passes(len(masks[0]), max_pass, burn_in)
            if prepared is not None and len(passes) > 1:
                prepared = None
                continue
            try:
                return self.write(user, frame, masks, passes, prepare_seconds, prepared)
            except torch.OutOfMemoryError as error:
                prepared = None
                torch.cuda.empty_cache()
                self.log({"user_id": user, "oom": str(error)[:200], "max_pass_reviews": max_pass})
                max_pass, burn_in = max_pass // 2, burn_in // 2
                print(f"[sm11] user {user}: out of memory, retrying with passes of {max_pass:,} reviews", flush=True)
        raise RuntimeError(f"user {user}: out of memory down to passes of {max_pass * 2:,} reviews")


class SinglePass(torch.utils.data.Dataset):
    """Worker-side PreparedBatch (+ coverage masks) of one single-pass user (the scorer's UserBatches)."""

    def __init__(self, users, data):
        self.users, self.data = users, data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        from rwkv.data_processing import get_rwkv_data
        user = self.users[index]
        started = time.perf_counter()
        try:
            if torch.utils.data.get_worker_info() is not None:
                torch.set_num_threads(1)
            frame = get_rwkv_data(self.data, user)
            prepared, masks = prepare_pass(frame, user, 0, len(frame)), expected_masks(frame)
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, prepared, masks, None, time.perf_counter() - started


def first(items):
    return items[0]


def load_run(path):
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    return run["train_users"], run["review_counts"], run["dataset_path"], run.get("dataset_revision", "unknown")


def select_users(args):
    users, counts, data, revision = load_run(args.run)
    if args.users:
        wanted = [int(u) for u in args.users.split(",")]
        unknown = sorted(set(wanted) - set(users))
        if unknown and not args.allow_non_training_users:
            raise SystemExit(f"users {unknown} are not training users of {args.run}")
        users = wanted
    return users, counts, Path(args.data or data), revision


def export(args):
    users, counts, data, revision = select_users(args)
    exporter = Exporter(args.out, args.teacher, data, revision, args.device, args.max_pass_reviews, args.burn_in,
                        args.threads)
    todo = [u for u in users if not exporter.path(u).exists()]
    for stale in exporter.users_dir.glob("*.tmp"):
        stale.unlink()
    if args.max_users:
        todo = todo[:args.max_users]
    long_users = [u for u in todo if counts.get(str(u), 0) > args.max_pass_reviews]
    short_users = [u for u in todo if u not in set(long_users)]
    total = sum(counts.get(str(u), 0) for u in todo)
    print(f"[sm11] {len(users)} users, {len(todo)} to export ({total:,} reviews; {len(long_users)} in several "
          f"passes), device {args.device}, out {args.out}", flush=True)
    started, done_reviews, done_users, stopped = time.perf_counter(), 0, 0, False

    def progress(meta):
        nonlocal done_reviews, done_users
        done_reviews += meta["reviews"]
        done_users += 1
        elapsed = time.perf_counter() - started
        print(f"[sm11] {done_users}/{len(todo)} user {meta['user_id']} ({meta['reviews']:,} reviews, "
              f"{len(meta['passes'])} pass(es), model {meta['model_seconds']:.1f}s) | {elapsed:.0f}s, "
              f"{done_reviews / max(elapsed, 1e-9):,.0f} reviews/s", flush=True)

    def out_of_time():
        return args.time_limit and time.perf_counter() - started > args.time_limit

    workers = args.workers if args.workers is not None else (3 if args.device == "cuda" else 0)
    loader = torch.utils.data.DataLoader(SinglePass(short_users, data), batch_size=1, num_workers=workers,
                                         collate_fn=first, prefetch_factor=1 if workers else None)
    failures = 0
    for user, prepared, masks, error, prepare_seconds in loader:
        if out_of_time():
            stopped = True
            break
        try:
            if error is not None:
                raise RuntimeError(error)
            progress(exporter.run_user(user, prepared, masks, prepare_seconds))
        except Exception as failure:  # one user's failure is logged; the rest continue
            failures += 1
            exporter.log({"user_id": user, "error": repr(failure)[:500]})
            print(f"[sm11] user {user} FAILED: {failure!r}"[:600], flush=True)
    del loader
    for user in long_users:
        if stopped or out_of_time():
            stopped = True
            break
        try:
            progress(exporter.run_user(user))
        except Exception as failure:
            failures += 1
            exporter.log({"user_id": user, "error": repr(failure)[:500]})
            print(f"[sm11] user {user} FAILED: {failure!r}"[:600], flush=True)
    elapsed = time.perf_counter() - started
    print(f"[sm11] pass finished: {done_users} users, {done_reviews:,} reviews in {elapsed:.0f}s "
          f"({done_reviews / max(elapsed, 1e-9):,.0f} reviews/s), {failures} failed"
          f"{', time limit reached' if stopped else ''}", flush=True)
    return status(args, quiet=False)


def status(args, quiet=False):
    users, counts, _, _ = select_users(args)
    out = Path(args.out)
    present = {int(p.stem) for p in (out / "users").glob("*.npz")} if (out / "users").exists() else set()
    missing = [u for u in users if u not in present]
    sizes = sum((out / "users" / f"{u}.npz").stat().st_size for u in users if u in present)
    reviews = sum(counts.get(str(u), 0) for u in users if u in present)
    errors = {}
    if (out / "progress.jsonl").exists():
        for row in map(json.loads, (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()):
            if "error" in row and row["user_id"] not in present:
                errors[row["user_id"]] = row["error"]
    result = {"out": str(out), "users": len(users), "complete_users": len(users) - len(missing),
              "missing_users": len(missing), "missing_sample": missing[:20],
              "reviews_done": reviews, "reviews_total": sum(counts.get(str(u), 0) for u in users),
              "bytes": sizes, "bytes_per_review": round(sizes / reviews, 3) if reviews else None,
              "errors": errors, "complete": not missing}
    if (out / "manifest.json").exists():
        temporary = out / "status.json.tmp"
        temporary.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, out / "status.json")
    if not quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "errors"} | {"errors": len(errors)}), flush=True)
    return 0 if not missing else 3


def decode(table):
    """(curve probabilities, rating probabilities) as float32, the values ReplayTargets feeds the loss."""
    table = np.asarray(table, dtype=np.float32)
    return np.clip(1 / (1 + np.exp(-table[:, 0])), CURVE_CLAMP, 1 - CURVE_CLAMP), table[:, 1:]


def verify(args):
    """Recompute sampled single-pass users on the CPU in float32 (reference recurrence) and compare with the stored
    targets: same NaN pattern, curve within 2.2e-4 and ratings within 3.5e-4 (float16 bound + GPU/CPU float32)."""
    import random
    from rwkv.data_processing import get_rwkv_data
    users, counts, data, _ = select_users(args)
    out = Path(args.out)
    torch.set_num_threads(args.threads)
    present = [u for u in users if (out / "users" / f"{u}.npz").exists() and counts.get(str(u), 0) <= args.max_reviews]
    sample = sorted(random.Random(args.seed).sample(present, min(args.sample, len(present)))) if not args.users else present
    model = cq.set_state_bits(cq.load_model(path=args.teacher, reference_wkv=True), 0).eval()
    rows, ok = [], True
    for user in sample:
        with np.load(out / "users" / f"{user}.npz") as stored:
            table = stored["targets"]
            meta = json.loads(stored["meta"].tobytes().decode())
        if not meta["single_pass"]:
            continue
        frame = get_rwkv_data(data, user)
        batch, own = prepare_pass(frame, user, 0, len(frame))
        with torch.inference_mode():
            stats = model.get_loss(batch)
        reference = np.full(table.shape, np.nan, dtype=np.float32)
        fill(reference, stats, own, 0, len(frame))
        same_holes = np.array_equal(np.isnan(reference), np.isnan(table))
        valid = ~np.isnan(reference[:, 0])
        (c_ref, r_ref), (c_got, r_got) = decode(reference), decode(table)
        curve_gap = float(np.abs(c_ref - c_got)[valid].max()) if valid.any() else 0.0
        rating_gap = float(np.nanmax(np.abs(r_ref - r_got))) if valid.any() else 0.0
        good = same_holes and curve_gap <= 2.2e-4 and rating_gap <= 3.5e-4
        ok &= good
        rows.append({"user_id": user, "reviews": len(frame), "device": meta["device"], "same_nan_pattern": same_holes,
                     "curve_max_abs": curve_gap, "rating_max_abs": rating_gap, "ok": good})
        print(f"[sm11 verify] {rows[-1]}", flush=True)
    result = {"out": str(out), "users": len(rows), "ok": ok and bool(rows), "rows": rows}
    temporary = out / "verify.json.tmp"
    temporary.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, out / "verify.json")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    return 0 if result["ok"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "status", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--run", default=str(GATE_RUN), help="run.json with train_users, review_counts, dataset_path")
        p.add_argument("--users", default="", help="comma list: only these users (checks)")
        p.add_argument("--allow-non-training-users", action="store_true")
        p.add_argument("--data", default="", help="dataset folder (default: the run's dataset_path)")
        p.add_argument("--max-pass-reviews", type=int, default=600_000)
        p.add_argument("--burn-in", type=int, default=300_000)
        if name in ("export", "verify"):
            p.add_argument("--teacher", default=str(TEACHER))
            p.add_argument("--threads", type=int, default=4, help="torch threads of the main process")
        if name == "verify":
            p.add_argument("--sample", type=int, default=8, help="users to recompute (with --users: those users)")
            p.add_argument("--max-reviews", type=int, default=20_000, help="only sample users up to this length")
            p.add_argument("--seed", type=int, default=0)
        if name == "export":
            p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            p.add_argument("--workers", type=int, default=None, help="preparation workers (default 3 on cuda, 0 on cpu)")
            p.add_argument("--time-limit", type=float, default=0, help="stop starting new users after N s (0 = none)")
            p.add_argument("--max-users", type=int, default=0, help="export at most K users in this pass (checks)")
    args = parser.parse_args(argv)
    return {"export": export, "status": status, "verify": verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
