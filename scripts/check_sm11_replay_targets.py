#!/usr/bin/env python
"""CPU check of SM11: full-history replay targets (scripts/sm11_replay_export.py) and Teacher "kd_source": "replay"
(scripts/expertium_options.ReplayTargets); docs/SMALL_MODEL.md, SM11.

  a. Option absent = today: the Teacher of the code before SM11 (expertium_options.py in the parent of the first
     commit adding kd_source; HEAD while uncommitted) and the new one give bitwise identical online targets, cache
     folder, cache miss/hit targets, and the CatchUp loss and every student gradient; "kd_source": "online" too.
  b. On real training users (float32 fork on CPU): the exported targets equal the fork's full-history forward through
     the scorer's prediction path (full_run_metrics.evaluate_with_predictions) within the float16 storage bound
     (curve logit: |dp| <= 1.2e-4, rating probabilities: |dp| <= 2.5e-4), with NaN exactly at reviews without a
     label; for a window that starts at the user's first review (encodings copied from the full-history batch,
     otherwise drawn independently) the online window teacher equals the full-history forward within 2e-5 (float32,
     different batch shapes) and the replay Teacher equals the window teacher within the float16 bound. Informational:
     window-local vs replay targets on later (cold-start) windows, and the encoding-draw noise floor (seed 2025 vs
     2026 full-history replay). The replay Teacher feeds the CatchUp loss end to end, zero at unlabelled rows,
     decay_alpha applied; refusals (cache + replay, another teacher, missing user, missing target, no batch key).
  c. Export resumes: a pass stopped after one user, a stale .tmp, then a rerun gives byte-identical targets to one
     uninterrupted export and leaves the finished user untouched; other settings on the same folder are refused;
     multi-pass (long-user) export covers every review exactly once.
  d. Disk per review, projected to the 5,000 gate training users and the 8k final run.

  .venv/Scripts/python.exe -X utf8 scripts/check_sm11_replay_targets.py [--output artifacts/small-model/check-sm11-replay-targets.json]
"""
from argparse import ArgumentParser
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
os.environ["RWKV_NO_JIT"] = "1"  # eager mode: two identical passes are bit-identical
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
GATE_REVIEWS, FINAL_REVIEWS, GATE_USERS, FINAL_USERS = 369_477_375, 581_942_738, 5000, 8000
CURVE_TOL, RATING_TOL, WINDOW_FP32_TOL = 1.2e-4, 2.5e-4, 2e-5


def previous_options_module(tmp):
    """scripts/expertium_options.py as before SM11."""
    log = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", "kd_source", "HEAD", "--",
                          "scripts/expertium_options.py"], cwd=ROOT, capture_output=True, text=True, check=True)
    commits = log.stdout.split()
    base = subprocess.run(["git", "rev-parse", f"{commits[0]}^" if commits else "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    source = subprocess.run(["git", "show", f"{base}:scripts/expertium_options.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    assert "kd_source" not in source, "the base still has kd_source"
    path = Path(tmp) / "expertium_options_before_sm11.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("expertium_options_before_sm11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, base


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-sm11-replay-targets.json"))
    parser.add_argument("--users", default="793,2188", help="two gate training users (3,463 and 3,367 reviews)")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    from scripts import sm11_replay_export as sx  # installs the compress_quant model patch before rwkv imports
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    from rwkv.data_processing import create_sample, get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import make_samples
    from scripts import compress_common as cc
    from scripts.full_run_metrics import evaluate_with_predictions
    from expertium import variants
    from scripts import expertium_options as xo
    started = time.perf_counter()
    users = [int(u) for u in args.users.split(",")]
    results, info = {}, {}
    cpu = torch.device("cpu")

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    def build_student():
        torch.manual_seed(2025)
        student = variants.build("w64")
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
        return student.train()

    def catch_up_pass(module, options, batch, key, labels, reviews_done=0):
        catch_up = module.CatchUp(options, None, [{"train_range": [0, 1]}])
        catch_up.load_teacher(cpu, torch.float32)
        student = build_student()
        torch.manual_seed(11)
        stats = catch_up.loss(student, batch, reviews_done, key, labels)
        stats.average_loss.backward()
        return (stats.average_loss.detach(), [p.grad.detach().clone() for p in student.parameters() if p.grad is not None],
                catch_up)

    def equal_pass(a, b):
        return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))

    def equal_targets(a, b):
        return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]

    spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32"}
    u0 = users[0]
    frame0 = get_rwkv_data(DATA, u0)
    span = min(len(frame0), 4 * 512)
    train_batch = prepare(make_samples(frame0.iloc[:span], u0, 512), seed=2025 + u0)
    train_key = (u0, 0, span, 2025 + u0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ------------------------------------------------------------------ a. option absent = today
        old, base = previous_options_module(tmp)
        info["base_commit"] = base
        new_teacher = xo.Teacher(spec, cpu, torch.float32)
        old_teacher = old.Teacher(spec, cpu, torch.float32)
        explicit = xo.Teacher({**spec, "kd_source": "online"}, cpu, torch.float32)
        for decaying in (False, True):
            a, b, c = (t.targets(train_batch, decaying) for t in (old_teacher, new_teacher, explicit))
            record(f"a. online targets (decaying={decaying}): new == code before SM11 ({base[:8]}) == kd_source online, bitwise",
                   {"alpha": b[2]}, equal_targets(a, b) and equal_targets(b, c))
        track = {"architecture": "w64", "batch_chunks": 4, "chunk_reviews": 512, "training_seed": 2025}
        cached_new = xo.Teacher({**spec, "cache": str(tmp / "cache-new")}, cpu, torch.float32, track)
        cached_old = old.Teacher({**spec, "cache": str(tmp / "cache-old")}, cpu, torch.float32, track)
        miss = [t.targets(train_batch, key=train_key, labels=train_batch.labels) for t in (cached_old, cached_new)]
        hit = [t.targets(train_batch, key=train_key) for t in (cached_old, cached_new)]
        files = [torch.load(t.cache / ("_".join(map(str, train_key)) + ".pt"), weights_only=True)
                 for t in (cached_old, cached_new)]
        record("a. KD cache: same folder digest, miss and hit targets and stored file bitwise equal to the code before SM11",
               {"folder": cached_new.cache.name, "stats": cached_new.cache_stats},
               cached_new.cache.name == cached_old.cache.name and equal_targets(*miss) and equal_targets(*hit)
               and all(torch.equal(files[0][k], files[1][k]) for k in ("curve", "rating"))
               and files[0]["fingerprint"] == files[1]["fingerprint"]
               and json.loads((cached_new.cache / "manifest.json").read_text()) ==
               json.loads((cached_old.cache / "manifest.json").read_text()))
        options = {"architecture": "w64", "distillation": spec}
        before = catch_up_pass(old, options, train_batch, train_key, train_batch.labels)
        after = catch_up_pass(xo, options, train_batch, train_key, train_batch.labels)
        record("a. CatchUp loss and every student gradient bitwise equal to the code before SM11 (online teacher)",
               {"loss": before[0].item(), "gradients": len(after[1])}, equal_pass(before, after))
        options_cached = {**track, "distillation": {**spec, "cache": str(tmp / "cache-new")}}  # same digest as cached_new
        hit_pass = catch_up_pass(xo, options_cached, train_batch, train_key, train_batch.labels)
        record("a. CatchUp with a cache hit: loss and gradients bitwise equal to the online pass",
               hit_pass[2].teacher.cache_stats, equal_pass(before, hit_pass) and hit_pass[2].teacher.cache_stats["hits"] == 1)

        # ------------------------------------------------------------------ b. replay targets on real users
        run_json = str(sx.GATE_RUN)
        export_dir = tmp / "export"
        code = sx.main(["export", "--out", str(export_dir), "--run", run_json, "--users", args.users, "--device", "cpu",
                        "--threads", str(args.threads)])
        record("b. CPU export of the check users completes (status exit 0)", code, code == 0)
        verify_code = sx.main(["verify", "--out", str(export_dir), "--run", run_json, "--users", args.users,
                               "--threads", str(args.threads)])
        record("b. `verify` (the post-GPU spot check) passes on the CPU export", verify_code, verify_code == 0)
        reference_model = sx.cq.set_state_bits(sx.cq.load_model(path=TEACHER, reference_wkv=True), 0).eval()
        replay_spec = {**spec, "kd_source": "replay", "replay": str(export_dir)}
        replay = xo.Teacher(replay_spec, cpu, torch.float32)
        window_teacher = xo.Teacher(spec, cpu, torch.float32)
        per_user = {}
        for user in users:
            frame = frame0 if user == u0 else get_rwkv_data(DATA, user)
            with np.load(export_dir / "users" / f"{user}.npz") as stored:
                table = stored["targets"].astype(np.float32)
                meta = json.loads(stored["meta"].tobytes().decode())
            full = cc.full_history_batch(DATA, user)  # the scorer's batch: seed 2025 + user
            _, predictions = evaluate_with_predictions(reference_model, [{"batch": full}])
            with torch.inference_mode():
                stats = reference_model.get_loss(full)
            q = stats.is_query.reshape(-1).bool() & stats.has_label.reshape(-1).bool()
            rating_ref = {int(th): p for th, p in zip(stats.label_review_th.reshape(-1)[q].tolist(),
                                                      stats.p_imm_all.reshape(-1, 4)[q].float().numpy())}
            curve_keys = sorted(predictions["ahead"])
            stored_curve = 1 / (1 + np.exp(-table[:, 0]))
            stored_curve = np.clip(stored_curve, 1e-5, 1 - 1e-5)
            curve_ref = np.clip(np.array([predictions["ahead"][k] for k in curve_keys]), 1e-5, 1 - 1e-5)
            curve_err = np.abs(stored_curve[np.array(curve_keys) - 1] - curve_ref)
            rating_keys = sorted(rating_ref)
            rating_err = np.abs(table[np.array(rating_keys) - 1, 1:] - np.stack([rating_ref[k] for k in rating_keys]))
            instant_keys = sorted(predictions["instant"])
            instant_err = np.abs(np.clip(1 - table[np.array(instant_keys) - 1, 1], 1e-5, 1 - 1e-5)
                                 - np.array([predictions["instant"][k] for k in instant_keys]))
            coverage = (set(np.flatnonzero(~np.isnan(table[:, 0])) + 1) == set(curve_keys)
                        and set(np.flatnonzero(~np.isnan(table[:, 1])) + 1) == set(rating_keys) == set(instant_keys))
            row = {"reviews": len(frame), "curve_targets": len(curve_keys), "rating_targets": len(rating_keys),
                   "curve_max_abs": float(curve_err.max()), "rating_max_abs": float(rating_err.max()),
                   "instant_max_abs": float(instant_err.max()),
                   "rating_sum_max_dev": float(np.abs(table[np.array(rating_keys) - 1, 1:].sum(1) - 1).max()),
                   "model_seconds": meta["model_seconds"]}
            record(f"b. user {user}: stored targets = full-history forward (scorer path) within float16 bounds "
                   f"(curve {CURVE_TOL}, rating {RATING_TOL}), NaN exactly where no label", row,
                   coverage and curve_err.max() <= CURVE_TOL and rating_err.max() <= RATING_TOL
                   and instant_err.max() <= RATING_TOL)
            # window from the first review, encodings copied from the full-history batch
            sample = create_sample(user, frame.iloc[:args.window], [], torch.float32, cpu)
            window = prepare([sample], seed=99)
            rows, width = sample.card_features.shape
            same_features = (torch.equal(window.start[:rows, :width], full.start[:rows, :width])
                             and torch.equal(window.labels[0, :rows], full.labels[0, :rows])
                             and torch.equal(window.label_review_th[0, :rows], full.label_review_th[0, :rows]))
            window.start = window.start.clone()
            window.start[:rows] = full.start[:rows]
            online = window_teacher.targets(window)
            key = (user, 0, args.window, 99)
            replayed = replay.targets(window, key=key, labels=window.labels)
            labels = window.labels[0, :rows]
            ahead = ((1 - labels[:, 6]) * labels[:, 4]).bool()
            ask = (labels[:, 6] * labels[:, 4]).bool()
            full_curve = stats.p_curve[0, :rows].float().clamp(1e-5, 1 - 1e-5)
            full_rating = stats.p_imm_all[0, :rows].float()
            fp32_gap = max((online[0][0, :rows][ahead] - full_curve[ahead]).abs().max().item(),
                           (online[1][0, :rows][ask] - full_rating[ask]).abs().max().item())
            curve_gap = (replayed[0][0, :rows][ahead] - online[0][0, :rows][ahead]).abs().max().item()
            rating_gap = (replayed[1][0, :rows][ask] - online[1][0, :rows][ask]).abs().max().item()
            masked_zero = bool((replayed[0][0, :rows][~ahead] == 0).all() and (replayed[1][0, :rows][~ask] == 0).all())
            record(f"b. user {user}: window of {args.window} reviews from the first review: online window teacher = "
                   f"full-history forward (float32, <= {WINDOW_FP32_TOL}); replay Teacher = window teacher within "
                   f"float16 bounds; zero at unlabelled rows",
                   {"card_features_equal": same_features, "window_vs_full_fp32_max_abs": fp32_gap,
                    "replay_vs_window_curve_max_abs": curve_gap, "replay_vs_window_rating_max_abs": rating_gap,
                    "labelled": [int(ahead.sum()), int(ask.sum())]},
                   same_features and fp32_gap <= WINDOW_FP32_TOL and curve_gap <= CURVE_TOL + WINDOW_FP32_TOL
                   and rating_gap <= RATING_TOL + WINDOW_FP32_TOL and masked_zero)
            # informational: window-local (training) targets vs replay, and the encoding-draw noise floor
            if user == u0:
                local = window_teacher.targets(train_batch)
                from_replay = replay.targets(train_batch, key=train_key, labels=train_batch.labels)
                lab = train_batch.labels
                gaps = []
                for chunk in range(lab.shape[0]):
                    a_mask = ((1 - lab[chunk, :, 6]) * lab[chunk, :, 4]).bool()
                    q_mask = (lab[chunk, :, 6] * lab[chunk, :, 4]).bool()
                    gaps.append({"chunk": chunk,
                                 "curve_mean_abs": round((local[0][chunk][a_mask] - from_replay[0][chunk][a_mask]).abs().mean().item(), 5),
                                 "curve_max_abs": round((local[0][chunk][a_mask] - from_replay[0][chunk][a_mask]).abs().max().item(), 4),
                                 "rating_mean_abs": round((local[1][chunk][q_mask] - from_replay[1][chunk][q_mask]).abs().mean().item(), 5)})
                info["window_local_vs_replay_training_batch_512x4"] = gaps
            other_seed = cc.full_history_batch(DATA, user, seed=2026 + user)
            with torch.inference_mode():
                noise = reference_model.get_loss(other_seed)
            has = stats.has_label.reshape(-1).bool()
            a_all = has & ~stats.is_query.reshape(-1).bool()
            per_user[user] = {"encoding_noise_curve_mean_abs": round((noise.p_curve.reshape(-1)[a_all] - stats.p_curve.reshape(-1)[a_all]).abs().mean().item(), 5),
                              "encoding_noise_curve_max_abs": round((noise.p_curve.reshape(-1)[a_all] - stats.p_curve.reshape(-1)[a_all]).abs().max().item(), 4)}
        info["encoding_noise_floor_full_history_seed_2025_vs_2026"] = per_user
        print(f"[info] window-local vs replay, training batch 4 x 512 of user {u0}: "
              f"{info['window_local_vs_replay_training_batch_512x4']}", flush=True)
        print(f"[info] encoding noise floor: {per_user}", flush=True)

        # replay Teacher in the trainer path
        replay_options = {"architecture": "w64", "distillation": replay_spec}
        replay_pass = catch_up_pass(xo, replay_options, train_batch, train_key, train_batch.labels)
        decayed = replay.targets(train_batch, decaying=True, key=train_key, labels=train_batch.labels)
        plain = replay.targets(train_batch, key=train_key)
        record("b. replay Teacher through CatchUp.loss: finite loss, gradients, no teacher model loaded, decay_alpha on decay",
               {"loss": replay_pass[0].item(), "gradients": len(replay_pass[1]), "alphas": [plain[2], decayed[2]],
                "describe": replay_pass[2].teacher.describe().get("kd_source")},
               torch.isfinite(replay_pass[0]) and len(replay_pass[1]) == len(after[1])
               and getattr(replay_pass[2].teacher, "model", None) is None and plain[2] == 0.9 and decayed[2] == 0.5
               and torch.equal(plain[0], decayed[0]))
        refusals = {}

        def refused(name, error_type, function):
            try:
                function()
                refusals[name] = False
            except error_type:
                refusals[name] = True
        refused("cache with replay", ValueError, lambda: xo.Teacher({**replay_spec, "cache": str(tmp / "c")}, cpu, torch.float32))
        refused("unknown kd_source", ValueError, lambda: xo.Teacher({**spec, "kd_source": "other"}, cpu, torch.float32))
        foreign = tmp / "foreign"
        shutil.copytree(export_dir, foreign)
        manifest = json.loads((foreign / "manifest.json").read_text())
        manifest["teacher_sha256"] = "0" * 64
        (foreign / "manifest.json").write_text(json.dumps(manifest))
        refused("another teacher", ValueError, lambda: xo.Teacher({**replay_spec, "replay": str(foreign)}, cpu, torch.float32))
        refused("missing user", RuntimeError, lambda: xo.Teacher(replay_spec, cpu, torch.float32).targets(
            train_batch, key=(123456,) + train_key[1:]))
        refused("no batch key (multi-user batch)", RuntimeError, lambda: xo.Teacher(replay_spec, cpu, torch.float32).targets(train_batch))
        holed = tmp / "holed"
        shutil.copytree(export_dir, holed)
        with np.load(holed / "users" / f"{u0}.npz") as stored:
            table, meta_bytes = stored["targets"].copy(), stored["meta"].copy()
        labelled_th = int(train_batch.label_review_th[0][train_batch.labels[0, :, 4] > 0][0])
        table[labelled_th - 1] = np.nan
        with open(holed / "users" / f"{u0}.npz", "wb") as f:
            np.savez(f, targets=table, meta=meta_bytes)
        refused("labelled row without a stored target", RuntimeError, lambda: xo.Teacher(
            {**replay_spec, "replay": str(holed)}, cpu, torch.float32).targets(train_batch, key=train_key, labels=train_batch.labels))
        os.environ["GPU_BATCH1_DATA_PATH"] = str(tmp)
        refused("another dataset", ValueError, lambda: xo.Teacher(replay_spec, cpu, torch.float32))
        os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
        record("b. refusals", refusals, all(refusals.values()))

        # ------------------------------------------------------------------ c. resume
        resumed = tmp / "resumed"
        first_code = sx.main(["export", "--out", str(resumed), "--run", run_json, "--users", args.users, "--device", "cpu",
                              "--threads", str(args.threads), "--max-users", "1"])
        first_file = resumed / "users" / f"{users[0]}.npz"
        first_stat = first_file.stat()
        first_bytes = first_file.read_bytes()
        (resumed / "users" / f"{users[1]}.npz.tmp").write_bytes(b"partial write from a killed run")
        second_code = sx.main(["export", "--out", str(resumed), "--run", run_json, "--users", args.users, "--device", "cpu",
                               "--threads", str(args.threads), "--workers", "1"])  # the GPU run's worker path
        identical = True
        for user in users:
            with np.load(resumed / "users" / f"{user}.npz") as a, np.load(export_dir / "users" / f"{user}.npz") as b:
                identical &= a["targets"].tobytes() == b["targets"].tobytes()
        log = [json.loads(line) for line in (resumed / "progress.jsonl").read_text().splitlines()]
        untouched = first_file.stat().st_mtime_ns == first_stat.st_mtime_ns and first_file.read_bytes() == first_bytes
        record("c. resume: stopped pass exits 3, rerun (through a loader worker) skips the finished user, ignores the "
               "stale .tmp, targets byte-identical to one uninterrupted export",
               {"exit_codes": [first_code, second_code], "exported_in_order": [r["user_id"] for r in log],
                "stale_tmp_left": (resumed / "users" / f"{users[1]}.npz.tmp").exists()},
               first_code == 3 and second_code == 0 and identical and untouched
               and [r["user_id"] for r in log] == users and not (resumed / "users" / f"{users[1]}.npz.tmp").exists())
        try:
            sx.main(["export", "--out", str(resumed), "--run", run_json, "--users", args.users, "--device", "cpu",
                     "--burn-in", "1000"])
            other_settings = False
        except SystemExit as stop:
            other_settings = "other settings" in str(stop)
        record("c. a folder written with other settings is refused", other_settings, other_settings)
        passes = tmp / "passes"
        code = sx.main(["export", "--out", str(passes), "--run", run_json, "--users", str(u0), "--device", "cpu",
                        "--threads", str(args.threads), "--max-pass-reviews", "1500", "--burn-in", "500"])
        with np.load(passes / "users" / f"{u0}.npz") as a, np.load(export_dir / "users" / f"{u0}.npz") as b:
            meta = json.loads(a["meta"].tobytes().decode())
            multi, single = a["targets"].astype(np.float32), b["targets"].astype(np.float32)
        same_holes = np.array_equal(np.isnan(multi), np.isnan(single))
        valid = ~np.isnan(single[:, 0])
        curve_gap = np.abs(1 / (1 + np.exp(-multi[valid, 0])) - 1 / (1 + np.exp(-single[valid, 0])))
        record("c. multi-pass export (passes of 1,500 reviews, 500 burn-in): every review covered exactly once, "
               "same NaN pattern as one pass (values differ: burn-in context and encoding draws)",
               {"passes": meta["passes"], "curve_mean_abs_vs_single_pass": round(float(curve_gap.mean()), 5),
                "curve_max_abs_vs_single_pass": round(float(curve_gap.max()), 4)},
               code == 0 and same_holes and len(meta["passes"]) == len(sx.plan_passes(len(frame0), 1500, 500)))
        plans = {n: sx.plan_passes(n, 600_000, 300_000) for n in (599_999, 600_000, 600_001, 3_910_718)}
        tiles = all(p[0][1] == 0 and p[-1][2] == n and all(a[2] == b[1] and b[1] - b[0] == 300_000
                    and b[2] - b[0] <= 600_000 for a, b in zip(p, p[1:])) for n, p in plans.items())
        record("c. pass plan tiles the history (production settings 600k / 300k)",
               {n: len(p) for n, p in plans.items()}, tiles)

        # ------------------------------------------------------------------ d. disk
        size = sum((export_dir / "users" / f"{u}.npz").stat().st_size for u in users)
        reviews = sum(len(get_rwkv_data(DATA, u)) if u != u0 else len(frame0) for u in users)
        payload = 10 * reviews
        overhead = (size - payload) / len(users)
        record("d. disk per review (float16 [reviews, 5] + per-file npz header and meta)",
               {"bytes": size, "reviews": reviews, "per_review": round(size / reviews, 3), "payload_per_review": 10,
                "per_file_overhead_bytes": round(overhead),
                "gate_5k_GB": round((10 * GATE_REVIEWS + overhead * GATE_USERS) / 1e9, 3),
                "final_8k_GB": round((10 * FINAL_REVIEWS + overhead * FINAL_USERS) / 1e9, 3)},
               overhead < 4096)
    results_ok = all(entry["ok"] for entry in results.values())
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ok": results_ok, "seconds": round(time.perf_counter() - started, 1), "users": users,
                                  "info": info, "checks": results}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{time.perf_counter() - started:.0f} s; {output}")
    print("ALL OK" if results_ok else "FAILED", flush=True)
    return 0 if results_ok else 1


if __name__ == "__main__":
    sys.exit(main())
