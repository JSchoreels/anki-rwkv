"""Grade-order sanity (SM33), stage 1: how often does the small model break interval(Again) <= interval(Hard) <=
interval(Good) <= interval(Easy)?

For sampled review rows of held-out users, the four what-if answers (rating 1..4) are scored exactly as the fork's
`answer_heads` does: the review row's own features with the rating replaced, run from the committed state before the
review, nothing committed. Here they are "probe" rows inserted into the full-history sequence right before the real
review row: copies of that row with skip = True (the same no-commit rule as query rows), has_label = False, rating
one-hot = g. Two details make them exact:
  * order within a review_th: query row, probes g = 1..4, real row; probes read the same committed state as the real
    row and never change it, so every real row's output is unchanged (checked: `check`);
  * token shift: prepare_batch points rows before an entity's first committed row at row 0. With probes before a
    card's (or deck's...) first review, that would read a probe's input, so rows up to and including the first
    committed row of each sub-sequence point at themselves (`fix_leading_shift`; a no-op when no sequence starts with
    a skip row, i.e. on every batch without probes - checked).
From each row's GRU curve R(t) = sum_i w_i (1 + t / S_i)^(-d_i) (no ahead residual in w64), the intervals at desired
retention 0.80 / 0.90 / 0.95 are solved per grade (bisection on log t, 1 s .. 36,500 days, unrounded), plus the fork's
displayed intervals at 0.90 with its default grade-order correction: on its search grid (sub-day points, days 1-30,
45 x 1.5^k, max) the four margins R_g(t) - 0.9 are pooled by PAVA (non-decreasing in g) at every time point and each
grade's crossing is refined on the pooled margin (rslib/src/rwkv/engine.rs unrounded_intervals_for_answer_curves,
enforce_grade_order default True), then ceil + clamp to 1..max days.

  measure  python scripts/sm33_grade_order.py measure --model <master.pth> --users-file <json> --out <dir>
           [--probes 3000] [--variant fp32|w8+s8] [--device cuda] [--time-limit 2400]
           writes <dir>/users/<user>.npz (resumable) and <dir>/measure-log.jsonl
  check    python scripts/sm33_grade_order.py check --model <master.pth> --user 4 --device cpu --max-rows 3000
           exactness of the probe construction (real rows unchanged; probe with the real rating == real row)
  report   python scripts/sm33_grade_order.py report --out <dir>   -> <dir>/sm33-stage1.{json,md}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SECONDS_PER_DAY = 86400.0
MAX_DAYS = 36500.0
TARGETS = (0.80, 0.90, 0.95)
HORIZONS = (1.0, 7.0, 30.0, 180.0)
GRADES = ("Again", "Hard", "Good", "Easy")
FORK_SUB_DAY = np.array([1.0 / 86400, 1.0 / 1440, 5.0 / 1440, 10.0 / 1440, 20.0 / 1440, 30.0 / 1440, 1.0 / 24,
                         2.0 / 24, 3.0 / 24, 4.0 / 24, 6.0 / 24, 8.0 / 24, 12.0 / 24, 16.0 / 24, 20.0 / 24])


def fork_grid(max_days=int(MAX_DAYS)):
    days = list(range(1, 31))
    day = 45
    while day < max_days:
        days.append(day)
        day = int(day * 1.5)
    days.append(max_days)
    return np.concatenate([FORK_SUB_DAY, np.array(days, dtype=np.float64)])


# ---------------------------------------------------------------- probe rows
_PLAN = {"review_ths": None}


def _install_probe_queries():
    import rwkv.data_processing as dp
    if getattr(dp.add_queries, "_sm33", False):
        return
    original = dp.add_queries

    def add_queries_with_probes(section_df, equalize_review_ths):
        import pandas as pd
        df = original(section_df, equalize_review_ths)
        wanted = _PLAN["review_ths"]
        df["sm33_order"] = np.where(df["is_query"].to_numpy() == 1, 0, 5)
        df["sm33_probe"] = 0
        if wanted is None:
            return df
        real = df[(~df["skip"].astype(bool)) & df["review_th"].isin(wanted)]
        copies = [df]
        for g in (1, 2, 3, 4):
            c = real.copy()
            c["rating"] = g
            for i in (1, 2, 3, 4):
                c[f"rating_{i}"] = int(i == g)
            c["y"] = int(g > 1)
            c["skip"] = True
            c["has_label"] = False
            c["sm33_order"] = g
            c["sm33_probe"] = g
            copies.append(c)
        df = pd.concat(copies, ignore_index=True)
        df = df.sort_values(by=["review_th", "sm33_order"], kind="stable").reset_index(drop=True)
        df["index"] = range(0, len(df))
        _PLAN["df"] = df
        return df

    add_queries_with_probes._sm33 = True
    dp.add_queries = add_queries_with_probes


def fix_leading_shift(batch):
    """Rows up to and including each sub-sequence's first committed row read their own input in the token shift
    (prepare_batch's rule for row 0). Returns the number of changed indices (0 on batches without leading skips)."""
    changed = 0
    for module_shifts, module_skips, lens in zip(batch.time_shift_selects, batch.skips, batch.sub_gather_lens):
        for i, (shift, skip, r) in enumerate(zip(module_shifts, module_skips, lens)):
            s2 = skip.view(-1, r)
            first = (~s2).to(torch.int8).argmax(dim=1)
            t = torch.arange(r, device=shift.device)
            lead = t[None, :] <= first[:, None]
            new = torch.where(lead, t[None, :].expand_as(s2).to(shift.dtype), shift.view(-1, r))
            changed += int((new != shift.view(-1, r)).sum())
            module_shifts[i] = new.reshape(-1).contiguous()
    return changed


def user_batch(data, user, probes, seed_offset=0, mode="neighbour", max_rows=0):
    """(batch, meta) for one user's full history with probe rows before `probes` sampled review rows (0 = none)."""
    from rwkv.data_processing import create_sample, get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts import sm71_leak
    _install_probe_queries()
    frame = get_rwkv_data(data, user)
    if mode != "keep":
        frame, _ = sm71_leak.neutralise(frame, mode, data, user)
    if max_rows:
        frame = frame.iloc[:max_rows]
    n = len(frame)
    if probes:
        rng = np.random.default_rng(33_000 + user)
        ths = frame["review_th"].to_numpy()
        pick = np.sort(rng.choice(ths, size=min(probes, n), replace=False)) if probes < n else ths
        _PLAN["review_ths"] = set(int(x) for x in pick)
    else:
        _PLAN["review_ths"] = None
    _PLAN.pop("df", None)
    sample = create_sample(user, frame, [], torch.float32, torch.device("cpu"))
    batch = prepare([sample], seed=2025 + user + seed_offset)
    changed = fix_leading_shift(batch)
    meta = {"n_reviews": n, "shift_changed": changed}
    if probes:
        df = _PLAN.pop("df")
        probe = df["sm33_probe"].to_numpy()
        real = (~df["skip"].astype(bool)).to_numpy()
        th = df["review_th"].to_numpy()
        rows = {g: np.flatnonzero(probe == g) for g in (1, 2, 3, 4)}
        pth = th[rows[1]]
        for g in (2, 3, 4):
            assert np.array_equal(th[rows[g]], pth)
        real_idx = np.flatnonzero(real)
        real_of = dict(zip(th[real_idx].tolist(), real_idx.tolist()))
        real_rows = np.array([real_of[int(x)] for x in pth])
        src = frame.set_index("review_th").loc[pth]
        meta.update(rows=rows, real_rows=real_rows, review_th=pth,
                    rating=src["rating"].to_numpy().astype(np.int8),
                    first=src["is_first_review"].to_numpy().astype(np.int8),
                    state=src["state"].to_numpy().astype(np.int8),
                    elapsed_days=src["elapsed_days"].to_numpy().astype(np.float32),
                    elapsed_seconds=src["elapsed_seconds"].to_numpy().astype(np.float32),
                    duration=src["duration"].to_numpy().astype(np.float32),
                    real_mask=real, length=len(df))
    return batch, meta


class Users(torch.utils.data.Dataset):
    def __init__(self, users, data, probes):
        self.users, self.data, self.probes = users, data, probes

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        t0 = time.perf_counter()
        try:
            batch, meta = user_batch(self.data, user, self.probes)
        except Exception as error:  # recorded, not fatal
            return user, None, repr(error), time.perf_counter() - t0
        return user, (batch, meta), None, time.perf_counter() - t0


def _first(items):
    return items[0]


# ---------------------------------------------------------------- model
def load_model(path, arch, device, variant):
    from scripts import compress_quant as cq
    from scripts import small_model_heldout as smh
    from scripts import state_vq
    base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    scheme, bits = smh.parse_variant(variant)
    if not isinstance(bits, int):
        raise SystemExit("fp32 / w8 / w8+s8 only")
    names = cq.linear_weight_names(base)
    wbits, group = cq.WEIGHT_SCHEMES[scheme]
    state = cq.quantize_state_dict(base, names, wbits, group)
    model = smh.build_expertium(arch, state)
    cpu = device.type == "cpu"
    if cpu:
        layout = state_vq.install(model)
        smh.set_expertium_state_bits(model, 0)
        state_vq.configure(layout, None, bits)
    else:
        layout = state_vq.install_scoring(model)
        state_vq.set_scoring(layout, None)
        smh.set_expertium_state_bits(model, bits)
    return model.to(device).eval()


@torch.inference_mode()
def heads(model, batch):
    """(w, s_raw, d_raw, p_logits) [T, ...] float64 numpy for a one-user batch."""
    out = model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects,
                                      batch.skips, batch.num_data)
    _, w, _, p_logits, s_raw, d_raw = out
    return tuple(x[0].double().cpu().numpy() for x in (w, s_raw, d_raw, p_logits))


# ---------------------------------------------------------------- curve math
def curve(w, s_raw, d_raw, days):
    """R(t) for params [..., N] at days [...] (broadcast): the model's gru_forgetting_curve, float64."""
    s = np.exp(np.clip(s_raw, -25, 25))
    d = np.exp(np.clip(d_raw, -25, 25))
    t = np.maximum(1.0, np.asarray(days) * SECONDS_PER_DAY)[..., None]
    r = np.sum(w * np.exp(-d * np.log1p(t / (1e-7 + s))), axis=-1)
    return 1e-5 + (1 - 2e-5) * r


def interval(w, s_raw, d_raw, target, iters=60):
    """Unrounded days with R(t) = target (R decreasing in t), clamped to [1 s, MAX_DAYS]."""
    shape = w.shape[:-1]
    lo = np.full(shape, np.log(1.0 / SECONDS_PER_DAY))
    hi = np.full(shape, np.log(MAX_DAYS))
    at_hi = curve(w, s_raw, d_raw, np.exp(hi)) > target
    at_lo = curve(w, s_raw, d_raw, np.exp(lo)) <= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = curve(w, s_raw, d_raw, np.exp(mid)) > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    out = np.exp(hi)
    out = np.where(at_hi, MAX_DAYS, out)
    return np.where(at_lo, 1.0 / SECONDS_PER_DAY, out)


def pava4(v):
    """Isotonic (non-decreasing) regression along the last axis (length 4, equal weights): max-min formula."""
    c = np.concatenate([np.zeros(v.shape[:-1] + (1,)), np.cumsum(v, axis=-1)], axis=-1)
    out = np.empty_like(v)
    for i in range(4):
        best = None
        for j in range(i + 1):
            low = None
            for k in range(i, 4):
                m = (c[..., k + 1] - c[..., j]) / (k + 1 - j)
                low = m if low is None else np.minimum(low, m)
            best = low if best is None else np.maximum(best, low)
        out[..., i] = best
    return out


def fork_intervals(params, target, iters=50):
    """The fork's displayed intervals with enforce_grade_order (PAVA on the four margins at each time point), unrounded.
    params = (w, s, d) each [n, 4, N]."""
    w, s, d = params
    grid = fork_grid()

    def pooled(days):  # days [n, 4] (one time per grade) -> pooled margins [n, 4] (grade i's margin at days[:, i])
        r = curve(w[:, None, :, :], s[:, None, :, :], d[:, None, :, :], days[:, :, None])  # [n, i, g]
        m = pava4(r - target)
        return m[:, np.arange(4), np.arange(4)]

    rg = curve(w[:, None, :, :], s[:, None, :, :], d[:, None, :, :], grid[None, :, None])  # [n, G, 4]
    mg = pava4(rg - target)
    hit = mg <= 0
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)  # [n, 4]
    hi = grid[first]
    lo = np.where(first > 0, grid[np.maximum(first - 1, 0)], 0.0)
    # the fork's bracket is (previous point, first point with margin <= 0); low = 0 for the first grid point
    lo_margin = pooled(lo)
    exact = lo_margin <= 0  # the fork returns the grid point itself when the low end is not above the target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = pooled(mid) > 0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    out = np.where(exact, grid[first], hi)
    return np.where(any_hit, np.minimum(out, MAX_DAYS), MAX_DAYS)


def displayed(days):
    return np.clip(np.ceil(np.asarray(days, dtype=np.float64) - 1e-9), 1, MAX_DAYS)


def user_arrays(w, s, d, meta):
    rows = meta["rows"]
    pw = np.stack([w[rows[g]] for g in (1, 2, 3, 4)], axis=1)  # [n, 4, N]
    ps = np.stack([s[rows[g]] for g in (1, 2, 3, 4)], axis=1)
    pd_ = np.stack([d[rows[g]] for g in (1, 2, 3, 4)], axis=1)
    iv = np.stack([interval(pw, ps, pd_, t) for t in TARGETS], axis=1)  # [n, 3, 4]
    cv = np.stack([curve(pw, ps, pd_, h) for h in HORIZONS], axis=1)    # [n, 4, 4]
    fk = fork_intervals((pw, ps, pd_), 0.90)                            # [n, 4]
    return {"intervals": iv.astype(np.float32), "curve_at": cv.astype(np.float32), "fork_pava_09": fk.astype(np.float32),
            "params": np.concatenate([pw, ps, pd_], axis=-1).astype(np.float32)}


# ---------------------------------------------------------------- commands
def cmd_check(args):
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    else:
        from scripts import compress_wkv_cuda
        compress_wkv_cuda.register_cuda()
        torch.cuda.set_per_process_memory_fraction(0.8)
    from scripts import compress_common as cc
    data = cc.data_path()
    model = load_model(args.model, args.arch, device, args.variant)
    report = {"user": args.user, "device": args.device, "variant": args.variant, "max_rows": args.max_rows}
    plain, pm = user_batch(data, args.user, 0, max_rows=args.max_rows)
    report["plain_shift_changed"] = pm["shift_changed"]  # must be 0: the shift fix is a no-op without probes
    probed, meta = user_batch(data, args.user, args.probes, max_rows=args.max_rows)
    report["probed_shift_changed"] = meta["shift_changed"]
    a = heads(model, plain.to(device))
    b = heads(model, probed.to(device))
    real_b = np.flatnonzero(meta["real_mask"])
    # plain row order = probed order with the probe rows removed
    keep = np.flatnonzero(~np.isin(np.arange(meta["length"]), np.concatenate([meta["rows"][g] for g in (1, 2, 3, 4)])))
    names = ("w", "s_raw", "d_raw", "p_logits")
    report["rows_plain"], report["rows_probed"], report["probes"] = int(a[0].shape[0]), meta["length"], len(meta["review_th"])
    report["non_probe_rows_unchanged_max_abs"] = {k: float(np.abs(x - y[keep]).max()) for k, x, y in zip(names, a, b)}
    actual = meta["rating"].astype(int)
    same = np.stack([meta["rows"][g] for g in (1, 2, 3, 4)], axis=1)[np.arange(len(actual)), actual - 1]
    report["probe_actual_vs_real_max_abs"] = {k: float(np.abs(y[same] - y[meta["real_rows"]]).max())
                                              for k, y in zip(names[:3], b[:3])}
    tol = 1e-5 if device.type == "cpu" else 1e-4
    ok = (report["plain_shift_changed"] == 0
          and max(report["non_probe_rows_unchanged_max_abs"].values()) <= tol
          and max(report["probe_actual_vs_real_max_abs"].values()) <= tol)
    # the curve math: unrounded intervals vs a direct check R(I) ~= target
    arr = user_arrays(*b[:3], meta)
    iv = arr["intervals"][:, 1, :].astype(np.float64)
    pw, ps, pd_ = np.split(arr["params"].astype(np.float64), 3, axis=-1)
    inside = (iv > 1.0 / SECONDS_PER_DAY * 1.01) & (iv < MAX_DAYS * 0.999)
    rr = curve(pw, ps, pd_, iv)
    report["interval_solve_max_abs_R_minus_0.9"] = float(np.abs(rr - 0.9)[inside].max()) if inside.any() else None
    raw_order = np.all(np.diff(iv, axis=1) >= 0, axis=1)
    fk = arr["fork_pava_09"]
    report["fork_pava_monotone_share"] = float(np.mean(np.all(np.diff(fk, axis=1) >= -1e-4 * fk[:, 1:], axis=1)))
    report["raw_monotone_share"] = float(raw_order.mean())
    report["tolerance"] = tol
    report["pass"] = bool(ok)
    print("[sm33-check] " + ("PASS " if ok else "FAIL ") + json.dumps(report), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if ok else 1


def cmd_measure(args):
    device = torch.device(args.device)
    cpu = device.type == "cpu"
    if cpu:
        torch.set_num_threads(4)
    else:
        from scripts import compress_wkv_cuda
        compress_wkv_cuda.register_cuda()
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    from scripts import compress_common as cc
    from scripts.experiment_provenance import process_provenance
    run = cc.run_info()
    counts = run["review_counts"]
    users = json.loads(Path(args.users_file).read_text())["users"]
    if args.max_reviews:
        users = [u for u in users if counts[str(u)] <= args.max_reviews]
    users = users[args.start:args.start + args.count] if args.count else users[args.start:]
    out = Path(args.out)
    (out / "users").mkdir(parents=True, exist_ok=True)
    todo = [u for u in users if not (out / "users" / f"{u}.npz").exists()]
    todo.sort(key=lambda u: counts[str(u)])  # short users first: most users done early if the time limit cuts
    log = out / "measure-log.jsonl"
    prov = process_provenance()
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "users": len(users), "todo": len(todo), "model": args.model,
                            "variant": args.variant, "probes": args.probes, "device": args.device,
                            "provenance": prov}) + "\n")
    model = load_model(args.model, args.arch, device, args.variant)
    loader = torch.utils.data.DataLoader(Users(todo, cc.data_path(run), args.probes), batch_size=1,
                                         num_workers=args.workers, collate_fn=_first,
                                         prefetch_factor=1 if args.workers else None)
    t0 = time.time()
    done = 0
    for user, item, error, prep in loader:
        if time.time() - t0 > args.time_limit:
            print(f"[sm33] time limit reached after {done} users", flush=True)
            break
        row = {"user": user, "reviews": counts[str(user)], "prepare_seconds": round(prep, 2)}
        if item is None:
            row["error"] = error
        else:
            batch, meta = item
            t1 = time.perf_counter()
            try:
                w, s, d, _ = heads(model, batch.to(device))
            except (torch.OutOfMemoryError, RuntimeError) as oom:
                if not isinstance(oom, torch.OutOfMemoryError) and "out of memory" not in str(oom):
                    raise
                torch.cuda.empty_cache()
                row["error"] = f"OOM: {oom}"[:200]
                w = None
            if w is not None:
                arr = user_arrays(w, s, d, meta)
                np.savez_compressed(out / "users" / f"{user}.npz", review_th=meta["review_th"], rating=meta["rating"],
                                    first=meta["first"], state=meta["state"], elapsed_days=meta["elapsed_days"],
                                    elapsed_seconds=meta["elapsed_seconds"], duration=meta["duration"],
                                    n_reviews=meta["n_reviews"], **arr)
                row.update(probes=len(meta["review_th"]), rows=meta["length"], model_seconds=round(time.perf_counter() - t1, 2),
                           shift_changed=meta["shift_changed"])
                done += 1
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if done % 50 == 0 and item is not None:
            print(f"[sm33] {done}/{len(todo)} users, {time.time() - t0:.0f} s", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "end", "done": done, "seconds": round(time.time() - t0, 1)}) + "\n")
    print(f"[sm33] measured {done} users in {time.time() - t0:.0f} s", flush=True)
    return 0


def _rate(mask, weight):
    return float(np.sum(mask * weight) / np.sum(weight)) if np.sum(weight) else None


def cmd_report(args):
    out = Path(args.out)
    files = sorted((out / "users").glob("*.npz"))
    if not files:
        raise SystemExit("no user files")
    cols = {k: [] for k in ("user", "n_reviews", "weight", "rating", "first", "state", "elapsed_days", "iv", "cv", "fk")}
    for f in files:
        z = np.load(f)
        n = len(z["review_th"])
        cols["user"].append(np.full(n, int(f.stem)))
        cols["n_reviews"].append(np.full(n, int(z["n_reviews"])))
        cols["weight"].append(np.full(n, float(z["n_reviews"]) / n))  # review-weighted: each row stands for N/n reviews
        for k, src in (("rating", "rating"), ("first", "first"), ("state", "state"), ("elapsed_days", "elapsed_days"),
                       ("iv", "intervals"), ("cv", "curve_at"), ("fk", "fork_pava_09")):
            cols[k].append(z[src])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    iv = c["iv"].astype(np.float64)  # [n, 3 targets, 4]
    fk = c["fk"].astype(np.float64)
    uw = np.ones_like(c["weight"])
    rw = c["weight"]
    users = np.unique(c["user"])
    per_user_w = np.zeros_like(rw)
    for u in users:  # user-averaged: every user weighs 1
        m = c["user"] == u
        per_user_w[m] = 1.0 / m.sum()
    pairs = ((0, 1, "Again>Hard"), (1, 2, "Hard>Good"), (2, 3, "Good>Easy"))
    res = {"users": int(len(users)), "probed_reviews": int(len(rw)), "reviews_represented": float(rw.sum()),
           "targets": {}, "curve_horizons": {}, "fork_display_0.90": {}, "breakdown_0.90": {}}

    def break_stats(lo_v, hi_v, weights):
        ratio = lo_v / np.maximum(hi_v, 1e-12)
        dl, dh = displayed(lo_v), displayed(hi_v)
        r = {}
        for wname, wt in weights.items():
            r[wname] = {"any": _rate(lo_v > hi_v * (1 + 1e-6), wt), "ratio>1.05": _rate(ratio > 1.05, wt),
                        "ratio>1.25": _rate(ratio > 1.25, wt), "ratio>2": _rate(ratio > 2, wt),
                        "displayed_days_inverted": _rate(dl > dh, wt), "displayed_inverted_by>=2d": _rate(dl - dh >= 2, wt)}
        brk = lo_v > hi_v * (1 + 1e-6)
        if brk.any():
            r["when_broken"] = {"median_ratio": float(np.median(ratio[brk])), "p90_ratio": float(np.quantile(ratio[brk], 0.9)),
                                "median_days_lo": float(np.median(lo_v[brk])), "median_days_gap": float(np.median((lo_v - hi_v)[brk])),
                                "p90_days_gap": float(np.quantile((lo_v - hi_v)[brk], 0.9))}
        return r

    weights = {"review_weighted": rw, "user_averaged": per_user_w, "probe_rows": uw}
    for ti, target in enumerate(TARGETS):
        t = {}
        for a, b, name in pairs:
            t[name] = break_stats(iv[:, ti, a], iv[:, ti, b], weights)
        anyb = np.zeros(len(rw), bool)
        for a, b, _ in pairs:
            anyb |= iv[:, ti, a] > iv[:, ti, b] * (1 + 1e-6)
        disp = displayed(iv[:, ti])
        t["any_pair"] = {w: _rate(anyb, wt) for w, wt in weights.items()}
        t["any_pair_displayed"] = {w: _rate(np.any(np.diff(disp, axis=1) < 0, axis=1), wt) for w, wt in weights.items()}
        t["median_intervals_days"] = {g: float(np.median(iv[:, ti, i])) for i, g in enumerate(GRADES)}
        res["targets"][f"{target:.2f}"] = t
    cv = c["cv"].astype(np.float64)
    for hi_, h in enumerate(HORIZONS):
        res["curve_horizons"][f"{h:g}d"] = {name: {"rate_review_weighted": _rate(cv[:, hi_, a] > cv[:, hi_, b] + 1e-6, rw),
                                                   "rate_user_averaged": _rate(cv[:, hi_, a] > cv[:, hi_, b] + 1e-6, per_user_w),
                                                   "median_gap_when_broken": (float(np.median((cv[:, hi_, a] - cv[:, hi_, b])[cv[:, hi_, a] > cv[:, hi_, b] + 1e-6]))
                                                                              if (cv[:, hi_, a] > cv[:, hi_, b] + 1e-6).any() else None)}
                                            for a, b, name in pairs}
    raw = iv[:, 1]
    raw_d, fk_d = displayed(raw), displayed(fk)
    fd = {}
    fd["fork_monotone_share"] = _rate(np.all(np.diff(fk_d, axis=1) >= 0, axis=1), rw)
    for i, g in enumerate(GRADES):
        rel = np.abs(fk[:, i] - raw[:, i]) / np.maximum(raw[:, i], 1e-9)
        fd[g] = {"changed_by_pava(>0.5%)": _rate(rel > 0.005, rw), "changed_by_pava(>5%)": _rate(rel > 0.05, rw),
                 "displayed_days_changed": _rate(fk_d[:, i] != raw_d[:, i], rw),
                 "median_rel_change_when_changed": float(np.median(rel[rel > 0.005])) if (rel > 0.005).any() else None}
    fd["ties_after_pava(Hard=Good displayed)"] = _rate(fk_d[:, 1] == fk_d[:, 2], rw)
    fd["ties_raw(Hard=Good displayed)"] = _rate(raw_d[:, 1] == raw_d[:, 2], rw)
    fd["ties_after_pava(Good=Easy displayed)"] = _rate(fk_d[:, 2] == fk_d[:, 3], rw)
    fd["ties_raw(Good=Easy displayed)"] = _rate(raw_d[:, 2] == raw_d[:, 3], rw)
    res["fork_display_0.90"] = fd
    # breakdowns at 0.90 (review-weighted)
    brk = {name: iv[:, 1, a] > iv[:, 1, b] * (1 + 1e-6) for a, b, name in pairs}
    ed = c["elapsed_days"]
    groups = {"first_review": c["first"] == 1, "not_first": c["first"] == 0}
    for r_ in (1, 2, 3, 4):
        groups[f"actual_{GRADES[r_ - 1]}"] = (c["rating"] == r_) & (c["first"] == 0)
    for lo_, hi2, nm in ((0, 1, "same_day(0d)"), (1, 7, "1-6d"), (7, 31, "7-30d"), (31, 181, "31-180d"), (181, 1e9, ">180d")):
        groups[f"elapsed_{nm}"] = (c["first"] == 0) & (ed >= lo_) & (ed < hi2)
    good_iv = iv[:, 1, 2]
    for lo_, hi2, nm in ((0, 1, "<1d"), (1, 7, "1-6d"), (7, 31, "7-30d"), (31, 181, "31-180d"), (181, 1e9, ">180d")):
        groups[f"good_interval_{nm}"] = (good_iv >= lo_) & (good_iv < hi2)
    nr = c["n_reviews"]
    qs = np.quantile(np.array([nr[c["user"] == u][0] for u in users]), [0.25, 0.5, 0.75])
    for lo_, hi2, nm in ((0, qs[0], "user_history_q1"), (qs[0], qs[1], "q2"), (qs[1], qs[2], "q3"), (qs[2], 1e12, "q4")):
        groups[f"{nm}"] = (nr >= lo_) & (nr < hi2)
    for gname, m in groups.items():
        res["breakdown_0.90"][gname] = {"share_of_reviews": _rate(m, rw),
                                        **{name: _rate(brk[name] & m, rw * m) if m.any() else None for name in brk}}
    # per-user distribution of the Hard>Good / Good>Easy rates
    per_user = []
    for u in users:
        m = c["user"] == u
        per_user.append((int(u), int(nr[m][0]), *(float(brk[name][m].mean()) for name in brk)))
    pu = np.array([p[2:] for p in per_user])
    res["per_user_rate_quantiles_0.90"] = {name: {q: float(np.quantile(pu[:, i], float(q))) for q in ("0.1", "0.5", "0.9", "0.99")}
                                           for i, name in enumerate(brk)}
    res["per_user_rate_quantiles_0.90"]["users_with_any_rate_over_10%"] = int(np.sum(pu.max(axis=1) > 0.10))
    worst = sorted(per_user, key=lambda p: -max(p[2:]))[:10]
    res["worst_users_0.90"] = [{"user": p[0], "reviews": p[1], **{name: round(v, 4) for name, v in zip(brk, p[2:])}} for p in worst]
    (out / "sm33-stage1.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("users", "probed_reviews")}), flush=True)
    print(json.dumps(res["targets"]["0.90"]["any_pair"]), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "check"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--arch", default="w64")
        s.add_argument("--variant", default="fp32")
        s.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        s.add_argument("--probes", type=int, default=3000)
    m = sub.choices["measure"]
    m.add_argument("--users-file", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--count", type=int, default=0)
    m.add_argument("--max-reviews", type=int, default=0)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--time-limit", type=float, default=2400)
    k = sub.choices["check"]
    k.add_argument("--user", type=int, default=4)
    k.add_argument("--max-rows", type=int, default=3000)
    k.add_argument("--json", default="")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = p.parse_args()
    return {"measure": cmd_measure, "check": cmd_check, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
