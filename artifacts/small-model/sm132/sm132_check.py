#!/usr/bin/env python
"""Again pressed as Hard (SM132): CPU proofs (a few minutes; run with CUDA_VISIBLE_DEVICES= ).

K1 state machine (scripts/sm31_inputs._anki_states) on synthetic cards: a flipped review-state Again turns the relearning
   streak after it into review-state rows; an unflipped Again inside that streak lapses the card for real; unexplained
   relearning starts, learning / new Agains and cards without relearning steps keep their states; no flip = unchanged.
K2 frames and samples on real users (show-time inputs, deleted cards neighbour, as the scorer builds them):
   p = 0 returns the frame unchanged (DataFrame.equals, same columns and dtypes) and samples equal to the clean ones;
   the draw is deterministic (two calls equal) and nested (flips at 5% <= 10% <= 20% <= 30% <= 50% <= 100% = every
   Again); p = 1.0 ("dry check"): every Again update row reads Hard (rating_1 0, rating_2 1), only the columns
   rating_1, rating_2 and scaled_state change, only on update rows (query rows equal), the state rows changed match
   the count reported, and every label tensor (global_labels, label_review_ths), review_ths, ids, day offsets and
   skips are bit for bit the clean ones (the targets stay true); state "keep" changes the two rating columns only.
K3 causality, in process (the show-time two-head cool-down for Small (SM118), float32 CPU): ONE benchmark Again review t
   flipped -> every instant prediction of a review <= t (t's own included: its query row precedes its update row) and
   every curve prediction made from an update row before t is bit for bit the clean one; the labels are equal; some
   prediction after t differs.
K4 scorer rows on CPU (users 2388, 4676; w8 weights + int8 state, the published format; cpu/smoke-cpu-rows.jsonl):
   SM118 and the show-time big model (SM91s2): no flag == --again-as-hard 0 bit for bit (benchmark, benchmark_existing,
   benchmark_deleted), benchmark_corrupt_labels at p = 0 == benchmark; Big no flag == the committed CPU rows of the
   compact state for Small and Big (SM131) bit for bit (the default path is unchanged); SM118 no flag within 1e-5 of
   its published GPU rows; p = 1.0 flips every Again (also every benchmark Again); p = 0.2 rows finite.
K5 rule selftest (sm132_rule.py).
-> check-cpu-sm132.json, exit 0 iff all pass.

`sm132_check.py smoke` (first step of the GPU ticket): gpu/smoke-gpu-rows.jsonl vs the CPU rows: every head within 1e-5,
the again_as_hard counts identical, and p = 0 within 1e-5 of the published GPU rows -> gpu/smoke-check.json.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
PY = sys.executable
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
CK118 = W / f"sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/{TRACK}/master.pth"
CK91 = W / "sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"
T118, T91 = "sm118-e26-dec2ep32k-t91s2", "sm91s2-big-showtime"
PUB118 = ROOT / "artifacts/small-model/sm118/gpu/sm118-rows.jsonl"
PUB91 = W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"
CPU131 = ROOT / "artifacts/small-model/sm131/cpu/smoke-cpu-rows.jsonl"
SMOKE = HERE / "cpu/smoke-cpu-rows.jsonl"
SMOKE_USERS = (2388, 4676)
FRAME_USERS = (2388, 4676, 2450)
HEADS = ("ahead", "instant")
CF = None   # CARD_FEATURE_COLUMNS, set lazily


def k1():
    from scripts.sm31_inputs import _anki_states as st
    a = np.array
    cases = {   # name: (state, true rating, corrupted rating, expected)
        "flipped_lapse_streak_becomes_review": (a([2, 2, 3, 3, 2]), a([3, 1, 3, 3, 3]), a([3, 2, 3, 3, 3]), a([2, 2, 2, 2, 2])),
        "unflipped_lapse_kept": (a([2, 2, 3, 3, 2]), a([3, 1, 3, 3, 3]), a([3, 1, 3, 3, 3]), a([2, 2, 3, 3, 2])),
        "real_lapse_inside_rewritten_streak": (a([2, 3, 3, 2]), a([1, 1, 3, 3]), a([2, 1, 3, 3]), a([2, 2, 3, 2])),
        "flipped_relearning_again_in_real_lapse": (a([2, 3, 3, 2]), a([1, 1, 3, 3]), a([1, 2, 3, 3]), a([2, 3, 3, 2])),
        "unexplained_relearning_start_kept": (a([3, 3, 2]), a([1, 3, 3]), a([2, 3, 3]), a([3, 3, 2])),
        "learning_and_new_again_kept": (a([0, 1, 1, 2]), a([1, 1, 3, 3]), a([2, 2, 3, 3]), a([0, 1, 1, 2])),
        "no_relearning_steps": (a([2, 2, 2]), a([1, 3, 3]), a([2, 3, 3]), a([2, 2, 2])),
        "two_lapses_second_flipped": (a([2, 3, 2, 3, 2]), a([1, 3, 1, 3, 3]), a([1, 3, 2, 3, 3]), a([2, 3, 2, 2, 2])),
    }
    res = {k: bool(np.array_equal(st(s, r, rc), e)) for k, (s, r, rc, e) in cases.items()}
    rng = np.random.default_rng(1)
    s = rng.integers(0, 4, 500)
    r = rng.integers(1, 5, 500)
    res["no_flip_identity_random"] = bool(np.array_equal(st(s, r, r), s))
    return all(res.values()), res


def frame_of(user):
    from scripts import compress_common as cc
    from scripts import sm31_inputs as si
    from scripts import sm71_leak
    data = cc.data_path()
    frame = si.load_frame(data, user, "e2s-query")
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    return si.apply_frame(frame, "e2s-query")


def samples_of(frame, user):
    from scripts import compress_common as cc
    from scripts import sm31_inputs as si
    with si.query_patch("e2s-query"):
        return cc.make_samples(frame, user, len(frame))


def same_sample_parts(a, b, skip_features=False):
    import torch
    ok = {"global_labels": torch.equal(a.global_labels, b.global_labels),
          "label_review_ths": torch.equal(a.label_review_ths, b.label_review_ths),
          "review_ths": torch.equal(a.review_ths, b.review_ths),
          "ids": all(torch.equal(a.ids[k], b.ids[k]) for k in a.ids) and a.ids.keys() == b.ids.keys(),
          "day_offsets": torch.equal(a.day_offsets, b.day_offsets) and torch.equal(a.day_offsets_first, b.day_offsets_first),
          "skips": torch.equal(a.skips, b.skips)}
    if not skip_features:
        ok["card_features"] = torch.equal(a.card_features, b.card_features)
    return ok


def k2():
    from rwkv.data_processing import CARD_FEATURE_COLUMNS
    from scripts import sm31_inputs as si
    col = {c: i for i, c in enumerate(CARD_FEATURE_COLUMNS)}
    res, detail = {}, {}
    for u in FRAME_USERS:
        f0 = frame_of(u)
        s0 = samples_of(f0, u)[0]
        fz, info0, fl0, again = si.again_as_hard(f0, u, 0.0)
        res[f"{u}/p0_frame_equal"] = bool(fz.equals(f0) and list(fz.columns) == list(f0.columns)
                                          and (fz.dtypes == f0.dtypes).all() and len(fl0) == 0)
        res[f"{u}/p0_samples_equal"] = all(same_sample_parts(samples_of(fz, u)[0], s0).values())
        m1, m2 = si.again_as_hard_flips(f0, u, 0.2), si.again_as_hard_flips(f0, u, 0.2)
        res[f"{u}/deterministic"] = bool(np.array_equal(m1, m2))
        masks = [si.again_as_hard_flips(f0, u, p) for p in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)]
        res[f"{u}/nested"] = all(bool((a <= b).all()) for a, b in zip(masks, masks[1:]))
        res[f"{u}/p1_flips_every_again"] = bool(np.array_equal(masks[-1], f0["rating"].to_numpy() == 1))
        n_again = int((f0["rating"] == 1).sum())
        detail[f"{u}/flip_share_at_p"] = {p: round(int(m.sum()) / max(n_again, 1), 4)
                                          for p, m in zip((0.05, 0.1, 0.2, 0.3, 0.5, 1.0), masks)}
        for state in ("anki", "keep"):
            f1, info1, fl1, _ = si.again_as_hard(f0, u, 1.0, state)
            s1 = samples_of(f1, u)[0]
            parts = same_sample_parts(s1, s0, skip_features=True)
            res[f"{u}/{state}/labels_ids_unchanged"] = all(parts.values())
            a, b = s0.card_features.numpy(), s1.card_features.numpy()
            diff = a != b
            cols = sorted(CARD_FEATURE_COLUMNS[j] for j in np.flatnonzero(diff.any(0)))
            allowed = {"rating_1", "rating_2"} | ({"scaled_state"} if state == "anki" else set())
            res[f"{u}/{state}/only_grade_columns"] = set(cols) <= allowed and {"rating_1", "rating_2"} <= set(cols)
            is_query = a[:, col["is_query"]] == 1
            res[f"{u}/{state}/query_rows_unchanged"] = not diff[is_query].any()
            ths = s0.review_ths.numpy()
            upd = ~is_query
            flipped_rows = set(ths[upd & (a[:, col["rating_1"]] == 1)].tolist())
            res[f"{u}/{state}/every_again_update_row_reads_hard"] = (
                flipped_rows == set(fl1.tolist()) == set(again.tolist())
                and not b[upd, col["rating_1"]].any()
                and bool((b[upd & (a[:, col["rating_1"]] == 1), col["rating_2"]] == 1).all()))
            n_state = int(diff[:, col["scaled_state"]].sum())
            res[f"{u}/{state}/state_rows_as_reported"] = n_state == info1["state_rows_changed"] and (
                (n_state > 0) if state == "anki" else n_state == 0)
            detail[f"{u}/{state}"] = {"again": info1["again"], "flipped": info1["flipped"],
                                      "state_rows_changed": info1["state_rows_changed"], "changed_columns": cols}
    return all(res.values()), {"checks": res, "detail": detail}


def k3():
    import lmdb
    import torch
    from rwkv.prepare_batch import prepare
    from scripts import compress_common as cc
    from scripts import small_model_heldout as smh
    from scripts import state_vq
    from scripts import sm31_inputs as si
    from scripts.compress_metrics import read_labels
    from scripts.heldout_fast_metrics import evaluate_with_arrays
    torch.set_num_threads(2)
    u = 2388
    base = {k: v.float() for k, v in torch.load(CK118, map_location="cpu", weights_only=True).items()}
    model = smh.build_expertium("w64", base)
    layout = state_vq.install(model)
    smh.set_expertium_state_bits(model, 0)
    state_vq.configure(layout, None, 0)
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    with env.begin() as txn:
        bench = set(read_labels(txn, u)[0])
    f0 = frame_of(u)
    th = f0["review_th"].to_numpy()
    nxt = f0.groupby("card_id")["review_th"].shift(-1).to_numpy()
    cand = [t for t, r, n in zip(th, f0["rating"].to_numpy(), nxt) if r == 1 and t in bench and n == n]
    t = int(cand[len(cand) // 2])
    f1, info, fl, _ = si.again_as_hard(f0, u, 1.0, "anki", flip=th == t)
    seed = 2025 + u
    with torch.inference_mode():
        _, p0 = evaluate_with_arrays(model, [{"batch": prepare(samples_of(f0, u), seed=seed)}])
        _, p1 = evaluate_with_arrays(model, [{"batch": prepare(samples_of(f1, u), seed=seed)}])
    prev = dict(zip(th.tolist(), f0.groupby("card_id")["review_th"].shift(1).fillna(-1).astype(int).tolist()))
    res = {"one_review_flipped": fl.tolist() == [t]}
    k0, v0 = p0.heads["instant"]
    k1_, v1 = p1.heads["instant"]
    early = k0 <= t
    res["instant_keys_equal"] = bool(np.array_equal(k0, k1_))
    res["instant_up_to_t_equal"] = bool(np.array_equal(v0[early], v1[early]))
    a0, w0 = p0.heads["ahead"]
    a1, w1 = p1.heads["ahead"]
    before = np.array([prev[int(j)] < t for j in a0])
    res["curve_from_rows_before_t_equal"] = bool(np.array_equal(a0, a1) and np.array_equal(w0[before], w1[before]))
    res["labels_equal"] = bool(np.array_equal(p0.heads["rating"][0], p1.heads["rating"][0])
                               and np.array_equal(p0.heads["rating"][1], p1.heads["rating"][1]))
    res["something_after_t_differs"] = bool((v0[~early] != v1[~early]).any() or (w0[~before] != w1[~before]).any())
    detail = {"user": u, "t": t, "instant_reviews_up_to_t": int(early.sum()), "instant_reviews_after_t": int((~early).sum()),
              "instant_after_t_changed": int((v0[~early] != v1[~early]).sum()),
              "curve_from_rows_before_t": int(before.sum()), "state_rows_changed": info["state_rows_changed"]}
    return all(res.values()), {"checks": res, "detail": detail}


def score(model, label_args, extra):
    cmd = [PY, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--device", "cpu", "--pool", "732",
           "--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour", "--nonfinite-rows",
           "--users-file", str(HERE / "smoke-users.json"), "--model", model, *label_args, *extra, "--out", str(SMOKE)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"[sm132-check] CPU scorer failed: {p.stderr[-800:]}")


def rows_of(path, users=SMOKE_USERS):
    out = {}
    for line in open(path, "rb"):
        try:
            r = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if r.get("user_id") in users and "benchmark" in r:
            out[(r["user_id"], r["label"])] = r
    return out


def maxdiff(a, b):
    return max(abs(a[h]["LogLoss"] - b[h]["LogLoss"]) for h in HEADS)


def k4():
    SMOKE.parent.mkdir(parents=True, exist_ok=True)
    m118, m91 = f"{T118}=expertium:w64:{CK118}", f"{T91}=fork::{CK91}"
    for model, tag in ((m118, T118), (m91, T91)):
        score(model, ["--only", f"{tag}/w8+s8"], [])
        score(model, ["--only", f"{tag}/w8+s8"], ["--again-as-hard", "0"])
    score(m118, ["--only", f"{T118}/w8+s8"], ["--again-as-hard", "0.2"])
    score(m118, ["--only", f"{T118}/w8+s8"], ["--again-as-hard", "1"])
    rows = rows_of(SMOKE)
    pub118, pub91, cpu131 = rows_of(PUB118), rows_of(PUB91), rows_of(CPU131)
    res, diffs = {}, {}
    for u in SMOKE_USERS:
        for tag in (T118, T91):
            a, b = rows[(u, f"{tag}/w8+s8@e2s-query")], rows[(u, f"{tag}/w8+s8@e2s-query+aah0")]
            res[f"{u}/{tag}/p0_equals_no_flag"] = all(a[k] == b[k] for k in ("benchmark", "benchmark_existing",
                                                                           "benchmark_deleted"))
            res[f"{u}/{tag}/p0_corrupt_labels_equal"] = b["benchmark_corrupt_labels"] == b["benchmark"]
            res[f"{u}/{tag}/p0_nothing_flipped"] = b["again_as_hard"]["flipped"] == 0
        nf = rows[(u, f"{T91}/w8+s8@e2s-query")]
        res[f"{u}/big_no_flag_equals_sm131_cpu_rows"] = nf["benchmark"] == cpu131[(u, f"{T91}/w8+s8@e2s-query")]["benchmark"]
        d = maxdiff(rows[(u, f"{T118}/w8+s8@e2s-query")]["benchmark"], pub118[(u, f"{T118}/w8+s8@e2s-query")]["benchmark"])
        diffs[f"{u}/sm118_no_flag_vs_published_gpu"] = d
        res[f"{u}/sm118_no_flag_vs_published_gpu_1e-5"] = d <= 1e-5
        diffs[f"{u}/sm91s2_no_flag_vs_published_gpu"] = maxdiff(nf["benchmark"], pub91[(u, f"{T91}/w8+s8@e2s-query")]["benchmark"])
        p1 = rows[(u, f"{T118}/w8+s8@e2s-query+aah100")]["again_as_hard"]
        res[f"{u}/p1_flips_every_again"] = (p1["flipped"] == p1["again"] > 0
                                            and p1["benchmark_flipped"] == p1["benchmark_again"])
        r2 = rows[(u, f"{T118}/w8+s8@e2s-query+aah20")]
        res[f"{u}/p20_finite_and_partial"] = (all(np.isfinite(r2["benchmark"][h]["LogLoss"]) for h in HEADS)
                                              and 0 < r2["again_as_hard"]["flipped"] < r2["again_as_hard"]["again"])
        diffs[f"{u}/p20_vs_p0"] = {h: round(r2["benchmark"][h]["LogLoss"] - rows[(u, f"{T118}/w8+s8@e2s-query+aah0")]["benchmark"][h]["LogLoss"], 6) for h in HEADS}
    return all(res.values()), {"checks": res, "diffs": diffs}


def k5():
    p = subprocess.run([PY, "-X", "utf8", str(HERE / "sm132_rule.py"), "selftest"], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout.strip().splitlines() or [p.stderr[-400:]])[-1:]


def smoke():
    cpu, gpu = rows_of(SMOKE), rows_of(HERE / "gpu/smoke-gpu-rows.jsonl")
    pub = {**rows_of(PUB118), **rows_of(PUB91)}
    want = [(u, f"{T118}/w8+s8@e2s-query+aah0") for u in SMOKE_USERS] + \
           [(u, f"{T118}/w8+s8@e2s-query+aah20") for u in SMOKE_USERS] + \
           [(u, f"{T91}/w8+s8@e2s-query+aah0") for u in SMOKE_USERS]
    res, diffs = {}, {}
    for key in want:
        name = f"{key[0]}/{key[1]}"
        if key not in gpu or key not in cpu:
            res[name] = False
            continue
        d = maxdiff(gpu[key]["benchmark"], cpu[key]["benchmark"])
        same_counts = gpu[key]["again_as_hard"] == cpu[key]["again_as_hard"]
        ok = d <= 1e-5 and same_counts
        if key[1].endswith("+aah0"):
            pkey = (key[0], key[1][:-len("+aah0")])
            dp = maxdiff(gpu[key]["benchmark"], pub[pkey]["benchmark"])
            ok &= dp <= 1e-5
            diffs[name + " vs published"] = dp
        diffs[name] = d
        res[name] = bool(ok)
    out = {"pass": all(res.values()), "checks": res, "max_abs_logloss_diff": diffs}
    (HERE / "gpu/smoke-check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm132-smoke] {'PASS' if out['pass'] else 'FAIL'} {json.dumps(diffs)}")
    return 0 if out["pass"] else 1


def main():
    if sys.argv[1:] == ["smoke"]:
        return smoke()
    from scripts import small_model_heldout  # noqa: F401  (installs the state-quant model before rwkv's model import)
    t0 = time.time()
    out = {}
    for name, fn in (("K1_state_machine", k1), ("K2_frames_samples", k2), ("K3_causality", k3),
                     ("K4_cpu_scorer_rows", k4), ("K5_rule_selftest", k5)):
        ok, detail = fn()
        out[name] = {"pass": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail)[:600]}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "check-cpu-sm132.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm132-check] {'ALL OK' if out['pass'] else 'FAILED'} ({out['seconds']} s)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
