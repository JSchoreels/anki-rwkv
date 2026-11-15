"""Time heads for the family (SM117): the CPU proof (scripts/sm117_head.py check). Checks, per model kind (Tiny w64-c1,
Small w64, Big fork; --models), before any GPU ticket:

  C0 masters: every family master's sha256 equals its package manifest's (all six), the shipped head of the show-time
     retrain (SM92) exists; the feature width per model (474 / 474 / 1306).
  C1 frozen: a CPU head training on 3 small plan users: no model parameter trainable or in the optimizer, the trunk
     written after training is bitwise the release (keys, dtypes, values), the trunk in memory unchanged, both heads
     moved, both first losses ln 16, head size = inputs x 128 + 128 + 128 x 64 + 64.
  C2 resume (first model only): stopped after one buffer pass and resumed = one run, bitwise.
  C3 capture: on 2 evaluation users (full history, show-time inputs) the capture path gives every model output bitwise
     the stock forward's (Big: the sequential capture vs SrsRWKV.forward_batch), features finite, [rows, inputs - 26].
  C4a show-time audit (the same-day question): on a training user, 6 reviews (3 on the same day as the card's previous
     review, 3 later) get a longer answer with the moment the card was shown kept (duration + 5 s, its answer-to-answer
     interval + 5 s, the card's next interval - 5 s). Show-time inputs: every input row up to the review's instant row
     and every frozen feature of instant rows predicting that review or an earlier one is bitwise unchanged (later ones
     change). Normal inputs (the contrast that made SM82 drop same-day reviews): the review's own instant row changes.
     Static: show-time vs normal batches differ only on instant rows and only in the 8 interval columns; instant rows
     carry a zero duration input.
  C4b explicit features: changing review j's duration and grade changes no row 0..j; prefix tables = the full table's
     first rows (as SM82c C4b).
  C5 labels: rows per user = the plan's (SM117 rule), grades in range, no duplicate review, same-day rows counted.
  C6 loss = metric: the training loss per row = the scorer's censored LogLoss.
  C7 scorer + grade: scripts/small_model_heldout.py (CPU, w8+s8, show-time, leak-free) gives equal rows and dumps for
     the release and the trunk written after training (1 user); the SM117 time scorer scores 2 users (finite, the
     baselines' reviews, every S).
  C8 train path = score path: the training batch builder's features (no cap) = the scorer's (LeakBatches e2s-query).
  C9 R-under-S arithmetic: cdf_at / event_prob vs a brute-force loop; LogLoss / Brier sums; ECE (20 equal-mass bins)
     and AUC from the fine histograms vs exact per-row computations; the running baselines (a) / (b) vs a brute-force
     loop and causal (changing review j changes no row 0..j).
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sm117_head as H  # noqa: E402
from scripts.sm117_timelib import (BINS, EXPLICIT_DIM, atomic_json, censored_ll, explicit_table, revlog,  # noqa: E402
                                   time_loss_rows, grade_compare)


def run_check(args):
    from scripts import small_model_heldout as smh  # noqa: F401  first
    import torch
    from scripts import compress_common as cc
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "models": args.models.split(",")}
    fails = []
    py = [sys.executable, "-X", "utf8", str(ROOT / "scripts/sm117_head.py")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    env.pop("EXPERTIUM_END_TO_START", None)
    data = cc.data_path()
    counts = cc.run_info()["review_counts"]
    (H.HERE / "work").mkdir(parents=True, exist_ok=True)

    def mark(name, ok, **rec):
        report["checks"][name] = {"pass": bool(ok), **rec}
        if not ok:
            fails.append(name)
        print(f"[sm117-check] {name}: {'PASS' if ok else 'FAIL'} ({time.perf_counter() - started:.0f} s)", flush=True)
        atomic_json(args.out, dict(report, fails=fails, partial=True))

    # C0 masters
    c0 = {}
    for key in H.ORDER:
        info = H.model_info(key)
        c0[key] = {"master": str(info["master"]), "sha256_ok": H.sha256(info["master"]) == info["sha256"], "label": info["label"]}
    c0["sm92_shipped_head_exists"] = Path(H.MODELS["sm92"]["shipped_head"]).exists()
    mark("C0_masters", all(v["sha256_ok"] for k, v in c0.items() if isinstance(v, dict)) and c0["sm92_shipped_head_exists"], masters=c0)

    plan_rec = json.loads((H.HERE / "plan.json").read_text())
    small = [r for r in plan_rec["order"] if 400 <= r["reviews"] <= 3000 and r["same_day"] > 0][:3]
    pool = json.loads(H.U732.read_text())["users"]
    eval_small = sorted((u for u in pool if 1500 <= counts[str(u)] <= 4000), key=lambda u: counts[str(u)])[:2]
    report["users"] = {"train": [r["user_id"] for r in small], "eval": eval_small}
    with tempfile.TemporaryDirectory(dir=str(H.HERE / "work")) as tmp:
        tmp = Path(tmp)
        mini_plan = tmp / "plan.json"
        mini_plan.write_text(json.dumps(dict(plan_rec, order=small)))
        for mi, key in enumerate(report["models"]):
            info = H.verified_master(key)
            released = torch.load(info["master"], map_location="cpu", weights_only=True)
            common = ["--model", key, "--plan", str(mini_plan), "--device", "cpu", "--batch", "128", "--buffer-rows", "600",
                      "--workers", "0", "--log-every", "5"]

            def run(cmd):
                r = subprocess.run(py + cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
                return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]
            rc_a, log_a = run(["train", "--out", str(tmp / f"{key}-a")] + common)
            ta = json.loads((tmp / f"{key}-a/train.json").read_text()) if (tmp / f"{key}-a/train.json").exists() else {}
            if rc_a != 0 or not ta.get("master"):
                mark(f"C1_frozen_{key}", False, log=log_a)
                continue
            trunk = torch.load(ta["master"], map_location="cpu", weights_only=True)
            head = torch.load(ta["head"], map_location="cpu", weights_only=True)
            abl = torch.load(ta["ablation"], map_location="cpu", weights_only=True)
            same_keys = set(trunk) == set(released)
            bitwise = same_keys and all(torch.equal(trunk[k], released[k]) and trunk[k].dtype == released[k].dtype for k in released)
            moved = [float(head["out.weight"].abs().max()), float(head["out.bias"].abs().max()), float(abl["out.weight"].abs().max())]
            inputs = ta["inputs"]["total"]
            want_inputs = 1306 if H.MODELS[key]["kind"] == "fork" else 474
            f1 = (bitwise and min(moved) > 0 and ta["trunk_unchanged_in_memory"] and ta["trainable_model_parameters"] == []
                  and ta["model_parameters_in_optimizer"] == 0 and ta["step"] >= 5 and abs(ta["first_loss"] - math.log(BINS)) < 1e-5
                  and abs(ta["first_loss_ablation"] - math.log(BINS)) < 1e-5 and inputs == want_inputs
                  and ta["head_values"] == inputs * 128 + 128 + 128 * 64 + 64)
            mark(f"C1_frozen_{key}", f1, steps=ta["step"], rows=ta["rows"], same_day_rows=ta["same_day_rows"],
                 trunk_bitwise_release=bitwise, out_layer_abs_max=moved, inputs=ta["inputs"], head_values=ta["head_values"],
                 trainable_model_parameters=ta["trainable_model_parameters"],
                 model_parameters_in_optimizer=ta["model_parameters_in_optimizer"],
                 first_loss=[ta["first_loss"], ta["first_loss_ablation"]], pass_losses=ta["pass_losses"])
            f5 = (ta["rows_planned_mismatch"] == [] and ta["grade_out_of_range"] == 0 and ta["duplicate_review_th"] == 0
                  and not ta["skipped"] and ta["rows"] == sum(r["labelled"] for r in small) and ta["nonfinite_features"] == 0
                  and ta["same_day_rows"] == sum(r["same_day"] for r in small) and ta["duration_input_nonzero"] == 0)
            mark(f"C5_labels_{key}", f5, rows=ta["rows"], planned=sum(r["labelled"] for r in small),
                 same_day=[ta["same_day_rows"], sum(r["same_day"] for r in small)], mismatch=ta["rows_planned_mismatch"],
                 skipped=ta["skipped"], duration_input_nonzero=ta["duration_input_nonzero"])
            if mi == 0:
                rc_b1, log_b1 = run(["train", "--out", str(tmp / f"{key}-b"), "--stop-after-passes", "1"] + common)
                rc_b2, log_b2 = run(["train", "--out", str(tmp / f"{key}-b")] + common)
                tb = json.loads((tmp / f"{key}-b/train.json").read_text()) if (tmp / f"{key}-b/train.json").exists() else {}
                if rc_b1 == 0 and rc_b2 == 0 and tb.get("head"):
                    hb = torch.load(tb["head"], map_location="cpu", weights_only=True)
                    ab = torch.load(tb["ablation"], map_location="cpu", weights_only=True)
                    f2 = (all(torch.equal(head[k], hb[k]) for k in head) and all(torch.equal(abl[k], ab[k]) for k in abl)
                          and tb["step"] == ta["step"] and tb["passes"] == ta["passes"])
                else:
                    f2 = False
                mark(f"C2_resume_bitwise_{key}", f2, steps=[ta.get("step"), tb.get("step")],
                     log=None if f2 else (log_b1 + log_b2)[-2000:])
            # C3 capture path == stock forward
            model = H.build_model(key, trunk, torch.device("cpu"))
            fz = H.Frozen(model, H.MODELS[key]["kind"])
            probes = []
            for user in eval_small:
                _, batch, error, _ = smh.LeakBatches([user], data, counts, "neighbour", 0, H.INPUTS)[0]
                with torch.inference_mode():
                    a = fz.stock(batch)
                    _, _, feats, c = fz.forward(batch)
                    hp_ok = True
                    if H.MODELS[key]["kind"] == "fork":   # the head_p output the capture computes feeds p_linear exactly
                        lab = batch.labels[0]
                        q = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
                        _, _, hp_all = H.fork_forward_capture(model, batch, q)
                        hp_ok = bool(torch.equal(model.p_linear(hp_all.float()), a[3]))
                probes.append({"user": user, "rows": int(a[3].shape[1]), "outputs": len(a), "head_p_feeds_p_linear": hp_ok,
                               "stock_vs_capture_path": [bool(torch.equal(a[i], c[i])) for i in range(len(a))],
                               "features_finite": bool(torch.isfinite(feats).all()), "feature_shape": list(feats.shape)})
            f3 = all(all(p["stock_vs_capture_path"]) and p["features_finite"] and p["head_p_feeds_p_linear"] and p["feature_shape"][1] == inputs - EXPLICIT_DIM
                     for p in probes)
            mark(f"C3_capture_bitwise_{key}", f3, probes=probes)
            # C4a show-time audit
            user = small[1]["user_id"]
            rev0 = revlog(data, user)
            raw0 = _raw(data, user)
            e = raw0["elapsed_days"].to_numpy(); d = raw0["duration"].to_numpy()
            nxt = _next_same_card(raw0["card_id"].to_numpy())
            same = [j for j in range(len(raw0)) if e[j] == 0 and d[j] > 0 and d[j] < 50000]
            later = [j for j in range(len(raw0)) if e[j] >= 1 and d[j] > 0 and d[j] < 50000]
            rng = np.random.default_rng(0)
            picks = sorted(rng.choice(same, 3, replace=False).tolist()) + sorted(rng.choice(later, 3, replace=False).tolist())
            base = {}
            for inputs_kind in ("e2s-query", "normal"):
                b0, _ = H.show_time_batch(data, user, inputs=inputs_kind)
                with torch.inference_mode():
                    qi0, th0, f0, _ = fz.forward(b0)
                o = torch.argsort(th0)
                base[inputs_kind] = (b0, qi0[o], th0[o], f0[o])
            audit = []
            for j in picks:
                def edit(frame, j=j):
                    frame = frame.copy()
                    cd, ce = frame.columns.get_loc("duration"), frame.columns.get_loc("elapsed_seconds")
                    frame.iloc[j, cd] = int(frame["duration"].iat[j]) + 5000
                    frame.iloc[j, ce] = int(frame["elapsed_seconds"].iat[j]) + 5
                    if nxt[j] >= 0 and int(frame["elapsed_seconds"].iat[nxt[j]]) >= 5:
                        frame.iloc[nxt[j], ce] = int(frame["elapsed_seconds"].iat[nxt[j]]) - 5
                    return frame
                rec = {"review_th": j + 1, "same_day": bool(e[j] == 0), "duration_ms": int(d[j])}
                for inputs_kind in ("e2s-query", "normal"):
                    b0, qi0, th0, f0 = base[inputs_kind]
                    with H.revlog_edit(edit):
                        b1, _ = H.show_time_batch(data, user, inputs=inputs_kind)
                    with torch.inference_mode():
                        qi1, th1, f1_, _ = fz.forward(b1)
                    o = torch.argsort(th1)
                    qi1, th1, f1_ = qi1[o], th1[o], f1_[o]
                    upto = th0 <= j + 1
                    own = th0 == j + 1
                    s0, s1 = b0.start.reshape(-1, b0.start.shape[-1]), b1.start.reshape(-1, b1.start.shape[-1])
                    rows_changed = torch.nonzero((s0 != s1).any(-1)).squeeze(-1)
                    qrow = int(qi0[own][0]) if bool(own.any()) else None
                    later_rows = th0 > j + 1
                    rec[inputs_kind] = {
                        "features_up_to_review_bitwise": bool(torch.equal(th0, th1) and torch.equal(f0[upto], f1_[upto])),
                        "own_instant_row_changed": bool(not torch.equal(f0[own], f1_[own])),
                        "later_features_changed": bool(not torch.equal(f0[later_rows], f1_[later_rows])) if bool(later_rows.any()) else None,
                        "input_rows_changed_all_after_query": qrow is not None and bool((rows_changed > qrow).all()),
                        "input_rows_changed": int(rows_changed.numel())}
                audit.append(rec)
            b_st, qi_st, _, _ = base["e2s-query"]
            b_no = base["normal"][0]
            s_st, s_no = b_st.start.reshape(-1, b_st.start.shape[-1]), b_no.start.reshape(-1, b_no.start.shape[-1])
            diff = s_st != s_no
            qmask = torch.zeros(s_st.shape[0], dtype=torch.bool)
            qmask[qi_st] = True
            static = {"rows_differ_only_on_instant_rows": bool(not diff[~qmask].any()),
                      "columns_differing": sorted(set(torch.nonzero(diff)[:, 1].tolist())),
                      "instant_rows_duration_input_zero": bool((s_st[qi_st, 8] == 0).all()),
                      "rows_with_a_difference": int(diff.any(-1).sum()), "instant_rows": int(qmask.sum())}
            f4a = (all(r["e2s-query"]["features_up_to_review_bitwise"] and r["e2s-query"]["input_rows_changed_all_after_query"]
                       and r["e2s-query"]["later_features_changed"] is not False and r["normal"]["own_instant_row_changed"]
                       for r in audit)
                   and static["rows_differ_only_on_instant_rows"] and set(static["columns_differing"]) <= set(range(2, 8))
                   and static["instant_rows_duration_input_zero"])
            mark(f"C4a_show_time_audit_{key}", f4a, user=user, picks=audit, static=static,
                 note="columns 2-7 = scaled_elapsed_seconds, its sin / cos, scaled cumulative, its sin / cos")
            # C8 train path == score path
            if mi == 0 or H.MODELS[key]["kind"] == "fork":
                bt, _ = H.show_time_batch(data, eval_small[0])
                _, bs, _, _ = smh.LeakBatches([eval_small[0]], data, counts, "neighbour", 0, H.INPUTS)[0]
                with torch.inference_mode():
                    _, tha, fa, _ = fz.forward(bt)
                    _, thb, fb_, _ = fz.forward(bs)
                mark(f"C8_train_path_equals_score_path_{key}", bool(torch.equal(tha, thb) and torch.equal(fa, fb_)),
                     user=eval_small[0], rows=int(len(tha)))
            # C7 scorer + grade (1 user) and the time scorer (2 users)
            ufile = tmp / "users1.json"
            ufile.write_text(json.dumps({"users": eval_small[:1]}))
            rows, dump = tmp / f"{key}-grade.jsonl", tmp / f"{key}-dump"
            cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file",
                   str(ufile), "--model", f"rel={H.spec(key, info['master'])}", "--model", f"new={H.spec(key, ta['master'])}",
                   "--only", "rel/w8+s8", "--only", "new/w8+s8", "--inputs", H.INPUTS, "--device", "cpu", "--workers", "0",
                   "--split-existing", "--deleted-cards", "neighbour", "--dump", str(dump), "--out", str(rows)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            cmp = (grade_compare(rows, dump, "rel", "new", eval_small[:1], variant=H.VARIANT)
                   if r.returncode == 0 and rows.exists() else {"pass": False})
            u2 = tmp / "users2.json"
            u2.write_text(json.dumps({"users": eval_small}))
            trows = tmp / f"{key}-time.jsonl"
            sc = ["score", "--model", key, "--head-dir", str(tmp / f"{key}-a"), "--users-file", str(u2), "--out", str(trows),
                  "--device", "cpu", "--workers", "0"]
            if key == "sm92":
                sc += ["--shipped-head", str(H.MODELS["sm92"]["shipped_head"])]
            r2 = subprocess.run(py + sc, capture_output=True, text=True, env=env, cwd=str(ROOT))
            tr = [json.loads(x) for x in trows.read_text().splitlines()] if trows.exists() else []
            time_ok = r2.returncode == 0 and len(tr) == 2 and all(
                "error" not in x and x["baseline_same_reviews"] and x["nonfinite_inputs"] == 0
                and math.isfinite(x["time"]["model"]["cond"]["ll"]) and len(x["rus"]["model"]) == len(H.S_GRID)
                and all(x["rus"][p][str(s)][0] == x["rus"]["model"][str(s)][0] for p in x["rus"] for s in H.S_GRID) for x in tr)
            mark(f"C7_scorer_grade_{key}", cmp["pass"] and time_ok, grade=cmp, scorer_exit=r.returncode,
                 scorer_log=None if r.returncode == 0 else (r.stdout + r.stderr)[-2000:], time_scorer_exit=r2.returncode,
                 time_rows=[{"user_id": x.get("user_id"), "reviews": x.get("reviews"), "later": x.get("later_day_reviews"),
                             "ll_per_review": x["time"]["model"]["cond"]["ll"] / max(x["time"]["model"]["cond"]["n"], 1),
                             "rus_10s": x["rus"]["model"]["10"]} for x in tr if "time" in x],
                 time_log=None if time_ok else (r2.stdout + r2.stderr)[-2000:])
            del model, fz
    # C4b explicit features (model independent)
    eb = []
    for u in [small[0]["user_id"], eval_small[0]]:
        rev = revlog(data, u)
        full = explicit_table(rev)
        r2 = np.random.default_rng(1)
        js = sorted(r2.choice(len(rev) - 1, size=min(10, len(rev) - 1), replace=False).tolist())
        ok_upto, changed = True, 0
        for j in js:
            pr = rev.copy()
            pr.iloc[j, pr.columns.get_loc("duration")] = int(rev["duration"].iat[j]) * 3 + 1000
            pr.iloc[j, pr.columns.get_loc("rating")] = int(rev["rating"].iat[j]) % 4 + 1
            t2 = explicit_table(pr)
            ok_upto &= bool(np.array_equal(full[:j + 1], t2[:j + 1]))
            changed += int(not np.array_equal(full[j + 1:], t2[j + 1:]))
        cuts = sorted(r2.choice(np.arange(1, len(rev)), size=5, replace=False).tolist())
        prefix_ok = all(np.array_equal(explicit_table(rev.iloc[:k]), full[:k]) for k in cuts)
        eb.append({"user": u, "perturbed": len(js), "rows_up_to_j_unchanged": ok_upto, "later_rows_changed": changed,
                   "prefix_equal": prefix_ok, "pass": bool(ok_upto and changed > 0 and prefix_ok and np.isfinite(full).all())})
    mark("C4b_no_leak_explicit_features", all(r["pass"] for r in eb), users=eb)
    # C6 loss = metric
    g = torch.Generator().manual_seed(0)
    n = 4096
    logits = torch.randn(n, 4 * BINS, generator=g, dtype=torch.float64) * 1.5
    grade = torch.randint(0, 4, (n,), generator=g)
    lower = torch.randint(0, BINS, (n,), generator=g)
    cens = (torch.rand(n, generator=g) < 0.2).long()
    ours = time_loss_rows(logits, grade, lower, cens).double().numpy()
    prob = torch.softmax(logits.view(n, 4, BINS)[torch.arange(n), grade], -1).numpy()
    gap = float(np.abs(ours - censored_ll(prob, lower.numpy(), cens.numpy())).max())
    mark("C6_loss_equals_metric", gap < 1e-5, rows=n, max_abs_gap=gap)
    # C9 R-under-S arithmetic
    ok9, rec9 = _c9(data, small[0]["user_id"])
    mark("C9_r_under_s_arithmetic", ok9, **rec9)
    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1), partial=False)
    atomic_json(args.out, report)
    print(f"[sm117-check] {'PASS' if not fails else 'FAIL ' + ','.join(fails)} ({report['seconds']} s; {args.out})")
    return 0 if not fails else 1


def _raw(data, user):
    import pandas as pd
    return pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}")


def _next_same_card(card):
    nxt = np.full(len(card), -1)
    last = {}
    for i in range(len(card) - 1, -1, -1):
        nxt[i] = last.get(int(card[i]), -1)
        last[int(card[i])] = i
    return nxt


def _c9(data, user):
    rng = np.random.default_rng(0)
    n = 300
    pg = rng.dirichlet(np.ones(4), n)
    pt = rng.dirichlet(np.ones(BINS) * 0.7, (n, 4))
    lo = [math.log(0.25)] + [math.log(e / 1000.0) for e in H.BIN_EDGES_MS]
    hi = [math.log(e / 1000.0) for e in H.BIN_EDGES_MS] + [float("inf")]
    gap_cdf = gap_ev = 0.0
    for s in H.S_GRID:
        ls = math.log(s)
        fast = H.cdf_at(pt, s)
        ev = H.event_prob(pg, pt, s)
        for i in range(n):
            tot_e = 0.0
            for gi in range(4):
                c = 0.0
                for k in range(BINS):   # brute force: the share of each bin's ln-time interval at or below ln s
                    if hi[k] <= ls:
                        c += pt[i, gi, k]
                    elif lo[k] < ls < hi[k]:
                        c += pt[i, gi, k] * (ls - lo[k]) / (hi[k] - lo[k])
                gap_cdf = max(gap_cdf, abs(c - fast[i, gi]))
                if gi >= 1:
                    tot_e += pg[i, gi] * c
            gap_ev = max(gap_ev, abs(tot_e - ev[i]))
    # metrics from histograms vs exact
    m = 200_000
    z = rng.normal(0.5, 1.6, m)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(m) < 1 / (1 + np.exp(-(z + 0.3 * rng.normal(size=m))))).astype(np.float64)
    nh, ph, sh = H.hist_of(p, y)
    st = H.ece_auc_from_hist(nh, ph, sh)
    o = np.argsort(p, kind="stable")
    groups = np.array_split(o, H.ECE_BINS)
    ece_exact = sum(abs(p[gix].sum() - y[gix].sum()) for gix in groups) / m
    ranks = np.empty(m); ranks[o] = np.arange(1, m + 1)
    P = y.sum(); N = m - P
    auc_exact = (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    s5 = H.binary_sums(p, y)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll_exact = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).sum())
    # running baselines vs a loop, and causality
    rev = revlog(data, user)
    pri = json.loads((H.HERE / "priors.json").read_text())
    F, E = H.running_baselines(rev, pri)
    d = rev["duration"].to_numpy(); r = rev["rating"].to_numpy(); e = rev["elapsed_days"].to_numpy()
    M = pri["pseudo_count"]; f0 = pri["F0_under_S_given_grade"]; e0 = pri["E0_event"]
    cnt = np.zeros(4); under = np.zeros((4, len(H.S_GRID))); nall = 0; evc = np.zeros(len(H.S_GRID))
    gap_base = 0.0
    for i in range(len(rev)):
        for k in range(len(H.S_GRID)):
            for gi in range(4):
                gap_base = max(gap_base, abs(F[i, gi, k] - (under[gi, k] + M * f0[gi][k]) / (cnt[gi] + M)))
            gap_base = max(gap_base, abs(E[i, k] - (evc[k] + M * e0[k]) / (nall + M)))
        if d[i] > 0 and e[i] >= 0:
            gi = r[i] - 1
            cnt[gi] += 1; nall += 1
            for k, s in enumerate(H.S_GRID):
                under[gi, k] += d[i] <= 1000 * s
                evc[k] += (r[i] >= 2) and d[i] <= 1000 * s
    causal = True
    for j in sorted(np.random.default_rng(2).choice(len(rev) - 1, 8, replace=False).tolist()):
        pr = rev.copy()
        pr.iloc[j, pr.columns.get_loc("duration")] = int(rev["duration"].iat[j]) * 3 + 1000
        pr.iloc[j, pr.columns.get_loc("rating")] = int(rev["rating"].iat[j]) % 4 + 1
        F2, E2 = H.running_baselines(pr, pri)
        causal &= bool(np.array_equal(F[:j + 1], F2[:j + 1]) and np.array_equal(E[:j + 1], E2[:j + 1]))
    rec = {"cdf_max_gap": gap_cdf, "event_max_gap": gap_ev, "ece_hist": st["ece"], "ece_exact": ece_exact,
           "auc_hist": st["auc"], "auc_exact": auc_exact, "logloss_sum": [s5[1], ll_exact], "baseline_max_gap": gap_base,
           "baselines_causal": causal, "baseline_user": user}
    ok = (gap_cdf < 1e-12 and gap_ev < 1e-12 and abs(st["ece"] - ece_exact) < 0.002 and abs(st["auc"] - auc_exact) < 0.001
          and abs(s5[1] - ll_exact) < 1e-6 * max(1.0, ll_exact) and gap_base < 1e-12 and causal)
    return ok, rec
