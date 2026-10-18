#!/usr/bin/env python
"""Build pages/data/r01.json for the R01 GPU-optimization GitHub Pages site.

CPU-only, stdlib-only, read-only on the repository (writes only pages/data/r01.json).
Sources:
  docs/GPU_OPTIMIZATIONS_LOG.md                           per-iteration run tables
  artifacts/retraining/gpu-retrain-r01/iteration-N/comparison.json  decisions, replicated blocks
  artifacts/retraining/gpu-retrain-r01/iteration-N/<model>/<track>/result.json  exact reviews/s
  docs/GPU_OPTIMIZATIONS.md                               ledger track names (F01 ...)

Rerun after new iterations land:
  .venv/Scripts/python.exe scripts/build_r01_pages_data.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs" / "GPU_OPTIMIZATIONS_LOG.md"
LEDGER = ROOT / "docs" / "GPU_OPTIMIZATIONS.md"
R01 = ROOT / "artifacts" / "retraining" / "gpu-retrain-r01"
OUT = ROOT / "pages" / "data" / "r01.json"

MODEL_KEYS = {"expertium": "expertium", "baseline": "alex"}
MODEL_LABELS = {"expertium": "Compact model (outside research)", "alex": "Alex (default)"}  # pages never name the outside researcher


def num(text):
    try:
        return float(str(text).replace(",", "").replace("+", "").replace("%", "").strip())
    except ValueError:
        return None


def model_key(label):
    return "expertium" if "xpertium" in label else "alex"


# ----------------------------------------------------------------------------- ledger names
def read_ledger_names():
    names = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(F\d+[a-z]*)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            names[m.group(1)] = m.group(2).replace("`", "")
    return names


# ----------------------------------------------------------------------------- log tables
def parse_log():
    """Return {iteration:int -> [row dict]} from the run log."""
    text = LOG.read_text(encoding="utf-8")
    sections = {}
    for m in re.finditer(r"<!-- r01-(\d+):start -->(.*?)<!-- r01-\1:end -->", text, re.S):
        it = int(m.group(1))
        header = None
        rows = []
        for line in m.group(2).splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0].startswith("Training start"):
                header = cells
                continue
            if header is None or set(cells[0]) <= set("-: "):
                continue
            row = dict(zip(header, cells))
            rows.append(
                {
                    "start": row.get("Training start (UTC)"),
                    "model": model_key(row.get("Model", "")),
                    "ids": row.get("Improvement IDs", ""),
                    "what": row.get("What was tried", ""),
                    "users": num(row.get("Training users reached")),
                    "reviews": num(row.get("Training reviews")),
                    "seconds": num(row.get("Training seconds")),
                    "rps": num(row.get("Reviews/s")),
                    "d_prev": num(row.get("Speed Δ vs previous")),
                    "d_base": num(row.get("Speed Δ vs baseline")),
                    "notes": row.get("Notes", ""),
                }
            )
        sections[it] = rows
    return sections


def full_rows(rows, model):
    return [
        r
        for r in rows
        if r["model"] == model and r["rps"] and r["users"] == 80 and "excluded from comparisons" not in r["notes"]
    ]


def norm_ids(s):
    return "-".join(p.strip() for p in s.split(",") if p.strip())


# ----------------------------------------------------------------------------- decision
def classify(comp, accepted):
    rb = comp.get("replicated_blocks") or {}
    rbd = rb.get("decision")
    if accepted:
        return "accepted"
    if rbd == "stack candidate":
        return "stack_candidate"
    if rbd == "extend":
        return "extend"
    if rbd == "reject":
        return "rejected"
    d = (comp.get("decision") or "").lower()
    if "fastmode" in d:
        return "fastmode"
    if "investigate" in d:
        return "investigate"
    if "retain previous" in d:
        return "rejected"
    if "promising" in d:
        return "screen"
    return "other"


def read_result(it, mdir, track):
    p = R01 / f"iteration-{it:04d}" / mdir / track / "result.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def unname(value):
    """Page text says "compact-model" (outside research) instead of the researcher's name; keys such as "expertium" stay."""
    if isinstance(value, str):
        return re.sub(r"\bExpertium\b", "compact-model", value)
    if isinstance(value, list):
        return [unname(v) for v in value]
    if isinstance(value, dict):
        return {k: unname(v) for k, v in value.items()}
    return value


def main():
    names = read_ledger_names()
    log = parse_log()
    iters_with_comp = {}
    for p in sorted(R01.glob("iteration-*/comparison.json")):
        it = int(p.parent.name.split("-")[1])
        try:
            iters_with_comp[it] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupt/partial file during a live run
            print(f"warn: {p}: {exc}", file=sys.stderr)

    all_iters = sorted(set(log) | set(iters_with_comp))

    # Original baselines: first complete 80-user "Baseline" control row per model.
    baselines = {}
    for it in sorted(log):
        for r in log[it]:
            if r["ids"].strip() == "Baseline" and r["users"] == 80 and r["rps"] and r["model"] not in baselines:
                baselines[r["model"]] = {"reviews_per_second": r["rps"], "iteration": it, "start": r["start"]}

    entries = []
    accepted_ref = {m: baselines[m]["reviews_per_second"] for m in baselines}
    accepted_ids = {m: set() for m in MODEL_LABELS}
    accepted_track = {m: "Baseline" for m in MODEL_LABELS}

    for it in all_iters:
        rows = log.get(it, [])
        comps = (iters_with_comp.get(it) or {}).get("comparisons", {})
        handled_models = set()
        for ckey, comp in comps.items():
            model = MODEL_KEYS.get(ckey, ckey)
            handled_models.add(model)
            track = comp.get("track", "")
            inc = comp.get("incumbent_after_iteration")
            accepted = bool(comp.get("accepted_main")) or (inc is not None and inc == track)
            if inc is None and "promising" in (comp.get("decision") or "") and it == 2:
                accepted = True  # first F01 screen became the working reference
            res = read_result(it, ckey, track)
            cand = [r for r in full_rows(rows, model) if norm_ids(r["ids"]) == track]
            row = cand[-1] if cand else None
            rps = (res or {}).get("reviews_per_second") or (row or {}).get("rps")
            started = (res or {}).get("training_started_utc") or (res or {}).get("started_utc") or (row or {}).get("start")
            ids = [t for t in track.split("-") if t]
            new_ids = [t for t in ids if not t.startswith("E") and t not in accepted_ids[model]] or ids
            first = new_ids[0] if new_ids else track
            base_id = re.match(r"F\d+", first)
            label = ", ".join(new_ids)
            tname = names.get(first) or (names.get(base_id.group(0)) if base_id else None)
            decision = classify(comp, accepted)
            rb = comp.get("replicated_blocks") or {}
            block = None
            if rb:
                block = {
                    "n": rb.get("n"),
                    "mean_percent": rb.get("mean_percent"),
                    "lower_bound_percent": rb.get("lower_bound_percent"),
                    "effects_percent": rb.get("effects_percent"),
                    "decision": rb.get("decision"),
                    "final": rb.get("final"),
                }
            prev_acc = accepted_ref.get(model)
            if accepted and rps:
                accepted_ref[model] = rps
                accepted_ids[model] = set(ids)
                accepted_track[model] = track
            base = baselines.get(model, {}).get("reviews_per_second")
            entries.append(
                {
                    "iteration": it,
                    "date": started,
                    "model": model,
                    "track": track,
                    "label": label,
                    "track_name": tname,
                    "description": (row or {}).get("what") or tname or track,
                    "reviews_per_second": rps,
                    "previous_accepted_reviews_per_second": prev_acc,
                    "vs_previous_accepted_percent": (100 * (rps / prev_acc - 1)) if rps and prev_acc else None,
                    "vs_baseline_x": (rps / base) if rps and base else None,
                    "decision": decision,
                    "decision_text": comp.get("decision"),
                    "accepted": accepted,
                    "block": block,
                    "accepted_reference_after": accepted_ref.get(model),
                    "accepted_track_after": accepted_track[model],
                    "speed_noise_band_percent": (comp.get("speed_noise_band") or {}).get("percent"),
                    "source": f"artifacts/retraining/gpu-retrain-r01/iteration-{it:04d}/comparison.json",
                }
            )
        # iterations / models without a comparison: expose the last complete log row, marked uncounted
        for model in MODEL_LABELS:
            if model in handled_models:
                continue
            fr = [r for r in full_rows(rows, model) if r["ids"].strip() != "Baseline"]
            if not fr:
                continue
            row = fr[-1]
            entries.append(
                {
                    "iteration": it,
                    "date": row["start"],
                    "model": model,
                    "track": norm_ids(row["ids"]),
                    "label": row["ids"],
                    "track_name": names.get(row["ids"].split(",")[-1].strip()),
                    "description": row["what"],
                    "reviews_per_second": row["rps"],
                    "previous_accepted_reviews_per_second": accepted_ref.get(model),
                    "vs_previous_accepted_percent": None,
                    "vs_baseline_x": row["rps"] / baselines[model]["reviews_per_second"] if model in baselines else None,
                    "decision": "no_comparison",
                    "decision_text": "No comparison.json (uncounted, failed or still running)",
                    "accepted": False,
                    "block": None,
                    "accepted_reference_after": accepted_ref.get(model),
                    "accepted_track_after": accepted_track[model],
                    "speed_noise_band_percent": None,
                    "source": f"docs/GPU_OPTIMIZATIONS_LOG.md#r01-iteration-{it:04d}",
                }
            )

    entries.sort(key=lambda e: (e["iteration"], e["model"]))
    exp = [e for e in entries if e["model"] == "expertium"]
    acc = [e for e in exp if e["accepted"] and e["reviews_per_second"]]
    current = acc[-1] if acc else None
    exp_base = baselines.get("expertium", {}).get("reviews_per_second")

    # ------------------------------------------------------------------ curated milestones
    by_it = {e["iteration"]: e for e in exp}
    MILESTONES = [
        (2, "Prefetch across users (F01)", "foundation",
         "Prepare the next user while the current one trains, so the GPU stops waiting for the CPU between users. Ordered updates and review exposure are preserved, so results are bitwise identical to the baseline."),
        (3, "More preparation workers (F16)", "host",
         "Raising Expertium batch preparation from six to eight workers with bounded queues was the first large host-side win: the model was starved by input preparation, not by the GPU."),
        (4, "Packed GPU feature encodings (F14)", "host",
         "Feature encodings for full histories are built on the GPU from packed buffers instead of by CPU workers. A tiny gain by itself, kept because it unlocks later communication savings."),
        (5, "Cheaper worker communication (F03)", "host",
         "Packed communication between preparation workers and the training process, trimming per-batch transfer overhead."),
        (12, "Sort-free exact backward kernels (F26 + F27)", "kernel",
         "The time-shift gather and hierarchy row-select backward passes no longer sort indices: custom CUDA kernels reproduce the exact same floating-point accumulation order. This is the first big kernel-level speedup with bitwise parity."),
        (20, "NumPy sample construction (F30) and early workers (F28)", "host",
         "Samples are built with NumPy in the workers and the worker pool starts during model setup, overlapping CPU preparation with GPU warm-up."),
        (22, "Exact row-tiled recurrence kernels (F31)", "kernel",
         "The RWKV recurrence kernels process rows in tiles with an identical reduction order, taking the dominant GPU cost down without changing a single bit of the results."),
        (24, "Deferred host reads (F33) and multi-SM GroupNorm backward (F32)", "host",
         "Loss, gradient-norm and finite-flag reads are delayed by one update so the host can enqueue the next step without stalling, while GroupNorm weight/bias backward is spread across multiple SMs."),
        (31, "Sync-free loss (F37)", "host",
         "NaN checks and label-count reads that forced a host-device synchronisation inside the loss were replaced with device-side logic."),
        (33, "Cheaper batch preparation (F42)", "host",
         "Columnar sample construction made worker-side batch preparation substantially cheaper, shortening both training time and the complete process."),
        (42, "Skip needless fills and padding sources (F48 + F50)", "kernel",
         "Skipping deterministic uninitialised-memory fills and padding sources in row-select backward removed thousands of tiny kernel launches. This bundle is the highest single-run result of the early campaign (see the 0042 bar)."),
        (72, "Native C++ launch path (F65b: native WKV)", "native",
         "The WKV recurrence is now launched through a native C++ dispatcher instead of Python, cutting host dispatch cost per kernel. First promotion in the replicated-block era and the start of the F65 series (WKV, GroupNorm, shift/row-select, gates). Its single-run level sits below 0042 because the machine state had drifted; the interleaved blocks, not the single run, carry the verdict."),
        (76, "Physical-core worker pinning (F71) with five workers (F73a)", "host",
         "One preparation worker per physical core, plus reserved cores for the main dispatch thread; five workers beat six. Approved by the human as a long-run option after it stacked positively in 0074."),
        (79, "Process QoS (F86) stacked with native GroupNorm (F65c)", "native",
         "Opting the main process and the pinned workers out of Windows power throttling (EcoQoS) and raising their priority, together with a native GroupNorm dispatcher. The two were promoted together; the split between them is not attributed."),
        (80, "Native shift-gather and row-select autograd (F65d)", "native",
         "Moves two more groups of Python autograd Functions into the native dispatcher while launching the very same kernels, removing Python backward nodes from the critical path."),
        (89, "Lean per-update gradient reset (F72a)", "host",
         "Gradients are cleared through cached parameter lists and train mode is set once, instead of walking the scripted module tree three times every update; the arithmetic is unchanged, so results stay bitwise identical."),
        (90, "Native glue (F65e) with WDDM queue flushes (F77a)", "native",
         "Gate, bonus and channel glue now run through the native dispatcher, and the Windows GPU command queue is flushed after the forward and the backward so the GPU starts earlier; both were small on their own and were promoted together after six interleaved blocks."),
    ]
    milestones = []
    for it, title, kind, comment in MILESTONES:
        e = by_it.get(it)
        if not e:
            continue
        milestones.append(
            {
                "iteration": it,
                "title": title,
                "kind": kind,
                "comment": comment,
                "reviews_per_second": e["reviews_per_second"],
                "gain_vs_previous_accepted_percent": e["vs_previous_accepted_percent"],
                "speedup_x": e["vs_baseline_x"],
                "date": e["date"],
                "block": e["block"],
            }
        )
    rejected = [e for e in exp if e["decision"] == "rejected" and e["iteration"] >= 48]
    stacks = [e for e in exp if e["decision"] == "stack_candidate"]
    if rejected or stacks:
        lo = min(e["iteration"] for e in rejected + stacks)
        hi = max(e["iteration"] for e in rejected + stacks if e["iteration"] <= 71) if any(e["iteration"] <= 71 for e in rejected + stacks) else lo
        milestones.append(
            {
                "iteration": 48,
                "end_iteration": 71,
                "title": "The plateau: replicated blocks replace single runs",
                "kind": "method",
                "comment": "After 0042 the easy wins were gone and single-run noise (about 7 percent between identical runs) exceeded most gains. From 0048 candidates are judged in interleaved replicated blocks with a lower confidence bound; most (F34 variants, F58, F62 flat buffers and more) were rejected or stayed 'stack candidates' with gains inside the noise.",
                "reviews_per_second": None,
                "gain_vs_previous_accepted_percent": None,
                "speedup_x": None,
                "date": by_it.get(48, {}).get("date"),
                "block": None,
            }
        )
    milestones.sort(key=lambda m: m["iteration"])

    data = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign": "R01 GPU training-speed optimization",
        "cohort": "80 training users, 7,654,975 training reviews (fixed cohort), reviews/s",
        "baselines": baselines,
        "models": MODEL_LABELS,
        "headline": {
            "model": "expertium",
            "accepted_reviews_per_second": current["reviews_per_second"] if current else None,
            "accepted_iteration": current["iteration"] if current else None,
            "accepted_track": current["track"] if current else None,
            "baseline_reviews_per_second": exp_base,
            "speedup_x": (current["reviews_per_second"] / exp_base) if current and exp_base else None,
            "iterations_logged": len(log),
            "iterations_counted": len(iters_with_comp),
            "latest_iteration": max(all_iters) if all_iters else None,
            "best_single_run": max(
                ({"iteration": e["iteration"], "track": e["track"], "reviews_per_second": e["reviews_per_second"]} for e in exp if e["reviews_per_second"]),
                key=lambda d: d["reviews_per_second"],
                default=None,
            ),
        },
        "decision_labels": {
            "accepted": "Accepted / promoted",
            "stack_candidate": "Stack candidate",
            "extend": "Extend (more blocks)",
            "rejected": "Rejected / not faster",
            "investigate": "Investigate retuning",
            "fastmode": "Opt-in fastmode",
            "screen": "Preliminary screen",
            "no_comparison": "No comparison (uncounted)",
            "other": "Other",
        },
        "iterations": entries,
        "milestones": milestones,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = unname(data)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} entries, {len(milestones)} milestones, "
          f"latest iteration {data['headline']['latest_iteration']}, "
          f"accepted {data['headline']['accepted_reviews_per_second']}")


if __name__ == "__main__":
    main()
