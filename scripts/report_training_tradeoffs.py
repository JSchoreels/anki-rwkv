"""Join paired held-out losses and measured throughput; preserve pending ideas."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import provenance_rows
DOCUMENT = ROOT / "docs/GPU_OPTIMIZATIONS.md"
QUALITY = ROOT / "artifacts/benchmarks/perf-optimizations/logloss/comparison.json"
TRACK_START = "<!-- optimization-tracks:start -->"
TRACK_END = "<!-- optimization-tracks:end -->"
QUALITY_START = "<!-- optimization-quality:start -->"
QUALITY_END = "<!-- optimization-quality:end -->"
SOURCE_START = "<!-- optimization-source:start -->"
SOURCE_END = "<!-- optimization-source:end -->"
VARIANTS = (
    "reference", "reference-repeat", "preparation-cpu", "preparation-gpu", "preparation-gpu-packed",
    "wkv-tile16", "wkv-tile8", "cast-fusion", "raw-launch", "fused-adamw",
    "cuda-graph", "combined", "combined-gpu-packed",
)


def marked_section(text, start, end):
    if start not in text and end not in text:
        return ""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one pair of section markers: {start}")
    begin, finish = text.index(start), text.index(end)
    if finish < begin:
        raise ValueError(f"Reversed section markers: {start}")
    return text[begin:finish + len(end)]


def remove_section(text, start, end):
    section = marked_section(text, start, end)
    return text.replace(section, "").rstrip() if section else text.rstrip()


def checked_metrics(item):
    """Audit the actual timed updates, rather than reuse historical speed data."""
    result = item["result"]
    durations = result["update_seconds"]
    steps = item["steps"]
    if result["measured_steps"] != steps or len(durations) != steps or not durations:
        raise ValueError("Measured update count does not match the quality protocol")
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Invalid measured update duration")
    seconds = sum(durations)
    reviews = steps * result["reviews_per_update"]
    rate = result["reviews_per_second"]
    if not math.isfinite(rate) or not math.isclose(rate, reviews / seconds, rel_tol=1e-10):
        raise ValueError("Review rate does not match measured review presentations / seconds")
    metrics = result["quality"]["metrics"]
    for key in ("curve_log_loss", "instant_log_loss"):
        if not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError(f"Invalid held-out {key}")
    if min(metrics["curve_labels"], metrics["instant_labels"]) <= 0:
        raise ValueError("Held-out evaluation requires Curve and Instant labels")
    return metrics, seconds, rate


def quality_section(report):
    lines = [
        QUALITY_START, "## Held-out log loss and training time", "",
        "Curve and Instant binary log loss are reported separately (lower is better). "
        "Elapsed measured training time and raw reviews/s describe the same runs "
        "as the validation scores. The composite training loss is not log loss.", "",
        "These one-user results are a preliminary regression screen. Marking a change "
        "or bundle as validated requires a separately requested multi-user comparison; "
        "the validation gate and agreed minimum cohort size are recorded below.", "",
    ]
    if report is None:
        lines += [
            "Status: **Pending**. The paired quality screen is being prepared/run in "
            "`scripts/check_gpu_logloss.py`; no held-out scores are inferred from the "
            "historical throughput measurements above.", "", QUALITY_END,
        ]
        return "\n".join(lines)

    seeds = report["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Quality comparison requires distinct paired seeds")
    latest = {(item["model"], item["variant"], item["seed"]): item
              for item in report["results"]}
    complete = {key: item for key, item in latest.items()
                if item.get("status") == "complete" and item.get("process_exit_code", 0) == 0}
    for item in complete.values():
        checked_metrics(item)
    lines += [
        f"Suite status: **{report['status']}**. Seeds: {', '.join(map(str, seeds))}. "
        f"Each run performs 6 training warmup updates plus **{report['steps']} measured updates**.",
        "", "User 17: train on reviews **4097–12288** and validate on later reviews "
        "**12289–20480**, with next-review targets confined to their split. The "
        "8,192-review training window is repeated. Validation uses common original "
        "features, dropout disabled, FP32 master weights and WKV tile32. "
        "This is a bounded quality regression screen.", "",
        "Times include CPU preparation waits, transfer, forward/backward, copies, "
        "gradient clipping and AdamW. They exclude file loads, process/model startup, "
        "compilation, discarded JIT primers, graph capture, training warmup and validation. Values are "
        "arithmetic means over the listed paired seeds; training seconds cover only "
        "the measured updates, while final losses include learning during warmup.",
    ]
    for model, title in (("baseline", "Default"), ("expertium", "Expertium iter36")):
        chunks = 128 if model == "baseline" else 256
        workers = 4 if model == "baseline" else 6
        span = chunks * 128
        lines += [
            "", f"### {title}", "",
            f"BF16 computation, FP32 master weights; **{chunks} chunks / {workers} workers**. "
            f"Each run measures **{report['steps'] * span:,} raw review presentations** "
            f"after **{6 * span:,} warmup presentations**. These are repeated exposures, "
            "not unique reviews.", "",
            "| Experiment | Paired seeds / status | Curve LL | Instant LL | Δ Curve | Δ Instant | Measured s | Reviews/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        variants = list(VARIANTS)
        variants += sorted({key[1] for key in latest if key[0] == model} - set(variants))
        for variant in variants:
            if variant == "combined-gpu-packed" and model == "baseline":
                continue
            if variant == "reference-repeat" and not any((model, variant, seed) in latest for seed in seeds):
                continue
            pairs = []
            for seed in seeds:
                item = complete.get((model, variant, seed))
                control = (complete.get((model, "reference-repeat", seed))
                           or complete.get((model, "reference", seed)))
                if item is None or control is None:
                    continue
                if item["validation_input_sha256"] != control["validation_input_sha256"]:
                    raise ValueError("Paired runs have different held-out inputs")
                for key in ("BATCH_CHUNKS", "CHUNK_REVIEWS", "DTYPE", "SEED"):
                    if item["config"][key] != control["config"][key]:
                        raise ValueError(f"Paired runs have different {key}")
                for key in ("initial_parameter_sha256", "training_input_sha256"):
                    if key in item and key in control and item[key] != control[key]:
                        raise ValueError(f"Paired runs have different {key}")
                if item["steps"] != control["steps"] or item["warmup_updates"] != control["warmup_updates"]:
                    raise ValueError("Paired runs have different training exposure")
                metrics, seconds, rate = checked_metrics(item)
                reference, _, _ = checked_metrics(control)
                if any(metrics[key] != reference[key] for key in ("curve_labels", "instant_labels")):
                    raise ValueError("Paired runs have different held-out label counts")
                pairs.append((metrics, reference, seconds, rate))
            failed = sum(latest.get((model, variant, seed), {}).get("status") == "failed"
                         for seed in seeds)
            status = f"{len(pairs)}/{len(seeds)}"
            if failed:
                status += f"; {failed} failed"
            elif len(pairs) < len(seeds):
                status += "; pending" if not pairs else "; partial"
            if not pairs:
                values = ["—"] * 6
            else:
                mean = statistics.mean
                curve = mean(pair[0]["curve_log_loss"] for pair in pairs)
                instant = mean(pair[0]["instant_log_loss"] for pair in pairs)
                curve_delta = mean(pair[0]["curve_log_loss"] - pair[1]["curve_log_loss"] for pair in pairs)
                instant_delta = mean(pair[0]["instant_log_loss"] - pair[1]["instant_log_loss"] for pair in pairs)
                values = [f"{curve:.6f}", f"{instant:.6f}", f"{curve_delta:+.6f}",
                          f"{instant_delta:+.6f}", f"{mean(pair[2] for pair in pairs):.2f}",
                          f"{mean(pair[3] for pair in pairs):,.0f}"]
            lines.append(f"| `{variant}` | {status} | {' | '.join(values)} |")
        controls = [(complete.get((model, "reference-repeat", seed))
                     or complete.get((model, "reference", seed))) for seed in seeds]
        controls = [item for item in controls if item is not None]
        if controls:
            counts = {(item["result"]["quality"]["metrics"]["curve_labels"],
                       item["result"]["quality"]["metrics"]["instant_labels"]) for item in controls}
            if len(counts) != 1:
                raise ValueError("Control seeds have different held-out label counts")
            curve_labels, instant_labels = counts.pop()
            used = ", ".join(f"{item['seed']}: `{item['variant']}`" for item in controls)
            lines += ["", f"Held-out labels per run: **{curve_labels:,} Curve / {instant_labels:,} Instant**.",
                      f"Controls used for loss deltas: {used}. Prefer the repeated control after JIT priming; "
                      "deltas against the first control remain preliminary until its repeat is available."]
    lines += [
        "", "Positive loss deltas mean worse validation quality than the same model's "
        "paired control. Default and Expertium have different batch sizes and twice "
        "as many review presentations per update for Expertium; their absolute scores "
        "do not establish an architecture ranking at matched training exposure. "
        "Equal-time quality and time to a target score remain future measurements (F11).",
        "", "Repeated controls, when present, are shown alongside the first control. "
        "If controls disagree, treat the corresponding quality deltas as inconclusive "
        "until setup/RNG effects are resolved; this table records observations rather "
        "than establishing unchanged quality.",
        "", "[Raw paired loss/time results](../artifacts/benchmarks/perf-optimizations/logloss/comparison.json), "
        "[per-seed regression summary](../artifacts/benchmarks/perf-optimizations/logloss/summary.md). "
        "Regenerate this section with `python scripts/report_training_tradeoffs.py` "
        "as runs finish. Missing and failed measurements remain visible.", "", QUALITY_END,
    ]
    return "\n".join(lines)


def source_section(throughput, quality):
    lines = [SOURCE_START, "## Experiment source revisions", "",
             "Commit IDs below come from saved per-run metadata. Missing historical "
             "revisions stay unrecorded. Dirty runs also require the linked source "
             "snapshot; implementation commits are recorded separately in the change ledger.", "",
             "| Suite | Runs | Cases | Experiment commit | Working tree | Source snapshot |",
             "|---|---:|---|---|---|---|"]
    for name, report in (("Throughput", throughput), ("Log loss", quality)):
        if report is not None:
            lines.extend(provenance_rows(report, name))
    lines.extend(["", SOURCE_END])
    return "\n".join(lines)


def decorate_report(base_text, previous_text, quality_path=QUALITY,
                    throughput_path=ROOT / "artifacts/benchmarks/perf-optimizations/comparison.json"):
    tracks = marked_section(previous_text, TRACK_START, TRACK_END)
    text = remove_section(base_text, TRACK_START, TRACK_END)
    text = remove_section(text, QUALITY_START, QUALITY_END)
    text = remove_section(text, SOURCE_START, SOURCE_END)
    report = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    throughput = json.loads(throughput_path.read_text(encoding="utf-8")) if throughput_path.exists() else None
    text += "\n\n" + source_section(throughput, report)
    text += "\n\n" + quality_section(report)
    if tracks:
        text += "\n\n" + tracks
    return text.rstrip() + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, default=QUALITY)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/perf-optimizations/loss-time-report.md")
    args = parser.parse_args()
    previous = DOCUMENT.read_text(encoding="utf-8")
    updated = decorate_report(previous, previous, args.quality)
    from scripts.artifact_markdown import rebase_links
    updated = rebase_links(updated, DOCUMENT.parent, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(args.output)
    print(f"Updated loss/time comparison; pending tracks retained in {args.output}")


if __name__ == "__main__":
    main()
