"""Add relative deltas without editing source consumed by the live GPU suite."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance

TRACKS = [f"F{index:02d}" for index in range(1, 9)]
BEGIN, END = "<!-- gpu-batch1-relative:start -->", "<!-- gpu-batch1-relative:end -->"


def delta(value, reference, digits=6, percent_digits=3, unit=""):
    if value is None or reference is None or not math.isfinite(value) or not math.isfinite(reference):
        return "—"
    change = value - reference
    relative = f"{100 * change / reference:+.{percent_digits}f}%" if reference else "—; control is zero"
    return f"{change:+,.{digits}f}{unit} ({relative})"


def value_with_percent(value, reference, digits=6, percent_digits=3):
    relative = (f" ({100 * (value - reference) / reference:+.{percent_digits}f}%)"
                if reference is not None and math.isfinite(reference) and reference != 0 else "")
    return f"{value:,.{digits}f}" + relative


def enrich_idea_table(text, suite, comparisons):
    """Keep the original table layout; append percentages to existing values."""
    results = {(result["model"], result["track"]): result for result in suite["results"]}
    decisions = {(value["model"], value["track"]): value for value in comparisons}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) < 9 or cells[1].strip() not in TRACKS:
            continue
        track = cells[1].strip()
        metrics, timings = [], []
        for model, label in (("baseline", "Default"), ("expertium", "Expertium")):
            result, control = results.get((model, track)), results.get((model, "control"))
            if not result or result.get("status") != "complete" or result.get("smoke_only"):
                metrics.append(f"{label} — ({result['status'] if result else 'pending'})")
                timings.append(f"{label} —")
                continue
            matched = (control and control.get("status") == "complete" and not control.get("smoke_only")
                       and decisions.get((model, track), {}).get("status") != "unmatched")
            scores = result["evaluation"]
            def pair(kind):
                values = []
                for task in ("curve", "instant"):
                    key = task + "_log_loss_" + kind
                    reference = control["evaluation"][key] if matched else None
                    values.append(value_with_percent(scores[key], reference))
                return " / ".join(values)
            metrics.append(f"{label} weighted {pair('label_weighted')}; user {pair('user_average')}")
            train = value_with_percent(result["training_seconds"], control["training_seconds"] if matched else None, 2, 2)
            speed = value_with_percent(result["reviews_per_second"], control["reviews_per_second"] if matched else None, 0, 2)
            timing = f"{label} {train} s / {speed} reviews/s"
            if result.get("cache_setup"):
                cold = value_with_percent(result["cold_training_seconds"], control["cold_training_seconds"] if matched else None, 2, 2)
                timing += f"; cold {cold} s"
            timings.append(timing)
        cells[5], cells[6] = " " + "; ".join(metrics) + " ", " " + "; ".join(timings) + " "
        lines[index] = "|".join(cells)
    return "\n".join(lines) + "\n"


def relative_rows(suite, comparisons):
    results = {(result["model"], result["track"]): result for result in suite["results"]}
    decisions = {(value["model"], value["track"]): value for value in comparisons}
    lines = ["| Model / track | Δ Curve LL weighted / user (%) | Δ Instant LL weighted / user (%) | Δ train s (%) | Δ reviews/s (%) | Δ cold train s (%) | Decision |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for model in ("baseline", "expertium"):
        control = results.get((model, "control"))
        for track in TRACKS:
            result = results.get((model, track))
            comparison = decisions.get((model, track), {})
            decision = comparison.get("decision", comparison.get("status", result.get("status", "pending") if result else "pending"))
            if suite.get("active_case") == {"model": model, "track": track} and not result:
                decision = "running"
            complete = (control and result and control["status"] == result["status"] == "complete"
                        and not control.get("smoke_only") and not result.get("smoke_only"))
            if not complete or comparison.get("status") == "unmatched":
                lines.append(f"| {model}/{track} | — | — | — | — | — | {decision} |")
                continue
            def loss(task):
                return " / ".join(delta(result["evaluation"][task + "_log_loss_" + kind],
                                        control["evaluation"][task + "_log_loss_" + kind])
                                  for kind in ("label_weighted", "user_average"))
            train = delta(result.get("training_seconds"), control.get("training_seconds"), 2, 2)
            speed = delta(result.get("reviews_per_second"), control.get("reviews_per_second"), 0, 2)
            cold = delta(result.get("cold_training_seconds"), control.get("cold_training_seconds"), 2, 2)
            lines.append(f"| {model}/{track} | {loss('curve')} | {loss('instant')} | {train} | {speed} | {cold} | {decision} |")
    return "\n".join(lines)


def document_block(suite, comparisons):
    return "\n".join([BEGIN, "### First-batch absolute and percentage changes", "",
                       "Each percentage is **100 × (candidate − matched control) / matched control**.",
                       "The reference is the first control for the same model. Positive log-loss",
                       "deltas mean worse quality; negative time deltas and positive throughput deltas",
                       "mean faster training. Throughput and elapsed-time percentages differ because",
                       "throughput is the reciprocal of time at equal review exposure. Cold time includes",
                       "F02 cache construction. Missing comparisons remain pending.", "",
                       relative_rows(suite, comparisons), "",
                       "[Live table with percentage changes](../artifacts/benchmarks/gpu-batch1-500/summary-relative.md).",
                       "[Original scores, adjusted uncertainty and experiment sources](../artifacts/benchmarks/gpu-batch1-500/summary.md).", END])


def write_reports(output, suite, comparisons, update_doc):
    block = document_block(suite, comparisons)
    lines = ["# First GPU batch: absolute and percentage changes", "",
             f"Suite: **{suite['status']}**; completed cases: **{sum(value['status'] == 'complete' for value in suite['results'])}/20**.", "",
             *block.splitlines()[3:-4], "",
             "[Absolute measurements, uncertainty and process-start source revisions](summary.md).", ""]
    (output / "summary-relative.md").write_text("\n".join(lines), encoding="utf-8")
    if update_doc:
        block = "\n".join([BEGIN, "### First-batch results", "", f"Suite: **{suite['status']}**; completed cases: **{sum(value['status'] == 'complete' for value in suite['results'])}/20**.", "", f"[Loss, timing and percentage comparisons](../{output.relative_to(ROOT).as_posix()}/summary-relative.md).", END])
        path = ROOT / "docs/GPU_OPTIMIZATIONS.md"
        existing = path.read_text(encoding="utf-8")
        if BEGIN in existing:
            start = existing.index(BEGIN)
            end = existing.index(END, start) + len(END)
            updated = existing[:start] + block + existing[end:]
        else:
            updated = existing.rstrip() + "\n\n" + block + "\n"
        updated = enrich_idea_table(updated, suite, comparisons)
        if updated != existing:
            path.write_text(updated, encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/gpu-batch1-500")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--update-doc", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a suite directory under workspace artifacts")
    provenance = process_provenance()
    state_path = output / "relative-reporter.json"
    state = {"status": "running", "pid": __import__("os").getpid(), "analysis_provenance": provenance,
             "started_utc": datetime.now(timezone.utc).isoformat(), "watch": args.watch,
             "purpose": "CPU-only percentage rendering; does not edit running training/report source"}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    last = None
    last_signature = None
    try:
        while True:
            try:
                suite_path, pair_path = output / "suite.json", output / "paired-comparisons.json"
                signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in (suite_path, pair_path))
                if args.watch and signature == last_signature and suite["status"] == "running":
                    time.sleep(3)
                    continue
                suite_text = (output / "suite.json").read_text(encoding="utf-8")
                pair_text = (output / "paired-comparisons.json").read_text(encoding="utf-8")
                suite, comparisons = json.loads(suite_text), json.loads(pair_text)
                digest = hashlib.sha256((suite_text + pair_text).encode()).hexdigest()
                last_signature = signature
            except (json.JSONDecodeError, FileNotFoundError):
                if not args.watch:
                    raise
                time.sleep(2)
                continue
            if digest != last:
                write_reports(output, suite, comparisons, args.update_doc)
                last = digest
                print(f"Percentage table refreshed: {suite['status']}, {len(suite['results'])} recorded cases", flush=True)
            if not args.watch:
                break
            # Wait for the launcher to exit too: terminal suite status is written
            # before the main reporter has finished computing all paired bounds.
            if suite["status"] != "running":
                process_path = output / "process.json"
                if process_path.exists():
                    process = json.loads(process_path.read_text(encoding="utf-8"))
                    if process.get("status") == "running":
                        import psutil
                        if psutil.pid_exists(process["pid"]):
                            time.sleep(3)
                            continue
                final_text = (output / "suite.json").read_text(encoding="utf-8")
                final_pairs = (output / "paired-comparisons.json").read_text(encoding="utf-8")
                write_reports(output, json.loads(final_text), json.loads(final_pairs), args.update_doc)
                break
            time.sleep(3)
        state["status"] = "exited"
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
