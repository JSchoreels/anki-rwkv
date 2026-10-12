"""Check report integrity without launching competing GPU training jobs."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_training_tradeoffs import (
    TRACK_START, TRACK_END, QUALITY_START, SOURCE_START, decorate_report, quality_section,
    source_section,
)


def fixture():
    control = {
        "model": "baseline", "variant": "reference", "seed": 2025,
        "status": "complete", "process_exit_code": 0, "steps": 2,
        "warmup_updates": 6, "validation_input_sha256": "common-validation",
        "initial_parameter_sha256": "common-weights",
        "config": {"BATCH_CHUNKS": 128, "CHUNK_REVIEWS": 128,
                   "DTYPE": "bfloat16", "SEED": 2025},
        "result": {
            "measured_steps": 2, "update_seconds": [0.25, 0.25],
            "reviews_per_update": 16384, "reviews_per_second": 65536.0,
            "quality": {"metrics": {"curve_log_loss": 0.6, "instant_log_loss": 0.5,
                                     "curve_labels": 10, "instant_labels": 20}},
        },
    }
    variant = copy.deepcopy(control)
    variant["variant"] = "cuda-graph"
    variant["result"]["quality"]["metrics"]["curve_log_loss"] = 0.61
    variant["result"]["update_seconds"] = [0.125, 0.125]
    variant["result"]["reviews_per_second"] = 131072.0
    return {"status": "running", "steps": 2, "seeds": [2025, 2026],
            "results": [control, variant]}


def rejected(report):
    try:
        quality_section(report)
    except ValueError:
        return
    raise AssertionError("Invalid experiment evidence was accepted")


def main():
    report = fixture()
    rendered = quality_section(report)
    assert "| `cuda-graph` | 1/2; partial | 0.610000 | 0.500000 | +0.010000 | +0.000000 | 0.25 | 131,072 |" in rendered
    assert "| `preparation-cpu` | 0/2; pending | — | — | — | — | — | — |" in rendered
    # A variant without a same-seed control must never look like a paired result.
    unpaired = copy.deepcopy(report)
    unpaired["results"] = unpaired["results"][1:]
    assert "| `cuda-graph` | 0/2; pending | —" in quality_section(unpaired)
    failed = copy.deepcopy(report)
    failed["results"][1]["status"] = "failed"
    assert "| `cuda-graph` | 0/2; 1 failed | —" in quality_section(failed)
    warmed = copy.deepcopy(report)
    repeat = copy.deepcopy(warmed["results"][0])
    repeat["variant"] = "reference-repeat"
    repeat["result"]["quality"]["metrics"]["curve_log_loss"] = 0.59
    warmed["results"].append(repeat)
    assert "| `cuda-graph` | 1/2; partial | 0.610000 | 0.500000 | +0.020000 |" in quality_section(warmed)
    assert "2025: `reference-repeat`" in quality_section(warmed)
    for key in ("validation_input_sha256", "initial_parameter_sha256"):
        invalid = copy.deepcopy(report)
        invalid["results"][1][key] = "different"
        rejected(invalid)
    invalid = copy.deepcopy(report)
    invalid["results"][1]["result"]["reviews_per_second"] = 999999
    rejected(invalid)
    invalid = copy.deepcopy(report)
    invalid["results"][1]["result"]["quality"]["metrics"]["curve_log_loss"] = float("nan")
    rejected(invalid)
    invalid = copy.deepcopy(report)
    invalid["results"][1]["result"]["quality"]["metrics"]["instant_labels"] = 99
    rejected(invalid)
    # Report regeneration retains untested ideas verbatim and is idempotent.
    ideas = (f"{TRACK_START}\nAn untested idea; keep this exact text.\n"
             "Implementation commit: recorded separately from the experiment.\n"
             f"{TRACK_END}")
    with TemporaryDirectory(dir=ROOT / "artifacts/diagnostics/check-temp") as folder:
        missing = Path(folder) / "absent.json"
        first = decorate_report("# New measured results\n", f"# Old results\n\n{ideas}", missing)
        second = decorate_report(first, first, missing)
        assert ideas in first and first == second
        assert first.count(QUALITY_START) == 1 and first.count(TRACK_START) == 1
        assert first.count(SOURCE_START) == 1
        try:
            decorate_report("# Results", TRACK_START, missing)
        except ValueError:
            pass
        else:
            raise AssertionError("An incomplete idea section would be overwritten")
    revision = "a" * 40
    report["results"][1]["provenance"] = {"git_commit": revision, "git_dirty": True}
    sources = source_section(None, report)
    assert revision in sources and "dirty" in sources and "— (not recorded)" in sources
    print("Loss/time pairing, provenance, rates, missing/failed results and persistent ideas passed.")


if __name__ == "__main__":
    main()
