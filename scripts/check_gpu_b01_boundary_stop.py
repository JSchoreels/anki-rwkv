"""CPU-only controller integration checks; child workloads are simulated."""

from argparse import ArgumentParser
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gpu_b01 as runner
from scripts import report_gpu_b01 as reporter
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / "artifacts") or destination.exists():
        parser.error("Use a new artifact directory; retain prior checks")
    destination.mkdir(parents=True)
    provenance = process_provenance()
    outcomes = []
    expression = ast.parse('(output / "STOP").exists()', mode="eval").body
    source = ast.parse(Path(runner.original.__file__).read_text(encoding="utf-8"))
    shared_loop = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_case")
    guards = [node for node in ast.walk(shared_loop) if isinstance(node, ast.If)
              and ast.dump(node.test) == ast.dump(expression)]
    if len(guards) != 1:
        raise AssertionError("Expected the actual shared loop's unique STOP guard")
    guard = compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])), "shared-stop-guard", "exec")

    class Watch:
        implementation_sha256 = "simulated-source"

        def __init__(self, ignored):
            pass

        def check(self):
            pass

    for name, initial_marker, during_child, expected_calls in (
        ("boundary-before-first", "STOP_AFTER_CASE", False, 0),
        ("boundary-during-child", None, True, 1),
        ("immediate-before-first", "STOP", False, 0),
        ("uninterrupted-six-cases", None, False, 6),
    ):
        fixture = destination / name
        output = fixture / "artifacts" / "suite"
        output.mkdir(parents=True)
        old_process = fixture / "artifacts" / "gpu-batch1-500" / "process.json"
        runner.original.atomic_json(old_process, {"status": "stopped"})
        runner.original.atomic_json(output / "protocol.json", {
            "protocol_status": "frozen before comparison", "selected_components": ["simulated"],
            "implementation_sha256": "simulated-source", "models": ["baseline", "expertium"],
        })
        if initial_marker:
            (output / initial_marker).write_text("CPU-only boundary fixture\n")
        milestones = []

        def child(command, **ignored):
            model = command[command.index("--model") + 1]
            track = command[command.index("--track") + 1]
            folder = output / model / track
            if during_child:
                (output / "STOP_AFTER_CASE").write_text("Requested while simulated child is active\n")
            for phase in ("training", "evaluation", "save"):
                exec(guard, {"output": output})
                milestones.append((model, track, phase))
            (folder / "master.pth").write_bytes(b"SIMULATED CHECKPOINT; NOT MODEL WEIGHTS")
            runner.original.atomic_json(folder / "result.json", {"model": model, "track": track,
                                                               "status": "complete", "synthetic_only": True})
            return SimpleNamespace(returncode=0)

        with patch.object(runner, "ROOT", fixture), patch.object(runner, "B01SourceWatch", Watch), \
                patch.object(runner, "process_provenance", return_value={"synthetic_only": True}), \
                patch.object(runner.subprocess, "run", side_effect=child) as launches, \
                patch.object(reporter, "write_report"), patch.object(reporter, "controls_stable", return_value=True), \
                patch.object(sys, "argv", ["run_gpu_b01.py", "--output", "artifacts/suite"]):
            code = runner.main()
        suite = json.loads((output / "suite.json").read_text())
        assert launches.call_count == expected_calls, name
        assert len(suite["results"]) == expected_calls, name
        assert len(milestones) == expected_calls * 3, name
        assert not (fixture / "artifacts" / "gpu-batch1.lock").exists(), name
        assert code == (0 if expected_calls == 6 else 1), name
        assert suite["status"] == ("complete" if expected_calls == 6 else "stopped"), name
        if during_child:
            assert suite["stopped_before_case"] == {"model": "baseline", "track": "control-repeat"}, name
            assert (output / "baseline" / "control" / "master.pth").exists(), name
        outcomes.append(name)

    immediate = destination / "immediate-guard"
    immediate.mkdir()
    (immediate / "STOP").write_text("Intentional immediate stop fixture\n")
    try:
        exec(guard, {"output": immediate})
    except InterruptedError:
        outcomes.append("shared immediate STOP retains its existing behavior")
    else:
        raise AssertionError("Immediate STOP stopped working")
    result = {"status": "passed", "synthetic_only": True, "checks": outcomes, "provenance": provenance,
              "scope": "Actual B01 controller with simulated children and the actual shared-loop STOP guard; no CUDA, training, quality or timing validation."}
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checks": len(outcomes), "synthetic_only": True}))


if __name__ == "__main__":
    main()
