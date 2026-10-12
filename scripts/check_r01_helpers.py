"""CPU-only regression checks for R01 token-saving helpers; no training."""
from argparse import ArgumentParser
from contextlib import redirect_stdout
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import r01_close_round as close, r01_preflight as pre, r01_workflow as flow
from scripts import r01_iterate as iterate
from scripts.r01_inspect import inspect
from scripts.r01_round_report import entry, round_documents, write_documents


class Helpers(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "artifacts/tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.env = patch.dict(os.environ)
        self.env.start()
        self.old_tempdir = tempfile.tempdir
        (self.root / "scripts").mkdir()
        for name in ("r01_workflow.py", "r01_preflight.py"):
            (self.root / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())

    def tearDown(self):
        tempfile.tempdir = self.old_tempdir
        self.env.stop()
        assert self.root.is_relative_to((ROOT / "artifacts/tmp").resolve())
        self.temp.cleanup()

    def fixture(self):
        base = close.R
        losses = {key: 0.3 + n * 0.01 for n, key in enumerate(__import__("scripts.r01_round_report", fromlist=["LOSSES"]).LOSSES)}
        for n, track, speed, accepted, previous in ((1, "control", 100, True, 1), (2, "A", 120, True, 1), (3, "B", 115, False, 2), (4, "C", 150, True, 2)):
            folder = self.root / base / f"iteration-{n:04d}"
            case = folder / "expertium" / track
            result = {"status": "complete", "model": "expertium", "training_seconds": 1000 / speed, "reviews_per_second": speed,
                      "process_wall_seconds": 20, "training_users": 3, "evaluation_users": 1, "review_presentations": 1000,
                      "training_curve_labels": 500, "training_instant_labels": 400, "updates": 5,
                      "evaluation": losses, "options": {"main_threads": 1, "batch_chunks": 512},
                      "config": {"SEED": 2025, "DTYPE": "bfloat16", "EPOCHS": 1, "CHUNK_REVIEWS": 128},
                      "provenance": {"git_commit": "2" * 40, "git_dirty": True, "source_archive": "artifacts/source.zip"}}
            flow.write_json(case / "result.json", result)
            flow.write_text(case / "updates.jsonl", json.dumps({"queue_wait_seconds": 1, "preparation_seconds": 20}) + "\n")
            flow.write_text(folder / "summary.md", "# Fixture case\n")
            flow.write_json(folder / "source-isolation.json", {"origin_git_commit": "1" * 40})
            prev_track = {1: "control", 2: "A"}[previous]
            prev_path = (base / f"iteration-{previous:04d}/expertium/{prev_track}/result.json").as_posix()
            reference = (base / "iteration-0001/expertium/control/result.json").as_posix()
            accepted_path = (case / "result.json").relative_to(self.root).as_posix() if accepted else prev_path
            comparison = {"track": track, "accepted_main": accepted, "decision": "retain main improvement" if accepted else "investigate retuning freed resources",
                          "references": {"previous": {"path": prev_path, "throughput_gain_percent": 100 * (speed / {1: 100, 2: 120}[previous] - 1)}, "original_baseline": {"path": reference}},
                          "parity": {"fields": {"weights": True, "evaluation": True}, "equal_update_counts": True, "update_mismatch_count": 0, "exact_recorded_parity": True},
                          "accepted_reference": {"path": accepted_path}, "no_20_percent_gain_streak": 0}
            flow.write_json(folder / "comparison.json", {"comparisons": {"expertium": comparison}})
            plan = base / f"plans/iteration-{n:04d}-unit.json"
            flow.write_json(self.root / plan, {"case_explanations": {"expertium": {track: "Cheaper preparation — unchanged arithmetic."}},
                                               "case_options_by_model": {"expertium": {track: result["options"]}}, "prerequisite_checks": []})
            flow.write_json(folder / "protocol.json", {"plan_path": plan.as_posix(), "validation_reviews": 200, "training_user_count": 3, "evaluation_users": 1,
                                                       "master_dtype": "float32", "cohort_sha256": "fixture-cohort", "round_authorization": "Human requested three fixture iterations"})
            flow.write_json(folder / "background-load.json", {"total_cpu_percent": 9, "python_gpu_processes": 0})
        flow.write_text(self.root / "artifacts/source.zip", "synthetic fixture, not an experiment archive")
        flow.write_text(self.root / base / "README.md", "# R01\n\n- **Open first:** [Old summary](old-summary.md).\n- Evidence is preserved.\n")
        return [entry(self.root, base, f"{n:04d}", "expertium") for n in (2, 3, 4)]

    def test_utf8_and_atomic_newlines(self):
        p = self.root / "artifacts/note.md"
        flow.write_text(p, "— pending\nMétadonnées\n")
        self.assertEqual(p.read_bytes(), "— pending\nMétadonnées\n".encode())
        self.assertFalse(list(p.parent.glob("*.tmp")))

    def test_preflight_reuses_without_rewriting(self):
        p = self.root / "artifacts/preflight.json"
        with patch.object(pre.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="driver, GPU")):
            first, reused = pre.preflight(self.root, p, 1)
            self.assertFalse(reused)
            original = p.read_bytes()
            _, reused = pre.preflight(self.root, p, 1)
            self.assertTrue(reused)
            self.assertEqual(original, p.read_bytes())
            with self.assertRaises(ValueError):
                pre.preflight(self.root, p, 2)
            self.assertEqual(original, p.read_bytes())
        self.assertIn("CUDA untested", first["scope"])
        self.assertTrue(Path(first["runtime"]["temporary_directory"]).is_relative_to(self.root / "artifacts"))
        self.assertTrue(Path(first["runtime"]["cupy_cache_directory"]).is_relative_to(self.root / "artifacts"))

    def test_preflight_rejects_invalid_paths_and_threads(self):
        for count in (0, -1, True, "1"):
            with self.assertRaises(ValueError):
                flow.configure_runtime(self.root, count)
        with self.assertRaises(ValueError):
            pre.preflight(self.root, self.root / "outside.json")

    def test_check_runner_rejects_training_entrypoint(self):
        argv = ["preflight", "--output", "artifacts/preflight.json", "--run", "scripts/run_gpu_r01.py"]
        with patch.object(pre, "ROOT", self.root), patch.object(pre, "preflight", return_value=({"runtime": {"temporary_directory": "fixture"}}, False)), patch.object(sys, "argv", argv), patch.object(pre, "run_logged") as run, redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            pre.main()
        run.assert_not_called()

    def test_log_resume_preserves_prior_attempt(self):
        log = self.root / "artifacts/task.log"
        first = flow.run_logged("task", [sys.executable, "-c", "print('first')"], log, root=self.root)
        second = flow.run_logged("task", [sys.executable, "-c", "print('second')"], log, root=self.root)
        self.assertNotEqual(first, second)
        self.assertIn("first", first.read_text(encoding="utf-8"))
        self.assertIn("second", second.read_text(encoding="utf-8"))

    def test_failure_is_bounded_and_not_retried(self):
        log = self.root / "artifacts/fail.log"
        with self.assertRaises(flow.StepFailure) as error:
            flow.run_logged("task", [sys.executable, "-c", "print('x'*20000); raise SystemExit(7)"], log, root=self.root)
        self.assertEqual(error.exception.returncode, 7)
        self.assertLess(len(str(error.exception)), 1000)
        self.assertGreater(log.stat().st_size, 19000)
        self.assertEqual(len(list(log.parent.glob("fail*.log"))), 1)
        self.assertIn("no automatic retry", flow.read_json(log.with_suffix(".log.run.json"))["status"])

    def test_one_internal_wait_heartbeat(self):
        process = Mock(pid=999999999, returncode=0)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 60), 0]
        process.poll.return_value = 0
        output = io.StringIO()
        with patch.object(flow.subprocess, "Popen", return_value=process) as launch, patch.object(flow.time, "monotonic", side_effect=itertools.count(0, 30)), redirect_stdout(output):
            flow.run_logged("task", ["fixture"], self.root / "artifacts/wait.log", root=self.root)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(process.wait.call_args_list[0].kwargs["timeout"], 60)

    def test_timeout_stops_owned_process_and_records_failure(self):
        process = Mock(pid=999999999, returncode=-9)
        process.poll.return_value = 0
        process.wait.return_value = -9
        log = self.root / "artifacts/timeout.log"
        with patch.object(flow.subprocess, "Popen", return_value=process), patch.object(flow.time, "monotonic", side_effect=itertools.count(0, 20)), self.assertRaises(flow.StepFailure):
            flow.run_logged("probe", ["fixture"], log, root=self.root, timeout_seconds=10)
        process.kill.assert_called_once()
        self.assertIn("timed out", flow.read_json(log.with_suffix(".log.run.json"))["status"])

    def test_short_poll_and_external_logs_rejected(self):
        for path, seconds in ((self.root / "artifacts/log", 5), (self.root / "outside.log", 60)):
            with self.assertRaises(ValueError):
                flow.run_logged("task", ["fixture"], path, root=self.root, heartbeat_seconds=seconds)

    def test_inspection_is_explicit_and_capped(self):
        p = self.root / "artifacts/data.json"
        flow.write_json(p, {"nested": {"status": "passed", "payload": "x" * 20000}})
        self.assertIn("passed", inspect(p, fields=["nested.status"]))
        self.assertLess(len(inspect(p, fields=["nested.payload"], chars=100)), 200)
        with self.assertRaises(ValueError):
            inspect(p)
        flow.write_text(p.with_suffix(".jsonl"), "{}\n")
        with self.assertRaises(ValueError):
            inspect(p.with_suffix(".jsonl"), fields=["status"])

    def test_markdown_section_stops_at_peer_heading(self):
        p = self.root / "artifacts/note.md"
        flow.write_text(p, "# Main\nsecret\n## Target\nkeep\n### Child\nkeep child\n## Next\nexclude\n")
        text = inspect(p, section="## Target")
        self.assertIn("keep child", text)
        self.assertNotIn("exclude", text)
        self.assertNotIn("secret", text)
        with self.assertRaises(ValueError):
            inspect(p, section="## Missing")

    def test_json_pointer_handles_artifact_keys_with_dots_or_slashes(self):
        p = self.root / "artifacts/fields.json"
        flow.write_json(p, {"counters": {"nvidia_utilization.gpu": {"mean": 69.1}, "source/file.py": "hash"}, "rows": ["large-value"]})
        self.assertIn("69.1", inspect(p, fields=["/counters/nvidia_utilization.gpu/mean"]))
        self.assertIn("hash", inspect(p, fields=["/counters/source~1file.py"]))
        self.assertNotIn("large-value", inspect(p, fields=["rows.@keys"]))

    def test_check_evidence_rejects_source_drift(self):
        for name, cls in (("run_gpu_batch1.py", "SourceWatch"), ("run_gpu_r01.py", "R01SourceWatch")):
            flow.write_text(self.root / "scripts" / name, f"class {cls}:\n    source = 'scripts/model.py'\n")
        flow.write_text(self.root / "scripts/model.py", "version = 1\n")
        archive = self.root / "artifacts/check.zip"
        archive.parent.mkdir(parents=True)
        with ZipFile(archive, "w") as z:
            z.writestr("source-manifest.json", json.dumps({"files": flow.training_inputs(self.root)}))
        result = self.root / "artifacts/check/result.json"
        flow.write_json(result, {"status": "passed", "provenance": {"source_archive": "artifacts/check.zip"}})
        self.assertIn("source compatible", flow.check_evidence(self.root, result)["status"])
        flow.write_text(self.root / "scripts/model.py", "version = 2\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            flow.check_evidence(self.root, result)
        flow.write_json(result, {"status": "failed"})
        with self.assertRaises(ValueError):
            flow.check_evidence(self.root, result)

    def test_complete_summary_and_handoff_use_accepted_reference(self):
        entries = self.fixture()
        summary = self.root / close.R / "unit-summary.md"
        text, handoff = round_documents(self.root, close.R, entries, "3" * 40, "unit", summary)
        self.assertNotIn("TODO", text)
        self.assertIn("Cheaper preparation — unchanged arithmetic.", text)
        self.assertIn("+25.00%", text)  # C compares with A, not rejected B.
        self.assertIn("Curve label weighted", text)
        self.assertIn("Instant user averaged", text)
        self.assertEqual(handoff["accepted_iteration"], "0004")
        self.assertEqual(handoff["accepted_options"], entries[-1]["result"]["options"])
        self.assertTrue(handoff["new_authorization_required"])
        self.assertEqual(handoff["next_iteration_hint"], "0005")

    def test_missing_metrics_are_not_substituted_with_zero(self):
        entries = self.fixture()
        entries[-1]["metrics"]["preparation_seconds"] = None
        entries[-1]["protocol"].pop("validation_reviews")
        text, _ = round_documents(self.root, close.R, entries, "3" * 40, "unit", self.root / close.R / "summary.md")
        self.assertIn("— (not recorded)", text)
        self.assertNotIn("| nan |", text)

    def test_counted_handoff_keeps_remaining_authorization(self):
        entries = self.fixture()
        entries[-1]["protocol"].update(iteration=entries[-1]["iteration"], round_start_iteration="0002",
                                       round_authorization="Human requested nine iterations",
                                       round_iteration_budget=9, round_iteration_index=3)
        text, handoff = round_documents(self.root, close.R, entries, "3" * 40, "unit", self.root / close.R / "unit-summary.md")
        self.assertIn("3 of 9 iterations completed; 6 remain authorized", text)
        self.assertFalse(handoff["new_authorization_required"])
        self.assertEqual(handoff["remaining_authorized_iterations"], 6)
        entries[-1]["protocol"]["round_iteration_budget"] = 3
        text, handoff = round_documents(self.root, close.R, entries, "3" * 40, "unit", self.root / close.R / "unit-summary.md")
        self.assertIn("budget is complete", text)
        self.assertTrue(handoff["new_authorization_required"])

    def test_update_mismatch_is_not_mislabeled_exact(self):
        self.fixture()
        p = self.root / close.R / "iteration-0003/comparison.json"
        c = flow.read_json(p)
        c["comparisons"]["expertium"]["parity"]["update_mismatch_count"] = 1
        flow.write_json(p, c)
        self.assertFalse(entry(self.root, close.R, "0003", "expertium")["exact"])

    def test_incomplete_round_is_rejected(self):
        self.fixture()
        p = self.root / close.R / "iteration-0004/expertium/C/result.json"
        r = flow.read_json(p)
        r["status"] = "running"
        flow.write_json(p, r)
        with self.assertRaises(ValueError):
            entry(self.root, close.R, "0004", "expertium")

    def test_existing_summary_preserved_and_index_short(self):
        entries = self.fixture()
        summary = self.root / close.R / "unit-summary.md"
        flow.write_text(summary, "Human-written explanation\n")
        with redirect_stdout(io.StringIO()):
            handoff = write_documents(self.root, close.R, entries, "3" * 40, "unit", summary)
        self.assertEqual(summary.read_text(encoding="utf-8"), "Human-written explanation\n")
        self.assertTrue(handoff.exists())
        self.assertLess(len((self.root / close.R / "README.md").read_text(encoding="utf-8")), 500)

    def test_commit_record_preserved_and_overwrite_archived(self):
        item = self.fixture()[0]
        directory = item["case"].parents[1]
        close.record_commit(item, "3" * 40, directory)
        p = directory / "implementation-commit.json"
        before = p.read_bytes()
        self.assertFalse(close.record_commit(item, "3" * 40, directory))
        with self.assertRaises(ValueError):
            close.record_commit(item, "4" * 40, directory)
        self.assertEqual(before, p.read_bytes())
        close.record_commit(item, "4" * 40, directory, overwrite=True)
        self.assertEqual(before, (directory / f"implementation-commit-before-{'3'*40}.json").read_bytes())

    def test_preview_does_not_modify_cases_or_call_reports(self):
        self.fixture()
        cases = list((self.root / close.R).glob("iteration-*/*.json"))
        before = {p: p.read_bytes() for p in cases}
        argv = ["close", "--iterations", "2-4", "--agent", "unit", "--preview", "artifacts/preview"]
        with patch.object(close, "ROOT", self.root), patch.object(close.subprocess, "run", return_value=SimpleNamespace(stdout="3" * 40)), patch.object(close, "run_logged") as run, patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            close.main()
        run.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in cases})
        self.assertFalse(list((self.root / close.R).glob("iteration-*/implementation-commit.json")))
        self.assertTrue((self.root / "artifacts/preview/unit-round-0002-0004-summary.md").exists())

    def test_failed_report_stops_round_before_summary_or_ledger(self):
        self.fixture()
        ledger = self.root / "docs/ledger.md"
        flow.write_text(ledger, "original ledger\n")
        log = self.root / "artifacts/failure.log"
        flow.write_text(log, "report failed\n")
        argv = ["close", "--iterations", "2-4", "--agent", "unit", "--ledger-change", "fixture"]
        with patch.object(close, "ROOT", self.root), patch.object(close, "LEDGER", ledger), patch.object(close.subprocess, "run", return_value=SimpleNamespace(stdout="3" * 40)), patch.object(close, "run_logged", side_effect=flow.StepFailure("report", 1, log)), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            close.main()
        self.assertEqual(ledger.read_text(encoding="utf-8"), "original ledger\n")
        self.assertFalse((self.root / close.R / "unit-round-0002-0004-summary.md").exists())

    def test_ledger_is_idempotent_and_preserves_later_sections(self):
        entries = self.fixture()
        ledger = self.root / "docs/ledger.md"
        header = "| Change | Tracks | Implementation | Experiments | Status |"
        flow.write_text(ledger, f"# Ledger\n\n## Implementation commits\n\n{header}\n|---|---|---|---|---|\n\n## Pending\n\n— preserve me\n")
        args = (ledger, self.root, self.root / close.R / "summary.md", entries, "3" * 40, "Change — exact", "T01")
        with redirect_stdout(io.StringIO()):
            close.append_ledger(*args)
            before = ledger.read_bytes()
            close.append_ledger(*args)
        self.assertEqual(before, ledger.read_bytes())
        self.assertIn(header, ledger.read_text(encoding="utf-8"))
        self.assertTrue(ledger.read_text(encoding="utf-8").endswith("— preserve me\n"))

    def test_iteration_selection_rejects_empty_or_duplicates(self):
        self.assertEqual(close.iteration_numbers("3,2,4"), ["0002", "0003", "0004"])
        for value in ("4-2", "0", "2,2"):
            with self.assertRaises(ValueError):
                close.iteration_numbers(value)

    def test_invalidated_execution_sidecar_is_not_reusable(self):
        result = self.root / "artifacts/check/result.json"
        flow.write_json(result, {"status": "passed"})
        flow.write_json(result.with_name("result.json.execution.json"), {"status": "source changed; incompatible evidence"})
        with self.assertRaisesRegex(ValueError, "Source changed"):
            flow.check_evidence(self.root, result)

    def test_cached_cuda_probe_with_changed_log_is_rejected(self):
        log = self.root / "artifacts/probe.log"
        flow.write_text(log, "changed")
        record = {"runtime": {"temporary_directory": "fixture", "gpu_driver": "fixture-driver"}, "cuda_probe": "artifacts/probe.log", "cuda_probe_sha256": "0" * 64}
        argv = ["preflight", "--output", "artifacts/preflight.json", "--cuda"]
        with patch.object(pre, "ROOT", self.root), patch.object(pre, "preflight", return_value=(record, True)), patch.object(pre, "run_logged") as run, patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            pre.main()
        run.assert_not_called()

    def test_check_source_change_invalidates_raw_success(self):
        script = self.root / "scripts/check_fixture.py"
        flow.write_text(script, "pass\n")
        result = self.root / "artifacts/raw/result.json"
        flow.write_json(result, {"status": "passed"})
        log = self.root / "artifacts/check.log"
        flow.write_text(log, "fixture check finished\n")
        flow.write_json(log.with_suffix(".log.run.json"), {"status": "complete"})
        argv = ["preflight", "--output", "artifacts/preflight.json", "--run", "scripts/check_fixture.py", "--output", "artifacts/raw"]
        with patch.object(pre, "ROOT", self.root), patch.object(pre, "preflight", return_value=({"runtime": {"temporary_directory": "fixture"}}, False)), patch.object(pre, "run_logged", return_value=log), patch.object(pre, "training_inputs", side_effect=[{"model": "old"}, {"model": "new"}]), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            pre.main()
        self.assertEqual(flow.read_json(result)["status"], "passed")  # Preserve original evidence.
        with self.assertRaisesRegex(ValueError, "Source changed"):
            flow.check_evidence(self.root, result)

    def test_helper_suite_reuses_compatible_checks_without_running(self):
        previous = self.root / "artifacts/previous/result.json"
        output = io.StringIO()
        argv = ["check", "--output", "artifacts/new-checks"]
        with patch.dict(globals(), ROOT=self.root), patch("scripts.gpu_r01_policy.check_inputs", return_value={}), patch("scripts.gpu_r01_policy.reusable_check", return_value=previous), patch.object(flow, "configure_runtime") as configure, patch.object(sys, "argv", argv), redirect_stdout(output):
            main()
        configure.assert_not_called()
        self.assertIn("reuse checks", output.getvalue())
        self.assertFalse((self.root / "artifacts/new-checks").exists())

    def test_frozen_explanation_wins_over_mutable_plan(self):
        self.fixture()
        p = self.root / close.R / "iteration-0004/protocol.json"
        protocol = flow.read_json(p)
        protocol["case_explanations"] = {"expertium": {"C": "Frozen selection explanation"}}
        flow.write_json(p, protocol)
        item = entry(self.root, close.R, "0004", "expertium")
        self.assertEqual(item["explanation"], "Frozen selection explanation")

    def test_background_query_failure_is_not_treated_as_no_competition(self):
        with patch.object(iterate.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")), self.assertRaisesRegex(RuntimeError, "query failed"):
            iterate.load_snapshot()

    def test_background_query_has_a_bounded_timeout(self):
        with patch.object(iterate.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)) as query, self.assertRaisesRegex(RuntimeError, "timed out"):
            iterate.load_snapshot()
        self.assertEqual(query.call_args.kwargs["timeout"], 5)

    def test_report_resume_reads_actual_log_and_needs_no_runtime_probe(self):
        self.fixture()
        log = self.root / "artifacts/report-attempt-0002.log"
        flow.write_text(log, "latest report decision\n")
        output = io.StringIO()
        argv = ["iterate", "--iteration", "4", "--previous", "3", "--candidate", "C", "--agent", "unit", "--from-step", "report"]
        with patch.object(iterate, "ROOT", self.root), patch.object(iterate, "preflight") as probe, patch.object(iterate, "run", return_value=log) as run, patch.object(sys, "argv", argv), redirect_stdout(output):
            iterate.main()
        probe.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertIn("latest report decision", output.getvalue())

    def test_smoke_mismatch_still_stops_before_full_run(self):
        self.fixture()
        update = {key: 1 for key in iterate.UPDATE_KEYS}
        result = {key: "fixture" for key in iterate.RESULT_KEYS}
        reference = self.root / close.R / "iteration-0002/functional-smoke-v1/expertium/A"
        candidate = self.root / close.R / "iteration-0004/functional-smoke-v1/expertium/C"
        for path in (reference, candidate):
            flow.write_json(path / "result.json", result)
            flow.write_text(path / "updates.jsonl", json.dumps(update) + "\n")
        changed = dict(update, gradient_norm=2)
        flow.write_text(candidate / "updates.jsonl", json.dumps(changed) + "\n")
        argv = ["iterate", "--iteration", "4", "--previous", "3", "--candidate", "C", "--agent", "unit", "--from-step", "smoke", "--check-reason", "fixture change"]
        with patch.object(iterate, "ROOT", self.root), patch.object(iterate, "preflight", return_value=({}, False)), patch.object(iterate, "run") as run, patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            iterate.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], "smoke")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-reason", help="Concrete reason to repeat unchanged successful helper checks")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use an artifacts directory")
    from scripts.gpu_r01_policy import check_inputs, reusable_check
    inputs = check_inputs(ROOT, tuple("scripts/" + name for name in
                          ("check_r01_helpers.py", "r01_workflow.py", "r01_preflight.py", "r01_inspect.py", "r01_round_report.py",
                           "r01_close_round.py", "r01_iterate.py", "r01_metrics.py", "check_r01_exact_options.py",
                           "run_gpu_batch1.py", "run_gpu_r01.py")))
    previous = reusable_check(ROOT, "R01 orchestration", inputs)
    if previous and not (args.repeat_reason or "").strip():
        print(json.dumps({"action": "reuse checks", "result": previous.relative_to(ROOT).as_posix()}))
        return
    if output.exists():
        parser.error("Preserve checks; use a new artifacts directory")
    flow.configure_runtime(ROOT, 1)
    from scripts.experiment_provenance import process_provenance
    result = {"provenance": process_provenance(), "scope": "synthetic CPU workflow regressions; no training or quality validation",
              "check_suite": "R01 orchestration", "check_inputs": inputs, "repeat_reason": args.repeat_reason}
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Helpers)
    run = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    result.update(status="passed" if run.wasSuccessful() else "failed", tests=run.testsRun, failures=len(run.failures), errors=len(run.errors))
    output.mkdir(parents=True)
    flow.write_text(output / "console.log", log.getvalue())
    flow.write_json(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "tests", "failures", "errors")}))
    if not run.wasSuccessful():
        print(flow.tail(output / "console.log", lines=12, chars=1800))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
