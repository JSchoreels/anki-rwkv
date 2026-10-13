"""F74 T3: result-store backfill is idempotent and consistent; lookups reuse only verified identical work.

The positive and negative lookups run in a temporary workspace copy of one recorded iteration, where a clearly
labelled fixture runtime manifest is added; historical records are never given a manifest they did not record.
"""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", default="0072")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    from scripts import r01_result_store as store
    from scripts.r01_evaluation_reuse import RUNTIME_FIELDS
    first, second = store.backfill(ROOT), store.backfill(ROOT)
    entries = list(store.entries(ROOT))
    inconsistent = [path for path, entry in entries
                    if [reason for reason in store.valid(ROOT, entry) if "manifest" not in reason and not reason.startswith("status")]]
    historical_reusable = sum(not store.valid(ROOT, entry) for _, entry in entries if entry["key"]["kind"] == "case")
    reusable_checks = sum(not store.valid(ROOT, entry) for _, entry in entries if entry["key"]["kind"] == "check")
    checks = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), **({"detail": detail} if detail is not None else {})})

    check("backfill indexes recorded cases and checks", first["case"] > 0 and first["check"] > 0, first)
    check("second backfill adds nothing (content-addressed, idempotent)", second["new"] == 0, second)
    check("every indexed result still has its recorded sha256", not inconsistent, len(inconsistent))
    check("no recorded case is reusable without a recorded runtime manifest (cases before F74 lack driver/NVRTC)", historical_reusable == 0, historical_reusable)
    check("passed checks with a preflight runtime record are indexed as reusable when inputs still match", reusable_checks > 0, reusable_checks)

    fixture_runtime = {key: "TEST FIXTURE ONLY" for key in RUNTIME_FIELDS}
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/tmp") as folder:
        temp = Path(folder)
        source = ROOT / store.R / f"iteration-{args.iteration}"
        target = temp / store.R / source.name
        for name in ("protocol.json",):
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target / name)
        replica = next(source.glob("replicates/r1/expertium/*/result.json"))
        case_dir = target / replica.relative_to(source).parent
        case_dir.mkdir(parents=True)
        shutil.copy2(replica.parents[2] / "protocol.json", case_dir.parents[1] / "protocol.json")
        shutil.copy2(replica.parent / "master.pth", case_dir / "master.pth")
        case = json.loads(replica.read_text(encoding="utf-8"))
        case["evaluation_context"] = {"runtime": fixture_runtime, "label": "TEST FIXTURE ONLY; not historical metadata"}
        (case_dir / "result.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        protocol_path = case_dir.parents[1] / "protocol.json"
        track = case["track"]
        store.record(temp, case_dir / "result.json")
        found, _ = store.lookup_case(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("identical full run is found and passes matching_case", len(found) == 1)
        origin, _ = store.evaluation_origin(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("evaluated run with checkpoint and manifest is an evaluation origin", origin is not None)
        found, refused = store.lookup_case(temp, protocol_path, "expertium", track, runtime={**fixture_runtime, "driver_version": "other"})
        check("other driver: different key, nothing reused", not found and not refused)
        found, refused = store.lookup_case(temp, protocol_path, "expertium", track, runtime={**fixture_runtime, "driver_version": None})
        check("incomplete current runtime manifest refuses", not found and refused)
        options = {**case["options"], "training_seed": 2026}
        found, _ = store.lookup_case(temp, protocol_path, "expertium", track, options=options, runtime=fixture_runtime)
        check("other seed: nothing reused", not found)
        found, _ = store.lookup_case(temp, protocol_path, "expertium", track, smoke=True, runtime=fixture_runtime)
        check("smoke lookup does not return a full run", not found)
        changed = deepcopy(case)
        changed["evaluation"] = {**changed["evaluation"], "curve_log_loss_label_weighted": 0.0}
        (case_dir / "result.json").write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
        found, refused = store.lookup_case(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("result edited after indexing is refused", not found and any("changed" in " ".join(row["reasons"]) for row in refused))
        (case_dir / "result.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        (case_dir / "master.pth").unlink()
        store.record(temp, case_dir / "result.json")
        origin, _ = store.evaluation_origin(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("origin without its checkpoint is not offered", origin is None)
        reused = deepcopy(case)
        reused["evaluation_reuse"] = {"status": "reused", "loss_provenance": {"origin_result": "fixture"}}
        shutil.copy2(replica.parent / "master.pth", case_dir / "master.pth")
        (case_dir / "result.json").write_text(json.dumps(reused, indent=2) + "\n", encoding="utf-8")
        store.record(temp, case_dir / "result.json")
        origin, _ = store.evaluation_origin(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("a run that itself reused scores is never an origin (no chains)", origin is None)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["implementation_sha256"] = "0" * 64
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        found, _ = store.lookup_case(temp, protocol_path, "expertium", track, runtime=fixture_runtime)
        check("other implementation sha256: nothing reused", not found)

    stats = {}
    for _, entry in entries:
        kind = stats.setdefault(entry["key"]["kind"], {"entries": 0, "keys": set()})
        kind["entries"] += 1
        kind["keys"].add(entry["key_sha256"])
    stats = {kind: {"entries": value["entries"], "distinct_keys": len(value["keys"])} for kind, value in stats.items()}
    passed = all(row["passed"] for row in checks)
    record = {"status": "passed" if passed else "failed", "checks": checks, "store": store.STORE.as_posix(), "store_stats": stats,
              "scope": "store consistency on recorded artifacts; fail-closed lookups on a labelled fixture copy; no training",
              "provenance": process_provenance()}
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "failed": [row["check"] for row in checks if not row["passed"]], "store": stats}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
