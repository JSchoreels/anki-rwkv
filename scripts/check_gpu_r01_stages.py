"""Check actual nested cohort drafts and invalid-input guards without training."""

from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.prepare_gpu_r01_stages import stages


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Preserve earlier checks; use a new artifact JSON")
    record = {"status": "running", "provenance": process_provenance(), "checks": [],
              "scope": "actual cohort draft integrity and invalid-input guards; no training or quality measurement"}
    try:
        path = (ROOT / args.draft).resolve()
        payload = path.read_bytes()
        draft = json.loads(payload)
        protocol_bytes = (ROOT / draft["source_protocol"]).read_bytes()
        protocol = json.loads(protocol_bytes)
        record.update(draft=path.relative_to(ROOT).as_posix(), draft_sha256=hashlib.sha256(payload).hexdigest())
        assert draft["source_protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
        record["checks"].append("draft refers to exact frozen source protocol bytes")
        assert draft["stages"] == stages(protocol, draft["sampling_seed"])
        record["checks"].append("recorded exact cohorts and budgets reproduce from fixed seed")
        for kind in ("overlap", "duplicate", "decile", "range"):
            bad = deepcopy(protocol)
            if kind == "overlap":
                bad["independent_final_test_users"][0] = bad["rows"][0]["user_id"]
            elif kind == "duplicate":
                bad["rows"][1]["user_id"] = bad["rows"][0]["user_id"]
            elif kind == "decile":
                bad["rows"][0]["history_decile"] = 0
            else:
                bad["rows"][0]["validation_range"][0] += 1
            try:
                stages(bad, draft["sampling_seed"])
            except ValueError:
                record["checks"].append("invalid " + kind + " rejected")
            else:
                raise AssertionError("invalid " + kind + " accepted")
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "checks": len(record["checks"])}))


if __name__ == "__main__":
    main()
