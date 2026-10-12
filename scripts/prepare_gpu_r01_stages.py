"""Prepare a nested R01 development-cohort draft; no model or run launch.

Candidate configurations, reference acceptance, trial/seed budgets and escalation
rules must still be frozen in the iteration before this draft can be executed.
"""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def stages(protocol, sampling_seed):
    rows = protocol["rows"]
    users = [row["user_id"] for row in rows]
    reserved = protocol["independent_final_test_users"]
    if (len(rows) != 500 or len(set(users)) != 500 or any(type(user) is not int or not 1 <= user <= 5000 for user in users)
            or len(reserved) != 100 or len(set(reserved)) != 100 or set(users) & set(reserved)):
        raise ValueError("Requires the preserved distinct 500-user cohort and disjoint reserved 100")
    groups = {}
    rng = random.Random(sampling_seed)
    for decile in range(1, 11):
        group = sorted((row for row in rows if row["history_decile"] == decile), key=lambda row: row["user_id"])
        if len(group) != 50:
            raise ValueError("Requires 50 users in every original history-size decile")
        rng.shuffle(group)
        groups[decile] = group
    result, previous = {}, set()
    for count in (100, 200, 500):
        selected = sorted((deepcopy(row) for group in groups.values() for row in group[:count // 10]), key=lambda row: row["user_id"])
        ids = {row["user_id"] for row in selected}
        if len(ids) != count or not previous <= ids:
            raise ValueError("Stages must have exact distinct nested user counts")
        previous = ids
        for row in selected:
            start, end = row["train_range"]
            vstart, vend = row["validation_range"]
            if start != 0 or end != vstart or end <= 0 or vend <= vstart or end % protocol["training"]["chunk_reviews"]:
                raise ValueError("Invalid retained chronological range")
        train = sum(row["train_range"][1] for row in selected)
        validation = sum(row["validation_range"][1] - row["validation_range"][0] for row in selected)
        updates = {}
        for model in ("baseline", "expertium"):
            reviews_per_update = protocol["training"][model + "_chunks_workers"][0] * protocol["training"]["chunk_reviews"]
            updates[model] = sum((row["train_range"][1] + reviews_per_update - 1) // reviews_per_update for row in selected)
        result[str(count)] = {"training_users": count, "evaluation_users": count, "rows": selected,
                              "user_ids": sorted(ids), "users_per_history_decile": count // 10,
                              "planned_training_reviews": train, "planned_heldout_reviews": validation,
                              "planned_one_epoch_updates": updates, "quality_status": "preliminary screen" if count < 500 else "confirmation pending; no run authorization granted"}
    if result["500"]["planned_training_reviews"] != protocol["training_reviews"] or result["500"]["planned_heldout_reviews"] != protocol["validation_reviews"]:
        raise ValueError("Full-cohort budgets differ from the preserved protocol")
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/benchmarks/gpu-baseline-b01/protocol.json"))
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Preserve previous drafts; use a new artifact directory")
    source = (ROOT / args.protocol).resolve()
    payload = source.read_bytes()
    protocol = json.loads(payload)
    selected = stages(protocol, args.sampling_seed)
    output.mkdir(parents=True)
    record = {"status": "draft; no experiments launched", "prepared_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": process_provenance(), "sampling_seed": args.sampling_seed,
              "source_protocol": source.relative_to(ROOT).as_posix(), "source_protocol_sha256": hashlib.sha256(payload).hexdigest(),
              "dataset_path": protocol["dataset_path"], "dataset_revision": protocol["dataset_revision"],
              "holdout": protocol["holdout"], "proposed_training": protocol["training"], "proposed_training_seeds": protocol["seeds"],
              "independent_final_test_users": protocol["independent_final_test_users"], "stages": selected,
              "pending": ["accepted frozen reference and speed incumbent", "exact candidate configurations/count",
                          "bounded total round/run/seed budget", "escalation/rejection/rerun limits",
                          "execution/source compatibility and control repeatability", "confidence/multiplicity/checkpoint protocol"],
              "interpretation": "Shrink training and held-out users together. Preserve full retained histories within each stage, chronological masks and original batching. Stage controls must train/evaluate at their own budget; 100/200-user screens cannot validate or promote."}
    (output / "cohorts.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    lines = ["# R01 nested stage-cohort draft", "", "Status: **draft; no experiments launched**. Reference acceptance, candidate and total run budgets remain pending.", "",
             f"Sampling seed: **{args.sampling_seed}**. Cohorts are nested subsets of B01's fixed 500 users; the reserved 100 remain unscored.", "",
             "| Stage | Training / evaluation users | Users per decile | Planned training reviews | Planned held-out reviews | Planned Default / Expertium updates |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, value in selected.items():
        updates = value["planned_one_epoch_updates"]
        lines.append(f"| {key} | {key} / {key} | {value['users_per_history_decile']} | {value['planned_training_reviews']:,} | {value['planned_heldout_reviews']:,} | {updates['baseline']:,} / {updates['expertium']:,} |")
    lines += ["", "These are exact planned workload counts, not measured speed or quality. Training remains one epoch with the original per-user ranges and batching; the smaller stages change total exposure and update count.", "",
              "[Exact user IDs, ranges, dataset/protocol hash and preparation provenance](cohorts.json).", "", "Pending before execution:", "", *["- " + item + "." for item in record["pending"]], ""]
    (output / "cohorts.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": record["status"], "stages": {key: {name: value[name] for name in ("training_users", "evaluation_users", "planned_training_reviews", "planned_heldout_reviews", "planned_one_epoch_updates")} for key, value in selected.items()}}))


if __name__ == "__main__":
    main()
