"""Open a numbered R01 step record before running its bounded search round."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, required=True)
    args = parser.parse_args()
    if args.number < 1:
        parser.error("Iteration numbers are positive")
    folder = ROOT / "artifacts/retraining/gpu-retrain-r01/iterations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"iteration-{args.number:04d}.md"
    if path.exists():
        raise FileExistsError("Resume and amend the existing record; never erase or reuse an iteration")
    provenance = process_provenance()
    def link(relative_path):
        return Path(os.path.relpath(ROOT / relative_path, folder)).as_posix()

    record = {"opened_utc": datetime.now(timezone.utc).isoformat(), "record_initialization_provenance": provenance,
              "status": "pending B01 acceptance; no R01 candidate experiments started"}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    content = f"""# R01 iteration {args.number:04d}

Opened UTC: {record['opened_utc']}. Status: **pending B01 acceptance**.
Human objective: continue finding improvements using R01. This record opens
before the round; no R01 candidate experiment or confirmation has started.
Save each step's outcomes here before moving to the next step. Retain failed,
inconclusive and superseded decisions as dated amendments. Resume this record
with resumed-run provenance; do not reuse its number for a different round.

Architectures: **Default** and **Expertium**, reported separately.
Frozen quality reference: **— (B01 has not passed the 500-user comparison)**.
Starting speed incumbent: **— (pending accepted B01)**.
Reference/candidate configuration, confirmed round budget and paired seeds:
**— (pending steps 1–2)**. The existing B01 scope remains 500 fixed known users,
26,705,280 training reviews and 6,716,892 held-out reviews/model, seed 2025,
chronological splits, BF16/FP32 and reserved 100 users unscored. No increased
user/seed/trial budget is inferred from opening this record.

[B01 frozen protocol]({link('artifacts/benchmarks/gpu-baseline-b01/protocol.json')}),
[B01 measurements and percentage deltas]({link('artifacts/benchmarks/gpu-baseline-b01/summary.md')}),
[R01 loop definition]({link('docs/GPU_RETRAIN_LOOP.md')}),
[persistent R01 ledger]({link('docs/GPU_OPTIMIZATIONS.md')}).
Record initialization provenance (not a training run): commit
`{provenance['git_commit']}`, dirty `{provenance['git_dirty']}`,
[source snapshot]({link(provenance['source_archive'])}).
The [adjacent JSON](iteration-{args.number:04d}.json) preserves the full metadata.

## 1. Reference and protocol

Status: **pending B01 acceptance**.

| Architecture | Frozen four-score quality reference | Starting incumbent | Protocol/budget decision |
|---|---|---|---|
| Default | — | — | pending B01 |
| Expertium | — | — | pending B01 |

Interpretation: B01's existing absolute 0.001 non-inferiority and repeated-control
gates apply first. Later candidates must satisfy all four total **<1% relative**
regression bounds against the fixed accepted reference; the allowance never
resets against successive speed winners. Exact round comparisons, cohorts,
review/update budgets, checkpoint cadence, seeds, confidence/multiplicity and
stopping budget must be recorded before execution. Decision: wait for B01.

## 2. Resource profile and bounded shortlist

Status: **pending step 1**. Default measurements: **—**. Expertium measurements: **—**.
Record representative full-history workload sufficiency, startup/warmup,
steady training, user transitions and evaluation. Link raw per-core/worker CPU,
GPU compute/memory activity, RAM/headroom, allocated/reserved/peak VRAM,
preparation/transfer waits, I/O/paging and sampler overhead. Mark unavailable
counters explicitly. Exact roughly 3–6 candidate configurations: **—**.
Interpretation/decision: pending profiling; target measured costs and retain
headroom, with complete time to quality as the objective.

## 3. Functional checks and development screening

Status: **pending step 2**. Default outcomes: **—**. Expertium outcomes: **—**.
For every candidate record configuration, provenance, input/gradient/RNG checks,
coverage and checkpoint learning curve, metrics, interpretation, decision and
raw evidence links. Retain unsuccessful candidates. Use a fixed stratified
100-user screen, then a nested 200-user stage for promising or unresolved
cases. Shrink and record training and evaluation workload together; compare
with the reference trained at the same stage budget. Both small stages remain
preliminary. Reserve 500-user confirmation for shortlisted winners and record
bounded escalation/rerun decisions and achieved experiments/hour. A few early
updates do not establish a learning curve's final rank.

## 4. Confirmation losses and confidence bounds

Status: **pending step 3 and the predeclared confirmation protocol**.
Default results: **—**. Expertium results: **—**.
Record at least 500 representative distinct evaluation users, paired seeds,
all four binary losses and relative-regression one-sided upper bounds, control
stability, seed variability and history-size groups. Proposed extra paired
seeds/trial budgets remain unrun. Interpretation/decision: pending; all four
bounds must remain below 1% against the frozen quality reference.

## 5. Complete time-to-quality ranking

Status: **pending qualifying confirmation**. Default ranking: **—**.
Expertium ranking: **—**.
Use actual saved checkpoints, with startup/preparation/transfers/capture,
required validation and saves included. Separate cold/reused-cache costs,
unique/repeated review exposure, updates, user coverage, and any prior
warm-start cost. Preserve the quality/time frontier and raw checkpoint links.
Interpretation/decision: pending; no interpolation or projected time-to-target.

## 6. Promotion, rejection or stop

Status: **pending step 5**. Default decision: **—**. Expertium decision: **—**.
Promote only a repeatably faster qualifying candidate, aiming for a meaningful
gain such as at least 5% beyond timing noise. Retain the previous incumbent and
failed/inconclusive attempts. Record the next action or predeclared budget/no-gain
stop condition. Reserved final-test data remain outside tuning.

## Dated amendments and resumed runs

- {record['opened_utc']}: record initialized before the first R01 round. B01 is
  the immediate prerequisite; all R01 measurements and decisions are pending.
"""
    # This template has no fenced code or intentional hard breaks. Preserve its
    # headings/tables; keep prose and list-item blocks on one physical line.
    content = "\n\n".join(block if block.startswith(("#", "|")) else " ".join(line.strip() for line in block.splitlines())
                          for block in content.rstrip().split("\n\n")) + "\n"
    path.write_text(content, encoding="utf-8")
    print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
