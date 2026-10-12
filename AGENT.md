# Optimization records

`docs/GPU_OPTIMIZATIONS.md` is the persistent list of GPU and general training
optimization tracks, including ideas that have not been implemented or tested.

- Record newly proposed optimization ideas in that file before finishing the
  task. Give each track a stable ID, a short description, status, and next step.
- Keep pending, running, failed, inconclusive, and completed tracks visible.
  Never remove an idea because it lacks measurements or a trial failed. Update
  its existing row rather than creating duplicates.
- Leave unmeasured values as `—` and label them pending. Do not substitute zero,
  projected gains, or another experiment's results for missing measurements.
- Compare held-out Curve and Instant binary log loss separately, together with
  elapsed training seconds and raw review presentations per second. The composite
  training objective and four-rating cross-entropy are different metrics.
- Record data splits, seeds, precision, batch/chunk sizes, update counts, label
  counts, and the timing scope. Compare optimizations against their own matched
  control. Disclose changes to training exposure or update schedules when
  comparing architectures or techniques that change training behavior.
- Distinguish measured throughput from complete training time and quality from
  numerical parity. Keep startup, preparation, transfer, validation, and capture
  costs explicit; do not divide log loss by time as an accuracy metric.
- Preserve the pending-track section when generating benchmark reports. Retain
  raw JSON/logs under `artifacts/` and link the evidence from the document.
- Record the full experiment Git commit ID when the process starts, together with
  whether the working tree is dirty and a preserved source snapshot for local
  edits/untracked code. Use `scripts/experiment_provenance.py` in experiment
  runners; keep this metadata in raw results and display it in the report.
- Record implementation commit IDs separately in `docs/GPU_OPTIMIZATIONS.md`
  after committing a change, with its track/bundle and experiment artifact links.
  A later implementation commit is not the revision tested by an earlier run.
  Until committed, label the change uncommitted. Never fill a missing historical
  experiment revision with the current HEAD or infer it from a timestamp.
- Preserve provenance for reused results and for each resumed training session.
  Restart experiment processes after source changes; their initial source
  snapshot cannot establish which later edits an already running process loaded.
- Preserve existing training checkpoints and unrelated workspace changes. Other
  chats can be working in this same checkout; inspect active runs before launching
  GPU benchmarks so timing is not distorted by competing jobs.
- Launch R01 cases through the current workspace's `scripts/run_gpu_r01.py` or `scripts/run_isolated_gpu_r01.py`; these apply live LOOP guards and isolate source before training. Do not directly launch archived runtime runners, which can retain obsolete rules. Reuse compatible recorded results and checks; provide a concrete repeat reason and a changed-behaviour purpose for any new smoke.

## Human-readable documentation

- Keep `docs/GPU_OPTIMIZATIONS.md` as a concise overview with stable tracks, statuses, decisions and links. Preserve full historical notes in `docs/GPU_OPTIMIZATION_HISTORY.md`.
- Put detailed progress, diagnostic attempts and per-step evidence in the relevant run or iteration file; avoid duplicating them in overview documents.
- Keep folder READMEs short: purpose, what to open first, and where supporting evidence lives. Preserve raw results and source snapshots even when they are not routine reading.
- Write each Markdown paragraph or list item on one source line, without fixed-width hard wrapping; retain line breaks required by Markdown structure.
- For tables with a user-specified schema, use only the requested or explicitly approved columns. Suggest additional columns and obtain the user's confirmation before adding them.

## Data quality records

- Record dataset anomaly investigations, review-pattern audits, and proposed
  filtering rules in `docs/DATA_QUALITY.md`, with stable IDs, status, evidence,
  and next steps. Keep these separate from GPU and training optimization tracks.
- Preserve pending and completed investigations. Any claimed training benefit
  from filtering remains subject to the validation requirements below.

## Validation requires an explicit user request

- Before treating a set of changes as validated, run a matched multi-user quality
  and speed comparison on the user-approved number of distinct evaluation users.
  Implementation, numerical parity, finite losses, throughput benchmarks, and a
  single-user quality screen do not satisfy this validation gate.
- The human user must explicitly request this broader validation phase. A request
  to implement changes, keep an idea list, or recommend a sample size does not
  authorize launching it. Existing explicit run authorization remains valid within
  its original scope; do not silently enlarge it.
- Several changes may be tested together against a control. Record the exact
  bundle/configuration; validation of a bundle does not establish independent
  validation of every component.
- Before the approved run, record the comparison/control, evaluation user count
  and IDs or sampling rule, training/evaluation review budgets, holdout protocol,
  seeds, and acceptance criterion in `docs/GPU_OPTIMIZATIONS.md`. Keep the status
  as implemented, screened, or pending broader validation until this gate passes.
- User-agreed validation minimum: 500 distinct evaluation users sampled from
  IDs 1–5000. A 100-user run is only a preliminary regression screen and does
  not satisfy this gate. Consider 1,000 or more for numerical, architecture, or
  training-strategy changes and inconclusive results. Agreement on the sample
  size does not authorize launching a run; an explicit request is still required.
- Count distinct held-out evaluation users separately from training users. Use
  representative history sizes and paired per-user loss deltas; preserve an
  independent final test cohort. Explain whether a result concerns unseen users
  or later reviews of known users. Report both label-weighted and user-averaged
  Curve/Instant losses, uncertainty at the user level, and end-to-end timing.

## Token efficiency

- Keep tool output small. Filter shell output with `tail`, `grep`, `head -c` or JSON field selection; never print full process lists, `resources.jsonl`, `updates.jsonl`, raw profiler dumps, PTX/SASS or whole `result.json` files into the conversation. Write large analyses to a file under `artifacts/` and read back only the summary lines.
- Read large files by section (`Read` with offset/limit, or `grep -n` first). `docs/GPU_OPTIMIZATIONS.md`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `scripts/gpu_r01_training.py` and iteration reports are too large to reread whole each step.
- Delegate log reading, telemetry summaries, file searches and similar mechanical lookups to Haiku subagents (Agent tool with `model: "haiku"`, Explore type for searches), with a brief that names the files and the exact numbers or answer to return. Keep design, kernel/exactness reasoning and decisions in the main agent.
- Wait for long runs with a single `run_in_background` command or an `until …; do sleep 5; done` loop, then read only the final status line, rather than polling repeatedly.
- Prefer starting a fresh session per iteration or two over one very long session; hand over through the round summary and this file instead of rediscovering context.

## R01 iteration quick reference

Run from the repository root with `.venv/Scripts/python.exe`; `R=artifacts/retraining/gpu-retrain-r01`.

1. Write a plan in `$R/plans/iteration-NNNN-<agent>.json`: control track = previous accepted track with its exact options, one candidate track and options, and a 1–3 sentence case explanation (it becomes the run-log "What was tried" cell; put evidence in `selection_evidence`).
2. `scripts/run_gpu_r01.py --prepare --output $R/iteration-NNNN --iteration NNNN --previous $R/iteration-PPPP --plan <plan>`; add `--round-authorization "<quoted human request>"` only at the start of a newly authorized round.
3. `scripts/select_gpu_r01_next.py --model expertium --output $R/iteration-NNNN`.
4. Smoke: `scripts/run_gpu_r01.py --case --output $R/iteration-NNNN --model expertium --track <track> --smoke --check-reason "<changed behaviour>"`, then compare its `updates.jsonl` and hashes with the previous accepted smoke.
5. Full run: the same command without `--smoke` (about 2.5 minutes), redirected to `$R/iteration-NNNN-expertium-console.log`.
6. `scripts/profile_gpu_resources.py --case $R/iteration-NNNN/expertium/<track> --protocol $R/iteration-NNNN/protocol.json --output $R/iteration-NNNN/profiles/expertium-<track>`, then `scripts/report_gpu_r01_next.py --output $R/iteration-NNNN` (writes `summary.md`, `comparison.json` and the run-log rows).
7. Commit accepted code with `git -c user.name=<agent> -c user.email=<agent noreply>` (no global identity is configured), write `implementation-commit.json`, rerun the report and add the implementation-commit row to the ledger.

Pitfalls: new option keys must be added to the allowlist in `scripts/gpu_r01_prefetch.py` before freezing a protocol (the protocol freezes the source hash, so a missed key costs an iteration); new source files used by training belong in the `R01SourceWatch` list in `scripts/run_gpu_r01.py`; new track IDs need names in `describe()` in `scripts/report_gpu_r01_next.py`. Exact-parity kernels depend on the NVRTC/PyTorch build: rerun `scripts/check_gpu_row_tiles.py --exact` and `scripts/check_gpu_exact_group_norm.py` after toolkit, driver or PyTorch changes. Close other GPU/CPU-heavy apps before timing runs; background load shifts throughput by several percent.
