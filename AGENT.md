**R01 speed-loop agents, start here:** read [Token efficiency](#token-efficiency) and [R01 iteration quick reference](#r01-iteration-quick-reference) below, then [docs/R01_HOST_PATH.md](docs/R01_HOST_PATH.md), the current round order, decisions and promotion gate in [docs/R01_NEXT_ROUND_PLAN.md](docs/R01_NEXT_ROUND_PLAN.md), and the latest round summary linked from `artifacts/retraining/gpu-retrain-r01/README.md`. The rest of this file still applies; skim it once and reread only the section you are about to act on. Data quality and broader validation do not arise in R01 speed iterations unless the human asks for them.

**Parallel sessions:** when several agents share this checkout, follow [docs/R01_PARALLEL_SESSIONS.md](docs/R01_PARALLEL_SESSIONS.md) for file ownership, the `scripts/r01_queue.py` run queue (launch a GPU job once as one background job; it waits its turn without polling and the session is notified when it ends), `scripts/r01_claim.py` edit claims and the commit rules.

**Subagent rotation (human decision 2026-10-14):** never keep a subagent alive while it waits on a long job.
- **Hand off before waiting:** before any wait on a GPU queue job, training or evaluation:
  1. enqueue the job detached, so the agent has no background children and the job's completion doesn't wake it;
  2. write its outputs to files;
  3. commit a short `HANDOFF.md` with tickets, output paths and the exact follow-up commands;
  4. stop.
- **Respawn fresh:** when results land, start a fresh agent from the handoff file.
- **Why:** long-lived agents re-sent 300–685k tokens of context on every wake, while a fresh agent finished a task in about 53k.
- **Model choice:** use Sonnet at medium effort for mechanical steps (scoring, tables, runbooks, simple merges). Keep Opus for design work.
- **Queue position:** detach jobs when first enqueued; re-registering a waiting ticket loses its queue position.

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

- Keep tool output small: aim for at most 6,000 characters across a call, usually less. Use `scripts/r01_inspect.py <file> --section '## Heading'` for Markdown, `--start N --lines 40` for source, or repeated `--field dotted.path` for JSON; `@keys` lists object keys/array indices, and `/JSON/pointer` handles keys containing dots or slashes. The inspector defaults to 40 lines/6,000 characters and refuses whole JSON/JSONL dumps. Scope `rg` to known files and cap matches. Never print full process lists, `resources.jsonl`, `updates.jsonl`, profiler dumps, PTX/SASS or whole `result.json` files. Save detailed analyses under `artifacts/` and read back their short conclusions.
- Read this file once, then revisit only the relevant sections. `docs/GPU_OPTIMIZATIONS.md`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `scripts/gpu_r01_training.py` and iteration reports must be read by section or selected fields; do not repeatedly dump them while waiting.
- When you know which fields or numbers you need, extract them with a short filtered script rather than a subagent.
- Delegate exploratory searches and reading to a lightweight subagent: use Haiku in Claude Code (`model: "haiku"`, Explore type for searches), or GPT-6 Luna in Codex (`spawn_agent` with `model: "gpt-6-luna"` and `fork_turns: "none"`). Use delegation when locating and summarizing evidence is substantial enough to justify a separate agent.
- Give each subagent a self-contained brief naming relevant files or search boundaries, the exact question, and the required output. Request concise findings with file paths and line numbers. Keep design, kernel/exactness reasoning, experiment decisions, and verification of consequential findings in the main agent.
- Before reading model or training code to pick R01 candidates, read [docs/R01_HOST_PATH.md](docs/R01_HOST_PATH.md) (structure, launch/sync counts, machine and noise facts) and update it when you learn something new.
- Run `scripts/r01_preflight.py --output artifacts/diagnostics/<round>/preflight.json --main-threads 1` once before diagnostics; it sets workspace-owned TMP/TEMP and CuPy cache, thread environment and UTF-8 before Torch/CuPy imports. Add `--cuda` only when a CUDA/NVRTC setup probe is needed; it is not a training or parity check. Compatible runtime records/probes are reused; changed runtime metadata requires a new output. Repeating a failed probe requires `--repeat-reason`. Helpers never change privileges or automatically retry failures. If a CUDA check stalls, inspect its `.log.run.json` resource counters and log tail once; diagnose system-call overhead early and use the available approval mechanism when a different execution environment is required.
- Finish all relevant source edits before exactness checks. Run a focused check through `scripts/r01_preflight.py --output <same preflight.json> --run scripts/check_<name>.py <arguments>`; it records command/exit/timing, preserves prior attempt logs and rejects training-source changes during the check. Do not edit training/check code until it finishes. Reuse compatible recorded checks; add a concrete repaired-failure or changed-behaviour reason for repeats. `r01_iterate.py --check <result.json>` records successful source-compatible evidence in the plan and checks it before freezing; runtime/build compatibility and purpose still need review. A worker-only change needs batch/RNG byte checks, not another identical GPU arithmetic check. New kernels still require their synthetic and real-model checks.
- Wait inside one helper/background job. Helpers emit one progress line every 60 seconds by default (`--heartbeat-seconds` permits 30–60). In Codex, after the initial session ID, use a 50–60 second `write_stdin` wait, with an outer `functions.exec` yield long enough to avoid unnecessary nested wait resumptions; never short-poll every 1–5 seconds or resample processes repeatedly. Keep user updates within the normal communication cadence. Check completion or actionable failure, then read only the final status and a bounded failure tail. Raw stdout and process counters remain in the log/sidecar; failures need an explicit resume after repair.
- Use `r01_close_round.py` to generate complete narratives, separate loss comparisons, provenance, a compact `*-handoff.json` and the README entry. Pass repeated `--next-step` arguments for concrete pending ideas. Do not write a new ad hoc report generator each round. Existing human summaries and commit records are preserved; use `--refresh-summary` or `--overwrite` only deliberately. `--preview <new artifacts directory>` tests rendering without changing recorded cases, the run log or the ledger.
- Prefer a fresh chat at a newly authorized round boundary, carrying the compact handoff, this file and the host-path notes. Creating another chat still requires the human's request. When continuing here, use those same small records instead of rereading the full history. A handoff closes the old budget; it never authorizes more iterations or broader validation.

## R01 iteration quick reference

An explicit human iteration count takes precedence over the three-consecutive-miss stop: complete the requested count, while retaining all acceptance, exactness and quality gates. Start such a round with `--round-authorization "<quoted request>" --round-iterations N` on `r01_iterate.py` (or `run_gpu_r01.py --prepare`); subsequent protocols inherit the count and progress. Never substitute the default three-miss stop for that count, infer a new budget from a handoff, or exceed the count. Uncounted rounds retain the three-miss stop.

For future rounds, bitwise-equal improvements are promoted when their measured throughput gain strictly exceeds the frozen measured timing-noise band, with no extra 2% or 5% floor. The original 5% target was a noise proxy. New protocols use `exact_promotion_rule=timing_noise`; missing measured noise leaves promotion pending. Preserve historical fixed-target protocols and decisions when reusing or reporting their evidence.

Run from the repository root with `.venv/Scripts/python.exe`; `R=artifacts/retraining/gpu-retrain-r01`.

Shortcut: `scripts/r01_iterate.py --iteration NNNN --previous PPPP --candidate <track> --set key=<json> --explanation "..." --evidence "..." --check-reason "..." --preflight artifacts/diagnostics/<round>/preflight.json` runs steps 1–6 below (control = previous accepted reference), compares the smoke bitwise, records background load and prints compact step/60-second progress lines. Optional repeated `--check <result.json>` arguments bind reusable evidence when creating the plan. Run it as one background job and resume failures with `--from-step`; prior console logs are preserved as numbered attempts. Pure profile/report resumes need no runtime probe. `scripts/report_gpu_r01_next.py --brief` prints one decision line. Fixed-window dispatch profiles are too noisy for effects under about 10%; pick small candidates on structure and let the full run measure them.

1. Write a plan in `$R/plans/iteration-NNNN-<agent>.json`: control track = previous accepted track with its exact options, one candidate track and options, and a 1–3 sentence case explanation (it becomes the run-log "What was tried" cell; put evidence in `selection_evidence`).
2. `scripts/run_gpu_r01.py --prepare --output $R/iteration-NNNN --iteration NNNN --previous $R/iteration-PPPP --plan <plan>`; add `--round-authorization "<quoted human request>"` only at the start of a newly authorized round.
3. `scripts/select_gpu_r01_next.py --model expertium --output $R/iteration-NNNN`.
4. Smoke: `scripts/run_gpu_r01.py --case --output $R/iteration-NNNN --model expertium --track <track> --smoke --check-reason "<changed behaviour>"`, then compare its `updates.jsonl` and hashes with the previous accepted smoke.
5. Full run: the same command without `--smoke` (about 2.5 minutes), redirected to `$R/iteration-NNNN-expertium-console.log`.
5b. Replicate blocks (rule `replicated_blocks`): `scripts/r01_replicates.py run --output $R/iteration-NNNN --sequence <control>,<candidate>,<control>,<candidate> --reason "<why>"` as one background job (each run about 90 s; it waits for system CPU at most 10%). Step 6's report reads the blocks; when its decision is "run replicate blocks to the next look", run two more blocks and rerun the report. Analyse ad hoc with `scripts/r01_replicate_stats.py --runs <case dirs> --calibrated-sd 1.01` and `scripts/r01_paired_timing.py`.
6. `scripts/profile_gpu_resources.py --case $R/iteration-NNNN/expertium/<track> --protocol $R/iteration-NNNN/protocol.json --output $R/iteration-NNNN/profiles/expertium-<track>`, then `scripts/report_gpu_r01_next.py --output $R/iteration-NNNN` (writes `summary.md`, `comparison.json` and the run-log rows).
7. Commit accepted code as the human, `git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@jisko.be` (no global identity is configured; keep the agent's Co-Authored-By trailer), then run `scripts/r01_close_round.py --iterations FIRST-LAST --commit HEAD --agent <agent> --ledger-change "<change>" --ledger-tracks "<IDs>" --next-step "<pending idea>"`: it writes implementation records, reruns the reports with bounded logs, generates a complete round summary/handoff, updates the README and appends the ledger row. A failed report stops closure before a summary/ledger success is published. Existing commit identities must match; deliberate replacement preserves the earlier record. Review the generated narrative and add only material interpretation or caveats the recorded data cannot establish. Commit the ledger/log changes separately.
8. Sync the GitHub Pages dashboard (https://jschoreels.github.io/anki-rwkv/campaign.html; the site root is a hub page) whenever new results are recorded, at least at every round closure: run `scripts/build_r01_pages_data.py` (CPU-only; rewrites `pages/data/r01.json`), add a `MILESTONES` entry in that script for each newly promoted iteration (one plain-language sentence on what changed and why it helped), rerun the builder, commit `pages/data/r01.json` with the script, and push `main`. The human asked for this standing sync, so pushing for it needs no further confirmation. Pushes to `pages/**` trigger `.github/workflows/pages.yml`, which redeploys the site.

Exactness before freezing a protocol: `scripts/check_r01_exact_options.py --a <reference options> --b <candidate options> --output artifacts/diagnostics/<new dir>` compares loss and every gradient bitwise on five real batches (options as JSON or a plan path plus `--a-track/--b-track`); add kernel-level synthetic cases only for new kernels. The wrapper's `[drift]` and `[profile]` lines print queue wait, summed preparation, validation, primer, GC, worker CPU and GPU against the reference (`scripts/r01_metrics.py`); read those instead of extracting numbers by hand.

Pitfalls: new option keys must be added to the allowlist in `scripts/gpu_r01_prefetch.py` before freezing a protocol (the protocol freezes the source hash, so a missed key costs an iteration); new source files used by training belong in the `R01SourceWatch` list in `scripts/run_gpu_r01.py`; new track IDs need names in `describe()` in `scripts/report_gpu_r01_next.py`. Exact-parity kernels depend on the NVRTC/PyTorch build: rerun `scripts/check_gpu_row_tiles.py --exact` and `scripts/check_gpu_exact_group_norm.py` after toolkit, driver or PyTorch changes. Close other GPU/CPU-heavy apps before timing runs; background load shifts throughput by several percent.
