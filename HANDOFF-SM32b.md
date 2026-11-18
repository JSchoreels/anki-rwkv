# HANDOFF SM32b: state-VQ QAT in the final decay (2026-10-17)

Design, checks and cost: docs/SMALL_MODEL.md "State quantization, QAT in the final decay (SM32b)" (including "Speed", S09). Nothing is launched or enqueued.

## State

- Code on branch `state-vq` (this worktree). S09 speed-up in `6b7231e`. The run uses recipe `vq378x412p` (the same codec as `vq378x412`, with the rank-1 direction from Gram squaring instead of `eigh`; the same picks on all 90,419 corpus entities; scorer metrics unchanged on 6 users), `factor_dtype` float32, and `impl` cuda (fused forward kernel `scripts/kernels/state_vq_wkv.cu`, backward = the K=16 WKV backward kernel at checkpoint 1).
- CPU check ALL OK: `artifacts/small-model/state-vq/qat/check.json` (log `check.log`; the runbook needs it, and it must be for `vq378x412p`). Loop shapes: `qat/timing.json` (32k) and `qat/timing-2048.json` (2,048 x 16); both are needed by `bench` and `cuda-codec`.
- Local inputs that are not committed: `artifacts/small-model/state-vq/codebooks.npz` (sha256 `eab6db585da4f2944b0abf26788a93da49bb81a8b8a44ba2f62819e11a73aee0`), the state corpus `state-vq/corpus/` (used by the check only).
- Reads only: epoch 10 of `C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont`, dec10 `.../small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay`, the KD cache `anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025/batches-458c9a61e4c9`.

## Step 1: GPU micro-benchmark (user; short ticket, no training)

```bash
WT=/c/Users/jonat/workspace/wt/state-vq; PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
"$PY" C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model --timing --claim gpu --expected-minutes 10 \
  --purpose "small-model: S09 state-VQ QAT micro-benchmark (kernel correctness + per-update timing, no training)" \
  -- "C:/Program Files/Git/usr/bin/bash.exe" -c "cd $WT && '$PY' -X utf8 scripts/check_state_vq_qat.py bench > artifacts/small-model/state-vq/qat/bench.log 2>&1"
```

Expected about 4-6 min of GPU time (queued as 10 min). The command writes `qat/bench.json` and `qat/bench.log`. The log ends with a table that has one row per chunking (32k, 2048) and per path (plain, old, old+kbwd, torch-p, torch-p+kbwd, cuda), with columns ms/update, added ms, and projected ticket minutes. Exit 1 means the kernel correctness failed; the per-case numbers are in `bench.json` `correctness`.

Then:
- Correctness OK and the 32k `cuda` row ≤ ~120 min: set `MINUTES` to that row's minutes (rounded up, plus ~15%) and launch step 2.
- Correctness failed: do not launch with `IMPL=cuda`. Look at the `hybrid` (`torch-p+kbwd`) row. The fallback is `IMPL=hybrid`, either at 32k if its row fits or at 2,048 x 16; the 2,048 runbook variant is not written yet.

## Step 2: launch (user)

```bash
# optional dry run (CPU: prepare in a scratch folder + compare with dec10, echo the ticket and the scoring); done 20:20, log runbook-2026-10-18-qat-dryrun.log
DRYRUN=1 bash $WT/artifacts/small-model/runbook-2026-10-18-qat.sh
# after the SM30 rep-flags runbook (waits for its "] done" / "ended with exit" line), then the runbook:
MINUTES=<from the bench> nohup bash $WT/artifacts/small-model/launch-after-repflags-qat.sh > $WT/artifacts/small-model/launch-after-repflags-qat.log 2>&1 &
# or directly: MINUTES=<...> nohup bash $WT/artifacts/small-model/runbook-2026-10-18-qat.sh > $WT/artifacts/small-model/runbook-2026-10-18-qat.log 2>&1 &
```

Knobs:
- `MINUTES`: queue estimate, default 110.
- `MAX_MINUTES`: default 180; the ticket's `cuda-codec` gate re-estimates from 4 sampled updates and stops before the smoke above it.
- `IMPL`: `cuda` | `hybrid` | `torch`; a non-cuda impl writes to a new folder `...-qat378x412p-<impl>`.
- `FACTOR=float64`.
- `NO_BREAK=1`. The break window is now 2026-10-17 22:00 to 02:00.

## While it runs

- Do not edit this worktree's `scripts/`, `expertium/` or `rwkv/`; the trainer's source watch stops the run.
- First minute of the ticket: `qat/cuda-codec.json` has the kernel correctness and the re-estimated minutes.
- Training output: `artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412p/expertium/E89-F65e-F77a/{result.json,updates.jsonl}`. The runbook logs reviews/s against dec10 (23,908 reviews/s, 65 min).

## After it (CPU; the runbook does this, these are the manual commands)

```bash
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/state-vq; cd $WT
QAT=$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412p/expertium/E89-F65e-F77a/master.pth
DEC10=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay/expertium/E89-F65e-F77a/master.pth
Q=$WT/artifacts/small-model/state-vq/qat; U=$WT/artifacts/small-model/state-vq/users34.json
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/qat-vq.jsonl --model qat=expertium:w64:$QAT --only qat/w8+vq378x412 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/dec10-vq.jsonl --model dec10=expertium:w64:$DEC10 --only dec10/w8+vq378x412 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/s8.jsonl --model dec10=expertium:w64:$DEC10 --model qat=expertium:w64:$QAT --only dec10/w8+s8 --only qat/w8+s8 &
wait   # ~35-50 min wall
$PY -X utf8 -m scripts.state_vq_report --jsonl artifacts/small-model/state-vq/fork.jsonl --jsonl $Q/qat-vq.jsonl --jsonl $Q/dec10-vq.jsonl --jsonl $Q/s8.jsonl \
  --labels dec10/w8+s8,dec10/w8+vq378x412,qat/w8+s8,qat/w8+vq378x412 --student dec10/w8+s8 --out artifacts/small-model/state-vq/qat/summary
```

The QAT model is scored with the deployed eigh codec `vq378x412`, which is the same codec as the PTQ row of dec10. Then:
- Put the table into docs/SMALL_MODEL.md SM32b ("Results: pending").
- Update the SM32 backlog row and pages/small-model.html (results with absolute LogLoss and eval size).
- Commit `qat/summary.md|json` and the small jsonl rows with `git add -f`, then push.
- Record the bench and run timings in docs/GPU_OPTIMIZATIONS.md S09.

## Follow-ups (not done)

- Learned catalogs, or a CPU k-means re-fit on the QAT model's states, if QAT closes less than half of the PTQ gap (+0.0012 / +0.0023 at vq378x412 on these 34 users).
- Kernel speed, if the bench shows the VQ search dominates (S09 untested ideas): codebook tiles in shared memory, or all 4 heads per block (4x fewer L2 reads); shift codec only on stored positions.
- The SM26 commit (branch sm26) also appends to the gpu_r01_prefetch allowlist and TEACHER_BLIND; merging both needs a one-line conflict resolution.
