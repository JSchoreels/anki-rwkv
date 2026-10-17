# HANDOFF SM32b: state-VQ QAT in the final decay (2026-10-17)

Design, checks and cost: docs/SMALL_MODEL.md "State quantization, QAT in the final decay (SM32b)". Nothing is launched or enqueued.

## State

- Code on branch `state-vq` (this worktree). CPU check ALL OK: `artifacts/small-model/state-vq/qat/check.json` (log `check.log`, needed by the runbook). CPU timing: `qat/timing.json` (needed by the ticket's `cuda-codec` estimate).
- Local inputs that are not committed: `artifacts/small-model/state-vq/codebooks.npz` (sha256 `eab6db585da4f2944b0abf26788a93da49bb81a8b8a44ba2f62819e11a73aee0`, pinned in the runbook and the option).
- Reads only: epoch 10 of `C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont`, dec10 `.../small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay`, the KD cache `anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025/batches-458c9a61e4c9`.

## Launch (user)

```bash
WT=/c/Users/jonat/workspace/wt/state-vq
# optional dry run (CPU: prepare in a scratch folder + compare with dec10, echo the ticket and the scoring)
DRYRUN=1 bash $WT/artifacts/small-model/runbook-2026-10-18-qat.sh
# after the SM30 rep-flags runbook (waits for its "] done" / "ended with exit" line), then the runbook:
nohup bash $WT/artifacts/small-model/launch-after-repflags-qat.sh > $WT/artifacts/small-model/launch-after-repflags-qat.log 2>&1 &
# or directly: nohup bash $WT/artifacts/small-model/runbook-2026-10-18-qat.sh > $WT/artifacts/small-model/runbook-2026-10-18-qat.log 2>&1 &
```

Knobs: `MINUTES` (queue estimate, 420), `MAX_MINUTES` (the ticket stops before training if the CUDA re-estimate is above, 720), `FACTOR=float32` (speed fallback; new folder `...-qat378x412f32`), `NO_BREAK=1`.

## While it runs

- Do not edit this worktree's `scripts/`, `expertium/` or `rwkv/` (the trainer's source watch stops the run).
- First minutes of the ticket: `qat/cuda-codec.json` (CUDA vs CPU codec, CUDA per-step times, re-estimated minutes for float64 / float32). If it stopped above `MAX_MINUTES`, rerun with `FACTOR=float32` (or a higher cap).
- Training: `artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412/expertium/E89-F65e-F77a/{result.json,updates.jsonl}`; the runbook logs reviews/s vs dec10 (23,908 reviews/s, 65 min).

## After it (CPU; the runbook does this, these are the manual commands)

```bash
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/state-vq; cd $WT
QAT=$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412/expertium/E89-F65e-F77a/master.pth
DEC10=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay/expertium/E89-F65e-F77a/master.pth
Q=$WT/artifacts/small-model/state-vq/qat; U=$WT/artifacts/small-model/state-vq/users34.json
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/qat-vq.jsonl --model qat=expertium:w64:$QAT --only qat/w8+vq378x412 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/dec10-vq.jsonl --model dec10=expertium:w64:$DEC10 --only dec10/w8+vq378x412 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/s8.jsonl --model dec10=expertium:w64:$DEC10 --model qat=expertium:w64:$QAT --only dec10/w8+s8 --only qat/w8+s8 &
wait   # ~35-50 min wall (VQ variants ~33 min model time each on 34 users)
$PY -X utf8 -m scripts.state_vq_report --jsonl artifacts/small-model/state-vq/fork.jsonl --jsonl $Q/qat-vq.jsonl --jsonl $Q/dec10-vq.jsonl --jsonl $Q/s8.jsonl \
  --labels dec10/w8+s8,dec10/w8+vq378x412,qat/w8+s8,qat/w8+vq378x412 --student dec10/w8+s8 --out artifacts/small-model/state-vq/qat/summary
```

Then: put the table into docs/SMALL_MODEL.md SM32b ("Results: pending"), update the SM32 backlog row and pages/small-model.html (results with absolute LogLoss + eval size), commit `qat/summary.md|json` and the small jsonl rows with `git add -f`, push. Optional: 464-user GPU scoring of `qat/w8+s8` (float-state quality only; the GPU scorer has no codec).

## Follow-ups (not done)

- Learned catalogs (gradient into the picked centroids or EMA k-means) or a CPU k-means re-fit on the QAT model's states, if QAT closes less than half of the PTQ gap (+0.0012 / +0.0023 at vq378x412 on these 34 users).
- Faster QAT step (docs/GPU_OPTIMIZATIONS.md S09): power iteration instead of eigh, fused CUDA step, CUDA graphs.
- The SM26 commit (branch sm26) also appends to the gpu_r01_prefetch allowlist and TEACHER_BLIND; merging both needs a one-line conflict resolution.
