# R01 0057–0059 preparation

Inherited human authorization covers 23 candidates 0048–0070. Start 0057/previous 0056 at index 9; this worker runs exactly 0057–0059. Accepted reference 0042 E33-F48-F50 and cumulative loss ceiling 0.001 remain frozen. F58 alone is retained unproven; omit all rejected slices. Alex/broader validation paused.

F58a recomputation pipeline preloads next scalar k/w/a/kd/skip and each row v before the current recomputation shuffle tree. F58b preloads each next per-row gradient before current backward barriers. F58c preloads next forward r/k/w/a/kd/skip before current forward shuffle trees. None changes FP intrinsics, shuffle/reduction order, checkpoints, skips or stores. All are independent opt-in modes under recurrence_load_pipeline=1/2/3, with inherited recurrence_prefetch=true.

Source edits finished before checks. Initial real-1 attempt failed on an accidental three-argument dict.get in checker integration; repaired before freezing. The failed output folder and attempt logs are preserved. A second launch refused that existing folder, so repaired checks use real-1-final. Synthetic-1 passed all 216 BF16/FP32 cases; remaining checks run sequentially on final source. Standalone timing now reuses the same at-most-10% quiet-machine gate as replicas.

All final checks pass: synthetic-1/2/3 each 216 BF16/FP32 skip/edge cases; real-1-final/2-final/3-final each five real losses/every-gradient bitwise equal to 0042. Runtime remains PyTorch 2.9.1+cu128 with existing CUDA build. Existing source watches cover kernel/backend. Each counted plan binds compatible synthetic and real evidence. Modes are alternatives rather than a combined bundle.

0057: initial two-block mean +0.149700%, lower -1.258430%; extended per frozen rule. Four-block mean -0.236315%, lower -1.230231%, reject; all eight replicas exact. Paired diagnostic best-of wall -0.376041%, preserved in 0057-paired.json. State flags show no retries/OOMs, same 3157 MHz median SM clocks and 12,188 MiB peak reserved; no concrete anomaly justifies repetition.

0058: smoke/full exact; report encountered transient Windows Errno 22 opening docs/GPU_OPTIMIZATIONS_LOG.md. Report-only resume passed; no training repeat. Orchestration artifact initially used wrong plan field control_track_by_model before any replicas, repaired to control_tracks. Failed helper/report attempts remain preserved.

0058 two-block mean -0.423905%, lower -1.823970%, reject; all four replicas exact. Paired wall -0.475467%; no retries/OOMs and matching 3157 MHz clocks. 0059 smoke/full exact; the same Errno 22 run-log failure was repaired through a report-only resume, preserving failed logs and avoiding training repeats.
