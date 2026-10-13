# Baseline drift: machine state, not code (2026-10-13)

The human asked why the historical best (0042, 197,499 reviews/s) now runs at about 166–170k, whether its exact speed can be retested, and whether 0072 must be re-recorded; they authorized runs and profiles for this case. Answer: the gap is a host-CPU scheduling state of the machine, not a source, configuration or instrumentation regression. 0072 does not need re-recording.

## Evidence from recorded runs (no new GPU work)

[history-telemetry.tsv](history-telemetry.tsv) tabulates every 0040–0049 and 0071–0073 run with GPU clock/power, per-process CPU, summed per-update host dispatch, queue wait and worker preparation ([telemetry_table.py](telemetry_table.py), [per_core_table.py](per_core_table.py)).

- **Same source, same protocol, step change.** 0048 replicates share source tree `537f8a6b311a`, protocol `518be81d63ac` and implementation `4e7873acce4e`. r1 (control) ran 199,486 and r2 (F58) 195,602 reviews/s; r3 (control, two minutes later, 2026-10-12 23:37 UTC = 01:37 local) ran 162,567, and every later control through 0073 stayed at 162–171k. The final parameter hash `8141e9b1` is identical throughout.
- **Instrumentation is not the cause.** 0048 r1 already carried the per-update CUDA events, elapsed-time reads and expanded GPU telemetry that the earlier audit suspected (F70), and it was faster than 0042.
- **GPU state did not change at the step.** 0048 r1–r12 all ran at about 3,090–3,160 MHz SM and 174–188 W. (0046/0047 were a different, separate slow state: about 2,450 MHz and 131 W, 161–184k.)
- **The training-time gap is main-thread dispatch time.** Training seconds track summed `host_dispatch_s` plus about 4 s: 0048 r1 38.37 s / 34.50 s, r3 47.09 s / 42.55 s, 0072 r1 46.02 s / 41.28 s. Dispatch rose 23% uniformly across update-size quartiles (1.19–1.29×), the signature of a slower host thread rather than one bad phase.
- **Worker preparation slowed at the same moment.** Summed preparation went 114–121 s → 155–163 s (per-update median 1.42×), with identical page faults and I/O. Per-core means show why: before, load spread over all 16 logical CPUs (for example 72 37 49 45 35 34 61 59 …); after, three SMT sibling pairs ran at 85–94% while other physical cores idled (0048 r3: 6/7 89/88, 10/11 86/85; 0072 r1: 4/5 94/92, 14/15 81/85). Two workers per physical core explains the ~1.4× preparation time.
- **No recorded cause in the OS.** Core parking is disabled (min cores 100%), the scheme is Balanced with the Best-performance overlay, R01 code never set affinity for the accepted options (F36 `host_placement` is off), and the System/Application logs show no power, lock or display event at 01:37 local. The machine rebooted at 09:44 local on 10-13; 0071–0073 after the reboot still ran in the packed state, but today's runs below did not, so the state is variable rather than permanent. The trigger (for example user absence, background apps or scheduler heuristics) remains unproven.

## Live check, current HEAD `af07fcf`, today

Seven uncounted full-history runs of the 0042 options (E33-F48-F50) were run through the normal `r01_replicates.py` → `run_isolated_gpu_r01.py` guards inside a disposable clone of HEAD (deleted afterwards), so they never entered this workspace's `gpu-retrain-r01` records, result reuse or noise band. Modes B/C applied OS affinity from outside with [pin_workers.py](pin_workers.py) (no source or option change); [run_blocks.sh](run_blocks.sh) and [run_blocks_abc.sh](run_blocks_abc.sh) drove them. Raw results, updates, resources, launch tickets and pin logs are in [runs/](runs/); the source snapshot is `artifacts/sources/experiment-sources/c14d5271…/source.zip`. Every run produced the bitwise-identical final parameter hash `8141e9b120b6b3c3`, validation input hash `d1511d463bc3`, 167 updates and 7,654,975 reviews.

| Run | Mode | Reviews/s | Training s | Dispatch s | Queue wait s | Preparation s |
|---|---|---:|---:|---:|---:|---:|
| r1 | A default scheduling | 174,601 | 43.84 | 39.24 | 0.72 | 127.2 |
| r2 | B workers pinned 1 per physical core | 179,387 | 42.67 | 38.91 | 0.43 | 107.8 |
| r3 | A default scheduling | 174,546 | 43.86 | 38.83 | 0.83 | 150.8 |
| r4 | B workers pinned | 183,486 | 41.72 | 37.73 | 0.12 | 106.1 |
| r5 | C workers pinned + main process on CPUs 0,1,14,15 | 188,775 | 40.55 | 36.12 | 0.38 | 100.3 |
| r6 | A default scheduling | 189,706 | 40.35 | 36.49 | 0.53 | 115.5 |
| r7 | C workers + main pinned | 180,412 | 42.43 | 37.70 | 0.44 | 100.8 |

([live-telemetry.tsv](live-telemetry.tsv) has GPU clock, power and CPU columns.)

- The current code and options still reach 189.7k with default scheduling (r6, whose per-core pattern matches 0042), so today's best configuration is not intrinsically ~166k. The exact 197.5k figure was a favourable-state draw; 0048 r1 shows the same state reproduces it (199.5k).
- Pinning workers one per physical core stabilizes preparation (100–108 s versus 115–151 s) and queue wait, but throughput follows main-thread dispatch, which still varied 36.1–39.2 s within each mode. Means: A 179.6k (n=3), B 181.4k (n=2), C 184.6k (n=2). With this spread the pinning gain is unproven; it needs interleaved blocks under the replicated gate.

## 0072 decision

0072 does not need re-recording. Its four replicate runs all ran in the packed state (preparation 162–163 s, dispatch 40.3–41.3 s, SM 3,137–3,160 MHz), so control and native-WKV arms are matched; final parameters and all four losses are bitwise identical, so the accepted change is exact. The +1.81% (lower bound +0.38%) is the gain under that state; its size in another state is unmeasured. Only absolute throughput comparisons across days are invalid: never compare a new run with 197,499 or 166,338 without a contemporaneous control.

Caution for future blocks: the state can switch between consecutive runs (0048 r2→r3), so a block whose control and candidate differ sharply in summed preparation or in per-core packing should be flagged before its timing is used.

## Next steps (recorded as F71 in the ledger)

- Option-level placement: one preparation worker per physical core plus a reserved main-thread core, as an exact host option, measured in interleaved blocks with the current gate.
- Record per-core means, summed preparation and dispatch in each replicate's background-load/report, and flag mismatched-state blocks.
- Diagnose main-thread dispatch variance (36–42.5 s for identical work): per-update `time.thread_time` versus wall time to separate descheduling from a slower core, ideally with CPU core clocks.
