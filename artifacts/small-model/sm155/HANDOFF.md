# Bounded-memory long-card scorer (SM155): handoff

The building agent stopped on 11-24 at about 00:20 (machine clock) while the exactness run was going. A fresh agent
(Sonnet, medium effort) processes the result. Branch `sm155` (worktree `C:/Users/jonat/workspace/wt/sm155`, cut from
origin/sm152). Do not edit `scripts/`, `rwkv/` or `expertium/` of this worktree while a run of it is going.

## What is built

- `scripts/stream_scorer.py`: the stream engine. `scripts/small_model_heldout.py --device cpu --engine stream
  --chunk-rows N` (default 16384, a multiple of 64) replays the history in chunks and carries only per-entity states
  (WKV matrix, update count, two token-shift vectors per stream and layer) between chunks. The model's own modules,
  codecs, heads and loss run; only the WKV reference and the shift gather are swapped during the evaluation. Rows are
  placed so the CPU kernels give every row the stock arithmetic (parity for the small GEMMs, one-entity replica blocks
  for rows near a stock block's end or thread split; details in `design.md`). Card and note states are stored as int8
  codes plus row scales or as low-rank factors, checked bit for bit at every write (float32 fallback per entity, row
  field `stream_engine`). The default engine (`parallel`) is today's scorer, unchanged.
- `scripts/state_lowrank.py`: `LowRankCodec.encode(..., return_right=True)` also returns the right factors (values
  unchanged, default off).
- Development tries (not part of the result, `design.md` "Development history"): bitwise equal on every review for Big
  two-head (SM91s2) on users 2388 and 2883 (fp32, REF, T24, tiered int8), Small two-head (SM118) and six-layer Tiny
  two-head (SM130) on user 2883 (fp32, REF, T24), and the state VQ label on Small two-head (SM118) user 2388.
- Memory probe (preparation only, no model): the largest Big user 3735 (531,091 reviews, 1,029,959 rows, 32,223
  cards, 22,544 notes) peaks at 2.66 GB working set for the batch and the row layout (`prep_probe.py`, normal inputs).
  Estimated state tables for Big at REF: about 17 KB per card and 11 KB per note, so about 0.8 GB for user 3735 and
  about 2 GB for user 4852 (77,865 cards); a stream process should stay near 3.5 to 4.5 GB on the largest users.

## Exactness rule (pre-registered in `design.md`)

PASS if and only if every (model, user, label) cell has complete rows from both engines, the same predicted reviews
per head, max |p_stream - p_stock| <= 1e-5 over every predicted review of both heads (`--dump-scope all`), and the
benchmark metrics equal within 1e-6. No averaging; one failing cell fails the rule.

## The running exactness run

Launched hidden via WMI at 00:16:50 Tue 11-24 (WMI PID 39132): `run-sm155-tests.sh`, 2 lanes = 2 CPU processes,
memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`).

- Lane a: Big two-head (SM91s2) and Big Instant-Only (SM99), main labels (`w8`, REF `w8+s8hx8`, T24
  `w8+lr2.16.4q8x8`, Q2 `w8+lr2q8x8`) on the 3 smoke users and on 5 long-card users (2883 3882 1215 280 4407), chunk
  rows 2048; then user 4407 at chunk rows 16384 (REF, T24); then the memory probe: user 1458 (92,307 reviews), Big
  two-head (SM91s2) REF, stock vs stream under `artifacts/small-model/sm136/peakrun.py`.
- Lane b: the same main matrix for Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130),
  six-layer Tiny Instant-Only (SM130); then the format families on the smoke users for Big two-head (SM91s2) and Small
  two-head (SM118), and the state VQ label on Small two-head (SM118).
- Expected end: about 01:15 to 01:45 (about 60 to 90 min in all).
- Outputs: `work/sm155/<cell>/{parallel,stream}.jsonl`, dumps `dump-*`, logs `*.out` (work/ is never committed);
  `cpu/run.log` one line per engine per cell.
- Done patterns in `cpu/run.log`: last line `done` (file `cpu/done`) or `done with failures` (file `cpu/failed`; see
  `cpu/cell-failures.txt` and `cpu/memguard-kill`); the line before is `compare: N/M pass; worst max|dp| X`. A cell
  line `... exit 1` names a failed engine run (its `work/sm155/<cell>/<engine>.out` has the traceback).
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like
  '*run-sm155-tests*' } | Select ProcessId`. The run is resumable: cells with `<engine>.ok` are skipped, so after a
  power event relaunch with the same command:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/run-sm155-tests.sh >> C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/cpu/run.nohup 2>&1"'; ProcessStartupInformation=$si}
```

## How to compare

The run ends by writing `sm155-exactness.json` and `sm155-exactness.txt`. To redo it by hand:

```bash
cd C:/Users/jonat/workspace/wt/sm155
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
$PY -X utf8 artifacts/small-model/sm155/sm155_compare.py $(ls -d work/sm155/main-* work/sm155/fam-* work/sm155/chunk16k-* work/sm155/mem-*) \
  --json artifacts/small-model/sm155/sm155-exactness.json > artifacts/small-model/sm155/sm155-exactness.txt
tail -1 artifacts/small-model/sm155/sm155-exactness.txt
grep -a PEAK work/sm155/mem-1458/*.out      # peak working set, stock vs stream
```

Each line gives user, label, reviews, max |dp| per head, the share of bitwise-equal reviews, the benchmark difference,
stock / stream model seconds and PASS / FAIL.

## What is left (fresh agent, Sonnet medium is enough)

1. When `cpu/done` or `cpu/failed` exists: read `sm155-exactness.txt` (and `cpu/cell-failures.txt` if any).
2. Write `RESULT.md` (short): exactness per model (cells passing, worst max |dp|, bitwise share), the format families,
   the default chunk size check; memory: peak working set stock vs stream on user 1458, the preparation probe above and
   the state-table estimate; speed: reviews/s per model, stock vs stream, from the rows' `model_seconds` (both
   engines ran 2 processes side by side); the verdict of the pre-registered rule. Naming: every model as "plain name
   (ID)" in every sentence, "%" never "points", never the outside researcher's name ("outside research").
3. Commit on `sm155`: `git add -f artifacts/small-model/sm155` (never `work/`, `.npz`, `.pth`, `quiet.json`,
   `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), `git -c user.name="Jonathan Schoreels" -c
   user.email="jonathan.schoreels@gmail.com" commit` with a message ending in a blank line and the Co-Authored-By line
   of the committing model; `git fetch origin && git merge origin/sm155`, then `git push origin sm155`. Never
   force-push, never bypass hooks.
4. If the rule fails: do not start the first use; report the failing cells to the main session (likely causes: a
   kernel path not covered by the placement rules; the dumps of the failing cell show which reviews differ).

## First use (decided by the user, NOT started)

Start it only when the main session says the CPU is free (a CPU speed measurement for the Anki fork and the GPU job of
smaller shift vectors and factors on T24 states (SM152) were running on 11-24) and only after the exactness rule passed.

`launch-sm155-firstuse.sh`: both Big models (Big two-head (SM91s2), Big Instant-Only (SM99)) x {T24, REF}, `--engine
stream`, chunk rows 16384, the SM148 long-card flags (`--inputs e2s-query --split-existing --deleted-cards neighbour
--nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `--dump-scope all`), first the 13 Big long-card users
missing from the low-rank int8 states test (SM148) (87k to 151k reviews, 1.55M in all), then the 5 users that never fit
on the GPU (296 2073 2267 3735 4852; 424k to 531k, 2.38M in all). It first runs a bitwise gate (user 2388, Big two-head
(SM91s2), REF and T24, stock vs stream in its own environment) and scores nothing if the gate is not bitwise equal.
2 processes at a time, memory guard 5.9 GB, quiet.json windows (scope all / cpu) respected, resumable, stop between
users with `touch artifacts/small-model/sm155/firstuse/STOP`. Rows: `firstuse/big/u-<user>.jsonl`, merged into
`firstuse/sm155-big-longcard-rows.jsonl`; dumps in `work/sm155-firstuse-dump/`; log `firstuse/launch-firstuse.log`;
end files `firstuse/done`, `firstuse/failed` or `firstuse/stopped`.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; mkdir -p C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/firstuse; bash C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/launch-sm155-firstuse.sh >> C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/firstuse/launch-firstuse.nohup 2>&1"'; ProcessStartupInformation=$si}
```

ETA: 18 users x 2 models x 2 labels = about 15.7M review-evaluations. Big under the stock CPU scorer ran at about
900 reviews/s per process (SM148); the stream engine ran at about 0.75 to 1.0 times that in the development tries, so
about 700 reviews/s per process, 2 processes: about 3 h (range 2.5 to 4 h): the 13 users about 1.2 h, the 5 large
users about 1.9 h. Refine with the measured speed in `RESULT.md`. Afterwards (fresh agent): add the 13 + 5 users' T24
and REF rows to the Big long-card read-out of the low-rank int8 states test (SM148) (R3 buckets by answer count, the
same statistic as `artifacts/small-model/sm148/sm148_report.py`), and report coverage (users per label) and the Big
T24 vs REF long-card cost with its CI.
