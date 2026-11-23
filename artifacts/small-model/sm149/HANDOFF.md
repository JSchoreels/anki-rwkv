# Rank-2 card and note states through a short cool-down (SM149): hand-off

Design: `design.md` (this folder). Nothing runs now. The main session reviews the design, then launches stage 1. Per
the token rule, no agent waits on the GPU: the launcher is detached, and each result step is a fresh agent started from
this file (Sonnet at medium effort for the scoring and the report; the rules are fixed).

## State at hand-off (2026-11-23)

- Branch `sm149` (design, proofs, scripts; this folder), pushed.
- Run tree: worktree `C:/Users/jonat/workspace/wt/sm149-run` on local branch `sm149-run` = `origin/sm148` (cea0f931),
  **unedited** (the launcher refuses source edits there). Recreate if missing:
  `git -C C:/Users/jonat/workspace/anki-rwkv worktree add -b sm149-run ../wt/sm149-run origin/sm148`.
- Prepared folders of stage 1 (in the run tree, not committed):
  `wt/sm149-run/artifacts/small-model/sm149-sm118-qat` and `sm149-sm118-ctl` (794 users, 67,256,032 reviews, peak
  7e-5, from Small two-head (SM118)'s final weights and Muon state; `pair` OK). Re-create with step 0 if missing.
- CPU proofs ALL OK (`check-cpu-sm149.json`); GPU: micro-benchmark, trainer smoke of both arms, real-data throughput
  of both arms done (design.md "Proofs").

## Step 0 (only if the prepared folders are missing)

```bash
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
S=C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149
R=C:/Users/jonat/workspace/wt/sm149-run
$PY -X utf8 $S/sm149_prepare.py arm --model sm118 --arm qat --root $R
$PY -X utf8 $S/sm149_prepare.py arm --model sm118 --arm control --root $R
$PY -X utf8 $S/sm149_prepare.py pair --model sm118 --root $R      # must print "ok": true
```

## Step 1: launch stage 1 (Small two-head (SM118), both arms + the 732-set scoring)

Dry run first, then detached (WMI ShowWindow=0, which survives Claude app restarts; check it after any gap):

```bash
cd C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149
DRYRUN=1 bash launch-stage1.sh && cat gpu/launch-stage1.log && rm gpu/launch-stage1.log
```

```powershell
# the launch line of the low-rank int8 states experiment (SM148), paths changed
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149/launch-stage1.sh >> C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149/gpu/launch-stage1.nohup 2>&1"'; ProcessStartupInformation=$si}
# alive: Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-stage1*' } | Select ProcessId, CommandLine
```

What it does (log `gpu/launch-stage1.log`, tickets owner `sm149`):
1. preconditions: `check-cpu-sm149.json` all_ok, `pair` OK, the run tree has no source edits;
2. trainer smoke of both arms (~1 min, `gpu/stage1-smoke.out`);
3. the QAT arm `sm149-sm118-qat` (expected about 2 h: 3.08 x the control's 38 min, measured), then the control arm `sm149-sm118-ctl` (expected about 40 min),
   each one ticket, snapshot every 100 users, snapshot-stopped by `gpu/STOP` or <= 2 min before a quiet window and
   resumed after it (running the launcher again resumes too); logs `gpu/train-<folder>.out`;
4. integrity: the original Small two-head (SM118) at REF on the 3 smoke users (`gpu/integrity-orig-smoke.jsonl`; must
   equal `wt/sm148/artifacts/small-model/sm148/gpu/smoke-gpu-rows.jsonl` for that model and label to 1e-6, checked
   in step 3);
5. the 732 set, both arms (QAT: REF, Q2, T24; control: REF, Q2), about 55 min, passes at cap 0.8 then 0.92; rows
   `gpu/sm149-rows-732.jsonl`; marker `gpu/stage1-gpu-DONE`.

Expected outputs: `wt/sm149-run/artifacts/small-model/sm149-sm118-{qat,ctl}/expertium/E89-F65e-F77a/{master.pth,
optimizer.pth,result.json,updates.jsonl}` (result.json `state_vq.recipe` = `lr2q8x8` for the QAT arm only, absent for
the control), the rows file with 723 users x 5 labels. The QAT train log ends with
`[sm149] low-rank hook calls: {...}` (window and carried counts > 0).

Stop: `touch gpu/STOP` (an arm snapshots at the next user boundary). Cancel: also `scripts/r01_queue.py cancel` for a
waiting ticket (a queue cancel does not kill a running command).

## Step 2: long-card users on the CPU (can start as soon as both arms are trained; alongside the GPU scoring)

A fresh agent copies the low-rank int8 states experiment (SM148)'s CPU chain (`wt/sm148/artifacts/small-model/sm148/launch-sm148.sh`, functions `lc_one`,
`memguard_on/off`, with `memguard.ps1` matching `small_model_heldout.py&--device cpu&sm149`) with only the models and
labels changed: models `sm149-sm118-qat` (REF, Q2, T24) and `sm149-sm118-ctl` (REF, Q2), the 50 users of
`wt/sm148/artifacts/small-model/sm148/users-longcard.json`, one fresh process per user, at most 2 at a time while the
GPU runs, `--device cpu --workers 1 --pool 732 --users-file <one user> --inputs e2s-query --split-existing
--deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000 --dump work/sm149-dump
--dump-scope all`, environment `-u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS
-u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0`, rows into `cpu/sm149-longcard-rows.jsonl`. The
original model's long-card rows and dumps: the low-rank int8 states experiment (SM148)'s (`cpu/sm148-longcard-rows.jsonl`; the dumps under
`wt/sm148/work/sm148-dump/` if still present, else re-score the original REF on the long-card users the same way).
About 2.5 CPU hours for Small (SM148: about 25 min per Small user-size pass for 4 labels on 2 models).

## Step 3: report and rules (fresh agent, after steps 1 and 2)

Adapt the low-rank int8 states experiment (SM148)'s `sm148_report.py` (statistic, bootstrap, rows loaders, bucket tables) and `sm148_dr.py` into
`sm149_report.py` / `sm149_dr.py` in this folder with the design's rules exactly: R1, R2 (vs the ORIGINAL model's
REF rows of the low-rank int8 states experiment (SM148)), R3, G (dR vs the original REF), R6 (new REF vs original REF), R5, and the stage gate (pass /
partial / no effect; design.md "Stage gates"), plus the reported comparisons vs the control. Write
`sm149-stage1.{json,md}`, update `RESULT.md`, commit + push on `sm149`, update the factory page row (main session).

## Step 4 (only after a stage-1 pass): stage 2

For each next model in the design's order (Small Instant-Only (SM113), Big two-head (SM91s2), Big Instant-Only
(SM99), six-layer Tiny two-head and Instant-Only (SM130)): step 0 with `--model sm113 | sm91s2 | sm99 | sm130th |
sm130io`, a copy of `launch-stage1.sh` with the model's folder names, architecture (`fork` scorer kind for Big,
`expertium:w64-c1-d2p1u1` for Tiny), track (`big-showtime`, `big-inst`, `E89-F65e-F77a`) and original checkpoint.
**Big first needs the memory check** (design.md "Memory"): a 3-minute throughput ticket of Big two-head's QAT arm on
a scratch copy (`gpu/timing-ticket.sh` with the folder names changed); if peak allocated exceeds 12.5 GB, stop and
implement the written-steps record (CPU proof first) before the Big launch.

## Clean-up done by this design

The scratch throughput folders `wt/sm149-run/artifacts/small-model/sm149-timing-{qat,ctl}` were deleted after the
read-out (`gpu/timing-sm149.json` keeps the numbers).
