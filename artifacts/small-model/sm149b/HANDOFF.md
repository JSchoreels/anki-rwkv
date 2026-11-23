# Rank-2 states with a tail-energy penalty (SM149b): hand-off

Design and pre-registered rules: `design.md` (committed f41def1c before the launch). The agent that built and launched
it has stopped; per the token rule nobody waits on the GPU. Each result step is a fresh agent started from this file
(Sonnet at medium effort is enough: the rules are fixed).

## State at hand-off (Mon 2026-11-23 14:01, machine clock)

- **Running:** `launch-stage1.sh`, started **14:00:36** detached via WMI (`Win32_Process.Create`, ShowWindow 0): WMI bash
  PID **30272**, launcher bash PIDs 30644 / 22116. Trainer smoke ticket 001520 passed (14:00:47); training ticket
  **001521** (owner sm149b) running since **14:00:48** (queue wrapper PID 23608, command PID 25460).
- **Log:** `C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b/gpu/launch-stage1.log` (raw stdout
  `gpu/launch-stage1.nohup`; training `gpu/train-sm149b-sm118-tail1.out`; scorer `gpu/score.out`).
- **Done marker:** `gpu/stage1-gpu-DONE` (and the log line `stage 1 GPU part done`). Failure: `grep STOP gpu/launch-stage1.log`.
- **Expected:** training about 123 min (range 115 to 140; measured 1.10 x SM149's QAT arm GPU time in the smoke, about
  9,100 reviews/s), integrity 0.3 min, scoring of REF / Q2 / T24 on the 732 set about 34 min plus a short pass at cap
  0.92. **GPU part done about 16:45 Mon 11-23** (range 16:30 to 17:20).
- Run tree: `C:/Users/jonat/workspace/wt/sm149-run` (= origin/sm148 cea0f931, unedited; the launcher refuses source
  edits). Prepared arm `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1` (not committed). Recreate if missing:
  `$PY -X utf8 sm149b_prepare.py arm --lam 1.0 --root C:/Users/jonat/workspace/wt/sm149-run` then `pair` (must print
  `"ok": true`).
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*sm149b*launch-stage1*' } | Select ProcessId, CommandLine`;
  queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

What the launcher does (ticket owner `sm149b`, cap 0.8, quiet windows respected as in SM149: no ticket starts inside or
within its expected length of a quiet.json window of scope all / gpu; a running arm is snapshot-stopped <= 2 min
before one and resumed after it):
1. preconditions (`check-cpu-sm149b.json` all_ok, `sm149b_prepare.py pair`, run tree unedited), trainer smoke
   (`gpu/stage1-smoke-ok`);
2. trains `sm149b-sm118-tail1` (snapshot every 100 users); expected outputs
   `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1/expertium/E89-F65e-F77a/{master.pth,optimizer.pth,result.json,updates.jsonl}`
   (result.json `state_vq.tail_penalty.lambda` = 1.0) and the tail log `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1/tail.jsonl`
   (one line per update pass; the train log ends with `[sm149b] low-rank hook calls: ...` and the process peak GPU memory);
3. integrity: SM149's QAT arm at REF on the 3 smoke users -> `gpu/integrity-sm149qat-smoke.jsonl` (checked in step 3);
4. the 732 set, new arm REF (`w8+s8hx8`), Q2 (`w8+lr2q8x8`), T24 (`w8+lr2.16.4q8x8`), passes at cap 0.8 then 0.92
   until 723 users per label -> `gpu/sm149b-rows-732.jsonl`; marker `gpu/stage1-gpu-DONE`.

Stop: `touch C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b/gpu/STOP` (the arm snapshots at the next
user boundary; note the trainer watches `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1/control/`, which the
launcher's `sm149b_run.py stop` writes). Cancel a waiting ticket with `r01_queue.py cancel` (a queue cancel does not
kill a running command). Resume: remove `gpu/STOP`, then the launch line again (a finished arm and the smoke marker
are skipped; the arm resumes from its snapshot). After a session restart or a power event, check the launcher and
the queue first.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b/launch-stage1.sh >> C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b/gpu/launch-stage1.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `launch-stage1.sh`, `sm149b_qat.py` or `sm149b_run.py` while it runs, and do not re-check out the `.sh`
files (they are LF in the working copy; autocrlf would make them CRLF).

## Step 2: long-card users on the CPU (as soon as `result.json` of the arm says `complete`; alongside the GPU scoring)

A fresh agent copies the CPU chain of the low-rank int8 states experiment (SM148)
(`wt/sm148/artifacts/small-model/sm148/launch-sm148.sh`, functions `lc_users`, `lc_one`, `lc_slots`, `mem_fits`,
`memguard_on/off`) into `cpu-longcard.sh` in this folder, with only the model and labels changed:

- model `sm149b-sm118-tail1=expertium:w64:C:/Users/jonat/workspace/wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1/expertium/E89-F65e-F77a/master.pth`,
  labels REF (`w8+s8hx8`), Q2 (`w8+lr2q8x8`), T24 (`w8+lr2.16.4q8x8`);
- the 50 users of `wt/sm148/artifacts/small-model/sm148/users-longcard.json`, one fresh process per user, at most 2 at
  a time while the GPU scoring runs (3 after it), `mem_fits` before each start;
- command: run tree `scripts/small_model_heldout.py --device cpu --workers 1 --pool 732 --users-file cpu/longcard/users-<u>.json
  --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000
  --dump C:/Users/jonat/workspace/wt/sm149b/work/sm149b-dump --dump-scope all --time-limit 1000000 --out cpu/longcard/u-<u>.part.jsonl`,
  environment `env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0`;
  rows merged into `cpu/sm149b-longcard-rows.jsonl`;
- **memory guard:** SM148's fixed `memguard.ps1` (`C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/memguard.ps1`,
  the one-group fix), started as `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <memguard.ps1> -Dir <this folder>\cpu
  -ParentPid <launcher Windows PID> -Match "small_model_heldout.py&--device cpu&sm149b-dump"` (one group naming only
  this CPU command's words; GPU tickets never match), limit 5.9 GB; a kill writes `cpu/memguard-kill` and `cpu/STOP`
  (that user counts as missing);
- dry-run, then detached via WMI (the line above with `cpu-longcard.sh` and `cpu/cpu-longcard.nohup`); quiet windows as
  in SM148's `wait_free ... cblock` (CPU scope).

The original model's long-card rows and dumps: SM148's (`wt/sm148/artifacts/small-model/sm148/cpu/sm148-longcard-rows.jsonl`;
dumps under `wt/sm148/work/sm148-dump/` if still present, else re-score the original Small two-head (SM118) REF on the
long-card users the same way). About 1.5 to 2 CPU hours (SM149 estimated 2.5 h for 5 labels; here 3).

## Step 3: report and rules (fresh agent, after steps 1 and 2)

```bash
cd C:/Users/jonat/workspace/wt/sm149b && git fetch origin
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm149b
grep -E 'complete|exit|STOP|732 set|done' $D/gpu/launch-stage1.log | cut -c1-300
$PY -X utf8 $D/sm149b_report.py          # dataset-wide R2 / R6 and the gate -> sm149b-stage1.json
```

1. **Integrity:** `gpu/integrity-sm149qat-smoke.jsonl` must equal SM149's rows
   (`wt/sm149/artifacts/small-model/sm149/gpu/sm149-rows-732.jsonl`, label `sm149-sm118-qat/w8+s8hx8`, the 3 users of
   `wt/sm148/artifacts/small-model/sm148/users-smoke.json`) to 1e-6; otherwise stop and report.
2. **Rules** exactly as design.md: R1, R2 and R6 (from `sm149b_report.py`), R3 and G from the long-card rows and dumps
   (adapt SM148's `sm148_report.py` long-card part and `sm148_dr.py` into `sm149b_dr.py`: dR = p(new, label) -
   p(original, REF), share of cards with a review |dR| > 0.10), R5 coverage. The gate: pass (Q2 or T24 passes all) /
   partial (C_new = new Q2 vs new REF, instant, <= +0.0395% = 0.5 x SM149's control +0.079%) / no effect. If only the
   dataset-wide part is in and it is already "no effect", the long-card step can be skipped (it cannot rescue a label
   that fails R2), as SM149 did.
3. Also report (no verdict): new vs SM149's QAT arm at each label (the penalty's own effect), the bucket table, the
   tail energy over training from `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1/tail.jsonl` (mean of the first
   and last 100 passes after the 3 primer passes, and the ratio; outside research saw about 45 x lower), training
   reviews/s and peak memory from result.json, GPU minutes per ticket.
4. Write `sm149b-stage1.json` (+ `.md`) and `RESULT.md` (aggregates only; "plain name (ID)" in every sentence, "outside
   research" never a name, "%" never "points"); commit on `sm149b` (`git add -f artifacts/small-model/sm149b`, never
   `work/`, `.pth`, `.npz`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`;
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending in a blank
   line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`), `git fetch origin && git merge origin/sm149b`, push.
   The factory page row is the main session's.

## Step 4 (only after a partial): the pre-registered follow-up, lambda = 3.0

```bash
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; cd C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b
$PY -X utf8 sm149b_prepare.py arm --lam 3.0 --root C:/Users/jonat/workspace/wt/sm149-run    # -> sm149b-sm118-tail3
$PY -X utf8 sm149b_prepare.py pair --lam 3.0 --root C:/Users/jonat/workspace/wt/sm149-run   # "ok": true
```

Then a copy of `launch-stage1.sh` (`launch-tail3.sh`) with `ARM=sm149b-sm118-tail3`, the pair check `--lam 3.0`, its
own rows file, log and markers; dry run, WMI launch, steps 2 and 3 for it. Same rules and gate; a second partial stops.
After a pass, the main session decides on the other five release models; after no effect, every size keeps rank 4 (Q4).
