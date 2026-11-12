# The big instant-only teacher (SM110): handoff (2026-11-12, prepared, NOT launched)

Design: `design.md` in this folder. In short:
- **Targets:** the show-time instant-only big model (SM99) replays the 8,000 cool-down users on show-time inputs
  (new folder `anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s`, ~4.3 GPU h).
- **Cool-down:** the Small Instant-Only model (SM98)'s cool-down is rerun from its epoch 35, with only those targets
  swapped (~8 GPU h).
- **Mid-check after 4,000 users:** SM110's snapshot minus SM98's own snapshot at the same point, show-time instant;
  continue if below 0.
- **Go:** SM110 minus SM98, show-time instant, 722 users, 95% CI below 0 and at least 0.05%.
- **Total:** about 13.9 GPU h, or 8.9 GPU h if the mid-check stops.

## Launch (NOT done; the caller launches after telling the user)

From `C:/Users/jonat/workspace/wt/sm110` at the pushed commit of branch `sm110`, hidden via WMI:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/launch-sm110.sh >> /c/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/gpu/launch-sm110.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Record the returned ProcessId (the wrapper) and the launcher bash pid here after the launch.

- **Before launching:**
  - Check the queue is empty: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
  - Check the source trees are clean: `git -C C:/Users/jonat/workspace/wt/sm110 status --porcelain -- scripts expertium rwkv` is empty.
  - The recipe folder `artifacts/small-model/sm110-recipe-e35st-dec-1ep32k-muon-t99` (untracked, on disk) must exist.
    If it is missing, recreate it with:
    `env RWKV_DELETED_CARDS=neighbour C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm77_warm.py cooldown --epoch 25 --start C:/Users/jonat/workspace/wt/sm98/artifacts/small-model/sm98-inst-e35-st --out artifacts/small-model/sm110-recipe-e35st-dec-1ep32k-muon-t99 --arch w64 --fraction 1.0 --extra-options "$(cat artifacts/small-model/sm110/extra-options.json)"`
- **Alive check:**
  - `tail -5 C:/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/gpu/launch-sm110.log`
  - `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm110*' } | select ProcessId,CommandLine"`
  - Export progress: `tail -2 gpu/export.out`. Cool-down progress: `tail -3 gpu/train.out`.
- **Stop:** `touch C:/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/gpu/STOP`.
  - It acts between tickets. A running cool-down gets a snapshot stop at the next user boundary (~1 min).
  - A running export finishes its current user and stops at its time limit. A hard kill loses at most the user in
    flight.
- **Relaunch:** only if the process is gone and the log has no `done`. If the stop was deliberate, remove `gpu/STOP`
  first, then run the same WMI command.
  - Finished steps are skipped. The export skips finished users. The cool-down resumes from its last snapshot.
  - A pending mid-check runs before the resume.
  - Detached launchers die when the Claude app restarts: check after a restart.
- **Breaks:** `sm110_block.py`, i.e. quiet.json plus the daily 20:00-01:00 (Sunday from 16:00). Thu 11-12's break is
  lifted by quiet.json's 1-minute window at 00:59-01:00 Fri. Every ticket must end before the next break. The cool-down
  snapshot-stops at most 2 min before a break and resumes after it.
- **Queue and GPU:**
  - Owner `sm110`. Every purpose starts with "the big instant-only teacher (SM110): ".
  - Caps 0.8: the exporter's default, the trainer's options, and the scorer's `HELDOUT_GPU_MEMORY_FRACTION`. The cap
    is 0.92 only for the 3 long users or for a retry pass that made no progress.
- **Do not edit or remove while it runs:**
  - `scripts/`, `expertium/`, `rwkv/` in `wt/sm110` (SourceWatch);
  - anything in `wt/sm98`: epoch 35 is the start, and the cool-down snapshot `users-04000` is the mid-check reference;
  - `wt/sm99` (the teacher).

## Where the output goes (under `artifacts/small-model/sm110/`)

- **Logs:**
  - main log `gpu/launch-sm110.log`. It ends `done (verdict ...)` or a `STOP:` line;
  - export `gpu/export.out`, trainer `gpu/train.out`, scorer `gpu/score.out`, smoke `gpu/check.out`, prepare
    `gpu/prepare.out`;
  - `gpu/segments.jsonl` (cool-down segments with nvidia-smi max) and `gpu/tickets.jsonl` (tickets with minutes and
    nvidia-smi max).
- **Export checks:** `gpu/check-targets-sm110.json`, plus copies of the targets' `gpu/replay-manifest.json`,
  `gpu/replay-status.json` and `gpu/replay-verify.json`.
- **Cool-down checks:** `gpu/check-replay-label-windows-sm110-inst-e35st-st-dec-1ep32k-muon-t99.json`,
  `gpu/check-muon-fresh-....json`, and the pair result in the main log.
- **Rows:**
  - `gpu/sm110-mid-rows.jsonl`: labels `sm98-dec-u4000/w8+s8@e2s-query` and `sm110-dec-u4000/w8+s8@e2s-query`;
  - `gpu/sm110-rows.jsonl` and `gpu/sm110-rows-1500.jsonl`: label `sm110-inst-e35st-st-dec1ep32k-t99/w8+s8[@e2s-query]`.
- **Rules:** `sm110-mid.json` / `.md` and `sm110-decision.json` / `.md`.
- **Cool-down folder:** `artifacts/small-model/sm110-inst-e35st-st-dec-1ep32k-muon-t99` (checkpoints: never committed).

## Checks done before launch

- CPU proofs `check-cpu-sm110.json` (+ `.out`): ALL OK in 1.3 min.
  - C1 code trees == 1e22b62c.
  - C2 sha256 pins: the teacher == the release package, epoch 35 == SM98's cool-down start, SM98's snapshot users-04000
    present.
  - C3 the recipe == SM98's cool-down except teacher + replay + paths, and 6 mutated copies are refused.
  - C4a SM99's show-time targets on CPU for users 4001 and 8081: manifest == the show-time teacher targets (SM91
    step 3)'s except the teacher, same NaN pattern as both earlier target sets, finite, moved.
  - C4b this tree reproduces the GPU targets of the show-time teacher targets (SM91 step 3) within one float16 step.
  - C5 the scorer reproduces SM98's committed rows exactly on CPU, and SM98's snapshot users-04000 loads and scores.
  - C6 rule selftest. C7 breaks on fixed times and on the real quiet.json. C8 disk.
- Pair: `sm110_pair.py --untrimmed` on the recipe: OK. The launcher repeats it on the real folder after the trim pass,
  without `--untrimmed`.
- Dry run: `gpu/launch-sm110-dryrun.log` ends `DRYRUN OK`.

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. **Check where the launcher is.**
   - Run `git -C C:/Users/jonat/workspace/wt/sm110 fetch` and fast-forward `sm110`.
   - `tail -8 C:/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/gpu/launch-sm110.log`
   - Mid-run: `grep -E "complete|mid-check|STOP|break|rule|export" gpu/launch-sm110.log | tail`.
2. **If the log ends `STOP:`,** read that line and the tails of `gpu/export.out`, `gpu/train.out`, `gpu/score.out` and
   `gpu/check.out`. Fix only what is clearly a transient (a reboot, a killed process) by relaunching with the WMI
   command above. Report anything else.
3. **If the log ends `done (verdict ...)`:**
   - If a rule did not run, run it yourself (CPU, seconds; exit 2 = rows missing):
     `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm110/sm110_rule.py decide`
     (or `mid`).
   - Write `artifacts/small-model/sm110/sm110-result.md` in the layout of
     `artifacts/small-model/sm99/sm99-result.md`, with:
     - the verdict;
     - the primary (SM110 minus the Small Instant-Only model (SM98), show-time instant) with its CI;
     - the share of the teacher gap closed;
     - the read-out table and the 1,500 pool, from `sm110-decision.md`;
     - the mid-check's numbers (`sm110-mid.md`);
     - the export's wall minutes and reviews/s (its `pass finished` line in `gpu/export.out`) and its nvidia-smi max
       (`gpu/tickets.jsonl`);
     - the cool-down's wall minutes (all segments in `gpu/segments.jsonl`), reviews/s (its `complete:` line),
       process-tree RSS and nvidia-smi max;
     - the 10-epoch ETA at the measured speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M);
     - the total GPU time (tickets + segments).
4. **Commit** with explicit paths and `git add -f`. Commit only what exists from this list:
   - `sm110-mid.json`, `sm110-mid.md`, `sm110-decision.json`, `sm110-decision.md`, `sm110-result.md`;
   - `gpu/sm110-mid-rows.jsonl`, `gpu/sm110-rows.jsonl`, `gpu/sm110-rows-1500.jsonl`;
   - `gpu/check-targets-sm110.json`, `gpu/replay-manifest.json`, `gpu/replay-status.json`, `gpu/replay-verify.json`;
   - `gpu/check-replay-label-windows-*.json`, `gpu/check-muon-fresh-*.json`;
   - `gpu/launch-sm110.log`, `gpu/launch-sm110.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`,
     `gpu/tickets.jsonl`, `gpu/score.out`, `gpu/export.out`.

   `gpu/train.out` is large: commit only its last 200 lines, as `gpu/train-tail.out`.

   Never commit checkpoints, snapshots, the cool-down or recipe folders, `sm110/work/`, the targets cache, `.claude/`,
   `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json.

   Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, ending the
   message with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

   Right before the push: `git fetch`, merge `origin/sm110`, then push `sm110`. Do not merge into main. Do not touch the
   Anki fork, the workload forecast or `pages/`.
5. **Report** in plain names with the ID in parentheses, in every sentence (e.g. "the big instant-only teacher
   (SM110)"). Give results as %, never "points". Quote reviews/s for the export and for the cool-down. Never write the
   outside research group's name.
