# The small model in the review simulator (SM41): handoff, part B (the policy search)

**Status: done** (ran 2026-11-05 02:01-09:31; results in [policy-result.md](policy-result.md)).

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41`. Always `git fetch` + `git merge --ff-only` first. If SSH
to GitHub fails:
`git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch|push https://github.com/JSchoreels/anki-rwkv.git sm41`.

Read first:
- the brief: [diag-brief.md](diag-brief.md);
- part A: [diag-result.md](diag-result.md);
- the plan, metric and decision rule, fixed before the runs: [policy-result.md](policy-result.md). Screen 1 is
  already in it.

## What runs (CPU only)

- **The chain:** `artifacts/small-model/sm41/policy/run-chain.sh`, launched hidden via WMI. Windows bash pid 78956;
  it logged MSYS pid 1532492 in `policy/chain.log`. Engine code: the commit that adds this file.
  1. **Screen 2:** `curve-R` and `hyb-R-a0.5` at R = 0.80 / 0.85 / 0.90 x seeds 0-3, on the 20 users of part A.
     Output `policy/screen2/rows/`.
  2. **Screen 3:** the time-aware target at β = 0.5 and 2 (`ihr-R-s0.5`, `ihr-R-s2`). Output `policy/screen3/rows/`.
  3. **Choice:** `sm41_policy.py choose` applies the plan's rule to all screen rows: the best time-aware family, plus
     the best other family whose gap vs FSRS-7 is < 0 or has a CI including 0, at most 2. Outputs
     `policy/lanes-confirm.txt`, `policy/lanes-confirm.json` (the screen gaps and the pick) and
     `policy/screen-tables.md`. "screens done" in `chain.log`.
  4. **Confirmation:** those lanes (3 targets x 4 seeds + tf) on every full-run user and split (117 users, 304
     splits, 60 days). Rows in `policy/confirm/rows/<user>.jsonl`, logs in `policy/confirm/logs/<user>.log`.
  5. **Reports:** `policy/confirm-tables.md` (all users) and `policy/confirm-heldout-tables.md` (the 97 users not in
     the screen), each with a `.json`. Then "done" in `chain.log`.
- **Load:** 4 processes x 1 thread, below-normal priority. `quiet.json` holds inside the engine (per user and per
  split, from 10 min before a window: next 20:00-01:00). A new split is held while less than 3 GB of RAM is free.
- **Expected end:**
  - screens about 02:50;
  - confirmation about 4 h (1 family) to 6.5 h (2 families) after that, so **about 07:00-09:30**.
  - Screen 1 took 27 min for 25 lanes on the 20 users. Those users are 7.1% of the full run's CPU time.
- **Resumable.** If the chain died (pid gone and no "done" in `chain.log`, e.g. after a Claude app restart),
  relaunch the same command. Finished splits are skipped.
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/run-chain.sh >> /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/chain.out 2>&1"'; ProcessStartupInformation=$si}
  ```
  A relaunch reruns `choose` on the same screen rows, so the pick is the same.

## Check commands

```
tail -n 3 /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/chain.log
ls /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/confirm/rows | wc -l      # 117 when done
grep -l Traceback /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/*/logs/*.log
cat /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/policy/lanes-confirm.json
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*sm41_engine.py run*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}}"
```

Rows per split: `cat policy/confirm/rows/*.jsonl | wc -l` should be 304.

## Follow-up (fresh agent, Sonnet medium)

1. **Checks.**
   - `chain.log` ends with "done".
   - 304 confirmation rows: 117 files, each user's 2-3 splits.
   - No Traceback in `policy/*/logs/`.
   - If anything is missing, relaunch (above) and stop. Do not wait on the run.
2. **Fill the "Results" section of `policy-result.md`.** Keep it short. Plain names with the ID in parentheses
   ("the small model in the review simulator (SM41)"), never a bare ID, never "Expertium".
   - **Screens 2 and 3:** from `policy/screen-tables.md`, the gap vs FSRS-7 and vs the instant head for every
     family. Name the families passed to confirmation (`lanes-confirm.json`).
   - **Confirmation:** from `confirm-tables.md` (all users) and `confirm-heldout-tables.md` (97 held-out users):
     - the gap vs FSRS-7 with its CI, median, and share of users cheaper;
     - the gap vs the instant head, and the 60-day cost gap;
     - the per-target table (retention, min/day, memorised).
   - **Verdict, by the plan's rule:**
     - "beats FSRS-7" if the gap is < 0 with the CI excluding 0;
     - "as good as FSRS-7" if the CI includes 0;
     - otherwise "FSRS-7 stays cheaper".
     - Say whether the held-out users agree.
   - **Could it become an Anki scheduler?** The time-aware target only needs the card's last answer time and the
     user's median answer time, both in the revlog. So it is a per-card desired-retention rule on top of any
     recall model: in the simulator it needs the instant head, and in Anki it would need the fork's RWKV runtime.
     Note the caveat: the simulator's learner is the same small model, so the gain is "inside the model's world".
     A real-data check comes next: offline, on the real revlog, are slow-answer cards cheaper to review late? It
     needs the user's go-ahead.
3. **Commit as Jonathan Schoreels:**
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, with the message
   ending in a blank line + `Co-Authored-By: Claude Sonnet <noreply@anthropic.com>` (your own model name). Use
   explicit paths and `git add -f`:
   - `policy/screen2/`, `policy/screen3/`, `policy/confirm/`: `rows/*.jsonl`, `logs/*.log`, `runner.log`;
   - `policy/lanes-confirm.txt`, `policy/lanes-confirm.json`, `policy/choose.out`, `policy/screen-tables.md`;
   - `policy/confirm-tables.md`, `policy/confirm-heldout-tables.md` and their `.json`;
   - `policy/chain.log`, `policy/chain.out`;
   - `policy-result.md`;
   - this file, with the status set to "done".

   Then push. Never stage `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `pages/` or
   `artifacts/coordination/queue/quiet.json`. Never bypass hooks. The main session updates the Factory page.
4. Reply in under 300 words. Include:
   - part A in two lines (from diag-result.md: the gap is seconds per review, +7.9 ± 1.6%; H1-H5 as stated not the
     cause);
   - the screens;
   - the confirmation gap vs FSRS-7 with its CI (all users and held-out);
   - the best policy;
   - the next step.
