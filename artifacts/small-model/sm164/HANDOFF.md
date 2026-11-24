# Release-candidate re-check of the final state formats (SM164): handoff

Status: GPU chain RUNNING (launched 11-24 23:34). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) does the read-out.

Design and rules: `design.md` (pre-registered in 855d0b8f before any row of this run existed). Question: do the six release-candidate models on their final state formats (Big and Small F6 `w8+lr2.16.4q6x8`, six-layer Tiny T24 `w8+lr2.16.4q8x8`) stay inside the user's budgets on the 732 set and on the 1,500 pool? Budgets, CI upper bound per head: Big <= +0.15% vs its own int8; Small <= +0.5% and Tiny <= +1.0% total vs the matching Big model at int8. On the 1,500 pool the Small / Tiny total is decomposed (upper(own int8 vs Big int8, 732 set) + upper(final label vs own int8, 1,500 set)) because the users with id >= 5000 are training users of the Big models.

## Done

- Worktree `C:/Users/jonat/workspace/wt/sm164`, branch `sm164`, cut from `origin/sm152` (8f4604c7, the branch that added the `q6` factor label; its F6 rows are reused). No source change.
- Reused, not rescored: SM148 rows (REF, T24, 732 set), SM152 rows (F6, 732 set), SM160 rows (REF, T24 on 100 users with id >= 5000, `--pool 1500`; sha256 pinned). SM155 has only long-card rows (nothing to reuse for the pools).
- `sm164_report.py selftest` OK; dry run OK (`gpu/launch-sm164-gpu-dryrun.log`). Without the new rows, the 732 rule reproduces SM152 / SM148 (all six cells pass).

## Running (GPU, via the queue)

- Launcher `launch-sm164-gpu.sh` from commit 855d0b8f, started hidden via WMI (`Win32_Process.Create`, ShowWindow 0) at 11-24 23:34: WMI PID 22664 (outer bash); launcher PIDs and the first tickets are in "Launch check" below.
- Queue purpose prefix (every ticket): `Release-candidate re-check of the final state formats (SM164): `, owner `sm164`.
- Steps: (1) GPU smoke, 5 users x 6 models x REF / T24 / F6, checked by `sm164_report.py smoke` (-> `gpu/smoke-check.{txt,json}`, `gpu/smoke-ok`); (2) the ids >= 5000 part of the 1,500 pool, one ticket per job: Tiny T24 + REF (665 users), Small F6 (765), Small REF (665), Big F6 (754), Big REF (654) -> `gpu/sm164-rows-new.jsonl`. Passes at cap 0.8, then 0.92 after a pass without progress.
- ETA: about 3.4 GPU hours: Tiny done about 00:25, Small F6 about 01:15, Small REF about 01:35, Big F6 about 02:45, Big REF about 03:10; `SM164 GPU DONE` around 03:00 to 04:00 on 11-25 (later if other tickets queue in between).
- Log: `gpu/launch-sm164-gpu.log` (raw stdout `gpu/launch-sm164-gpu.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`).
- Patterns in the log: done `SM164 GPU DONE`; stopped or failed `SM164 GPU STOP` (the cause is on an earlier line starting `STOP:`).
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm164*' } | Select ProcessId, Name`. Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM164 GPU (DONE|STOP)' C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/launch-sm164-gpu.log; do sleep 120; done; tail -5 C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/launch-sm164-gpu.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/STOP` (before the next ticket); to stop the running ticket too, `r01_queue.py cancel` on its ticket id (touch STOP first: a cancel does not stop the launcher).
- Resume after a stop or a machine restart (finished users and `gpu/smoke-ok` are kept; remove `gpu/STOP` first; check that no sm164 process is still alive):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/launch-sm164-gpu.sh >> C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/launch-sm164-gpu.nohup 2>&1"'; ProcessStartupInformation=$si}`
  (Without the `>> ... 2>&1` redirect the hidden bash hangs before its first line.)
- Do not change anything under `scripts/`, `expertium/`, `rwkv/` or the sm164 scripts, json files and design.md of this worktree while the chain runs (the launcher checks a clean tree at start). Do not touch `wt/sm160`: the SM160 rows file is pinned by sha256.

## Launch check

- 23:34:03 smoke ticket 001583-sm164; exit 0 at 23:35:27; `[sm164] smoke OK: 45 (model, label, head, source) cells within 1e-05 (worst 0.00e+00)`: the new rows equal SM152's GPU smoke rows and SM160's rows exactly (so `--pool 1500` and `--pool 732` give the same rows).
- 23:35:28 first real ticket 001584-sm164 (Tiny T24 + REF, 665 users): queue wrapper PID 13164, command PID 15212, scorer python PID 20320 / 38992; nvidia-smi 7.8 GB, 96% at 23:35:48.
- Launcher bash PIDs: 28020 (main), 22664 (WMI outer bash). Later tickets get the next free ids (001585 and up), each with the purpose prefix above.

## Follow-up steps (fresh agent, Sonnet, medium effort)

1. `git -C C:/Users/jonat/workspace/wt/sm164 fetch` (merge `origin/sm164` if it moved). Wait for `SM164 GPU DONE` or `SM164 GPU STOP` (command above). On a STOP, read the `STOP:` line, `gpu/smoke-check.txt` and the tail of `gpu/score.out`; fix nothing in source; report to the main session.
2. Read-out (CPU, a few minutes, under 1 GB):
   `cd C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm164_report.py report`
   -> `sm164-result.json`, `sm164-result.md`. Exit 0 = decided; 2 = pending (coverage short: a label has fewer than 715 users on the 732 set or fewer than 97% of its list on the ids >= 5000; the lines are printed) -> rerun the launcher once with the resume command, then report again; if still pending, report to the main session; 3 = invalid (SM160 rows changed or no `gpu/smoke-ok`).
   It prints one line per size: `[sm164] Big F6 (w8+lr2.16.4q6x8): GO|HOLD|NO`, same for Small and Tiny.
3. Write `RESULT.md` from `sm164-result.json` / `.md`, in full plain sentences:
   - the question and the rules (design.md, plain words, including why the Small / Tiny 1,500 rule is decomposed);
   - the verdict per size (GO / HOLD / NO) with, per model and head, the 732 rule value and 95% CI against its ceiling, and the 1,500 rule upper bound (Big: vs own int8 on the 1,500 set; Small / Tiny: own int8 vs Big int8 on the 732 set + final label vs own int8 on the 1,500 set) and its margin to the ceiling;
   - report-only lines: the direct Small / Tiny vs Big comparison on the 1,500 set and on the ids >= 5000 alone, and own int8 vs Big int8 on the ids >= 5000 (Big's in-sample advantage), each with its CI;
   - warnings, copied (not rerun): long-card and per-card from `../sm152/RESULT.md` (Small and Tiny F6; F6 is the Small format), `../sm148/RESULT.md` (T24, the Tiny format) and SM155 (Big T24 long cards, `C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/RESULT.md`); say that Big F6 has no long-card read-out;
   - row sources (`row_sources` in the json), users per comparison, GPU minutes per ticket (`gpu/tickets.jsonl`).
   Every model as "plain name (ID)": Big two-head (SM91s2), Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130); "%" never "points"; never name the outside research's author ("outside research").
4. Update the status at the top of this HANDOFF.md. Commit with `git add -f`: `RESULT.md`, `HANDOFF.md`, `sm164-result.json`, `sm164-result.md`, `gpu/sm164-rows-new.jsonl`, `gpu/smoke-gpu-rows.jsonl`, `gpu/smoke-check.json`, `gpu/smoke-check.txt`, `gpu/smoke-ok`, `gpu/tickets.jsonl`, `gpu/launch-sm164-gpu.log`, `gpu/launch-sm164-gpu.nohup`, `gpu/score.out`. Never `work/`, `.npz`, `.pth`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `__pycache__/`. Identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm164` right before the push; push `sm164`. No force-push, no hook bypass.
5. Do NOT edit `pages/` or the factory page (the main session does that). Report to the main session: the verdict per size, the tightest cell per size with its upper bound and ceiling, any HOLD with its numbers, and the commit.
