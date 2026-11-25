# Deleted-card note ids (SM173): handoff

Status: DONE 11-25 22:35, verdict HARMLESS (RESULT.md); no phase 2.

## Done

- Cause found (design.md "Cause"): `rwkv/data_processing.get_rwkv_data` stores `ID_PLACEHOLDER + card_id` in a float64 `note_id` column (spacing 64 at 3.14e17): deleted cards in one 64-wide card-id block share a note lane; the int32 id cast of `create_sample` turns every placeholder into -2^31 (one note code for all deleted cards in scoring). The fork is not affected (i64 / Python ints).
- Phase 1a (CPU): `sm173-measure.json`, per-user counts `sm173-measure-users.jsonl`. Training users (8,000): 27.0% of reviews are deleted-card reviews, 26.97% on a shared placeholder lane; 93.5% of users have deleted cards (90.9% a shared lane); 46.7 deleted cards per lane (max 65). 732 set: 24.7% of reviews, 95.0% of users. 1,500 pool: 25.4% of reviews, 93.7% of users.
- CPU smoke of the hook (3 users, Tiny, CPU): `cpu/smoke-cpu-rows.jsonl`; the user without deleted cards is bitwise equal in all arms.

## Running

- Launcher `launch-sm173-gpu.sh`, started hidden via WMI (`Win32_Process.Create`, ShowWindow 0). Four arms (cur, lanes, fix, reseed) x two models, one queue ticket per arm (owner sm173, purpose prefix `Deleted-card note ids (SM173): `), about 10 min each; then the report runs on the CPU (`sm173_report.py report` -> `sm173-result.json`, `sm173-result.md`, `gpu/report.out`).
- Run tree `C:/Users/jonat/workspace/wt/sm173-run` (detached at e8ab44d4, never edit it while the chain runs). Do not edit `sm173_score.py` while it runs.
- Log `gpu/launch-sm173-gpu.log` (raw stdout `gpu/launch-sm173-gpu.nohup`, scorer output `gpu/score.out`, rows `gpu/sm173-rows-732.jsonl`). Patterns: `SM173 DONE` (the line before gives the report exit and the VERDICT line), `SM173 STOP` (cause on an earlier `STOP:` line).
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM173 (DONE|STOP)' C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/gpu/launch-sm173-gpu.log; do sleep 120; done; tail -4 C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/gpu/launch-sm173-gpu.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/gpu/STOP` (a queue cancel does not stop the launcher).
- Resume (remove `gpu/STOP` first; finished arms are skipped):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/launch-sm173-gpu.sh >> C:/Users/jonat/workspace/wt/sm173/artifacts/small-model/sm173/gpu/launch-sm173-gpu.nohup 2>&1"'; ProcessStartupInformation=$si}`

## Follow-up

Read `sm173-result.md`; verdict per design.md "Rules". MATTERS or MIXED: design phase 2 (design.md "Phase 2") and launch it. HARMLESS: write RESULT.md, record the issue as harmless, stop. Commit with `git add -f` (rows, result files, logs), never `work/`, `.pth`, `.npz`, `quiet.json`, `.claude/`.
