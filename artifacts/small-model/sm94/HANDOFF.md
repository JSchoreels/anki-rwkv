# Fast-recall prediction (SM94): handoff

Branch `sm94` (from `sm92`), worktree `C:/Users/jonat/workspace/wt/sm94`. Design and fixed rules: [DESIGN.md](DESIGN.md). Code `scripts/sm94_fastpass.py` (commit `7abcb6be`).

## What runs (detached, WMI hidden, below-normal priority)

Launcher `launch-sm94.sh`, WMI pid **21020** (bash), launched 2026-11-07 00:45. Log `gpu/launch-sm94.log`, nohup `gpu/launch-sm94.nohup`.

1. GPU ticket **000947-sm94** (owner sm94, cap 0.8, expected 25 min), started 00:45:28: `sm94_fastpass.py dump` on the 723 users of the 732 set -> `work/dump/users/<user>.npz` (~5-6 GB, not committed) + `work/dump/dump-rows.jsonl`; stdout `gpu/dump.out`. A pass 2 runs only if users are missing. Expected end ~01:05.
2. CPU, right after: `sm94_fastpass.py analyze` -> `stage1-result.json`; stdout `gpu/analyze.out`. ~10-20 min, one process, < 3 GB RAM. Expected end ~01:25.

The launcher holds in quiet windows and while free RAM < 3 GB. Stop between steps: `touch C:/Users/jonat/workspace/wt/sm94/artifacts/small-model/sm94/gpu/STOP`. Relaunch (dumped users are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0; PriorityClass=[uint32]16384}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm94/artifacts/small-model/sm94/launch-sm94.sh >> C:/Users/jonat/workspace/wt/sm94/artifacts/small-model/sm94/gpu/launch-sm94.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `scripts/` of this worktree or anything in `C:/Users/jonat/workspace/wt/sm82c` (the scorer's code, imported read only) while it runs.

## Follow-up (fresh Sonnet medium agent, CPU only, a few minutes)

1. `cd C:/Users/jonat/workspace/wt/sm94 && git fetch && git merge --ff-only`. Check `gpu/launch-sm94.log` ends with `done (rc 0)`. If the analysis step failed, rerun it (below-normal priority, after fixing the cause): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm94_fastpass.py analyze --dump-dir artifacts/small-model/sm94/work/dump --out artifacts/small-model/sm94/stage1-result.json`.
2. Speed: from `gpu/dump.out` the last line `done N users in S s, M scored reviews (R/s)`; also the queue's elapsed time. Quote scored reviews/s.
3. Rule A0 (`identity_vs_sm82c` in `stage1-result.json`): `same_reviews` = `users` and `max_rel_gap` <= 1e-5, and >= 722 users. If it fails, report and stop.
4. Write `stage1-result.md` applying DESIGN.md's rules exactly (A1, A2, the recall rule). Per threshold (under 3.9 / 5.5 / 10.9 / 21.5 s, under the user's typical time): users, rows, event rate, LogLoss of C / B0 / B1 / S0 / S, points vs C with 95% CI (`vs_C_points`), S vs S0 (`S_vs_S0_points`), calibration of C (mean predicted vs observed, ECE). Recall: `Rb_vs_Rb0_points` (the rule), `Rb0_vs_R_points` (calibration note), in points of the big model (1 leak-free epoch). One verdict line per rule, plain names with IDs in parentheses, the limitation sentence from DESIGN.md.
5. Commit (`git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`), explicit paths, `git add -f`: `stage1-result.json`, `stage1-result.md`, `gpu/launch-sm94.log`, `gpu/dump.out`, `gpu/analyze.out`, `work/dump/dump-rows.jsonl` (not the npz files). Push. The main session updates the Factory page (results row with times + detail) and asks the user about stage 2.
6. Stage 1c (the FSRS7-Instant x speed baseline) and stage 2 are not started without the user's go.

## Notes

- CPU smoke (3 users, before launch): dump identical to SM82c's recorded rows (relative gap <= 4e-8, same review sets); the analysis ran end to end (numbers on 3 users are meaningless: each half's stack is fit on 1-2 users).
- The GPU was free at launch (queue empty), so the ticket started at once; a later ticket of another experiment waits behind it in the FIFO queue.
