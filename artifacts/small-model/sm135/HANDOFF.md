# Handoff: six-layer Tiny with longer recovery (SM135)

Design: `design.md` (pre-registered 11-20 ~01:55, before any GPU ticket). More constant-LR recovery epochs for the six-layer Tiny (SM130) (`w64-c1-d2p1u1`, 185,057 parameters), then SM130's two-pass cool-down and final scoring (w8+s8, compact w8+s4c8nhx8, card-only w8+s4hgx4c8hx8n), one chain after the other:
- **Instant-Only (io), first:** SM130's epoch 41 -> epochs 42..45 at most -> the longer cool-down for Tiny (SM115)'s two-pass cool-down (the show-time Instant-Only big model (SM99)'s targets) -> `rule decide (Instant-Only)`.
- **two-head (th), only if the Instant-Only chain goes:** SM130's epoch 32 -> epochs 33..36 at most. Its cool-down + scoring is **held** (budget ~14 GPU h): it runs only after a relaunch with `TH_COOLDOWN=1`, which needs the main session's (user's) OK.

Rules (`sm135_rule.py`): after every recovery epoch an early check (constant LR, w8+s8): stop if the first new epoch is significantly worse than SM130's last epoch; cool down from the previous epoch if a later one is significantly worse; stop after the second new epoch if any trained head gained less than 0.05% vs SM130's last epoch; cool down after the fourth. Main rule per chain: **go iff every trained head's 95% CI upper bound of 100 x (SM135 / SM130 - 1) < 0 AND the point estimate <= -0.05%**, published w8+s8. Report only: vs Small (the longer Small Instant-Only model (SM113) / the show-time two-head cool-down for Small (SM118)) for all three formats with the +1.00% budget check, vs SM130 in the same format (card-only: the six-layer Tiny with the card-only state (SM134)'s rows), vs the unmodified Tiny (the longer cool-down for Tiny (SM115) / the show-time two-head cool-down for Tiny (SM124)).

## The running job

- Launched 02:01 Fri 11-20 from branch `sm135` at b52aefc6, hidden via WMI. Windows PIDs: WMI wrapper bash **3276**, launcher bash **28384** (`bash .../launch-sm135.sh`), memory guard powershell **12244** (ends with the launcher). Smoke passed 02:01:49; first training ticket **001392-sm135** (Instant-Only epoch 42) running since 02:01:49 (queue wrapper pid 11868, command pid 28064). Queue owner `sm135`, purposes start with `Six-layer Tiny with longer recovery (SM135): `.
- Main log `C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/gpu/launch-sm135.log` (lines `[sm135 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm135.nohup`; trainer `gpu/train.out`; scorer `gpu/score.out`; smokes `gpu/check.out`; `gpu/segments.jsonl` (minutes, nvidia-smi max per training segment), `gpu/tickets.jsonl`; rows `gpu/sm135-early-rows.jsonl` (constant-LR checkpoints), `gpu/sm135-rows.jsonl` (final students); decisions `sm135-early-{io,th}-eN.{json,md}`, `sm135-decision-{io,th}.{json,md}`; state files `gpu/cool-start-{io,th}` (cool-down start epoch), `gpu/verdict-{io,th}`; memory guard `gpu/memguard.log`, `gpu/memguard-kill` (only if it killed something).
- Folders (`wt/sm135/artifacts/small-model/`, not committed): `sm135-io-e42-st` ... `sm135-io-e45-st`, `sm135-io-dec2ep32k-p1`, `-p2`; `sm135-th-e33-st` ... `sm135-th-e36-st` (and later `sm135-th-dec2ep32k-p1`, `-p2`).

## Expected timeline (central; ranges in design.md)

| step | ends about | GPU |
|---|---|---|
| io epoch 42 + scoring + early check | ~03:15 Fri 11-20 | 1.1 h |
| io epochs 43 / 44 / 45 (+ checks) | ~04:20 / ~05:25 / ~06:35 Fri | 1.1 h each |
| io cool-down pass 1 / pass 2 (~150 min each, ~65,000 reviews/s) | ~09:10 / ~11:35 Fri | 5.0 h |
| **io result** (`rule decide (Instant-Only)`, 3 formats ~15 min) | **~11:50 Fri 20 Nov** (11:00-13:30; ~03:15 / ~04:20 if an early stop fires) | 0.25 h |
| th epochs 33-36 + checks (only if io went go) | ~16:15 Fri, then the `done (` line with two-head "held" | 4.4 h |
| th cool-down + scoring, only after `TH_COOLDOWN=1` | +5.3 h after the relaunch (e.g. ~21:30 Fri) | 5.3 h |

Recovery epochs ran at ~178,000-182,000 reviews/s in SM130 (53-55 min per 581.9M-review epoch); expect the same.

## Log lines to watch (regex, on `gpu/launch-sm135.log`)

- **Done:** `\] done \(verdict` -> `done (verdict Instant-Only: <...>; two-head: <...>)`. Two-head "held: recovery done, cool-down from epoch N waits for an OK (relaunch with TH_COOLDOWN=1)" means it stopped at the budget, as planned.
- **Failure:** `\] STOP:` (the last one is `STOP: ended with exit N`). `STOP: pass N ended at lr` = a broken schedule; `STOP: ... is not the reference pass 1` / `is not the chain's step` = a pair check failed: report, do not relaunch blindly.
- **Early stops / recovery end:** `EARLY STOP \(|early check after epoch [0-9]+ \(.*\): (continue|cooldown)|rule early` (each early check's line has both comparisons).
- **Chain verdicts:** `rule decide \((Instant-Only|two-head)\)`; held: `HELD: two-head cool-down`; skipped: `two-head chain skipped`.
- **Throughput:** `complete: .*reviews/s` (one per training folder).
- **Memory guard:** `memory guard killed` (then a STOP file exists).

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm135.sh*' } | Select ProcessId, CommandLine`
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/gpu/launch-sm135.log`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Rules by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/sm135_rule.py early --chain io --epoch 42` / `decide --chain io` / `decide --chain th` (exit 2 = rows missing).
- Stop: `touch C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/gpu/STOP` (between tickets; a training folder snapshot-stops at the next user boundary). Remove it before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:` that needs a look; finished steps are skipped, a folder resumes from its last snapshot), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/launch-sm135.sh >> C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/gpu/launch-sm135.nohup 2>&1"'; ProcessStartupInformation=$si}`
- **Two-head cool-down after the OK** (after `done (`; `gpu/cool-start-th` exists): the same command with `export CHAINS=th TH_COOLDOWN=1;` inserted after `export PATH=...;`. It skips the finished recovery and runs pass 1, pass 2, the three-format scoring and `decide --chain th`, then a new `done (` line.
- Run the two-head chain although the Instant-Only chain did not go (only on request): delete `gpu/verdict-th`, then relaunch with `export CHAINS=th TH_FORCE=1;` (add `TH_COOLDOWN=1` for the cool-down).
- Never edit the running launcher or `scripts/ expertium/ rwkv/` of `wt/sm135` while it runs (SourceWatch stops the training). `wt/sm130` is read only (start epochs). Read-only inputs that must not be cleaned up: design.md, Risks 4.

## After each step (a fresh Sonnet medium agent per step)

1. **After the Instant-Only verdict** (`rule decide (Instant-Only)` or an `EARLY STOP (Instant-Only` line), on branch sm135 (`cd C:/Users/jonat/workspace/wt/sm135; git fetch origin; git merge origin/sm135`): `git add -f` the io decision files (`sm135-decision-io.{md,json}`, `sm135-early-io-*.{md,json}`), `gpu/*.jsonl`, the logs (`launch-sm135.log`, `launch-sm135.nohup`, `score.out`, `check.out`, `prepare.out`, `trim.out`), `gpu/check-*.json`, `gpu/verdict-io`, `gpu/cool-start-io`, `gpu/memguard.log` if present, a `gpu/train-tail.out` (last ~200 lines of `train.out`), and a short `RESULT.md` (Instant-Only section: verdict, main-rule table vs SM130, the three formats vs Small with the budget check and vs SM130 same format, vs the unmodified Tiny, every early check, reviews/s per epoch and pass from the `complete:` lines with the 8k / 2k 10-epoch ETA). Never `.pth`, `.npz`, `work/`. Identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push. Do not touch `pages/` (the main session updates the factory page).
2. **After `done (`:** add the two-head early checks (`sm135-early-th-*`), `gpu/cool-start-th`, `gpu/verdict-th` if present, and the updated logs / jsonl; extend RESULT.md with the two-head recovery (and say plainly that its cool-down is held for an OK, with the start epoch and the +5.3 GPU h cost). Commit + push as above.
3. **After the two-head result** (only if `TH_COOLDOWN=1` was run): the two-head decision files and a two-head RESULT.md section; commit + push.
4. Report to the main session in % (never "points"), every model as "plain name (ID)": verdicts with CIs, the early checks, the card-only numbers vs Small (the headroom question), reviews/s per run with the 8k / 2k 10-epoch ETA, CPU use alongside reviews/s where known, GPU minutes (segments.jsonl + tickets.jsonl).
