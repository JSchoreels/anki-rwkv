# Handoff: six-layer Tiny with the card-only state (SM134)

Design: `design.md` (pre-registered 11-20 before any GPU ticket). Training-free: the two six-layer Tiny (SM130) final models (the two-head six-layer Tiny, `sm130-th-dec2ep32k-t91s2`; the Instant-Only six-layer Tiny, `sm130-io-dec2ep32k-t99`) scored with the card-only state `w8+s4hgx4c8hx8n` (619 B per card, notes 1,315 B). Rule (`sm134_rule.py decide`): go(chain) iff every head's 95% CI upper bound of 100 x (card-only / Small published w8+s8 - 1) <= +1.00%; verdict go iff both chains go. Small = the show-time two-head cool-down for Small (SM118) and the longer Small Instant-Only model (SM113). Reported only: card-only vs the six-layer compact rows and vs the six-layer published rows (SM130's rows, `inputs/sm130-rows.jsonl`).

## The running job

- Launched 01:30 Fri 11-20 from branch `sm134` at f242d1c3 (design + code; this HANDOFF is a later commit), hidden via WMI. Wrapper bash 30404 (the WMI process), launcher bash 33068 (`bash .../launch-sm134.sh`; children are its subshells).
- Queue ticket `001390-sm134` (owner `sm134`, GPU cap 0.8), started 01:31 with the GPU free; GPU smoke PASSED (`gpu/smoke-ok`, `gpu/smoke-check.json`), scoring both models in one process (`gpu/score.out`, lines `[small-model] N/723 user ...`, ~70-80k review-evals/s).
- Logs: main `gpu/launch-sm134.log` (lines `[sm134 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm134.nohup`; scorer + ticket steps `gpu/score.out` (`[sm134-ticket ...] step ...`); CPU proofs `gpu/check.out` + `check-cpu-sm134.json`; `gpu/tickets.jsonl`; rows `gpu/sm134-rows.jsonl`. All under `C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/`.
- Quiet windows: quiet.json scope all / gpu only (`sm134_block.py`), re-read before every ticket; no daily break. A running ticket is not interrupted.

## Expected end

About 20-45 min of scoring (SM130's two-format final scoring took ~35 min per model; one format each): **ends about 02:00-02:20 Fri 11-20** (`done (verdict ...)`), +~20 min per retry ticket if users are missing.

## End lines

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/gpu/launch-sm134.log` -> `done (verdict go)` or `done (verdict no go (th|io|th, io))`; the line before it: `rule decide exit 0: [sm134-rule] verdict ...` with the CIs.
- Failure: `grep -E '\] STOP:' .../gpu/launch-sm134.log` (`STOP: GPU smoke mismatch`, `STOP: scoring ticket failed`, `STOP: the rule is pending` = card-only rows missing for some of the 722 users).

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm134.sh*' } | Select ProcessId, ParentProcessId, CommandLine`
- Progress: `tail -5 .../gpu/launch-sm134.log`; `grep -a 'small-model\] [0-9]*/' .../gpu/score.out | tail -2`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Verdict by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/sm134_rule.py decide` (exit 2 = rows missing)
- Stop: `touch C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/gpu/STOP` (between tickets; a running ticket ends normally; to end it now cancel it through `r01_queue.py`). Remove STOP before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; scored users and a passed smoke are skipped), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/launch-sm134.sh >> C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/gpu/launch-sm134.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/` of `wt/sm134` while a ticket runs. `wt/sm130`, `wt/sm113`, `wt/sm72s` are read only inputs. Do not touch `pages/` or `factory-curated.json` (main session).

## After `done` (a fresh Sonnet medium agent is enough)

1. Branch sm134 (`cd C:/Users/jonat/workspace/wt/sm134; git fetch origin; git merge origin/sm134`): `git add -f` `sm134-decision.{md,json}`, `gpu/sm134-rows.jsonl`, `gpu/smoke-gpu-rows.jsonl`, `gpu/smoke-check.json`, `gpu/launch-sm134.log`, `gpu/launch-sm134.nohup`, `gpu/score.out`, `gpu/check.out`, `gpu/tickets.jsonl`, `gpu/memguard.log` (if any), and a short `RESULT.md`: verdict per chain with the CIs (card-only vs Small, vs six-layer compact, vs six-layer published), bytes (card-only 619 / 1,315, compact 803 / 1,315, fork default 1,055 / 1,567), users covered, GPU minutes from `tickets.jsonl`, reviews/s. Never `.pth`, `.npz`. Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push right after committing.
2. Report in % (never "points"), every model as "plain name (ID)": the two-head six-layer Tiny (SM130) and the Instant-Only six-layer Tiny (SM130) with card-only vs the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model (SM113), the go / no go per chain, and the report-only comparisons. The main session updates the factory page.
