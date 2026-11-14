# The longer big instant-only teacher (SM113): handoff (2026-11-13, launched 16:17 Fri)

Design: `design.md` in this folder. In short:
- The big instant-only teacher (SM110)'s cool-down stretched over two passes of the 8,000 users: one one_minus_sin decay over 2 x 581,942,724 reviews, from the Small Instant-Only model (SM98)'s epoch 35, with the show-time instant-only big model (SM99)'s show-time targets, as two chained folders (pass 1: 7e-4 -> 0.293 x peak; pass 2: from pass 1's end, new order, 0.293 x peak -> 0).
- **Early stop after pass 1:** pass 1's final model minus SM98's cool-down snapshot after 4,000 users (matching learning rate), show-time instant; continue if below 0.
- **Go:** SM113 minus the Small Instant-Only model (SM98), show-time instant, 722 users, 95% CI below 0 and at least 0.05%.
- **Total:** about 17.1 GPU h, or about 8.3 GPU h if the early stop fires.

## The running job

- Launched 16:17 Fri 11-13 from branch `sm113` at d99fe356, hidden via WMI. **WMI pid 28476** (the `bash -c` wrapper); **launcher bash pid 6428** (`bash .../launch-sm113.sh`).
- **Main log:** `C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/gpu/launch-sm113.log` (every line starts `[sm113 MM-DD HH:MM:SS]`). Trainer `gpu/train.out`, scorer `gpu/score.out`, smokes `gpu/check.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`.
- Queue owner `sm113`; every purpose starts with "the longer big instant-only teacher (SM113): ". First pass-1 ticket 001186.
- At launch: smoke OK (11.5 s); pass 1 running at ~19.3k reviews/s; the GPU learning rate matches the 2R schedule exactly (checked on the first 31 updates).

## Expected timeline (breaks: Fri 20:00 -> Sat 01:00, Sat 20:00 -> Sun 01:00, Sun 16:00 -> Mon 01:00)

| step | ends about |
|---|---|
| pass 1, first segment; snapshot stop <= 2 min before 20:00 (log: `requesting a snapshot stop (break ...)`) | 19:58 Fri |
| break; the launcher waits by itself (log: `...: break, Ns left`) | 01:00 Sat |
| pass 1 resumes from its last snapshot (train.out: `[decay] resuming ...`) | ~05:45 Sat |
| `lr check: pass 1 ended at ...`, then the early stop: pass 1 scored (723 users, ~12 min), `rule mid` | ~06:00 Sat |
| pass 2 prepare + pair (`[sm113-pair] p2 OK`), smoke, pass 2 | ~14:10 Sat |
| `lr check: pass 2 ...`, final scoring (show-time, normal, 1,500 pool), `rule decide` | **~15:00 Sat** |

If pass 2 slips past Sat 20:00 it snapshot-stops and resumes at Sun 01:00 by itself.

## Log lines to watch

- **Early stop:** `rule mid (after pass 1 of 2) exit 0: [sm113-rule] mid continue: ...` or `... mid stop: ...`; on stop also `early stop says stop: pass 2 does not run; verdict no go (stopped after pass 1)` and the last line `done (verdict no go: stopped after pass 1, ...)`.
- **Verdict:** `rule decide (final) exit 0: [sm113-rule] verdict go: ...` or `verdict no go: ...`, then the last line `done (verdict go)` / `done (verdict no go)`.
- **Failure:** any line containing `STOP:` (the last one is `STOP: ended with exit 1`).

## Commands

- Verdict (CPU, seconds; exit 2 = rows missing): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/sm113_rule.py decide` (early stop: `... sm113_rule.py mid`). Output: `sm113-decision.md` / `.json`, `sm113-mid.md` / `.json`.
- Alive check: `tail -5 .../gpu/launch-sm113.log`; `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm113*' } | select ProcessId,CommandLine"`; progress `tail -c 300 .../gpu/train.out`.
- Stop: `touch C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/gpu/STOP` (between tickets; a running pass snapshot-stops at the next user boundary, ~1 min).
- Relaunch (only if the process is gone and the log has no `done`; after a deliberate stop remove `gpu/STOP` first). Finished steps are skipped and a pass resumes from its last snapshot:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/launch-sm113.sh >> /c/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/gpu/launch-sm113.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Detached launchers die when the Claude app restarts: check after a restart.
- **Do not edit or remove while it runs:** `scripts/`, `expertium/`, `rwkv/` in `wt/sm113` (SourceWatch); `wt/sm98` (epoch 35, SM98's cool-down); `wt/sm110` (its cool-down folder is the pass-1 pair reference); `wt/sm99` (the teacher); the targets cache `anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s`.

## What to do on each outcome (fresh Sonnet agent, medium effort)

1. `git -C C:/Users/jonat/workspace/wt/sm113 fetch` and fast-forward `sm113`; `tail -8 .../gpu/launch-sm113.log`.
2. **`STOP:`** — read that line and the tails of `gpu/train.out`, `gpu/score.out`, `gpu/check.out`. Relaunch only for a clear transient (reboot, killed process, app restart). A `STOP: pass N ended at lr ...` or a pair REFUSE is a design bug: report it, do not relaunch.
3. **Early stop fired (`done (verdict no go: stopped after pass 1 ...)`)** — write `sm113-result.md` (layout of `artifacts/small-model/sm99/sm99-result.md`): verdict no go (stopped after pass 1), the `sm113-mid.md` numbers (vs SM98's snapshot and vs SM110's), pass 1's wall minutes (`gpu/segments.jsonl`), reviews/s (its `complete:` line), RSS and nvidia-smi max, total GPU time. Update the SM113 plan row to done and add a results row in `pages/data/factory-curated.json` (main checkout).
4. **`done (verdict ...)`** — if `rule decide` did not run, run it. Write `sm113-result.md` with the verdict, the primary with its CI, SM113 minus the big instant-only teacher (SM110) (the length effect with the same targets), the share of the teacher gap closed, the read-out and 1,500-pool tables from `sm113-decision.md`, the early-stop numbers, per-pass wall minutes / reviews/s / RSS / nvidia-smi max, the 10-epoch ETA at the measured speed (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M) and the total GPU time (tickets + segments).
   - **Go:** the main session tells the user (a candidate to replace the Small Instant-Only model (SM98) in the family; a release needs the user's direct OK). Note the attribution caveat: a two-epoch cool-down with SM98's own targets (~17 GPU h) separates the length from the teacher.
   - **No go:** report it; the backlog re-ranking (Opus, high effort) decides whether the length test on old targets is still worth running.
5. **Commit** with explicit paths and `git add -f`, only what exists: `sm113-mid.json`, `sm113-mid.md`, `sm113-decision.json`, `sm113-decision.md`, `sm113-result.md`, `gpu/sm113-mid-rows.jsonl`, `gpu/sm113-rows.jsonl`, `gpu/sm113-rows-1500.jsonl`, `gpu/launch-sm113.log`, `gpu/launch-sm113.nohup`, `gpu/check.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`, `gpu/score.out`; `gpu/train.out` only as its last 200 lines in `gpu/train-tail.out`. Never commit checkpoints, snapshots, the pass folders, the targets cache, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm113`, push `sm113`. Do not merge into main.
6. **Report** every model as "plain name (ID)" in every sentence, results in %, never "points"; quote reviews/s per pass; never name the outside research group.
