# Low-rank int8 card and note states, post-training (SM148): handoff

Design and rules: `design.md` (pre-registered before any scoring of the 732 set or the long-card users). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) processes the result.

## Launch

Status 11-23 01:55 (machine clock): **running**. First launch 01:42 stopped (the memory guard killed the GPU self-test; fixed, design.md "Launch notes"); relaunched **01:50:28 Mon 11-23** from branch `sm148` at a500a887 (source commit 379e548a, clean).

- Launcher: `launch-sm148.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 7760, launcher bash Windows PID 3156 (memory guard parent). CPU proofs ALL OK (`check-cpu-sm148.json`); GPU smoke ticket **001492** passed at 01:51 (54 checks; REF and U8 GPU vs CPU max 1.0e-5, the low-rank labels max 4.1e-5). Tiny scoring ticket **001493** running since 01:51 (expected about 65 to 80 min), then Small and Big (ids assigned when enqueued; about 95 and 185 min). GPU done about 08:00 to 08:30 Mon 11-23 (range 07:00 to 09:30).
- ETA of `done`: about 08:00 to 09:30 Mon 11-23 (the GPU chain is the long pole: the CPU long-card chain ran about 1 min per Tiny user, so it should end before the GPU chain).

- GPU chain (queue, owner sm148, cap 0.8, `PYTORCH_JIT=0`): the smoke ticket (`sm148_check.py gpu-selftest` + 3 smoke users x 6 models x REF, Q4, T24, Q2, U8, F4 vs the CPU proof rows), then one ticket per size on the 732 set (723 users, 2 models x REF / Q4 / T24 / Q2): Tiny (expected about 65 min), Small (about 95 min), Big (about 185 min), plus short retry passes (the largest user runs out of memory for Big at cap 0.8; the retry at 0.92 covers it). About 5.7 GPU hours in all (range 4.5 to 7.5). FIFO, other owners' tickets can slip in.
- CPU chain (in parallel): the 50 long-card users x Tiny, Small, Big x REF, Q4, T24, Q2 (every label scored directly), 2 processes at a time while the GPU chain runs and 3 after it, memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/<size>/killed.txt`), `--dump-scope all` into `work/sm148-dump/` (per-review values, never committed; a killed or failed user's dumps are deleted). About 8 to 9 hours.
- Then `sm148_dr.py aggregate` -> `sm148-dr.json` (per-review / per-card |dR| vs int8, aggregates only), then `sm148_report.py report` -> `sm148-result.{json,md}`.

## Patterns

- Log: `C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/gpu/launch-sm148.log` (raw stdout `gpu/launch-sm148.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`).
- Done: `grep -E '\] done \(status' gpu/launch-sm148.log` (the status line names the go pick per size).
- Pending: `report pending` (exit 2: fewer than 720 GPU users or 48 long-card users for a label, or an incomplete dR read-out); the main session decides.
- Failure / STOP: `grep -E '\] STOP' gpu/launch-sm148.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure files: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm148*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP` (and `cpu/STOP` if the memory guard wrote one), then the launch command again; finished users (GPU rows, `cpu/longcard/<size>/u-<user>.jsonl` with their dumps) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/launch-sm148.sh >> C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/gpu/launch-sm148.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm148 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source at start; a mid-run change would not be in the processes that already loaded the code). Never edit `launch-sm148.sh` while it runs. The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm148 && git fetch origin
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm148
grep -E 'smoke|done|STOP|report|dR read-out|killed|failed' $D/gpu/launch-sm148.log | cut -c1-300
cat $D/sm148-result.md
$PY -X utf8 $D/sm148_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go pick per size with bytes vs int8 (Q4 -74.4 / -52.3 / -51.1%, T24 -81.7 / -66.3 / -64.7%, Q2 -82.8 / -68.2 / -66.7% for Big / Small / Tiny; the current best tiered int formats are -43.5 / -38.6 / -42.7%), for Tiny both the go under the user's Tiny rule (R2T) and the strict go (R2); per model and label R2 (dataset-wide vs int8), R2T for Tiny (vs the matching Small at int8), R3 (cards with 128+ answers, CPU, point <= +0.5% and upper bound <= +1.0%), G (share of cards with a review |dR| > 0.10, at most 1%); the bucket table (where the cost sits by answer count); the per-review / per-card |dR| tables in plain words (does any card lose a real retention prediction?); coverage (killed or failed long-card users); integrity vs SM145's REF rows; the speed cost per write (`report_only.speed` in `sm148-result.json`: review-evals/s of each label vs REF on the GPU and on the CPU long-card rows, plus the CPU per-write microbenchmark in `check-cpu-sm148.json` `info`); GPU minutes per ticket (`gpu/tickets.jsonl`), CPU process-minutes. Quote reviews/s.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, the per-card read-out in plain words, the speed cost per write, what it means for the fork, e.g. whether a cool-down trained with the compressed state is worth it for a label that failed), aggregates only. Naming rules: every model as "plain name (ID)" in every sentence (e.g. "Big two-head (SM91s2)", "six-layer Tiny Instant-Only (SM130)"); say "outside research", never the researcher's name; "%" never "points".
3. **Commit** on branch `sm148`: `git add -f artifacts/small-model/sm148` (never `work/`, `.npz`, `.pth`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with a message ending in a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch origin && git merge origin/sm148` (if it moved), then `git push origin sm148`. Never force-push, never bypass hooks.
4. **Factory page** (main repo `C:/Users/jonat/workspace/anki-rwkv`, `git fetch` + fast-forward first): close the SM148 plan row (match prefix "Low-rank int8 states (SM148): "; `when`: done time and verdict per size; `state` done; `ended_at`) and add a results row in the format of the SM145 / SM146 results rows (per model and label table, times, structured detail), by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check` (SYNC 0); commit factory-curated.json + factory.json only; fetch + merge origin, push main. No "Expertium"; "plain name (ID)" in every sentence; "%" not "points".
5. If the report is pending, a label fails only on one head or one model, or the Tiny go under R2T and the strict go differ, the decision is the user's (via the main session).
6. After the write-up is pushed, the dumps in `work/sm148-dump/` (about 1 to 2 GB) can stay for a later look; deleting them is the user's call.
