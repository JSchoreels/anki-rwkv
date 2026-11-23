# Smaller shift vectors and factors on T24 states (SM152): handoff

Design and rules: `design.md` (pre-registered before any scoring of the new labels on the 732 set). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) processes the result. A narrow copy of the rank-1 int8 states (SM150) tree: new labels on top of T24 (`w8+lr2.16.4q8x8`, the user's 11-23 choice): S4 (`w8+lr2.16.4q8x4`, int4 shift vectors), F6 (`w8+lr2.16.4q6x8`, int6 factors), F6S4 (`w8+lr2.16.4q6x4`, both). REF and T24 are SM148's rows (REF, T24 and Q2 rescored on the 3 smoke users only). Codec change: `scripts/state_lowrank.py` (q6, x4), `scripts/state_lowrank_cuda.py` (factor qmax as the kernel argument), one line of `scripts/small_model_heldout.py` (`set_ref`).

## Launch

Status: __STATUS__

## Patterns

- Log: `C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/gpu/launch-sm152.log` (raw stdout `gpu/launch-sm152.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`).
- Done: `grep -E '\] done \(status' gpu/launch-sm152.log` (the status line names the go pick per size, e.g. `done (status decided {'Big': 'F6S4', ...})`).
- Pending: `report` exit 2 (fewer than 715 GPU users for a label or its comparison REF); the main session decides.
- Failure / STOP: `grep -E '\] STOP' gpu/launch-sm152.log` (the last is `STOP: ended with exit 1 (gpu part)`; the one before names the step). Smoke failure files: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm152*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop between steps: `touch C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/gpu/STOP` (a running ticket finishes). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`; the launcher then stops at its next check.
- Resume: remove `gpu/STOP`, then the launch command again; finished users (GPU rows) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/launch-sm152.sh >> C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/gpu/launch-sm152.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm152 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source at start). Never edit `launch-sm152.sh` while it runs. The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF). The `work/` folder is not git-ignored: never `git add` under it.
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm152 && git fetch origin
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm152
grep -E 'smoke|done|STOP|report|users scored|failed' $D/gpu/launch-sm152.log | cut -c1-300
cat $D/sm152-result.md
$PY -X utf8 $D/sm152_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go pick per size (F6S4, F6, S4, or the T24 fallback) with bytes vs int8 and vs T24 (`sm152-bytes.json`: S4 -84.49% / -70.38% / -69.52% vs int8 and -15.12% / -12.20% / -13.77% vs T24; F6 -84.60% / -71.44% / -69.74% and -15.75% / -15.34% / -14.38%; F6S4 -87.36% / -75.56% / -74.60% and -30.87% / -27.55% / -28.15%; T24 -81.72% / -66.26% / -64.65%, for Big / Small / Tiny); per model and label the cost vs own int8, vs the matching Big model at int8 for Small and Tiny (CI upper bound per head against +0.15% for Big, +0.5% for Small, +1.0% for Tiny) and vs T24 (the cost of this step alone); which labels pass for both models of each size; the bucket tables (vs own int8 and vs T24: where the cost sits by answer count); coverage (users per label); the integrity lines (CPU: REF, T24 and Q2 rescored on the 3 smoke users equal SM148's rows within 1e-6, `check-cpu-sm152.json`; GPU: REF and T24 equal SM148's GPU smoke rows within 1e-5, `gpu/smoke-check.txt`); speed (`report_only.speed` in `sm152-result.json`; quote reviews/s); GPU minutes per ticket (`gpu/tickets.jsonl`). Say plainly that the long-card and per-card warnings are **not run yet** (later CPU step below). Explain in full plain sentences.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, what is not run yet, what it means for the fork), aggregates only. Naming rules: every model as "plain name (ID)" in every sentence (e.g. "Big two-head (SM91s2)", "six-layer Tiny Instant-Only (SM130)"), never a bare ID; say "outside research", never a researcher's name; "%" never "points".
3. **Commit** on branch `sm152`: `git add -f artifacts/small-model/sm152` (never `work/`, `.npz`, `.pth`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with a message ending in a blank line and `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; `git fetch origin && git merge origin/sm152` (if it moved), then `git push origin sm152`. Never force-push, never bypass hooks.
4. **Not the factory page**: the main session owns `pages/` (plan-row match prefix "Smaller shift vectors and factors on T24 states (SM152): "). Do not edit `pages/` in the main repo.
5. If the report is pending, a label fails only on one head or one model, or the pick sits between two labels by a hair, the decision is the user's (via the main session).

## Later step, NOT started: long-card warnings (CPU)

A CPU speed measurement for the fork is due, so the long-card chain is not run with the GPU part. When the main session says the CPU is free (and the GPU part is done), launch it hidden; it scores the 50 long-card users x {Tiny, Small} x 2 models x {S4, F6, F6S4} on the CPU (2 processes, memory guard 5.9 GB per process, dumps into `work/sm152-dump/`, never committed), copies SM148's REF dumps next to them, writes `sm152-dr.json`, then re-runs the report with the warnings filled in. About 3 to 4 h (SM150: about 2 h for 2 labels). Log `cpu/launch-sm152-longcard.log`; done pattern `grep -E '\] longcard done \(status' cpu/launch-sm152-longcard.log`; STOP pattern `grep -E '\] STOP' cpu/launch-sm152-longcard.log`; stop with `gpu/STOP` as above. Dry run first: `PART=longcard DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/launch-sm152.sh` (log `cpu/launch-sm152-longcard-dryrun.log`).

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; PART=longcard bash C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/launch-sm152.sh >> C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/cpu/launch-sm152-longcard.nohup 2>&1"'; ProcessStartupInformation=$si}
```

After `longcard done`: report the warnings in plain words (long cards 128+ for Small and Tiny: point and upper bound vs own int8, flagged above +0.5% / +1.0%; per-card share of cards with a review |dR| > 0.10, flagged above 1%; Big not run), update `RESULT.md`, commit and push as in step 3.
