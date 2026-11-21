# Tiered card state by answer count (SM141): handoff

Status 11-21 17:45: **running** (launched 17:38:09 Sat 11-21 from branch `sm141` at b2bc1db6). Design and rules: `design.md` (pre-registered before any scoring). The agent that launched it has stopped; a fresh agent processes the result.

## Launch

- Launcher: `launch-sm141.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 34612, launcher bash Windows PID 2756 (memory guard parent). Log `gpu/launch-sm141.log`, raw stdout `gpu/launch-sm141.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`.
- GPU chain (queue, owner sm141, cap 0.8): smoke ticket **001445** (queued behind 001444 of the answer-time expressiveness check (SM142), which was expected to end about 18:01), then one ticket per size Tiny / Small / Big on the 732 set (expected 60 / 80 / 130 min; FIFO, so other owners' tickets can slip in between). GPU done about 23:00 Sat if the queue stays free.
- CPU chain (started 17:38, in parallel): the 50 long-card users x Tiny, Small, Big, 3 processes at a time, memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/<size>/killed.txt`). Tiny users ran at 2,600-3,800 review evaluations per second; expected end about 22:00-24:00 Sat.
- ETA of `done`: about 23:00 Sat 11-21 to 01:00 Sun 11-22.

## Patterns

- Done: `grep -E '\] done \(status' gpu/launch-sm141.log` (the status line names the pick per size).
- Pending: `report pending (coverage)` (exit 2 of the report: fewer than 720 GPU users or 48 long-card users for a label); the main session decides.
- Failure: `grep -E '\] STOP' gpu/launch-sm141.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm141*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm141/artifacts/small-model/sm141/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP`, then the launch command again; finished users (GPU rows, `cpu/longcard/<size>/u-<user>.jsonl`) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm141/artifacts/small-model/sm141/launch-sm141.sh >> C:/Users/jonat/workspace/wt/sm141/artifacts/small-model/sm141/gpu/launch-sm141.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm141 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source). The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).
- After a session restart or power event: check the launcher and the queue (detached launchers have died at the 11-17 power-off); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough for steps 1-3)

```bash
cd C:/Users/jonat/workspace/wt/sm141
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm141
grep -E 'smoke|done|STOP|report|killed|failed' $D/gpu/launch-sm141.log | cut -c1-300
cat $D/sm141-result.md
$PY -X utf8 $D/sm141_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go per size (pick and bytes vs int8), and per model and scheme R2 (dataset-wide), R3 (long cards 128+, CPU), R4 (young cards); non-finite users; coverage (killed or failed long-card users); the report-only lines: REF vs SM139's GPU rows (`report_only.ref_vs_sm139_gpu_rows`, should be about 0) and GPU vs CPU on the long-card bucket. Quote the GPU minutes per ticket (`gpu/tickets.jsonl`) and the CPU minutes.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, what it means for the fork); commit with `git add -f artifacts/small-model/sm141`, fetch + merge, push `sm141`.
3. **Factory page** (main repo): update the SM141 plan row (`when`: done time and verdict; `state` done; `ended_at`) and add a results row with times and the structured detail per scheme and size, by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check`; commit factory-curated.json + factory.json only; push main. No "Expertium" on the page; "plain name (ID)" in every sentence; "%" not "points".
4. If the report is pending, or a scheme fails only on one head or one model, the decision is the user's (via the main session). Open questions are in `design.md` (int2 ternary, A / B4 long-card check by dominance, discrete GPU rule for int3 / int2 tiers).

## Fork work if a scheme passes (estimate; not started; the fork worktrees were not touched)

- The fork already writes one precision byte per stored module state (0x18 Int8Compact, 0x14 Int4Compact) and decodes each state by its own byte, so mixed widths in one collection need no decode change.
- **A / AN:** select Int4Compact below 128 answers and Int8Compact from 128 at every encode, for cards (A) and notes (AN). The encoder needs the entity's update count: from the replay (card: its revlog answers; note: the answers of its cards) or a counter in the state header (u16 / u32, +2-4 B, about 0.02-0.4% of the bytes). Tests: the boundary (answer 127 int4, 128 int8), mixed-width decode, the bulk path, the synthetic long-card probe of the fork compact state check (SM133). About 0.5-1 agent-day.
- **B4 / B8 (int3):** a new Int3Compact codec (e.g. precision byte 0x13): `encode_matrix` / `decode_matrix` / `row_bytes` with 3-bit packing (8 codes in 3 bytes, codes -3..3, float16 row scale, int8 shift vectors unchanged); parity vectors from `sm141_bytes.py` (`pack` / `unpack`) and the scorer's `TierQuant`. About 1-1.5 agent-days on top of A.
- **D4 (int2):** Int2Compact (e.g. 0x12, ternary codes -1..1, 4 per byte). About 0.5 agent-day more if int3 exists.
- Per-size mapping: only the sizes whose pick is not REF change; the release format stays int8 for the others.
