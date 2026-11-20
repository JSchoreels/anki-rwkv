# Handoff: long-card state drift on real users (SM136)

Design: `design.md` (pre-registered Fri 11-20 ~02:30, before any GPU ticket). Training-free.

Question: how much LogLoss do int4 card rows cost against int8 card rows (`w8+s8`, the published format) on reviews of
cards with many prior answers? Two int4 formats are tested: the fork default (`w8+s4c8n`) and compact (`w8+s4c8nhx8`).
Two more are report only: int8 cards with the compact blocks (`w8+s8hx8`) and float32 state (`w8`).

Models: the show-time big model (SM91s2), the show-time instant-only big model (SM99), the show-time two-head
cool-down for Small (SM118), the longer Small Instant-Only model (SM113), the show-time two-head cool-down for Tiny
(SM124) and the longer cool-down for Tiny (SM115).

Key design facts:
- **The benchmark reviews stop at 127 prior answers of a card**, so the readout buckets every review the scorer predicts
  (new opt-in scorer flag `--prior-buckets`).
- Main pool: the 66 users of the 732 set with a card of >= 128 prior answers.
- Rule (`sm136_rule.py decide`): int4 card rows are acceptable for a size iff, for both of its models:
  - no main-pool user is non-finite for the format;
  - every head's 95% CI upper bound of 100 x (format / w8+s8 - 1) is <= +0.25% in the rule buckets 128-299 and 128+.
- Otherwise the rule recommends int8 card rows. The release decision follows compact.
- The 300-999 bucket (6 users), the 1000+ bucket (1 user) and the extension pool (10 validation users with id >= 5000)
  are report only.

## The running job

- **Launched 02:32:42 Fri 11-20** from branch `sm136` at 314c539f, hidden via WMI (ShowWindow 0).
- Windows PIDs:
  - wrapper bash 30580 (the WMI process);
  - launcher bash 3480 (`bash .../launch-sm136.sh`). Its children are subshells: 1880 is the GPU-memory sampler, and
    13624 runs `r01_queue.py run` (python 26036).
- CPU proofs passed at 02:34:58 (`CPU proofs ALL OK`, 112.7 s). The memory guard peak was 1.1 GB, so the CPU smoke rows
  were reused from the 02:22 run.
- Queue ticket **`001393-sm136`** (owner `sm136`, expected 105 min) waits behind the six-layer Tiny with longer recovery
  (SM135)'s Instant-Only epoch 42 (`001392-sm135`, started 02:01, expected 70 min).
- ONE chained ticket: `sm136_ticket.sh 0.8 smoke big small tiny ext`.
  - It runs the GPU smoke (users 1866, 2883, 30 labels, vs `cpu/smoke-cpu-rows.jsonl`), then Big, Small and Tiny on the
    66 main-pool users (10 labels each), then the extension (10 users, 30 labels, report only).
  - In-ticket retries for missing users run at cap 0.8, then 0.92.
  - Further tickets run only if main-pool users are still missing, up to 4 passes. The extension is tried in pass 1
    only.
- Quiet windows: quiet.json scope all / gpu only (`sm136_block.py`), re-read before every ticket. No daily or Sunday
  break. A running ticket is not interrupted. To pause, add a quiet.json window (the next ticket waits) or touch STOP.
- Logs (all under `C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/`):
  - main log `launch-sm136.log` (lines `[sm136 MM-DD HH:MM:SS] ...`). The selftest's synthetic `[sm136-rule] verdict`
    lines near the top are the selftest, not a result;
  - nohup `launch-sm136.nohup`;
  - scorer and ticket steps `score.out`;
  - CPU proofs `check.out` + `../check-cpu-sm136.json`;
  - GPU smoke `smoke-gpu-rows.jsonl` + `smoke-check.json` (marker `smoke-ok`);
  - rows `sm136-rows.jsonl` (main) and `sm136-ext-rows.jsonl` (extension);
  - `tickets.jsonl`, `memguard.log`.

## Expected timeline (no pause planned)

| step | ends about |
|---|---|
| queue wait behind the six-layer Tiny with longer recovery (SM135)'s current ticket | ~03:15 Fri 11-20 (SM135's epoch 42 expected 70 min from 02:01) |
| ONE chained ticket: GPU smoke (~5 min), Big (~30 min), Small (~18 min), Tiny (~18 min), extension (~22 min), in-ticket retries | |
| `rule decide` -> `done (verdict ...)` | **~04:45 Fri 20 Nov** (04:15-06:30; +~1.5 h if a retry ticket is needed) |

This is about 1.5 GPU h. If more than 2 main-pool users miss for some model and format, `decide` exits 2
(`STOP: the rule is pending`) and the main session decides.

## Log lines to watch

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/launch-sm136.log`
- Failure: `grep -E '\] STOP:' C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/launch-sm136.log`
- Done or STOP, one regex: `grep -E '\] (done \(verdict|STOP:)' C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/launch-sm136.log`
- Milestones:
  - in `launch-sm136.log`: `CPU proofs ALL OK`, `enqueue: scoring pass 1`, `exit 0: scoring pass 1`,
    `all main-pool labels scored`, `rule decide exit 0`;
  - in `gpu/score.out`: `[sm136-smoke] PASS`, `[sm136-ticket ...] step big` / `small` / `tiny` / `ext`, `ticket done`.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm136.sh*' } | Select ProcessId, ParentProcessId, CommandLine`
- Progress:
  - `tail -5 C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/launch-sm136.log`
  - `grep -a 'small-model\] [0-9]*/\|sm136-ticket' C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/score.out | tail -3`
  - `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Verdict by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/sm136_rule.py decide`
  - exit 2 = pending (fewer than 64 of 66 users for some model and format);
  - exit 3 = the bucket counts differ between formats (a bug; do not decide by hand).
- Stop: `touch C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/STOP`.
  - The STOP file acts between tickets; a running ticket ends normally.
  - To end a running ticket now, cancel it through `r01_queue.py`.
  - Remove STOP before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; scored users and a passed
  smoke are skipped), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/launch-sm136.sh >> C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/gpu/launch-sm136.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the launcher or `scripts/` of `wt/sm136` while a ticket runs.
- Read-only inputs: `wt/sm91s2`, `wt/sm99`, `wt/sm118`, `wt/sm113`, `wt/sm124`, `wt/sm115` (models, published rows),
  `wt/sm72s` (users file). `wt/sm135` and the fork are never touched.
- If `STOP: GPU smoke mismatch`: read `gpu/smoke-check.json` (per user, model and format: GPU vs CPU benchmark LogLoss
  and all-scope bucket mean). The tolerances are 1e-5 for w8+s8 (also vs published) and 1e-4 otherwise. Do not relax a
  tolerance without the user.
- If `STOP: the rule is pending`: `sm136_rule.py decide` prints the model and format. Check `score.out` for `OOM` rows;
  a relaunch retries them (cap 0.8, then 0.92).

## After `done` (a fresh Sonnet medium agent is enough)

1. Branch sm136 (`cd C:/Users/jonat/workspace/wt/sm136; git fetch origin; git merge origin/sm136`): `git add -f` these
   files:
   - `sm136-decision.{md,json}`, `check-cpu-sm136.json`;
   - `gpu/sm136-rows.jsonl`, `gpu/sm136-ext-rows.jsonl`, `gpu/smoke-gpu-rows.jsonl`, `gpu/smoke-check.json`;
   - `gpu/launch-sm136.log`, `gpu/launch-sm136.nohup`, `gpu/score.out`, `gpu/check.out`, `gpu/tickets.jsonl`;
   - `gpu/memguard.log` (if any);
   - a short `RESULT.md`.

   The rows hold aggregate bucket sums per user only (no card ids, no raw reviews). Never add `.pth` or `.npz` files.

   `RESULT.md` covers:
   - the verdict per size (compact; fork default reported);
   - per model, the 128-299 and 128+ numbers with CIs per head for compact and fork default;
   - w8+s8hx8 and float32 vs w8+s8;
   - the 300-999 / 1000+ / main + extension 300+ numbers (report only);
   - the non-finite users per format;
   - the bytes table;
   - the users covered, the unparseable lines, and the GPU minutes from `tickets.jsonl`.

   Commit identity: `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`. The message ends
   with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Fetch + merge, then push.
2. The main session updates the factory page (`pages/data/factory-curated.json`: close the SM136 plan row, add a
   results row, then publish and `--check`). The agent of step 1 does not touch `pages/`.
3. Report to the user:
   - in % (never "points"), every model as "plain name (ID)";
   - the verdict per size and what it means for the fork. If compact fails for a size, the fork should store int8 card
     rows there (`w8+s8hx8` bytes per card: Tiny 1,315, Small 2,545, Big 13,891, against compact 803 / 1,521 / 7,747);
   - the long-bucket costs with CIs, and the 300+ evidence (report only, few users);
   - the non-finite users (user 3876 on the fork default for Big is expected);
   - that the benchmark LogLoss cannot see cards past 127 prior answers.
