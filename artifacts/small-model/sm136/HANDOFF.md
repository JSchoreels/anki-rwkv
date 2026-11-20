# Handoff: long-card state drift on real users (SM136), CPU scoring

**STATUS 11-20 ~17:00, pass 2 NOT launched (blocked): the memory guard kills are not a label-count problem.** Findings (details
below the line): the scorer's peak memory is per evaluation and grows with the user's review count (about 3.2 GB at 102,545
reviews, 5.6 GB at 150,500 for one label; 3 labels in one process only add ~0.3 GB), so splitting labels across processes cannot
bring the 15-16 killed users per size (186k to 531k reviews) under 5.9 GB. Coverage stays at 50-51 of 66 users per model and format
(< 64), so `decide` stays pending. The real fix needs an exactly time-chunked layer forward (state carry for the token shift and
the WKV recurrence) or a machine with ~20 GB free for the largest users. Nothing is running. Interim, non-decisive readout:
`sm136-interim.json` (made by `sm136_interim.py`, never `sm136-decision.json`).

## Memory finding (measured 11-20, CPU, 4 threads, one user per process, `peakrun.py` polls the working set every 2 s)

| user (reviews), model | labels | peak working set |
|---|---|---|
| 1866 (5,673), Tiny (SM124) | w8+s4c8n | 0.67 GB |
| 1015 (102,545), Tiny (SM124) | w8 / w8+s8 / w8+s4c8n | 3.23 / 3.23 / 3.24 GB |
| 1015, 1 thread instead of 4 | w8 | 3.19 GB |
| 1015, 3 labels in one process | w8, w8+s4c8n, w8+s8hx8 | 3.51 GB |
| 1692 (150,500), Tiny (SM124) | w8+s4c8n | 5.57 GB |

Cause: one `model.get_loss` over the whole history. The batch splits rows by sequence length; user 1015 has a single row of
193,671 steps (the global stream) plus short-card splits. A time mixer holds about 40 tensors of [rows, steps, 64] floats at
once (the 8-way shifted-lerp stack alone is 8 copies), about 10 KB per step of the longest stream, and the allocator keeps the
high-water mark for the rest of the process (live tensors after a layer are ~0.9 GB while the working set stays at 3.2 GB). The
label format does not change it (w8 without any state rounding is the same), the thread count does not, and the number of labels
per process only adds ~0.3 GB. Row-chunking (at most 256 rows per layer call; tried, reverted) changes nothing, because the
memory is one long row, not many rows. Heap trimming (`_heapmin`/`HeapCompact`) and `MIMALLOC` variables changed nothing (torch
uses the CRT heap here). A 531,091-review user would need ~15-20 GB at this per-step cost.

Options (main session decides; the pre-registered rule is untouched): (a) a CPU-only exact time-chunked layer (chunk the
sequence, keep the layer-normed input for the shifted gathers, carry the WKV state and v0; the check is bitwise equality with the
unchunked rows on users 1866 and 1015 at a tiny chunk size); the fork (Big) mixers have the same structure but a separate class;
(b) a machine with >= 24 GB free RAM for the 15-16 killed users per size (Big ones are the largest per step); (c) amend the design
(pool without the killed users) - changes the pre-registered coverage rule, needs the user.
The old CPU run's state is unchanged below.

---

**STATUS 11-20 03:06: the GPU run stopped (`STOP: GPU smoke mismatch` 02:56: all-scope bucket means differed up to 1.85e-3 on
long-card user 1866 for quantized formats, limit 1e-4; benchmark LogLoss agreed within 4.4e-5). Scoring was moved to the CPU
(`design.md` amendment 11-20 03:00): same rule, pool, models, formats; no tolerance relaxed. The GPU plan below the CPU section
is kept for reference (rule, design facts and after-done steps still apply).**

## The CPU run (launch-sm136-cpu.sh)

- **Launched 03:05:45 Fri 11-20** from branch `sm136` at f54d4c39, hidden via WMI. Free RAM at launch 9.6 GB (the six-layer
  Tiny with longer recovery (SM135) training holds ~11 GB; scorers use 1.3-2 GB each at the start).
- Windows PIDs: wrapper bash 16612 (WMI process), launcher bash 28156, memory guard powershell 16880 (5.9 GB per scorer),
  workers 13128 and 24920 (2 workers, one fresh python scorer per user).
- Log: `C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/cpu/launch-sm136-cpu.log`
  (nohup `cpu/launch-sm136-cpu.nohup`; per-user score output `cpu/rows/<stage>/score-<user>.out`; rows `cpu/rows/<stage>/u-<user>.jsonl`,
  merged after each stage into `gpu/sm136-rows.jsonl` and `gpu/sm136-ext-rows.jsonl`, which the unchanged rule reads;
  also `cpu/progress.out`, `cpu/memguard.log`, `cpu/oom.txt`, `cpu/failed.txt`).
- Done or STOP regex: `grep -E '\] (done \(verdict|STOP:)' <the log>`
- Design as scored: per user ONE process = one stage (2 models x 5 formats w8+s8, w8+s4c8n, w8+s4c8nhx8, w8+s8hx8, w8), `--device cpu`,
  show-time inputs, leak-free (neighbour), `--nonfinite-rows`, `--prior-buckets 0,32,128,300,1000`. Stages in the order of
  `cpu/STAGES` = `tiny small big ext` (Tiny: SM124 + SM115; Small: SM118 + SM113; Big: SM91s2 + SM99; ext = the 10 extension users with
  all six models, report only). Users from the smallest to the largest review count (`cpu/order.json`; the 66 main users range from
  5,673 to 531,091 reviews).
- Quiet windows: quiet.json scope all / cpu / gpu only (`sm136_block.py --scopes all,cpu,gpu`); a user starts only if it fits before
  the next window. No daily or Sunday break. The GPU queue is not used.
- Resume: complete users are skipped (all labels present, no error). Users killed by the guard are in `cpu/oom.txt` and skipped on
  resume (delete their line to retry); the run goes on and STOPs before the rule if any.

## Runtime estimate (measured on CPU, 4 threads per scorer; user 1866 = 5,673 reviews, user 954 = 85,171 reviews)

| model x 5 formats | ms per review |
|---|---|
| the show-time big model (SM91s2) | ~5.8 (33 s for user 1866; the show-time instant-only big model (SM99) assumed equal) |
| the show-time two-head cool-down for Small (SM118) | ~3.4 (23 s for user 1866, 292 s for user 954; the longer Small Instant-Only model (SM113) assumed equal) |
| the show-time two-head cool-down for Tiny (SM124) | ~3.2 (18 s for user 1866; the longer cool-down for Tiny (SM115) assumed equal) |

All six models: ~25 ms per review. Main pool 8.41 M review rows: ~58 h serial = **~29 h with 2 workers**; extension 2.78 M rows: ~19 h
serial = ~10 h with 2 workers (report only). Per stage with 2 workers: Tiny ~7.5 h, Small ~8 h, Big ~14 h (the largest Big user alone
~50 min), extension ~10 h. **This exceeds 14 h. Proposed reduction (the main session decides; the pre-registered design is untouched
until then):** Tiny + Small first (~15.5 h); raise to 3 workers once SM135's training ends (~1.5x faster); optionally skip the
report-only formats w8+s8hx8 and w8 (saves 40%) or the extension. Edit `cpu/STAGES` (read at every stage start) to cut stages; the
rule stays `pending` (exit 2) unless Tiny, Small and Big all have >= 64 of 66 users.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm136-cpu.sh*' } | Select ProcessId, CommandLine`
- Progress: `tail -5 <the log>`; `wc -l C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/cpu/progress.out`
- Verdict by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/sm136_rule.py decide` (exit 2 = pending, 3 = bucket counts differ: a bug)
- Stop: `touch C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/cpu/STOP2` (workers end after their current user); remove before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/launch-sm136-cpu.sh >> C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136/cpu/launch-sm136-cpu.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Do not edit the launcher or `scripts/` of `wt/sm136` while it runs. Never touch `wt/sm135` or the fork.
- After `done` (a fresh Sonnet medium agent is enough): the steps in "After `done`" below, with the CPU files: add `cpu/launch-sm136-cpu.log`,
  `cpu/progress.out`, `cpu/order.json`, `cpu/oom.txt` (if any), `gpu/sm136-rows.jsonl`, `gpu/sm136-ext-rows.jsonl`,
  `sm136-decision.{md,json}`, `gpu/smoke-check.json` (the GPU mismatch evidence); `RESULT.md` mentions the GPU smoke mismatch and the move to the CPU.

---

## Original GPU plan (stopped, reference only)


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
