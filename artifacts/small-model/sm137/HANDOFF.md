# Handoff: trained smaller state for Tiny (SM137), full plan

**CLOSED 11-21 09:30:** the full plan ended at 09:16 with the Instant-Only chain **no go** (main rule passed, long-card check failed on a wide interval); the two-head chain did not run. Write-up: `RESULT.md`. Nothing is left to watch or relaunch.

Design: `design.md` (pre-registered 11-21; section "Pilot result and full-plan launch" records what changed since: checks only, **no rule changed**). The pick: the six-layer Tiny (SM130)'s card and note WKV state cut to 16 value rows x 8 of 16 key lanes per head (`w64-c1-d2p1u1-v16k8`), **803 B per card and per note** instead of 1,315 B at the release-candidate format.

## State on 11-21 03:20

- Pilot done 02:58: **go**, instant +0.065% [+0.054, +0.077] vs the six-layer Tiny with longer recovery (SM135)'s epoch 42 (w8+s8, 722 users); readout `sm137-pilot-io-v16k8.md`, log `gpu/launch-sm137.log`.
- **Full plan launched 03:19** (branch sm137 at 4b466ade): `launch-sm137-full.sh`, launcher bash Windows PID **15372** (WMI hidden), first queue ticket **001404** (trainer smoke of pass 1, exit 0), then **001405** = Instant-Only cool-down pass 1 (enqueued 03:19:42).
- Prepared and proved before launch (CPU): the Instant-Only pass 1 folder `artifacts/small-model/sm137-io-v16k8-dec2ep32k-p1` (Muon fresh check 7 PASS, `sm137_muon_zero.py` PASS, `sm137_pair.py p1` OK vs SM115's and SM130's pass 1); the two-head cut `artifacts/small-model/sm137-th-v16k8-e32` (sha256 101523b5a8b3, local, not committed) with `check-cpu-sm137-th-v16k8.json` ALL OK, CPU smoke rows `cpu/smoke-cpu-th-v16k8-rows.jsonl`, `sm137_epoch.py pair --chain th` OK.

## Stages and ETA (Sat 11-21)

| stage | ETA end |
|---|---|
| Instant-Only pass 1 (~150-165 min at ~60-66k reviews/s) | ~06:00-06:25 |
| pass 2 (prepared + paired by the launcher, then trained) | ~08:40-09:10 |
| final scoring (student w8+s8 + w8+s8hx8; SM130 final w8+s8hx8), ~30 min | ~09:40 |
| long-card CPU rows (50 users, 2 processes, ~25-35 min), `longcard --chain io`, `decide --chain io` | **Instant-Only verdict ~10:15** |
| two-head (only if io = go): GPU smoke, prepare + pair, smoke, lane epoch 33 (~60-65 min), w8+s8, `early-th` | ~11:40 |
| two-head pass 1 + pass 2 (~2 x 165 min) | ~17:15 |
| final scoring + long-card + `decide --chain th` | **two-head verdict ~18:30** |

## Watch

- Log: `C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/launch-sm137-full.log` (stdout/stderr of the launcher: `gpu/launch-sm137-full.nohup`); trainer `gpu/train.out`, scorer `gpu/score.out`, smokes `gpu/check.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`; long-card per-user rows and logs `cpu/longcard/<chain>/`.
- Done: `grep -E '\] done \(full' .../gpu/launch-sm137-full.log` -> `done (full: io <verdict>, th <verdict>)`. Per chain: `gpu/verdict-full-io`, `gpu/verdict-full-th`, and the line `<chain> chain verdict: ...`.
- Failure: `grep -E '\] STOP:' .../gpu/launch-sm137-full.log` (the last one is `STOP: ended with exit 1`; the one before it is the reason). Report; do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm137-full.sh*' } | Select ProcessId, CommandLine`
- Stop: `touch C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/STOP` (a running pass snapshot-stops at the next user boundary). Resume: `rm` the STOP file, then the same launch command (finished steps are skipped: `p1-<chain>-ok`, `smoke-*-ok`, complete folders, scored rows, complete long-card users, `verdict-full-<chain>`).
- Launch command (PowerShell):

      $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/launch-sm137-full.sh >> C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/launch-sm137-full.nohup 2>&1"'; ProcessStartupInformation=$si}

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm137 worktree, and do not merge into it, while a training ticket runs (SourceWatch); edits under `artifacts/` are fine.
- A memory-guard kill (any CPU process of this run above 5.9 GB) writes `gpu/memguard-kill` + `gpu/STOP`; the long-card step then stops the launcher before the rule.

## After each chain: write-up (CPU, seconds; fresh agent, Sonnet medium is enough)

    cd C:/Users/jonat/workspace/wt/sm137
    PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm137
    cat $D/gpu/verdict-full-io                        # or verdict-full-th
    $PY -X utf8 $D/sm137_rule.py longcard --chain io --lanes v16k8   # re-print (writes sm137-longcard-io-v16k8.json)
    $PY -X utf8 $D/sm137_rule.py decide --chain io --lanes v16k8     # re-print (writes sm137-decision-io-v16k8.{json,md}); exit 2 = rows missing
    $PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_inspect.py $D/sm137-decision-io-v16k8.json --field vs_small_release_format --field vs_small_published --field vs_sm130_release_format --field long_card --field verdict
    grep -E 'complete:|lr check|verify|final scoring|rule ' $D/gpu/launch-sm137-full.log | cut -c1-400   # reviews/s, minutes, dropped-row checks

For the two-head chain also `sm137-early-th-v16k8.json` (epoch 33 vs SM130's epoch 32). Then:

- write `RESULT.md` (both chains: verdict, the rule numbers vs Small at w8+s8hx8 and w8+s8, vs SM130 at both formats, long-card excess, reviews/s per pass and the 8k/2k 10-epoch ETA, bytes 803 vs 1,315), commit with `git add -f` the readouts, `gpu/launch-sm137-full.log`, `gpu/sm137-rows.jsonl`, `gpu/tickets.jsonl`, `gpu/segments.jsonl`, `gpu/verdict-full-*`, `cpu/sm137-longcard-rows.jsonl` (never .pth / .npz), fetch + merge origin, push sm137;
- factory page in the MAIN repo (git fetch + ff first): in `pages/data/factory-curated.json` add a results row per chain (times, reviews/s, the rule table, long-card excess) like the SM137 pilot row of 11-21, and update the SM137 plan row (`when`, `state`, `ended_at` once both chains are decided). Targeted byte edits only (CRLF + HTML entities; never json.dumps the whole file); then `.venv/Scripts/python.exe scripts/publish_factory_page.py` and `--check` (must print SYNC 0); commit only `pages/data/factory-curated.json` and `pages/data/factory.json`; push main. Never "Expertium" in page text; models as "plain name (ID)".
- The two-head chain runs only if `verdict-full-io` starts with `go`; otherwise it writes `not run (...)`. A two-head run without the io go needs the main session's OK: relaunch with `export TH_FORCE=1;` before `bash` in the launch command.
