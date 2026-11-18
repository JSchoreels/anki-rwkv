# Handoff: six-layer Tiny (SM130)

Design: `design.md` (pre-registered 11-18 before any GPU ticket). Tiny (w64-c1) cut to `w64-c1-d2p1u1` (card 1, deck 2, note 1, preset 1, user 1 layers; 185,057 parameters, 6 layer steps), both chains one after the other:
- **two-head (th):** cut of the Tiny two-head model (SM106)'s epoch 29 -> recovery epochs 30-32 (SM106's epoch step) -> the show-time two-head cool-down for Tiny (SM124)'s two-pass cool-down with the show-time big model (SM91s2)'s targets;
- **Instant-Only (io):** cut of the Tiny Instant-Only model (SM101)'s epoch 38 -> epochs 39-41 -> the longer cool-down for Tiny (SM115)'s two-pass cool-down with the show-time Instant-Only big model (SM99)'s targets.

Rules (`sm130_rule.py`): early stop of a chain if a trained head is above +1.50% vs the uncut start epoch after the first recovery epoch, above +1.20% after the last; final go per chain if every trained head's 95% CI upper bound of 100 x (six-layer / Small - 1) <= +1.00% (published w8+s8; the compact w8+s4c8nhx8 rows are scored and judged too if the compact state for Tiny (SM129) went go for that model). Small = the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model (SM113).

## The running job

- Launched 22:58 Wed 11-18 from branch `sm130` at be9a4929, hidden via WMI. Windows PIDs: wrapper bash 30868 (the WMI process), launcher bash 11452 (`bash .../launch-sm130.sh`). It waits for the break to end (01:00 Thu 11-19), runs the CPU proofs (beside SM129, memory guard 5.9 GB), then waits for the final line of SM129's log (`wt/sm129/artifacts/small-model/sm129/gpu/launch-sm129.log`, at most 6 h) before its first GPU ticket.
- Main log `artifacts/small-model/sm130/gpu/launch-sm130.log` (lines `[sm130 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm130.nohup`; trainer `gpu/train.out`; scorer `gpu/score.out`; smokes `gpu/check.out`; CPU proofs `gpu/check-cpu.out` + `check-cpu-sm130.json`; `gpu/segments.jsonl` (minutes, nvidia-smi max per training segment), `gpu/tickets.jsonl`; rows `gpu/sm130-early-rows.jsonl` (constant-LR checkpoints), `gpu/sm130-rows.jsonl` (final students); per-chain verdict files `gpu/verdict-th`, `gpu/verdict-io`. Queue owner `sm130`, purposes start with `Six-layer Tiny (SM130): `.
- Folders (`wt/sm130/artifacts/small-model/`, not committed): cuts `sm130-th-cut-e29`, `sm130-io-cut-e38`; epochs `sm130-th-e30-st` ... `sm130-th-e32-st`, `sm130-io-e39-st` ... `sm130-io-e41-st` (30 and 39 prepared before the launch, the rest by the launcher); cool-downs `sm130-{th,io}-dec2ep32k-p1`, `-p2`.

## Expected timeline (central; ranges in design.md)

| step | ends about |
|---|---|
| CPU proofs (beside SM129) | 01:20 Thu 11-19 |
| SM129 done; SM106 epoch 29 reference scoring | ~03:00 Thu |
| two-head epoch 30 + early check 1 | ~04:30 Thu |
| epochs 31-32 + early check 2 | ~07:15 Thu |
| two-head cool-down passes 1 + 2 | ~16:00 Thu |
| **two-head result** (`rule decide (two-head)`) | **~16:30 Thu 19 Nov** (13:30 Thu - 02:30 Fri) |
| Instant-Only epochs 39-41 (41 pauses at the 20:00 break) + early checks | ~02:15 Fri 20 Nov |
| **Instant-Only result**, CPU timing, `done (...)` | **~12:00 Fri 20 Nov** (07:00 Fri - ~10:00 Sat 21 Nov) |

About 27 GPU h (21-43); ~1.7 h per chain if the first early stop fires, ~4.4 h if the second does.

## Log lines to watch

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/launch-sm130.log` -> `done (verdict two-head: <...>; Instant-Only: <...>)`
- Failure: `grep -E '\] STOP:' C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/launch-sm130.log`
- Per chain: `grep -E 'rule early|EARLY STOP|rule decide|complete: .*reviews/s' .../gpu/launch-sm130.log` (early checks, chain verdicts, reviews/s of every training folder).
- Start: `[sm130-cut] quick OK`, `[sm130-rule] selftest ... -> OK`, `CPU proofs ALL OK`, `SM129 finished: ...`.
- `STOP: pass N ended at lr` = a broken schedule; `STOP: ... is not the reference pass 1 ...` = a pair proof failed: report, do not relaunch blindly.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm130.sh*' } | Select ProcessId, CommandLine`
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/launch-sm130.log`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Rules by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/sm130_rule.py early --chain th --epoch 30` / `decide --chain th` / `decide --chain io` (exit 2 = rows missing).
- Stop: `touch C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/STOP` (between tickets; a running training folder gets a snapshot stop at the next user boundary). Remove it before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; finished steps are skipped, a folder resumes from its last snapshot; `CHAINS="io"` runs one chain only), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/launch-sm130.sh >> C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130/gpu/launch-sm130.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/ expertium/ rwkv/` of `wt/sm130` while it runs (SourceWatch stops the training). Read-only inputs that must not be cleaned up: see design.md, Risks 5.

## After each chain's verdict and after `done` (a fresh Sonnet medium agent per step)

1. Branch sm130 (`cd C:/Users/jonat/workspace/wt/sm130; git fetch origin; git merge origin/sm130`): `git add -f` the decision files (`sm130-decision-{th,io}.{md,json}`, `sm130-early-*.{md,json}`), `check-cpu-sm130.json`, `sm130-timing.json`, `gpu/*.jsonl`, the logs (`launch-sm130.log`, `launch-sm130.nohup`, `score.out`, `check.out`, `check-cpu.out`, `prepare.out`, `trim.out`, `timing.out`, `quick.out`), `gpu/check-*.json`, `gpu/verdict-*`, a `gpu/train-tail.out` (last ~200 lines of train.out) and a short `RESULT.md` (verdict per chain, the tables, parameters, scorer `weight_bytes` from the final rows, state bytes (same as Tiny: 1,055 / 1,567 B fork default, 803 / 1,315 B compact), reviews/s per epoch and pass from `launch-sm130.log` `complete:` lines with the 8k / 2k 10-epoch ETA, the CPU timing). Never `.pth`, `.npz`, `work/`. Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push.
2. Main session: close the SM130 plan row (`state` done, `when` with end time and verdicts, `ended_at`) and add one results row per chain (or one with both), then publish + `--check` (SYNC 0). Template:

```json
{
 "date": "<MM-DD>",
 "kind": "gpu-train",
 "id": "SM130",
 "track": "small model",
 "name": "Six-layer Tiny (SM130): Tiny with the deck / preset / user streams cut to 2 / 1 / 1 layers (the context-stream pruning (SM15) cut; 185,057 parameters, 6 layer steps instead of 12), three recovery epochs, then the two-pass cool-down of the show-time two-head cool-down for Tiny (SM124) (two-head) / the longer cool-down for Tiny (SM115) (Instant-Only)",
 "eval": "Leak-free w8+s8, 722 users of the 732 set, show-time inputs; % = 100 &times; (six-layer / Small &minus; 1) of the review-weighted LogLoss, paired user bootstrap 95% CI. Go = every trained head&rsquo;s CI upper bound &le; +1.00% vs the show-time two-head cool-down for Small (SM118) (two-head) or the longer Small Instant-Only model (SM113) (Instant-Only). Early stops: +1.50% / +1.20% vs the uncut epoch after the first / last recovery epoch",
 "fork": "Not used: the fork's engine has the layer counts compiled in (MODULE_LAYERS); held for a later release decision",
 "takeaway": "<strong><two-head verdict></strong> two-head: curve <c>% [<lo>, <hi>], instant <i>% [...] vs Small (<x>% / <y>% vs the unmodified Tiny); <strong><io verdict></strong> Instant-Only: <io>% [...]. 41% fewer parameters (185,057), half the layer steps; training ran at <r> reviews/s in the cool-down (Tiny: 21-29k). CPU review step (PyTorch, 1 thread): <t6> vs <tt> ms.",
 "inputs": "show-time",
 "started_at": "<first GPU ticket>",
 "ended_at": "<done line>",
 "gpu_minutes": <segments.jsonl + tickets.jsonl minutes>,
 "tickets": <count>,
 "detail": {
  "verdict": "<verdicts>; branch sm130 <commit> (sm130-decision-th.md, sm130-decision-io.md, RESULT.md)",
  "table": {
   "head": ["Show-time", "Curve %", "Curve 95% CI", "Instant %", "Instant 95% CI", "Instant better / worse"],
   "rows": [
    ["six-layer two-head minus the show-time two-head cool-down for Small (SM118) (the rule)", "...", "...", "...", "...", "..."],
    ["six-layer two-head minus the show-time two-head cool-down for Tiny (SM124)", "...", "...", "...", "...", "..."],
    ["six-layer two-head, compact state, minus SM118 (if scored)", "...", "...", "...", "...", "..."],
    ["six-layer Instant-Only minus the longer Small Instant-Only model (SM113) (the rule)", "", "", "...", "...", "..."],
    ["six-layer Instant-Only minus the longer cool-down for Tiny (SM115)", "", "", "...", "...", "..."],
    ["early check 1 / 2 (six-layer epoch minus the uncut epoch), two-head", "...", "...", "...", "...", ""],
    ["early check 1 / 2, Instant-Only", "", "", "...", "...", ""]
   ]
  }
 }
}
```

3. Report to the user in % (never "points"), every model as "plain name (ID)": verdicts with CIs, the early checks, reviews/s per run with the 8k / 2k 10-epoch ETA, parameters / bytes / CPU timing, and CPU use alongside reviews/s where known.
