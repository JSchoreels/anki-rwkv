# HANDOFF: better teacher (SM65) stage 1

Branch `sm65` (worktree `C:/Users/jonat/workspace/wt/sm65`), cut from `sm72` (9dfa232e). Design:
`artifacts/small-model/sm65/DESIGN.md` on main (3f2316e7, 4be454e3).

## State 2026-10-29 15:13: launcher started hidden, waiting for the leak-free small model (SM72s)

The launcher waits until the last line of `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.log`
ends with `] done` or holds a `] STOP` line (checked every 2 min, logged every hour). It then runs:

0. GPU smoke via the queue (`sm72_big_epoch.py smoke`: trainer smoke mode on a scratch copy, 2 users x 512 reviews,
   ~5 min), once; `gpu/smoke-ok` marks success, a failure stops the launcher.
1. Training: `env RWKV_DELETED_CARDS=neighbour sm72_big_epoch.py run --out artifacts/small-model/sm65-big-moreusers`,
   in queue tickets cut at the quiet windows. The watcher asks for a snapshot stop when a window starts or is 2 min
   away. The next ticket resumes from the snapshot after the window (`expertium_decay.resume_point`), the same
   mechanism as the big model's leak-free epoch (SM72 step 1). A segment starts only with >= 30 min before a window.
2. Scoring: `small_model_heldout.py --split-existing --deleted-cards neighbour`, label `big-moreusers/w8+s8`, 732 pool
   (723 fit users), then the 3 long users at the 0.92 cap -> `artifacts/small-model/sm65/gpu/bigmore-neighbour.jsonl`.
3. Rule (CPU): `scripts/sm65_rule.py decide` -> `artifacts/small-model/sm65/sm65-decision.{json,md}` and one log line
   with GO / STOP / GREY; then `done`. **No export**: stage 2 is started by hand after the decision.

Queue purposes all start with `small-model: better teacher (SM65) stage 1:` (the factory page's SM65 plan row matches "SM65").

## Run

- Run folder: `artifacts/small-model/sm65-big-moreusers` (decay.json committed). 3,969 users (train_users of
  `anki-rwkv/artifacts/full-run/run-8k-20261013/run.json` with id < 5000), 285,886,387 reviews (285,884,532 after 14
  unlabelled tails were trimmed). Start: `wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth`
  (sha256 ac7f0d8a..., read only). Recipe as SM72 step 1: 2 strided lanes x 8,192 carried state, fresh AdamW, warmup
  200 updates, 3.5e-4 -> 0 one_minus_sin, cap 0.8, carry table above 768 MiB in host memory (a0a953ea), shuffle seed 1.
  `decay.json` "users_check": 0 overlap with the run's validation (1,500), quick validation, final test (500), preserved
  final test, and the 732 (incl. never fit) / 464 / 3-long lists.
- New checkpoint: `artifacts/small-model/sm65-big-moreusers/expertium/big-moreusers/master.pth`.
- Logs: `artifacts/small-model/sm65/gpu/launch-sm65-teacher.log` (launcher), `launch-sm65-teacher.nohup`, `smoke.out`,
  `train.out`, `score.out`. Stop: `touch artifacts/small-model/sm65/gpu/STOP` (a running segment snapshots first).
- Launch command (PowerShell, hidden):
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm65/artifacts/small-model/sm65/launch-sm65-teacher.sh >> C:/Users/jonat/workspace/wt/sm65/artifacts/small-model/sm65/gpu/launch-sm65-teacher.nohup 2>&1"'; ProcessStartupInformation=$si}
  ```
  Relaunching after a session restart is safe: the smoke is skipped once `smoke-ok` exists, training resumes from the
  last snapshot, scoring skips users already in the rows file.
- Dry run: `DRYRUN=1 bash artifacts/small-model/sm65/launch-sm65-teacher.sh` (log `gpu/launch-sm65-teacher-dryrun.log`).

## ETAs

Training ~416 min at 11,460 reviews/s (SM72 step 1's rate) + smoke ~5 min + scoring ~15-18 min.
- SM72s done ~13:00 Fri 10-30: the epoch ends ~20:15 before the break, decision ~20:40 Fri.
- SM72s done ~15:00: ~5.7 h until the snapshot stop at ~20:58 (~235M reviews), ~50M left after 02:00 (~75 min),
  decision ~03:45 Sat 10-31.

## Checks done (CPU, 10-29)

- Without the new flags, `prepare` writes SM72 step 1's decay.json (every key equal, minus timestamps and the later
  trim records) and its protocol.json (equal to protocol.untrimmed.json except the output folder name).
- `sm72_big_epoch.py check --init <start> --report artifacts/small-model/sm65/check-init-load-sm65.json`: bit-exact load,
  same keys / shapes / dtypes as the shipped file, scorer rows identical (users 4676, 516, fp32 and w8+s8): PASS.
- `check_state_carry_lanes.py --architecture fork --checkpoint <start> --skip-base`: ALL OK
  (`check-state-carry-lanes-fork-init.json`, 7 min).
- A held-out user in the list is refused (tested with user 4 added).
- `sm65_rule.py decide` on the reference rows relabelled gives +0.00% / +0.00% (stop) and on the shipped big model's
  rows +0.23% [+0.07, +0.37] / +0.16% [+0.14, +0.18] on 722 users, as in the SM72s summary.
- No GPU smoke yet: at 15:00 the SM72s epoch ticket held 14.5 of 16.3 GB, so the smoke is the launcher's first ticket.

## Decision rule (pre-registered, `scripts/sm65_rule.py`)

722 users (the 732 fit users in the reference rows; reference 0.29897 / 0.27576), paired user bootstrap 2,000, seed 0.
Go if curve <= -0.20% or instant <= -0.15%, that head's CI clear of zero, the other head <= +0.03%. Stop if both heads
> -0.10%. Grey otherwise: report and ask.

## Follow-up

- **Stage 2 (only on GO, or a user "go" in the grey zone): teacher targets, ~4 h GPU.** New folder, never overwrite
  an existing one. Tickets via the queue, cut before a window with `--time-limit` (copy step 3 of
  `wt/sm72/artifacts/small-model/sm72/launch-sm72-big.sh`):
  ```
  CK=C:/Users/jonat/workspace/wt/sm65/artifacts/small-model/sm65-big-moreusers/expertium/big-moreusers/master.pth
  R8N=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-moreusers-final8k-neighbour
  RUN8K=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json
  env RWKV_DELETED_CARDS=neighbour $PY -X utf8 C:/Users/jonat/workspace/wt/sm65/scripts/sm11_replay_export.py export --out $R8N --run $RUN8K --teacher $CK --device cuda --time-limit <seconds>
  env RWKV_DELETED_CARDS=neighbour $PY -X utf8 .../sm11_replay_export.py status --out $R8N --run $RUN8K
  env RWKV_DELETED_CARDS=neighbour $PY -X utf8 .../sm11_replay_export.py verify --out $R8N --run $RUN8K --teacher $CK --sample 8   # CPU
  ```
- **Stage 3: paired cool-down of the leak-free small model (SM72s), ~8.5 h GPU.** The SM72s cool-down (`prepare_decay`
  in `wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.sh`: `expertium_decay.py prepare --source-run <SM72s run>
  --epoch K --fraction 1.0 --shape one_minus_sin --evaluate none --seed 1 --fresh-optimizer --train-tier final
  --extra-options "$EXTRA"`) from the same start epoch as the cool-down SM72s already has (25, or 20 if that one
  loses), with only `distillation.teacher` -> the new master.pth and `distillation.replay` -> `$R8N` in `$EXTRA`
  (`DIST` line 75 of that launcher). Label-window check against the new replay, then the 732 score (723 + 3 long)
  and the pair on 722 users against the existing cool-down from that epoch (the control). Not written yet.
