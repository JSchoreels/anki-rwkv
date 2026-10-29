# HANDOFF: tiny architecture additions (SM77) stage 1

Branch `sm77` (worktree `C:/Users/jonat/workspace/wt/sm77`), cut from `origin/sm72s` (6a074a0b). Design:
`artifacts/small-model/sm77/DESIGN.md` on main. Code and CPU checks only: **not launched, no GPU used**. The user
launches it after the better teacher's (SM65) stage 1 (Friday night or Saturday).

## What the arm is

The leak-free small model (SM72s) from its constant-LR epoch K (default 20) with the preset `w64-sm77`
(`expertium/variants.py`, `expertium/model/sm77.py`), all new parameters zero (function preserving):
- (A) key extras, every time mixer: removal key `k_scale * normalize(k * (1 + sm77_k_remove))`, insertion key
  `k_scale * normalize(k) * (a + sm77_k_insert * (1 - a)) + sm77_k_bias`, `r + sm77_r_bias`; 4 x 64 per layer, 3,328 values.
- (B) learned initial state, card (2 layers) and note (1 layer): a never-seen entity starts from `sm77_init_wkv`
  [4, 16, 16] instead of zeros, and its first-review token shift is `x + sm77_init_shift` (time and channel mixers;
  card layer 1 has no channel mixer); 3,392 values. No extra stored state.
- Flags off (`w64`): today's classes, parameters and kernels, bit for bit.

## Steps of `artifacts/small-model/sm77/launch-sm77.sh`

0. Waits until the last line of `wt/sm65/artifacts/small-model/sm65/gpu/launch-sm65-teacher.log` is `] done` or a
   failure / STOP line (checked every 2 min, logged hourly); `touch artifacts/small-model/sm77/gpu/NO-WAIT` skips it.
1. CPU: `sm77_warm.py extend` (SM72s epoch K master + AdamW state re-indexed, new parameters zero) ->
   `artifacts/small-model/sm77-ext-eK`; `sm77_warm.py epoch` -> `sm77-arm-eK+1` (SM72s's epoch-(K+1) protocol rebuilt:
   same user order, seeds, options, trims; start = the extended files); for K = 25 also `sm77-ref-e26` (flags off).
   Pair check: arm protocol == SM72s epoch K+1 protocol except the start files and paths.
1. Ticket 0 (~15 min): `sm77_warm.py gpu-check --ext <ext>` (the new dS0 backward kernel vs CPU autograd, generic
   and row tiles, float32 and bf16; S0 = 0 forward and gradients bitwise = the SM21 path; the scorer's long kernel
   from S0; trainer-path function preservation on GPU, 2 users x 512 reviews, bf16, row tiles) then the trainer smoke
   of the arm (`expertium_decay.py smoke`). Writes `gpu/check-gpu-sm77.json`, `gpu/gpu-check-ok`.
2. Arm epoch K+1 (~110-120 min, started only if it fits before the next window; snapshot-resumable).
3. Reference epoch: SM72s's own epoch K+1 when it exists (K = 20..24, free); else trained here.
4. Quarter-epoch 32k Muon cool-downs (SM72s's cool-down options, fraction 0.25, ~145M reviews, seed-1 users, fresh
   Muon), arm then reference, each ~125 min, segmented at windows; label-window trims, Muon fresh-state CPU check and
   a pair check (identical except start weights / architecture) first.
5. Scoring: 723 fit users of the 732 set + the 3 long users at the 0.92 cap, leak-free inputs, w8+s8, labels
   `sm77-arm-eK+1-q32k/w8+s8` and `sm77-ref-eK+1-q32k/w8+s8` -> `gpu/sm77-neighbour.jsonl`.
6. Rule: `scripts/sm77_rule.py decide` -> `sm77-decision.{json,md}`, one log line GO / STOP / GREY, then `done`.

GPU: K = 20: ~7.0 h (15 min + 120 + 2 x 125 + 2 x 18). K = 25: ~9.0 h (the reference epoch is trained too).

## Launch (PowerShell, hidden; default start epoch 20)

```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/launch-sm77.sh >> C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/launch-sm77.nohup 2>&1"'; ProcessStartupInformation=$si}
```

From epoch 25 instead (exists from Fri ~06:00; +2 h GPU for the reference epoch): replace `bash C:/...` with
`START_EPOCH=25 bash C:/...` inside the command line.

- Log `artifacts/small-model/sm77/gpu/launch-sm77.log`; `prepare.out`, `check.out`, `train.out`, `score.out` next to it.
- Stop: `touch artifacts/small-model/sm77/gpu/STOP` (a running segment is snapshot-stopped).
- Relaunch after a session restart is safe: done steps are skipped, an interrupted folder resumes from its snapshot,
  scoring skips scored users. Check the launcher is alive after a Claude app update.
- Dry run: `DRYRUN=1 bash artifacts/small-model/sm77/launch-sm77.sh` (log `gpu/launch-sm77-dryrun.log`; prepares and
  deletes scratch folders `sm77-dryrun-*`).

## Follow-up commands

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm77
tail -20 $WT/artifacts/small-model/sm77/gpu/launch-sm77.log
$PY -X utf8 $WT/scripts/sm77_rule.py decide --rows $WT/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl \
    --arm-label sm77-arm-e21-q32k/w8+s8 --ref-label sm77-ref-e21-q32k/w8+s8      # e26 labels with START_EPOCH=25
cat $WT/artifacts/small-model/sm77/sm77-decision.md
```
- GO: split the arm (presets `w64-sm77-keys`, `w64-sm77-init`, same launcher with ARCH changed) or go to a full
  1-epoch cool-down release candidate; the fork runtime needs the changes listed in DESIGN.md section 4 first.
- Grey: report and ask. STOP: close SM77 (A)+(B); the two-step delta (literature item 4) stays shelved.

## Checks done (CPU, 10-29; no GPU)

- `sm77_warm.py cpu-check` -> `artifacts/small-model/sm77/check-cpu-sm77.json`, ALL OK (3 min):
  - T0 flags off, trainer path (carry, 2 lanes x 128, users 4745 / 9037, 8 updates): loss, outputs and all gradients
    bitwise equal to the `wt/sm72s` tree (run read-only, no bytecode written).
  - S0 flags off, scorer (fp32, w8+s8, leak-free, users 2388 / 4676 of the 732 set): metrics and per-review dumps
    bitwise equal to the `wt/sm72s` tree.
  - T1 / T1j zero-init `w64-sm77` vs `w64` epoch 20, trainer path, eager and TorchScript: loss and outputs bitwise
    equal, old gradients within 2.8e-6 relative (summation order), every new tensor gets a gradient.
  - S1 zero-init scorer: metrics and per-review dumps bitwise equal; weight bytes 410,459 -> 437,339 (w8).
- `check_state_carry_lanes.py --architecture w64-sm77`: perturbed new parameters (normal 0.05,
  `check-state-carry-lanes-sm77-perturbed.json`): D (carry vs full history) every row within 3.4e-5; zero init
  (`...-zero.json`): L, D, F, G pass, B differs only in the init-state gradients (by design: lane-start entities with
  history in an earlier lane keep the zero start in training; confirmed tensor by tensor).
- `sm77_warm.py check-protocol --epochs 2,11,21,22`: rebuilt protocols equal to disk.
- `sm77_rule.py decide` on SM72s's rows (cool-down e20 as "arm", e15 as "reference") reproduces the SM72s summary
  (0.30019 / 0.30038, +0.41% / +0.47% vs the big model (1 leak-free epoch)).
- Dry run OK (`gpu/launch-sm77-dryrun.log`): extend (354 AdamW states moved), arm epoch 21 prepared and pair-checked
  against SM72s's epoch-21 protocol, reference cool-down prepared (1,991 users, 145,578,259 reviews).
