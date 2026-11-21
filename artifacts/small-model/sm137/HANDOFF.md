# Handoff: trained smaller state for Tiny (SM137)

Design: `design.md` (pre-registered 11-21, before any GPU ticket). **Nothing is launched.** The pilot needs the main session's (or the user's) OK and a free GPU.

The pick: the six-layer Tiny (SM130)'s card and note WKV state cut to all 16 value rows x 8 of 16 key lanes per head (architecture `w64-c1-d2p1u1-v16k8`): **803 B per card and per note** at the release-candidate format, instead of 1,315 B (39% less). The model zeroes the dropped lanes, so the stored state keeps exact zeros; the fork needs a storage (codec) change only.

## State on 11-21

- CPU proofs: `check-cpu-sm137-v16k8.json` (log `cpu/check-cpu-v16k8.out`), ALL OK (see design.md section "CPU proofs").
- Cut checkpoint (local, not committed): `C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137-io-v16k8-e41/` (`master.pth`, `optimizer.pth`, `extend.json` with the sha256 pins and the lane permutations), from SM130's Instant-Only epoch 41 (sha256 af16139e28bb).
- CPU smoke rows for the GPU check: `cpu/smoke-cpu-v16k8-rows.jsonl` (users 2388, 4676).
- Zero-shot screen (17 users outside the 732 set, information only): `sm137-zero-shot-screen.json`.
- The pilot folder `sm137-io-v16k8-e42-st` is written by the launcher (`sm137_epoch.py prepare`); `sm137_epoch.py pair --chain io` already passes (== SM135's epoch 42 except the architecture and the start).

## Launch the pilot (about 1.5 GPU h)

Dry run first (no GPU, no folder written):

    DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/launch-sm137.sh; tail -30 C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/launch-sm137-dryrun.log

Launch, hidden via WMI (PowerShell):

    $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/launch-sm137.sh >> C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/launch-sm137.nohup 2>&1"'; ProcessStartupInformation=$si}

Steps (all GPU steps through `scripts/r01_queue.py`, owner `sm137`, purposes start with `Trained smaller state for Tiny (SM137): `; GPU memory cap 0.8): checks -> memory guard -> prepare + pair -> GPU scorer smoke (~5 min) -> trainer smoke (~3 min) -> zero-shot score of the cut (~10 min) -> Instant-Only lane epoch 42 (~55 min at ~179k reviews/s) -> dropped-row check -> scoring w8+s8 + w8+s8hx8 (~20 min) -> `sm137_rule.py pilot`.

## Watch

- Log: `C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/launch-sm137.log` (lines `[sm137 MM-DD HH:MM:SS] ...`); trainer `gpu/train.out`, scorer `gpu/score.out`, smokes `gpu/check.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`, rows `gpu/sm137-early-rows.jsonl`.
- Done: `grep -E '\] done \(pilot' .../gpu/launch-sm137.log` -> `done (pilot v16k8: go | pilot_v16k12 | stop)`; verdict file `gpu/verdict-pilot-v16k8`; readout `sm137-pilot-io-v16k8.{json,md}`.
- Failure: `grep -E '\] STOP:' .../gpu/launch-sm137.log`. Report, do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm137.sh*' } | Select ProcessId, CommandLine`
- Stop: `touch C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/gpu/STOP` (a running epoch snapshot-stops at the next user boundary); resume = the same launch command after removing STOP.
- Rule by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137/sm137_rule.py pilot --lanes v16k8` (exit 2 = rows missing).
- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm137 worktree while the epoch runs (SourceWatch).

## After the pilot

- `go`: prepare the full plan (design.md "Full plan"): the Instant-Only two-pass cool-down from `sm137-io-v16k8-e42-st` with SM115's / SM130's recipe (adapt `artifacts/small-model/sm135/launch-sm135.sh`'s cool-down functions and `sm135_pair.py` / `sm135_prepare.py` with `ARCH=w64-c1-d2p1u1-v16k8`; the cool-down's fresh Muon check `scripts/check_muon_fresh_decay.py` must accept the lane preset), final scoring (student w8+s8 + w8+s8hx8, SM130 final w8+s8hx8), the long-card CPU check (`users-longcard.json`, `--prior-buckets 0,32,128,300,1000`, under `memguard.ps1`), `sm137_rule.py longcard --chain io` and `decide --chain io`; then the two-head chain (cut of SM130's epoch 32: `sm137_lanes.py mask --chain th --value-lanes 16 --key-lanes 8 --stats work/stats-th-src.json --out artifacts/small-model/sm137-th-v16k8-e32`, `sm137_epoch.py prepare --chain th`, epoch 33, `early-th`, cool-down with SM91s2's targets, `decide --chain th`). About 12 GPU h.
- `pilot_v16k12`: make the milder cut and its CPU proofs, then relaunch with `LANES=v16k12`:

      cd C:/Users/jonat/workspace/wt/sm137
      PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm137; export CUDA_VISIBLE_DEVICES=
      $PY -X utf8 $D/sm137_lanes.py mask --chain io --value-lanes 16 --key-lanes 12 --stats $D/work/stats-io-src.json --out artifacts/small-model/sm137-io-v16k12-e41
      $PY -X utf8 $D/sm137_lanes.py stats --chain io --users-file $D/users-screen.json --arch w64-c1-d2p1u1-v16k12 --checkpoint artifacts/small-model/sm137-io-v16k12-e41/master.pth --value-lanes 16 --key-lanes 12 --keep-states --tag io-v16k12-cut
      SM137_TMP=C:/Users/jonat/AppData/Local/Temp $PY -X utf8 $D/sm137_lanes.py cpu-check --cut artifacts/small-model/sm137-io-v16k12-e41 --cut-stats $D/work/stats-io-v16k12-cut.json --value-lanes 16 --key-lanes 12 --out $D/check-cpu-sm137-v16k12.json
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START PYTHONDONTWRITEBYTECODE=1 $PY -X utf8 scripts/small_model_heldout.py --device cpu --pool 732 --users-file $D/users-smoke.json --count 2 --model "smoke=expertium:w64-c1-d2p1u1-v16k12:C:/Users/jonat/workspace/wt/sm137/artifacts/small-model/sm137-io-v16k12-e41/master.pth" --variants w8+s8,w8+s8hx8 --inputs e2s-query --time-limit 600 --split-existing --deleted-cards neighbour --out $D/cpu/smoke-cpu-v16k12-rows.jsonl
  then the WMI launch with `export LANES=v16k12;` added before `bash`.
- `stop`: write the result (no cool-down).
- In every case: update `pages/data/factory-curated.json` (results row with times and detail; the SM137 plan row's state), publish, `--check` SYNC 0.
