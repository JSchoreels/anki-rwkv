# Grade-order sanity (SM33): handoff

Branch `sm33` (cut from `sm80`, merges `sm33p`), worktree `C:/Users/jonat/workspace/wt/sm33`. Started 2026-11-01 ~10:30.

## What it asks

The four answer buttons must satisfy interval(Again) <= interval(Hard) <= interval(Good) <= interval(Easy). The fork computes them from four what-if answer rows (`rslib/src/rwkv/engine.rs` `answer_heads` / `simulated_answer_input`: the review row with the rating replaced, from the committed state, nothing committed) and, by default (`rwkv_review_enforce_grade_order` = true), hides inversions with PAVA: at each point of its interval search grid the four margins R_g(t) - target are pooled to be non-decreasing in g (`unrounded_intervals_for_answer_curves` / `pava_non_decreasing`). So the buttons never show an inversion, but a pooled button shows a blend of two grades' curves instead of its own.

## Stage 1 (measure): `scripts/sm33_grade_order.py`

- Probe rows: four copies of a sampled review row (rating 1..4, skip = True like query rows, no label) inserted right before the real row, in the full-history scorer layout (the 732 pool, leak-free inputs). Probe-led sequences (a card's first review) get the token-shift fix `fix_leading_shift`. CPU check PASS: non-probe rows unchanged (<= 4e-6), the probe with the real rating = the real row.
- Per sampled review: unrounded intervals at 0.80 / 0.90 / 0.95 from each grade's curve, curve values at 1 / 7 / 30 / 180 days, and the fork's PAVA display at 0.90 (its grid, ceil, 1..36,500 days).
- Launcher `launch-sm33-stage1.sh` (one GPU ticket: CUDA check then 723 users x 3,000 reviews), then CPU `report` -> `sm33-stage1.json` (copied here). CPU preview on 8 short users: displayed order broken on ~38% of reviews at 0.90; PAVA changes Good's displayed interval on ~26%.

## Stage 2 (training fix), pre-registered, runs only if stage 1 passes the gate

- Gate (`launch-sm33-stage2.sh` `gate`): displayed order broken on >= 5% of reviews at 0.90, or PAVA changes Good's displayed interval on >= 5%.
- Option `"grade_order": {"weight": 0.1, "density": 0.04, "horizons_days": [1, 3, 7, 15, 30, 60, 120, 240, 480, 960, 1920, 3650], "seed": 33}`: `scripts/gpu_r01_samples.add_grade_probes` puts the four probe rows before a seeded 4% of the reviews of every training window (about +8% rows); `expertium/model/variant_model.grade_order_penalty` adds weight x mean relu(logit R_g(t) - logit R_{g+1}(t)) over adjacent probe pairs and the 12 horizons (zero when ordered). Absent = nothing runs.
- CPU proof `scripts/check_grade_order.py` -> `check-cpu-grade-order.json`: PASS, 17 checks (option absent and weight 0 bitwise equal to the code before for samples, loss, loss tensor, gradients; probe rows leave non-probe inputs bitwise and outputs within 3e-6, eval-mode loss within 2e-7 relative; penalty = float64 NumPy recompute; exactly 0 with zero gradient when ordered; TorchScript compiles). Train-mode dropout masks shift with the extra rows, so the arm is not bitwise the reference even at weight 0 (disclosed).
- Arm: SM72s's own epoch 21 + the quarter-epoch 32k Muon cool-down `artifacts/small-model/sm33-arm-e21-dec-q32k-muon` (prepared, CPU checks PASS; `sm33_pair.py`: identical to the reference `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon` except `grade_order` and paths: 1,991 users, 145,578,259 reviews, same start weights and update offset 463,638).
- Rule (`sm33_decide.py`): go = both heads <= +0.03 points (722 users, points of % of the big model (1 leak-free epoch)) and the review-weighted displayed break share at 0.90 drops with the paired user bootstrap CI clear of zero; cost = drops but a head > +0.03; else no effect. Output `sm33-decision.{json,md}`.

## What runs (detached, WMI hidden)

1. `launch-sm33-stage1.sh` (log `gpu/launch-stage1.log`, output `gpu/stage1.out`), queue ticket purpose "grade-order sanity (SM33): stage 1, CUDA probe check + what-if intervals of the released small model on the 732 pool (723 users, 3,000 reviews each)", ~40 min.
2. `launch-sm33-chain.sh` (log `gpu/launch-sm33-chain.log`): waits for stage 1, then `launch-sm33-stage2.sh` (log `gpu/launch-sm33-stage2.log`). Ticket purposes in order, each prefixed "grade-order sanity (SM33): ":
   - "trainer smoke of the cool-down with the grade-order hinge (2 users x 512 reviews)" (12 min)
   - "arm quarter-epoch 32k Muon cool-down from SM72s epoch 21 with the grade-order hinge (w64, the reference's options and order) (segment to the next window or the end)" (~170 min in all; snapshot stop before 16:00 Sunday, resumes 01:00)
   - "scoring sm33-arm-e21-q32k on the 732 pool minus 9 never-fit (723) (pass 1)" (12 min), "... (pass 2)" if needed, "scoring sm33-arm-e21-q32k on the 3 long users of the 732 pool (cap 0.92)" (6 min)
   - "stage 2, what-if intervals of the arm on the 732 pool (723 users, 3,000 reviews each)" (40 min)
   - "stage 2, what-if intervals of the reference on the 732 pool (723 users, 3,000 reviews each)" (40 min)
   - CPU rule -> `sm33-decision.{json,md}`; log line `rule exit 0: [sm33-rule] ...`.

Expected end: the cool-down is behind other agents' tickets (frozen-model grade side head (SM87) 122 min, ST01a 25 min), so it likely starts ~14:30, pauses at ~15:55 for the Sunday break and resumes at 01:00; scoring and both measures follow, end ~Monday 04:00-05:00.

## Follow-up (fresh agent)

```bash
cd C:/Users/jonat/workspace/wt/sm33 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm33/gpu/launch-stage1.log artifacts/small-model/sm33/gpu/launch-sm33-chain.log artifacts/small-model/sm33/gpu/launch-sm33-stage2.log
cat artifacts/small-model/sm33/sm33-decision.md     # when stage 2 is done
```

Write up stage 1 from `sm33-stage1.json` (break rates per pair at 0.80/0.90/0.95, displayed vs raw, the PAVA effect on Good, breakdowns by first review / real rating / elapsed / Good interval / user history quartile, the per-user distribution) and stage 2 from `sm33-decision.md`; commit `gpu/*.log`, `gpu/*.out`, the JSON results and `gpu/sm33-neighbour.jsonl`; push.

## Relaunch (resumes: finished steps are skipped, an interrupted cool-down resumes from its snapshot)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/launch-sm33-stage2.sh >> C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/gpu/launch-sm33-stage2.nohup 2>&1"'; ProcessStartupInformation=$si}
```

(If stage 1 itself died, relaunch `launch-sm33-stage1.sh` the same way first, then the chain.) To stop between tickets: `touch C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/gpu/STOP` (remove it before relaunching). After a Claude app restart, check the launchers are alive; if a log has been silent for more than 10 minutes outside a ticket, relaunch. Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs.
