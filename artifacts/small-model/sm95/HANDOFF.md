# The grade order by construction (SM95): handoff

Branch `sm95`, worktree `C:/Users/jonat/workspace/wt/sm95`, cut from `sm33` (`e8f56632`, the grade-order sanity check (SM33)'s tip). Main has none of the small-model code (every small-model branch forks from `eb429978`), and `sm33` carries the probe-row machinery, its order tool and the exact code that passed the pair check against the reference, so it is the base that keeps the three-way comparison exact. Started 2026-11-07 ~16:15. Do not merge into main.

## Question

The four buttons must satisfy interval(Again) <= Hard <= Good <= Easy at any desired retention, by construction, without an accuracy cost. Today each button's curve comes from its own what-if row, and the curves are inverted on 26.8% of reviews at 0.90 (18.6% displayed). The fork hides that with PAVA, which changes Good on 19%. The soft hinge tried by the grade-order sanity check (SM33) cut displayed breaks to 0.9%, but at +0.14 points on the curve head (verdict cost).

## Design (chosen: an in-model rectifier, candidate A)

- **Operator** (`expertium/model/grade_rectify.py`): for one review and one time t, the four what-if values R_1..R_4(t) go through unweighted isotonic regression in max-min form, `out_i = max_{j<=i} min_{k>=i} mean(x_j..x_k)`. Rows that are already ordered are returned bit for bit. The model's curve for grade g is R'_g(t) = out_g.
- **Proof of exact order, in floating point too.** Every mean m(j, k) is computed once per row. out_{i+1} takes its max over a superset of j and its min over a subset of k compared with out_i. So out_i <= out_{i+1} holds for any rounded m table, because min and max are exact operations.
- **Each R'_g is still a forgetting curve.** Means, minima and maxima of non-increasing functions of t are non-increasing, and the values stay within [min R_g, max R_g]. So the intervals are ordered at every target retention: interval_g(r) = inf{t : R'_g(t) <= r}, and R'_g <= R'_{g+1} pointwise.
- **Relation to the fork's PAVA.** With one target retention for all buttons, this is exactly the fork's PAVA on the margins R_g(t) - r. The fork's PAVA run on the rectified curves is the identity, which the check confirms bitwise.
- **Scored probability = the scheduler's.** A review's curve is its own grade's pooled value. That pooled curve is what the four buttons (the scheduler) use. The real row itself fills its own grade's slot, so it keeps its own dropout mask and gradient path. The other three slots are what-if rows: the real row with the rating replaced, skip = True (no state commit, like `answer_heads`), has_label = False, label time copied. They sit right before the real row as query row, the three probes in grade order, then the real row.
- **Training** (`"grade_rectify": {"density": 0.08, "seed": 95}`). On a seeded 8% of the labelled reviews, the curve loss (and KD target mix) is computed on the rectified value, the scored quantity. All other rows are unchanged. Why 8%: the outside research used the same density; SM33's +8% rows fit the 16 GB card; here it is about +11% rows (3 probes on ~53% of rows x 0.08). Gradients reach the probe rows only when they pool with the real row.
- **Scoring** (`small_model_heldout.py --grade-rectify`). The three probes are inserted before every labelled review, about 2.5x rows, and the probed rows' curve is rectified. The instant head and the set of scored reviews do not change.
- **Rejected designs.**
  - (B) A joint monotone head, e.g. shared mixture weights and decay with S_g non-decreasing in g, which is exact at every t. It needs one rating-free representation, i.e. the query row. First reviews have no query row, the rating would no longer enter the stack, and the model would need retraining from scratch. That costs too much accuracy risk and cannot be paired with the SM72s warm start.
  - Learned power-mean pooling (outside research). It needs a new operator in the fork; classic pooling is already the fork's.
  - Raising density towards 1. That would make training cost about 2.5x rows.
- **Fork change needed (small).**
  - Buttons: none. `answer_heads` already runs the 4 what-if rows, and `pava_non_decreasing` with one target retention is the rectifier. It must stay on: `rwkv_review_enforce_grade_order` = true becomes part of the model definition.
  - Stored card curve (`curves.insert` after an answer). It should become the pressed grade's rectified curve: keep the 4 what-if curves from the show-time `answer_heads` (or run the 3 other what-if rows at commit from the pre-review state) and evaluate PAVA at t in `predict_curve`. That is 4 x 9 floats per card instead of 9. In bulk replay, this is needed only at each card's last review.
  - Per-button target retentions: margins of rectified curves can still need pooling, and the existing code handles that.
  - Duration: the fork's what-if rows carry no answer duration (query input), while training and scoring use the real row's. That gap already exists and is separate (answer-time head / show-time inputs).
- **Costs.** Training: about +11% rows, so roughly -10% reviews/s (pending, measured by the cool-down). Scoring: about 2.5x rows per rectified scoring pass. The order measure is unchanged: the arm's raw what-if curves are measured once, then rectified offline on the CPU (`scripts/sm95_rectify.py derive`, about 1.4 s per user per mode).

## CPU proof (stage 0): PASS, 21 checks

`scripts/check_grade_rectify.py` writes `check-cpu-grade-rectify.{json,out}`. Run at `1b31fb02`. The base for "the code before" is `e8f56632`.

- **Option absent is bitwise equal to the code before**:
  - training samples;
  - the scorer's leak-free full-history batch (user 4);
  - loss, loss tensor, p_curve and every gradient;
  - `grade_rectify` False, and True on a batch without probes;
  - `fix_leading_shift(probes=3)` changes nothing without probes.
- **Probe rows**:
  - training layout (density 0.3: 159 probed reviews) and scorer layout (every labelled review: 1,594);
  - non-probe inputs bitwise, outputs within 4e-6;
  - slot g holds grade g;
  - probe-led sequences are fixed.
- **Rectified p_curve** equals a float64 stack-PAVA recompute within 1.3e-7. 108 of 159 probed reviews were raw-broken. Unprobed and ordered rows keep the raw curve bitwise, and the instant head is bitwise. Train-mode gradients are finite and different, and the gradient reaches the pooled probe rows only.
- **Order exact**: zero rectified breaks at 242 horizons (1 s .. 100 years, plus the label time), in float32 and float64. This holds for the trained weights (raw curves broken on 95.5% of reviews somewhere on that range) and for 3 random initialisations. The rectified curves are non-increasing in t (max increase 0.0).
- **Adversarial inputs**: 1,000,000 4-vectors (ties, reversals, extremes, near-ties) give zero breaks in float32 and float64, match the stack PAVA within 2.2e-16, and are the identity on ordered rows bitwise.
- **Intervals**: the rectified intervals are ordered at 12 target retentions from 0.50 to 0.99, against 29.5% raw breaks at 0.90. The fork's display is bitwise identical with and without its PAVA, and the fork's PAVA on the raw curves matches the rectified display (1.9e-14 relative).
- **TorchScript** compiles with the option, and its loss equals the eager loss.
- **Preview** (one user, fp32, not a gate): curve LogLoss 0.39219 raw vs 0.39179 rectified; 17.6% of reviews change.

## Stage 1 (GPU), pre-registered rule (`sm95_decide.py`, written before any GPU run)

Pair protocol as SM33.

- **Arm**: the leak-free small model (SM72s) epoch 21 plus the quarter-epoch 32k Muon cool-down with `grade_rectify` (`artifacts/small-model/sm95-arm-e21-dec-q32k-muon`). `sm95_pair.py` checks it is identical to the reference except `grade_rectify` and paths.
- **Reference**: `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon`, read only, already scored raw.
- **Third comparator**: the SM33 hinge arm (raw).
- **Accuracy**: leak-free w8+s8, review-weighted LogLoss on the 722 users of the 732 set (minus users missing in any of the four labels; at least 700 required). Points are % of the big model (1 leak-free epoch) LogLoss (curve 0.298966, instant 0.275757). Paired user bootstrap, 2,000 resamples, seed 0.
- **Order**: the SM33 tool on the same 722 x 3,000 sampled reviews (seed 33,000 + user, so identical to SM33's). Read-outs: displayed at 0.90 before PAVA, raw, and Good changed by the fork's PAVA (fork grid with vs without PAVA). Also any pair at 0.80 and 0.95.
- **Scored probability**: the arm is scored rectified, because that is the curve the scheduler uses (the four buttons pooled). The reference is scored both ways: raw (today's number, already on disk) and rectified (= today's model behind the fork's PAVA).
- **go**: zero breaks of the arm's rectified curves on every sampled review (raw and displayed, at 0.80 / 0.90 / 0.95, and zero Good changes by the fork's PAVA) AND both heads (arm rectified - reference raw) <= +0.05 points (point estimates; CIs reported).
- **cost**: order zero but a head above +0.05.
- **fail**: any break (a bug).
- **incomplete**: fewer than 700 users.
- **Reported, not gating**:
  - arm rectified vs reference rectified (the training effect, both as the scheduler sees them);
  - reference rectified vs reference raw (the cheap no-training arm: rectifying today's model);
  - arm raw vs reference raw;
  - SM33 hinge arm vs reference raw;
  - all of them vs the big model.

## What runs

Launcher `launch-sm95.sh`, log `gpu/launch-sm95.log`, detached via WMI (hidden). It was launched 2026-11-07 16:44: WMI wrapper bash pid 41936, launcher bash pid 57072 (`bash .../launch-sm95.sh`); its first log line says "started: 1b31fb02". A dry run (`gpu/launch-sm95-dryrun.log`) went through every ticket. Queue owner `sm95`. Each ticket purpose is prefixed "the grade order by construction (SM95): ".

1. CPU: preconditions, prepare the arm cool-down, pair check (done 11-07 with `PREPARE_ONLY=1`).
2. Wait until `wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57-stage4.log` has a line containing `] done`, polling every 5 min. `touch gpu/GO-NOW` skips the wait if SM57 died.
3. "GPU smoke 1/2: trainer smoke of the rectified cool-down (2 users x 512 reviews)", 12 min.
4. "GPU smoke 2/2: rectified scorer on 2 users of the 732 pool (reference checkpoint, w8+s8, raw and rectified)", 12 min. Then `sm95_smoke.py` must print PASS: the raw label reproduces the reference rows, the rectified label keeps the instant head and changes the curve. It also prints reviews/s.
5. "arm quarter-epoch 32k Muon cool-down from SM72s epoch 21 with the grade rectifier (w64, the reference's options and order) (segment to the next break or the end)", about 150-175 min. It takes a snapshot stop at most 2 min before a break and resumes after it; the log line on completion has reviews/s and peak memory.
6. "scoring sm95-arm-e21-q32k ..." (raw, 723 users in 2 passes, plus the 3 long users at a 0.92 cap).
7. "scoring sm95-arm-e21-q32k-rect ..." (rectified, `--grade-rectify --workers 3`, up to 4 passes of 45 min, plus the long users at a 0.92 cap, 25 min; those may run out of memory and are then left out).
8. "scoring sm77-ref-e21-q32k-rect ..." (the same, reference checkpoint).
9. "what-if intervals of the arm on the 732 pool (723 users, 3,000 reviews each, ...) (pass 1/2)", 40 min.
10. CPU: `sm95_decide.py` derives the order read-outs and writes `sm95-decision.{json,md}`; the log line is `rule exit 0: [sm95-rule] ...`. The reference's and SM33's derived folders were being pre-computed on 11-07 16:25-~17:50 in `work/derived/{ref_raw,ref_rect,sm33_raw}` (log `work/derive-prederive.out`). Derive skips users already done, so the rule completes whatever is missing; the arm's two folders take about 35 min CPU at the rule.

Breaks are daily 20:00-01:00 and Sunday from 16:00 (`sm95_block.py`, quiet.json scope all/gpu replaces a night's default). No ticket starts unless it fits before the next break, and CPU steps respect breaks too. A RAM gate holds while less than 3 GB is free, for at most 30 min.

**Expected timeline.**
- SM57 stage 4 ends about Sun 11:50.
- Smoke until about 12:15.
- Cool-down about 12:15-14:50.
- Arm raw scoring about 15:20.
- The rectified scoring (45 min tickets) does not fit before 16:00, so it waits for Mon 01:00.
- Arm rectified until about 03:00, reference rectified until about 05:00, measure until about 05:45, rule about 06:30.
- So: end about **Mon 2026-11-09 06:30-07:30**.

## Is it alive / what if it died

```bash
tail -5 C:/Users/jonat/workspace/wt/sm95/artifacts/small-model/sm95/gpu/launch-sm95.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm95*' } | select ProcessId,CommandLine"
python C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status    # tickets of owner sm95
```

- While waiting for SM57, the log prints a line every hour. During a break it prints one every hour too.
- If no bash process runs `launch-sm95.sh` and the log does not end with `done` or `STOP`, relaunch. After a Claude app restart, always check.
- On a `STOP` line, read its reason (train.out / score.out / check.out), fix, then relaunch. Finished steps are skipped and an interrupted cool-down resumes from its snapshot.
- If the cool-down runs out of GPU memory, prepare a new arm at density 0.05 (and say so in the write-up).
- Stop between tickets with `touch .../gpu/STOP`; remove the file before relaunching.

Relaunch (PowerShell; the same command started it):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm95/artifacts/small-model/sm95/launch-sm95.sh >> /c/Users/jonat/workspace/wt/sm95/artifacts/small-model/sm95/gpu/launch-sm95.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch refuses uncommitted source, and edits mid-run stop the training).

## Follow-up for a fresh Sonnet agent (medium effort), when the log says `done`

```bash
cd C:/Users/jonat/workspace/wt/sm95 && git fetch && git merge --ff-only
tail -20 artifacts/small-model/sm95/gpu/launch-sm95.log
cat artifacts/small-model/sm95/sm95-decision.md
```

1. If the rule did not run (no `rule exit` line, or exit 2 = incomplete), check the user counts per label in `gpu/sm95-neighbour.jsonl`. Re-run the missing scoring by relaunching the launcher, then re-run the rule by hand:
   `OMP_NUM_THREADS=2 .venv/Scripts/python.exe -X utf8 artifacts/small-model/sm95/sm95_decide.py --rows artifacts/small-model/sm95/gpu/sm95-neighbour.jsonl --ref-rows C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl --sm33-rows C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/gpu/sm33-neighbour.jsonl --arm-measure artifacts/small-model/sm95/work/measure-arm --ref-measure C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/work/stage2-reference --sm33-measure C:/Users/jonat/workspace/wt/sm33/artifacts/small-model/sm33/work/stage2-arm --derived artifacts/small-model/sm95/work/derived`
   (main checkout venv: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe`).
2. Write `artifacts/small-model/sm95/README.md`, short and in plain names with the ID in parentheses:
   - the question and the design, from this file;
   - the accuracy table from `sm95-decision.md` (all comparisons, CIs, vs the big model (1 leak-free epoch));
   - the order table (arm rectified counts must be 0; reference raw about 19% displayed at 0.90);
   - training cost: the cool-down's reviews/s and minutes from the log line and `train.out`, vs the SM77 reference's and SM33's 17,497 reviews/s;
   - scoring cost: rectified vs raw `model_seconds` per user from `gpu/sm95-neighbour.jsonl`, and reviews/s;
   - the 8k/2k 10-epoch ETA impact;
   - users left out (out-of-memory long users);
   - the verdict and the fork change (above).
3. Update the SM95 row in `docs/GPU_OPTIMIZATIONS.md` with status, measured quality, reviews/s and the result link.
4. Commit `README.md`, `sm95-decision.{json,md}`, `gpu/*.log`, `gpu/*.out`, `gpu/*.jsonl`, the check JSONs and `work/measure-arm/measure-log.jsonl` with `git add -f` and explicit paths. Never commit checkpoints, snapshots, `work/derived` or `work/measure-arm/users`. Commit as Jonathan Schoreels, with the message ending in a blank line plus `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push branch sm95 right away. Do not merge into main; the main session updates the Factory page.

## Retry of the rectified scoring (2026-11-08)

The launcher ended 11-08 17:40 with `rule exit 2: INCOMPLETE: 694 of 722`. Both rectified labels (`sm95-arm-e21-q32k-rect`, `sm77-ref-e21-q32k-rect`) lack the same 29 users: 276, 296, 481, 499, 510, 551, 968, 1125, 1354, 1830, 1870, 2073, 2267, 2423, 2582, 2594, 2912, 3156, 3289, 3301, 3433, 3735, 3923, 4071, 4390, 4577, 4655, 4661, 4852 (file `gpu/retry-users.json`). The 3 long users (650, 4501, 1601) also hit out-of-memory but are not in the 722 the rule needs.

- **Cause**: GPU out-of-memory (`OOM` error rows in `gpu/sm95-neighbour.jsonl`): the rectified scoring has about 2.5x the rows and ran with `--workers 3` at cap 0.8 (12.7 GiB allowed, 10-12 GiB allocated); the 29 users have 260k-530k reviews. Passes 2-4 retried the same 29 under the same limits.
- **Fix (memory only, same numerics)**: same scorer, model, users, `--grade-rectify`, `--deleted-cards neighbour`; only `--workers 1` and `HELDOUT_GPU_MEMORY_FRACTION=0.92`.
- **Launcher**: `gpu/retry-rect.sh` (never edits `launch-sm95.sh`). It checks breaks itself, queues behind the show-time scoring (owner small-model, until about 20:00), scores the arm then the reference (up to 3 tickets each, 45 min expected), then runs the rule, and logs `rule exit N` and `done`. Started 11-08 17:43, hidden via WMI, pid 51188 (bash). Log `gpu/retry-rect.log`, stdout `gpu/retry-rect.nohup`, scorer output `gpu/score.out`.
- **Expected**: the break is 21:00-01:00, so if the GPU is free before about 20:15 the arm retry runs then, otherwise it starts about Mon 01:00. End of scoring about Mon 02:30-03:30, rule about 1 h of CPU after.
- **Alive check**: `tail gpu/retry-rect.log`. If the process died, relaunch: same command as above with `gpu/retry-rect.sh` (finished work is skipped).
- **Follow-up**: when the log says `done`, continue with the follow-up section above from step 2 (the rule already ran). If it says `rule exit 2`, check the missing ids and rerun the script; if the rule did not run, use the rule command in step 1 above.
