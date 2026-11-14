# Codebook state compression sweep (SM90): handoff (2026-11-01)

Design, ladder, knobs, decision rule: [DESIGN.md](DESIGN.md) (pre-registered before any model-side result).
Worktree `C:/Users/jonat/workspace/wt/sm90`, branch `sm90` (= origin/main + the leak-free small model (SM72s) tree +
state compression at a large budget (SM46) fixes + this work). Always `git fetch` + `git merge --ff-only` first.

## State at handoff
- Codebooks: `books/codebooks-sm90.npz` (not committed, ~40 MB; the corpus `books/corpus/` 0.9 GB neither), sha256 in
  `books/codebooks-sm90.sha256` = c12b659b19c348a28421f6077ecf5dfdd82ce25f449d2b297af2002ed54b368b (committed; the
  runbook refuses other books). 200k WKV fitting points. Regeneration: `collect-sm90.sh`, `fit-sm90.sh` + `fit-c67.sh`,
  `finalize-sm90.sh` (seeded; determinism proof bitwise at 4 threads). The 900k refit (`fit-big-sm90.sh`,
  `finalize2-sm90.sh`) was killed for RAM (see DESIGN.md) and must NOT be rerun as is: one fitting process at a time,
  < 6 GB RAM, free RAM > 8 GB while GPU training runs.
- CPU proof, ALL PASS (`proof/`, `check-sm90-cpu-{a,b,c}.json`):
  - (a) two seeded k-means fits bitwise equal (`proof-fit-det.json`);
  - (b) encode deterministic, explicit decode of the picks = codec output bitwise, held-out residual falls with every
    stage (`proof-codec.json`; held-out / in-sample 1.15x -> 4.4x over 20 stages = deep stages overfit at 200k points);
  - (c) `w8+s8` / `w8+s4c8n` rows of this tree bitwise = the released tree (wt/sm72s), 3 users (`proof-int8.json`);
  - (d) CPU `w8+s8` vs the recorded released GPU rows: max |d| 1e-6 (<= 1e-5);
  - (e) GPU codec path on CPU tensors = CPU scorer, all 9 screen recipes x 2 users, d = 0 (the codec is active: float
    state moves p by up to 0.01-0.05).
- Model-free proxy: `proxy-sm90-wkv.json`, `proxy-sm90-shift.json`.
- Sizes / encode cost: `cost-sm90.json` (`cost_sm90.py`).

## Running (detached, WMI hidden)
- Launched 11-01 13:57 from commit ad9acc40 (WMI pid 60884). First ticket: 000840 (GPU check), queued behind
  000838 (duration-leak scoring (SM31)) and 000839 (grade-order sanity (SM33)), max wait 84 min; if it cannot start
  before the 16:00 break the runbook waits and re-enqueues after 01:00.
- Runbook: `C:/Users/jonat/workspace/wt/sm90/artifacts/small-model/sm90/runbook-sm90.sh`
- Log: `artifacts/small-model/sm90/runbook-sm90.log` (+ `.nohup`); ticket output `gpu/check.out`, `gpu/screen.out`,
  `gpu/final.out`, rule output `gpu/rule.out`. Dry run: `runbook-sm90-dryrun.log`, `gpu/*-dryrun.out`.
- It waits for a free window itself (quiet.json + daily 20:00-01:00; today's break 16:00 to 01:00 Monday) and only
  enqueues a ticket that fits before the next window. Expected: nothing starts before 01:00 Monday 11-02 (the queue
  holds grade-order sanity (SM33) and duration-leak scoring (SM31) tickets ahead).
- Tickets, in order (purpose prefix "codebook state compression sweep (SM90): "):
  1. "GPU check of the codec path vs the CPU references (every screen recipe, 2 users) and reviews/s" (35 min) ->
     `check-sm90-gpu.json`; STOP on failure (discrete codec rule of AGENT.md);
  2. "screen of the budget ladder on 40 users of the 732 set (pass N)" (75 min, up to 4 passes) ->
     `gpu/screen-sm90.jsonl`; then `sm90_rule.py screen` -> `screen-summary.{md,json}`, `finalists.txt`;
  3. "scoring <label> on the 723 users of the 732 set (pass N, K done)" (110 min each, up to 6 passes per label),
     labels from `finalists.txt` (w8+s4c8n first) -> `gpu/final-sm90.jsonl`;
  4. `sm90_rule.py final` -> `sm90-summary.{md,json}`; log line `done`.
- Expected end: GPU check ~01:40, screen ~03:00, finalists (2-5 labels; a coded label ~2-3 h on the 723 users, the
  int one ~0.5 h) Monday ~10:00-16:00, before the 20:00 break if the queue is otherwise free. A label that cannot
  finish before 20:00 resumes after 01:00 Tuesday (passes skip users already scored).

## Update 2026-11-02 05:55: stalled scoring step, fixed, relaunched
- Symptom: `rel/w8+vqC4430N3020` stuck at 721/723 users; the runbook stopped with "no progress" at 05:52.
- Root cause (not a scorer bug): users 4852 and 3735 (~0.5M reviews each, the two longest) hit CUDA out of memory at the
  scorer's default 0.8 memory cap (12.72 GiB; 10 GiB allocated + 2.5 GiB reserved-but-fragmented). The scorer writes an
  error row (`"error": "OOM: ..."`, no metrics) and the runbook only counts error-free rows; the "0 review-evals/s" lines
  of pass 3 are those OOM rows. Earlier labels got these users through on a lone retry pass (3735 also OOM'd once for
  `vqC4430` and `s4c8n`); the 3 note-coded stages of `vqC4430N3020` needs more memory, so both fail at 0.8.
- Fix (commit 1c307a69, runbook only, scripts untouched): after a pass with no progress, ONE more pass at
  `HELDOUT_GPU_MEMORY_FRACTION=0.92` (the cap other runbooks use for their "3 long users" tickets; memory only, the
  numerics are unchanged), then STOP if it still makes no progress. The cap resets for each label. Consequence: every
  coded label now costs one extra short ticket that OOMs at 0.8 before the 0.92 pass. Rows of already scored
  users/labels are untouched (the scorer skips them).
- Relaunched 05:55 (WMI pid 52328, hidden) from commit 1c307a69; queued behind the robustness-to-other-apps (SM29)
  scoring ticket. Remaining: `vqC4430N3020` (2 long users), `rel/w8+vqC5902`, `rel/w8+vqN3020`, then the rule read-out.
  Expected end: ~13:00-15:00 today (before the 20:00 break unless the queue is busier), else resumes after 01:00.

## Follow-up commands
```bash
cd C:/Users/jonat/workspace/wt/sm90 && git fetch -q && git merge --ff-only -q
H=artifacts/small-model/sm90; PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
tail -n 30 $H/runbook-sm90.log
$PY -c "import json; r=json.load(open('$H/check-sm90-gpu.json')); print(r['passed'], r['fails'], r.get('timing'))"
cat $H/screen-summary.md $H/finalists.txt
cat $H/sm90-summary.md
# re-pair after extra rows (CPU, seconds; never inside a break)
$PY -X utf8 $H/sm90_rule.py screen; $PY -X utf8 $H/sm90_rule.py final
# runbook stopped or died (Claude app restarts kill detached jobs): relaunch, it resumes (passed GPU check reused)
#   PowerShell: $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm90/artifacts/small-model/sm90/runbook-sm90.sh >> C:/Users/jonat/workspace/wt/sm90/artifacts/small-model/sm90/runbook-sm90.nohup 2>&1"'; ProcessStartupInformation=$si}
git add -f $H/gpu/screen-sm90.jsonl $H/gpu/final-sm90.jsonl $H/screen-summary.md $H/screen-summary.json $H/finalists.txt \
  $H/sm90-summary.md $H/sm90-summary.json $H/check-sm90-gpu.json $H/runbook-sm90.log
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit -m "Codebook state compression sweep (SM90): <result>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

## Decision rule (copy of DESIGN.md)
On the 723 fit users of the 732 set, leak-free, w8 weights: a budget is nearly free if its extra cost vs int4 cards +
int8 notes (`w8+s4c8n`, the fork default) is <= +0.02 points on curve AND instant (points = % of the big model (1
leak-free epoch) LogLoss; paired user bootstrap 2,000, seed 0) AND it stays inside the smaller-model bar (<= +0.5%
curve / <= +0.4% instant vs the big model (1 leak-free epoch)). Answer per stream: the smallest nearly-free card budget
(`vqC*`, notes int8) and note budget (`vqN*`, cards int4), and the both-coded pair. Practical = codebooks <= 10 MB at
float16 and estimated encode <= 1 ms per review; report state MB, codebook MB, GPU reviews/s and encode ms with every
budget.

## If something fails
- GPU check fails: read `check-sm90-gpu.json` (`base_float_state_gpu_vs_cpu` large = base numerics; only +vq off = near-tie
  flips, see `n_dp_over_1e-4` per user). Card-only / note-only recipes run the int kernel for the other stream (the
  `+s4c8n` path) through the slots.
- Screen read-out exit 3 (rows missing): rerun the runbook (it resumes).
- Do not edit `scripts/`, `expertium/` or `rwkv/` in wt/sm90 while a ticket runs (the runbook checks a clean tree at its
  start; the scorer imports at ticket start). Never edit the running runbook.
- Not done this round (DESIGN.md "Knobs"): codebook size per stage, Lloyd iterations (`knobs-sm90.sh`, proxy only),
  10-bit shift codes, relative norms, a note rung above 1.3x.
