# SM100b (the show-time error sensitivity test (SM100), follow-up): model comparison without timeout reviews (2026-11-09 19:35)

## Question
Does the decision pair "the show-time instant-only small model (SM96) minus the show-time retrain of the release (SM92), instant
head, show-time inputs (`e2s-query`), 722 users" change when the timeout reviews are excluded? Timeout reviews = SM100's exact
definition (`sm100_rule.cap_of`: user cap >= 10,000 ms, repeated >= 20 times and >= 0.5% of reviews; reviews at the cap;
1,010,999 of 30,126,145 reviews).

## What runs
- Launcher `launch-sm100b.sh` (committed 1e7050d9), started 2026-11-09 19:34 hidden via WMI: wrapper pid **8804**, launcher bash
  pid **48504**. It sleeps until 01:00 itself (`sm100b_block.py` hard hold 2026-11-10 01:00 + the daily 20:00-01:00 break + quiet.json),
  then queues ONE ticket (owner `sm100b`, purpose "the show-time error sensitivity test (SM100), follow-up: scoring ...", expected
  25 min, GPU cap 0.8, memory-only retry 0.92) behind whatever the stage 2 (SM98) ticket holds, then runs `sm100b_rule.py decide` (CPU).
- Expected: start ~01:01 Tue 2026-11-10 (+ queue wait), scoring ~12-15 min, end ~01:20-01:40, rule 1-2 min.
- Log `gpu/launch-sm100b.log` (ends `done (verdict ...)` or `STOP:`); scorer output `gpu/score.out`; tickets `gpu/tickets.jsonl`;
  rows `gpu/sm100b-rows.jsonl`; dumps `gpu/dump/` (~0.5 GB, never committed).
- Model: SM92, `sm92-e26-dec1ep32k/w8+s8`, checkpoint `wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth`
  (sha256 2091f8a879a5, checked). Scorer = SM100's path (723 users of `eval-users-732-fit.json`, `--split-existing`, `--deleted-cards neighbour`, `--dump`).

## Pre-registered rule (`sm100b_rule.py`, fixed before launch)
SM96 minus SM92, instant, `e2s-query`, review-weighted, paired user bootstrap 2,000 seed 0, 95% CI, points = % of the big model
(1 leak-free epoch) normal-input instant LogLoss 0.275757, on (a) all reviews (must reproduce -0.214 [-0.233, -0.195]; checked on the
rows by the report's `compare`, the dump estimator must be within 0.002), (b) without timeout reviews, (c) timeout reviews only.
Verdict **unchanged** if |(b) - (a)| < 0.05 points, else **changed**. Also reported: (b)'s CI, SM96's absolute instant LogLoss with
and without timeouts, the reproduction of SM92's existing e2s rows (new rows vs old rows must be bit for bit; dumped per-user
LogLoss vs old rows within ~1e-6).
Self-test before launch (CPU): (a) on the existing rows -0.214 [-0.233, -0.195] OK; SM96 side from the SM100 dumps OK (SM96 instant
LogLoss show-time: all 0.276287, without timeouts 0.274182, timeouts only 0.336900; estimator reproduces SM96's show-time cost
+0.055). The SM92 side can only be tested after the ticket.

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)
1. `git -C C:/Users/jonat/workspace/wt/sm100 fetch`, fast-forward `sm100`. `tail -8 artifacts/small-model/sm100b/gpu/launch-sm100b.log`.
   `done (verdict ...)` -> step 3. `STOP:` -> read it and the tail of `gpu/score.out`; relaunch only for a clear transient
   (same WMI command as `HANDOFF.md` of SM100, with `launch-sm100b.sh`; scored users are skipped); otherwise report. Still waiting (before ~01:30) -> report and stop.
2. If scoring finished but the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm100b/sm100b_rule.py decide` (exit 2 = rows / dumps missing).
3. Write `sm100b-result.md` from `sm100b-decision.md/json`: verdict, the (a)/(b)/(c) table, (a) reproduction, SM96 absolute LogLoss
   with/without timeouts, SM92 reproduction, ticket minutes (`gpu/tickets.jsonl`). One plain sentence of interpretation.
4. Commit with `git add -f` explicit paths: `sm100b-decision.json`, `sm100b-decision.md`, `sm100b-result.md`, `gpu/sm100b-rows.jsonl`,
   `gpu/launch-sm100b.log`, `gpu/launch-sm100b.nohup`, `gpu/tickets.jsonl`, `gpu/score.out` (else its last 200 lines). Never `gpu/dump/`,
   checkpoints, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md`, quiet.json. Commit as
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line +
   `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm100`, push `sm100`.
5. Plain names with IDs in parentheses, units stated (points = % of the big model (1 leak-free epoch) normal-input instant
   LogLoss), never "Expertium". Never touch `wt/sm98`, `wt/sm96`, `wt/sm92` files, anki-rwkv main or the Anki fork.
