# Handoff: Training-free state check (SM116)

Design (pre-registered, committed before the score in `04fbebff`): `design.md` in this folder. Code: `sm116_state.py`
(directions, basis, projected scoring through the stock CPU scorer), `sm116_rule.py` (pool, proof, rule, self-test),
`collect.sh` + `collect-b2.sh` (directions, done), `proof.sh` (done, PASS), `launch-sm116.sh` + `memguard.ps1` (the
score, running). Branch `sm116` (from `origin/main` + `origin/sm106` merged for the scorer code). Worktree
`C:/Users/jonat/workspace/wt/sm116`.

In short: the show-time retrain (SM92)'s card and note state (17 blocks: 12 WKV heads of 16 x 16, 5 token-shift
vectors of 64; 3,392 floats) got a centered PCA per block from 94 unseen validation users (ids >= 5000, outside every
scored user). The 70% state keeps the first 179 of 256 / 45 of 64 directions of every block (2,373 floats, 69.96%).
The stored state is projected at every stored step, then the stock int8 rounding (exact fold for the shift vectors;
for the WKV state no exact fold exists, so the projection is the upper bound of a fold). Variance kept: 98.65% to
99.98% per block. CPU proof PASS: the 100% projection equals the unmodified model bit for bit on the CPU scorer, and
its GPU rows within 2e-6. One score: the 629 users of the show-time retrain (SM92)'s 722-user rule pool with <= 125,000
reviews (the 6 GB RAM limit; 51% of the pool's reviews), vs SM92's GPU rows. Rule: "starting point" iff both heads'
upper 95% CI bound of the cost is below +0.05% of the big model (1 leak-free epoch) show-time LogLoss; else "drop".

## The running job

- Launched 15:58 Wed 11-18 from `04fbebff`, hidden via WMI (wrapper pid 16780). 3 CPU scorer processes (2 torch
  threads each, about 1-4 GB each), fixed slices of the scorer's stratified order (a: 0-209, b: 210-419, c: 420-628),
  plus `memguard.ps1` (kills a scorer above 5.9 GB, then the launcher stops).
- Expected: about 1,550 reviews/s per process, 21.4M reviews: **done about 17:20-17:45 Wed 11-18** (well before the
  20:00 break). New users start only until 19:34 (25 min before the break); if anything is left then, the launcher
  sleeps through 20:00-01:00 and resumes at 01:00 by itself (scored users are kept).
- Log: `rows/launch-sm116.log` (`[sm116 MM-DD HH:MM:SS]` lines); stdout `rows/launch-sm116.nohup`; per part
  `rows/score-{a,b,c}.out` (`[small-model] N/210 user U (R reviews) | Ts, X review-evals/s`); rows
  `rows/sm116-k70-{a,b,c}.jsonl`, merged `rows/sm116-k70.jsonl`; memory `rows/memguard.log`.
- Patterns: `pass N ended: exit codes 0 0 0; 0 users left`; final line **`done (verdict starting point|drop)`**;
  failure **`STOP: ...`** (a scorer failed, error rows, the memory guard killed a scorer: `rows/memguard-kill`, or a
  `rows/STOP` file).
- Outputs at the end: `sm116-decision.json`, `sm116-decision.md` (also `rows/decide.out`).

## Alive check / stop / relaunch

- Alive: `tail -3 C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/rows/launch-sm116.log`;
  `for p in a b c; do grep review-evals C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/rows/score-$p.out | tail -1; done`;
  `powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { \$_.CommandLine -like '*launch-sm116*' -or \$_.CommandLine -like '*sm116_state.py score*' } | select ProcessId,Name"`.
  After a Claude app restart or a power event check it (detached jobs died before).
- Stop: `touch C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/rows/STOP` (seen between passes), or kill
  the launcher bash and the three `sm116_state.py score` python processes; scored users are kept.
- Relaunch (only if no launcher runs and the log has no `done`; first remove `rows/STOP`, `rows/memguard-kill` and
  `rows/memguard.off` if present; finished users are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/launch-sm116.sh >> C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/rows/launch-sm116.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- If the memory guard killed a scorer: report it (user id from the part's last `score-*.out` line); do not raise the
  cap or drop the user without the main session's OK (the pool is pre-registered).
- Do not edit `scripts/`, `expertium/` or the sm116 scripts while it runs. Never touch other worktrees (read only:
  `wt/sm92`, `wt/sm72`, `wt/sm72s`, `wt/sm31`).

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. If the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/sm116_rule.py decide` (exit 2 = rows missing).
2. Write `sm116-result.md`: verdict and both heads (cost in % of the big model (1 leak-free epoch) show-time LogLoss
   with CI, and in % of the unmodified model's own LogLoss), users better / worse, the directions table from
   `design.md`, the proof, the pool caveat (629 users <= 125k reviews; the 93 heaviest not scored), wall time and
   review-evals/s per part (`rows/score-*.out`), peak RAM (`rows/memguard.log`).
3. Commit on `sm116` with `git add -f` (only those that exist): `sm116-decision.json`, `sm116-decision.md`,
   `sm116-result.md`, `HANDOFF.md`, `rows/launch-sm116.log`, `rows/launch-sm116.nohup`, `rows/score-a.out`,
   `rows/score-b.out`, `rows/score-c.out`, `rows/decide.out`, `rows/memguard.log`, and the rows gzipped
   (`gzip -k rows/sm116-k70.jsonl` -> `rows/sm116-k70.jsonl.gz`). Never `work/`, `.npz`, `.pth`.
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a
   blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch`, merge `origin/sm116`, push
   `sm116`. Never force-push, never bypass hooks.
4. Factory page: the main session owns `pages/data/factory-curated.json`; give it the results row below with the
   numbers filled in (and the plan row closed: `state` done, `ended_at`, `when`).
5. Text rules: every model as "plain name (ID)", "%" never "points", never the word "Expertium" (say "outside
   research").

## Plan row to insert now (by the agent that owns `pages/data/factory-curated.json`; this agent did not edit it)

```json
{
 "id": "SM116",
 "track": "small model",
 "since": "11-18T15:25",
 "kind": "cpu",
 "minutes": 80,
 "match": [
  "training-free state check (SM116): "
 ],
 "what": "Training-free state check (SM116): how much of the show-time retrain (SM92)&rsquo;s card and note state carries information, and what keeping only 70% of it costs with no training at all. Per-block state directions (a PCA of the stored states, 17 blocks) from 94 unseen validation users outside the scored pool: 90% of the variance sits in 4 to 74 of 256 directions per state head, but 99.9% needs 98 to 242; the 70% state keeps 98.65% to 99.98% of the variance. The stored state is projected onto the kept directions at every review (an exact fold into the weights for the token-shift part; for the main state no exact fold exists, so this is the upper bound of a fold); the 100% projection reproduces the model bit for bit. One pre-registered CPU score on show-time inputs, curve and instant, 629 users of the 732 set (those with at most 125,000 reviews, the CPU memory limit). Rule fixed before the score: it becomes the start of a trained smaller model only if both heads&rsquo; cost has its 95% CI upper bound below +0.05%; otherwise it is dropped",
 "when": "Started 15:58 Wed 11-18 (directions and CPU proof 15:25-15:57); CPU score about 80 min on 3 processes, result about 17:20-17:45 Wed 11-18",
 "decides": "Whether a 70% card and note state, folded in with no training, is cheap enough (both heads under +0.05%, CI included) to start a trained smaller model from; otherwise the idea is dropped",
 "state": "running",
 "started_at": "2026-11-18T15:25+01:00"
}
```

## Results row template (fill the `<...>` from `sm116-decision.json` / `sm116-result.md` after `done`)

```json
{
 "date": "11-18",
 "kind": "cpu-score",
 "id": "SM116",
 "track": "small model",
 "name": "Training-free state check (SM116): the show-time retrain (SM92) with a 70% card and note state, no training",
 "eval": "629 users of the 732 set (the show-time retrain (SM92)&rsquo;s 722-user rule pool with at most 125,000 reviews, the CPU memory limit; 21.4M reviews, 51% of the pool&rsquo;s reviews), show-time inputs, leak-free, w8+s8, CPU scorer; against the show-time retrain (SM92)&rsquo;s own rows; % of the big model (1 leak-free epoch) show-time LogLoss, paired user bootstrap 2,000, 95% CI. State directions from 94 other validation users (ids from 5,000 up), none of them scored.",
 "fork": "Nothing changed in the fork",
 "takeaway": "<strong><verdict>.</strong> Keeping 70% of the card and note state (2,373 of 3,392 values) with no training costs <curve %> [<lo>, <hi>] on the curve and <instant %> [<lo>, <hi>] on instant. <one sentence: starting point for a trained smaller model, or dropped>. The state is concentrated (90% of the variance in 4 to 74 of 256 directions per state head) but its tail matters: 99.9% needs 98 to 242.",
 "inputs": "show-time",
 "started_at": "2026-11-18T15:25+01:00",
 "ended_at": "<2026-11-18THH:MM+01:00>",
 "gpu_minutes": 0,
 "tickets": 0,
 "detail": {
  "verdict": "<starting point | drop>; branch sm116 <commit> (design.md, sm116-decision.md, sm116-result.md)",
  "table": {
   "head": [
    "Head",
    "Unmodified",
    "70% state",
    "Cost (CI)",
    "% of own LogLoss",
    "Users better / worse"
   ],
   "rows": [
    ["curve", "<0.xxxxxx>", "<0.xxxxxx>", "<+x.xxx% [lo, hi]>", "<+x.xxx%>", "<n / n>"],
    ["instant", "<0.xxxxxx>", "<0.xxxxxx>", "<+x.xxx% [lo, hi]>", "<+x.xxx%>", "<n / n>"]
   ]
  },
  "notes": [
   "State directions: a centered PCA per block (12 state heads of 16 x 16, 5 token-shift vectors of 64) from 94 unseen validation users; the 70% state keeps 179 of 256 and 45 of 64 directions of every block, 98.65% to 99.98% of the variance",
   "Fold: exact for the token-shift vectors (static lerps followed by linear maps); for the main state no exact fold into the weights exists (data-dependent decay and removal, normalised keys and values, a per-head group norm), so the score projects the stored state at every review: the upper bound of a fold",
   "CPU proof: the 100% projection equals the unmodified model bit for bit on 8 users (and its GPU rows within 2e-6)",
   "Pool caveat: the 93 heaviest users of the 722 (above 125,000 reviews) are not scored, the CPU scorer needs more than 6 GB for them",
   "CPU: 3 processes, <wall minutes> min, about <rate> review-evals/s each"
  ]
 }
}
```
