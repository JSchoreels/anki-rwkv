# HANDOFF: SM55 (SM36's recipe on the 8k final tier), launched 2026-10-21

Worktree `C:/Users/jonat/workspace/wt/sm55` (branch `sm55`, from `small-model`). One detached chain:
`artifacts/small-model/runbook-2026-10-21-sm55.sh`, log `artifacts/small-model/runbook-2026-10-21-sm55.log` (dry run: `runbook-2026-10-21-sm55-dryrun.log`, ends `DRYRUN OK`).
Do not edit `scripts/ expertium/ rwkv/` in wt/sm55 while it runs (SourceWatch). Stop between tickets: `touch artifacts/small-model/sm55-STOP`.

## What runs (all GPU steps are queue tickets, owner small-model, claim gpu)
| Step | Expected min | Output |
|---|---:|---|
| Replay export of the 3,000 final-tier users not in the 5k export (resumable tickets, `--time-limit 5100`) | 95 per ticket, ~105 total (2 tickets) | `C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k` (5,000 hard links to `fork-fp32-gate5k` + 3,000 new files; `status.json`, `progress.jsonl`) |
| SM11 `verify` on 8 new users (CPU) | ~5 | `fork-fp32-final8k/verify.json` |
| Decay (smoke first), 581.9M reviews | 530 (SM36: 20,183 reviews/s -> ~486 actual) | `artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon-final8k/` |
| Scoring 732 pool minus 9 never-fit (723; the 464 are a subset), pass 2 only if users are missing | 45 (+20) | `artifacts/small-model/heldout-sm55-pools.jsonl`, label `s-w64kd-final-sc-lanes-dec12-1ep32k-muon/w8+s8` |
| Summary (CPU, seconds) | - | `artifacts/small-model/sm55-summary.{md,json}` |

Breaks: before each ticket the chain checks `quiet.json` and the daily 21:00-01:00 rule; a ticket that would end inside a break sleeps to 01:00, and `--max-wait-minutes` makes a queue wait give up (exit 75) and re-plan instead of running into the break.
Expected timeline (queue free): export ~09:00-10:45, decay ~10:50-~18:55 (nominal end 19:40; latest start 12:10), scoring ~18:55-19:15, `done` ~19:20. If the decay cannot start by 12:10: decay 01:00-~09:05 on 10-22, `done` ~09:30.

## Log lines to watch
- Done: `[sm55 MM-DD HH:MM:SS] done` (after the `[sm55-summary]` block).
- Failure: any line `[sm55 ...] STOP: ...` (e.g. `STOP: verify failed`, `STOP: decay ticket failed`, `STOP: ended with exit N`).

## Follow-up (fresh Sonnet medium agent)
- Resume after a stop or a crash: `nohup bash C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.sh >> C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.log 2>&1 & disown` (appends to the log; exported users, the finished decay and scored users are skipped).
- Summary again: `SUMMARY_ONLY=1 bash C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.sh`.
- Report from `sm55-summary.md`: SM55 vs SM36 (Δ, 95% CI) and both vs the fork on 464 and 732 (723 users); training reviews/s and minutes from `sm55-summary.json`. Commit `sm55-summary.*`, `heldout-sm55-pools.jsonl` (benchmark users only), the log, then update the SM55 section in `docs/SMALL_MODEL.md`.
