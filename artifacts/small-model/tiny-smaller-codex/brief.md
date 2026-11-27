You are reviewing an ML research repository (read-only: do not edit files, do not run git commands that change anything, do not start training/scoring/GPU jobs, do not install anything). Your output is one written analysis.

## Context

This repository trains RWKV-based spaced-repetition models (predicting whether an Anki card will be recalled) that ship in a fork of Anki as a "family" of sizes: Big, Small, Tiny. Each size has a two-head model (curve = recall probability over time, instant = recall at the next review) and an Instant-Only model. Metric: LogLoss on the 732-user evaluation set, show-time inputs; differences quoted in % with a paired user bootstrap 95% CI.

Current candidates (curve / instant LogLoss):
- Small two-head: the show-time two-head cool-down for Small (SM118): 0.300016 / 0.276599 (331,511 parameters, 3,530 bytes of card+note state per card).
- Small Instant-Only: the longer Small Instant-Only model (SM113): instant 0.275933.
- Tiny two-head: the show-time two-head cool-down for Tiny (SM124): 0.300369 / 0.277105 (311,786 parameters, 2,622 bytes of state).
- Tiny Instant-Only: the longer cool-down for Tiny (SM115): instant 0.276446.
So Tiny today is only about +0.12% curve / +0.18% instant worse than Small, and it is not much smaller in parameters; its saving is mostly the per-card state.

Newest relevant result (branch sm116, readable at `C:/Users/jonat/workspace/wt/sm116/artifacts/small-model/sm116/`: `design.md`, `sm116-result.md`, `sm116-decision.md`): the training-free state check (SM116) kept only 70% of the show-time retrain (SM92)'s card and note state (PCA per block, 2,373 of 3,392 values, no training) and lost only +0.015% curve / +0.032% instant. Verdict "starting point" for a trained smaller model.

## The question

The user wants the Tiny model to become clearly smaller, accepting slightly worse accuracy: the budget is **at most +1% LogLoss vs the Small model**, on both curve and instant (two-head vs Small two-head SM118; Instant-Only vs Small Instant-Only SM113). That is roughly: Tiny two-head curve <= 0.30302, instant <= 0.27936; Tiny Instant-Only instant <= 0.27869.

Based on the backlog and the past results in this repository, which ideas could make Tiny much smaller within that budget? Consider all size dimensions that matter for shipping in Anki: parameter count / model file size, per-card and per-note state bytes (stored for every card in a collection), compute per review (CPU inference speed in the client), and quantization. For each idea:
1. what it is and which past experiments (by ID and plain name) support or argue against it, with their measured numbers;
2. expected size saving (estimate) and expected LogLoss cost vs the budget;
3. cost to test it here (GPU hours on one 16 GB RTX 5080, CPU hours), and whether it needs training or is training-free;
4. risks (e.g. leaks, the curve head degrading more than instant, users with very long histories).
Then give a ranked short list (top 3-5) and a concrete first experiment with a pre-registered go rule (go only if the CI upper bound is within the budget).
Also say which backlog items are already dropped/done and should NOT be retried, and why.

## Where to look

Working root: `C:/Users/jonat/workspace/anki-rwkv`.
- `pages/data/factory-curated.json`: keys `backlog` (73 items, status next/idea/parked/done/dropped), `results` (169 result rows with numbers), `plan`, `family` (the shipped/candidate models). This is the best index.
- `artifacts/small-model/<id>/` (design.md, result/report md files) for experiments; other worktrees with experiments live under `C:/Users/jonat/workspace/wt/<id>/artifacts/small-model/<id>/` (read only).
- `docs/GPU_OPTIMIZATIONS.md` and `AGENT.md` for how training/scoring works and what it costs.
- The "smaller model" track (SM73 and the pruning experiments), state-size experiments, quantization, distillation results.

## Rules

- Never read anything under `C:/Users/jonat/AppData/Roaming/Anki2` (a private collection). Do not quote per-user raw review data; aggregate numbers only.
- Name every model as "plain name (ID)", e.g. "the show-time retrain (SM92)", never a bare ID. Say "%" never "points". Do not use the word "Expertium" in your write-up; say "outside research".
- English, full plain sentences. Markdown. Be concrete with numbers and file paths you relied on.
