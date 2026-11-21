# Tiered card state by answer count (SM141): result

Verdict (report decided 00:52 Sun 11-22): **no scheme passes; int8 cards and notes (`w8+s8hx8`) stay for every size.** Details and per-model table: `sm141-result.md`, `sm141-result.json`; rules and bytes: `design.md`.

## Question

Can each card and note pick its state width by its own answer count (int4 below 128 answers, int8 from 128, optionally int3 or int2 for very young cards), so the state shrinks by 28% to 54% with no measurable LogLoss cost against int8 everywhere? No training; the six release-candidate models; the 732 set (723 users) on the GPU, the 50 long-card users on the CPU.

## Result

- Go per size (Big, Small, Tiny): none passes, so int8 stays (+0.0% bytes).
- Scheme A (int4 cards below 128 answers, notes int8) passes R2 (dataset-wide, all CI upper bounds <= +0.045%) for all six models and fails only R3 (cards with 128+ answers, limit +0.15%). A's R3 was not measured directly: it was taken from AN (same cards, int4 notes as well) by dominance.
- AN, B8 and D4 have R3 point estimates of +0.14% to +0.48% with wide CIs (50 users). The young int3 tier (B4 and B8) mostly passes R4 (young cards) but pushes R2 over +0.05% for several models; D4 (int2) clearly fails R2 and R4.
- Plain reading: the cost of int4 lands on long cards even though they switch to int8 at 128 answers, so the damage is done during the int4 period. A follow-up (SM144) tests an earlier switch (16 or 32 answers).

## Run facts

- The 17:47 GPU smoke stopped on two tolerance edges, fixed in commit d7d02555 (design.md "Smoke fix 11-21"); the GPU chain resumed with `launch-sm141-gpu-resume.sh`.
- The report crashed once at 00:49 on a report-only integrity loop (scalar fields in the benchmark dict); fixed in `sm141_report.py`, the decision is unaffected.
- Long-card user 1714 was killed by the memory guard: 49 of 50 long-card users scored, R5 coverage (48 of 50) still met. GPU coverage 722 users.
- Tickets (`gpu/tickets.jsonl`): smoke 8 min; Tiny 61 + 16 + 15 min; Small 115 + 8 min; Big 149 + 49 min (about 7 GPU hours in all). CPU long-card chain 17:38 to 00:44 (about 7 h, 3 processes).

## For the fork

Nothing to build: the release-candidate format stays int8 cards and notes for all sizes. The cost per answer-count bucket motivates SM144.
