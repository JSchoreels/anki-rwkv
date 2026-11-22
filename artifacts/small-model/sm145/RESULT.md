# Int3 young tier on the early switch (SM145): result

Verdict (report decided 22:42 Sun 11-22), pre-registered rules: **go pick Big Y4N, Small Y2N, Tiny Y4C. Recommended under the pre-registered "worth its bytes" criterion: Big Y4C (not Y4N), Small Y2N, Tiny Y4C.** Details: `sm145-result.md` (every table), `sm145-result.json`, `sm145-dr.json` (per-card read-out), `design.md` (rules, bytes), `sm145-tiny-reread.json` (Tiny re-read).

Labels (all keep int8 from 16 answers, int4 for 5-15 answers, int8 shift vectors): Y2N = int3 for 1-2 answers on cards and notes; Y4C = int3 for 1-4 answers on cards only (notes stay int4); Y4N = int3 for 1-4 answers on cards and notes; AN16 = int4 below 16 answers on cards and notes, the earlier int8 switch (SM144) pick; REF = int8 everywhere (cards and notes, `w8+s8hx8`).

Run note: the launcher's report step failed at 22:31 with `KeyError: 'ahead'` (`dataset_cmp` in `sm145_report.py` indexed the result of `st.compare` by the row key "ahead" instead of "curve" / "instant", unlike the earlier int8 switch (SM144)). The main session fixed that one line at 22:42, reran `selftest` (OK) and `report`, checked the launcher's final outputs by hand and appended two lines to `gpu/launch-sm145.log` ending `done (status decided ...)`. The fixed `sm145_report.py` is committed with this result.

## Question

On top of the int4 switch at 16 answers (earlier int8 switch, SM144), does an int3 state for the youngest cards still pass, and is it worth its bytes? No training; the six release-candidate models; the 732 set (723 users, 722 rule users) on the GPU; the 50 long-card users on the CPU, every label scored directly; new: a per-review and per-card |dR| read-out against int8.

## Result under the pre-registered rules

Rules: R1 every user finite; R2 dataset-wide benchmark LogLoss vs int8, 95% CI upper bound <= +0.05% per head; R3 reviews of cards with 128+ answers (CPU), point <= +0.5% and upper <= +1.0% per head (the user's 11-22 tolerance, registered before scoring); R4 reviews of young cards (1-M answers, the int3 tier) upper <= +0.1%. Go: the smallest bytes that pass for both models of a size. Recommended: the same among AN16 and the int3 labels that are worth their bytes (saves at least 2% of the int8 bytes vs AN16, extra cost vs AN16 point <= +0.02% and upper <= +0.03%, share of cards with a review |dR| > 0.10 at most twice AN16's).

| size | passing labels | go pick (bytes vs int8) | recommended (bytes vs int8) |
|---|---|---|---|
| Big (show-time big model (SM91s2) and show-time instant-only big model (SM99)) | Y4N, Y4C, Y2N, AN16 | Y4N (-45.34%) | **Y4C (-43.48%)** |
| Small (show-time two-head cool-down for Small (SM118) and longer Small Instant-Only model (SM113)) | Y2N, AN16 | Y2N (-38.57%) | Y2N (-38.57%) |
| Tiny (six-layer Tiny two-head model (SM130) and six-layer Tiny Instant-Only model (SM130)) | Y4C, Y2N, AN16 | Y4C (-37.74%) | Y4C (-37.74%) |

- Big: Y4N passes every rule but fails the per-card part of "worth its bytes": its share of cards with a review |dR| > 0.10 is 0.0648% (show-time big model (SM91s2), instant head) against 0.0251% for AN16, more than twice (2.6x); Y4C is 0.0294% (1.2x) and Y2N 0.0330% (1.3x). Y4C saves 3.23% of the int8 bytes vs AN16, Y4N 5.09%. The go pick and the recommendation differ, so the choice between them is the user's.
- Small: Y4N and Y4C fail R3 on the show-time two-head cool-down for Small (SM118) curve head (upper bounds +1.110% and +1.017%, over +1.0%); Y2N passes (+0.152% [-0.009, +0.917]).
- Tiny: Y4N fails R2 on the six-layer Tiny Instant-Only model (SM130): +0.058% [+0.054, +0.063] (upper bound over +0.05%). Y4C passes on both models.
- Dataset-wide cost vs int8 (R2, value [95% CI], curve / instant): Big Y4C +0.008% [-0.001, +0.017] / +0.015% [+0.013, +0.016]; Small Y2N +0.012% [+0.009, +0.016] / +0.016% [+0.014, +0.018]; Tiny Y4C +0.019% [+0.016, +0.023] / +0.029% [+0.026, +0.032]. Every other value, the young-card (R4) and long-card (R3) numbers and the answer-count bucket tables are in `sm145-result.md`.
- The extra cost of each int3 label vs AN16 is small everywhere (dataset-wide +0.000% to +0.028% of AN16's LogLoss; largest for Y4N on the six-layer Tiny Instant-Only model (SM130), +0.028% [+0.025, +0.031]).

## Per-card read-out (|dR| = how far a review's predicted retention moves against int8; 50 long-card users, CPU, aggregates only)

Does any card lose a real retention prediction? Typical moves are tiny: P50 of a review 0.0008 to 0.0021, P99 0.011 to 0.028 for every label and model. The share of reviews off by more than 0.10 is at most 0.0061% (Y4N, show-time big model (SM91s2), instant head). A few single reviews are off by more: the largest |dR| is 0.387 (Y4C, show-time big model (SM91s2), instant head; AN16 on the same head 0.282). Counted per card by its worst review, the share of cards with a review |dR| > 0.10, largest per size (AN16 for comparison):

| size | largest share of cards > 0.10 | label, model, head | AN16 same model and head | users with such a card |
|---|---|---|---|---|
| Big | 0.0648% | Y4N, show-time big model (SM91s2), instant | 0.0251% | 6 of 49 |
| Small | 0.0020% | Y4N, longer Small Instant-Only model (SM113), instant | 0.0000% | 5 of 50 |
| Tiny | 0.0086% | Y4N, six-layer Tiny Instant-Only model (SM130), instant | 0.0008% | 6 of 50 |

Among the labels that pass or are recommended: Big Y4C 0.0294% (show-time big model (SM91s2), instant; 0.0203% on the show-time instant-only big model (SM99)); Small Y2N 0.0016% (longer Small Instant-Only model (SM113)); Tiny Y4C 0.0024% (six-layer Tiny Instant-Only model (SM130)). So a small number of cards (well under 0.1% of cards) have one review whose predicted retention moves by more than 0.10; the instant head is the more affected one, and Y4N the most. Cards with 128+ answers never move by more than 0.08 on any review.

## Tiny re-read under the user's 11-22 Tiny rule (not pre-registered)

After the design was registered, the user set a Tiny-specific rule on 11-22: for Tiny, int3 or int4 state may cost more than the +0.05% dataset-wide bar as long as Tiny with that state format stays within +1.00% (95% CI upper bound, each head) of the matching Small model at int8. Matching pairs: the six-layer Tiny two-head model (SM130) against the show-time two-head cool-down for Small (SM118), curve and instant; the six-layer Tiny Instant-Only model (SM130) against the longer Small Instant-Only model (SM113). Computed by `sm145_tiny_reread.py` from the GPU rows of the 732 set (`gpu/sm145-rows-732.jsonl`, same 723 users and inputs, Small int8 rows from the same file), each Tiny label paired by user with the matching Small int8 model, ratio of summed LogLoss x size, 2,000-draw user bootstrap (the method of `_bench_cmp` in `sm145_report.py`). R1, the long-card tolerance (R3: point <= +0.5%, upper <= +1.0%), R4 and the per-card numbers are kept as they are; only R2 is replaced.

| Tiny label (state bytes vs int8) | two-head model (SM130) curve vs Small | two-head model (SM130) instant vs Small | Instant-Only model (SM130) instant vs Small | passes |
|---|---|---|---|---|
| REF, int8 (0%) | +0.467% [+0.398, +0.538] | +0.519% [+0.478, +0.564] | +0.561% [+0.519, +0.606] | yes (reference) |
| AN16, int4 below 16 answers (-35.34%) | +0.478% [+0.410, +0.550] | +0.538% [+0.497, +0.584] | +0.591% [+0.547, +0.638] | yes |
| Y2N, int3 for 1-2 answers, cards and notes (-37.53%) | +0.487% [+0.419, +0.559] | +0.549% [+0.507, +0.595] | +0.604% [+0.562, +0.651] | yes |
| Y4C, int3 for 1-4 answers, cards only (-37.74%) | +0.486% [+0.417, +0.559] | +0.548% [+0.506, +0.595] | +0.603% [+0.560, +0.650] | yes |
| Y4N, int3 for 1-4 answers, cards and notes (-39.80%) | +0.493% [+0.425, +0.565] | +0.560% [+0.517, +0.607] | +0.619% [+0.576, +0.667] | yes |

The six-layer Tiny models already sit +0.47% to +0.56% above their Small counterparts at int8; every Tiny label stays inside the +1.00% bound (largest upper bound +0.667%), and each also passes R1, R3 and R4 on both Tiny models. **Under this rule the Tiny pick is Y4N (-39.80% state bytes)**, the smallest bytes passing for both Tiny models; it is not the pre-registered pick (Y4C, -37.74%). The per-card cost of that choice: Y4N on the six-layer Tiny Instant-Only model (SM130) is the Tiny label with the most cards moved by more than 0.10 (0.0086%, 6 of 50 users); the pre-registered "worth its bytes" criterion also rejects Y4N for Tiny (extra cost vs AN16 above the limit). A Tiny-only follow-up with more aggressive tiers (SM146) is being prepared separately.

## Run facts

- Smoke ticket 001474 passed (66 checks, GPU vs CPU within tolerance). Scoring tickets (`gpu/tickets.jsonl`): Tiny 41 + 1 + 0 min, Small 74 + 2 min, Big 110 + 1 min (about 229 GPU minutes, 12:57 to about 17:00); Big at about 64,600 review-evals/s. CPU long-card chain 12:57 to 22:31 (9.6 h wall, 2 processes, about 1,000 review-evals/s per big-model user).
- Coverage: 723 GPU users per model (722 rule users), no non-finite user; long-card CPU users 50 of 50 for Tiny and Small, 49 of 50 for Big (user 1714 killed by the memory guard; the 48-user minimum is met).
- Integrity: REF and AN16 reproduce the earlier int8 switch (SM144) rows exactly (max |difference| 0.0 in benchmark LogLoss on the GPU and CPU rows, every model). GPU vs CPU on the 128+ bucket: mean LogLoss differs by at most 0.0015.

## For the fork

The largest saving that passes: Big Y4N (-45.3%), Small Y2N (-38.6%), Tiny Y4N under the 11-22 Tiny rule (-39.8%) or Tiny Y4C under the pre-registered rules (-37.7%). If the per-card criterion is applied, Big takes Y4C (-43.5%) instead of Y4N; Small and Tiny are unchanged by it. The decision is the user's; the Tiny rule was not part of the registered design.
