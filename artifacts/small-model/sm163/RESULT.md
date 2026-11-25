# Clean re-measure of state-building speed in the fork (SM163): result

Written Wed 11-25, 04:40. Decision D-1124-06. Fork commits: **before** = fork main fb9f59a18 (T24 with int8 factors,
the faster codec (SM156), the Learning fix (SM161)); **after** = fork main eda153ddb (adds 6-bit factors for Big and
Small (SM162)). Release builds, worktree `wt/fork-lowrank-speed` (checked out detached for the builds, then back on
`rc-lowrank-speed`; no fork source or branch changed). Synthetic data only.

## Verdict

- **Full first-launch rebuild (app path, pairs, all threads, 205,785 synthetic reviews):** the new default
  (`compact_lowrank`) runs at Big 14,346 reviews/s (+38.8% vs int8 rows), Small 45,156 (-3.9%), Tiny 80,569
  (-3.9%). Before vs after on the default: Big -2.9%, Small -1.1%, Tiny +0.6%. A whole rebuild takes 14.6 s (Big),
  4.7 s (Small) and 2.7 s (Tiny), against 20.4 s, 4.6 s and 2.6 s at int8 rows.
- **SM162 changed nothing on the existing formats:** before vs after on int8 rows and on T24 int8 factors stays within
  ±2% everywhere (inside the run-to-run spread), with one exception at 3.3%: Big pair, T24, four threads.
- **The 6-bit decode cost on the clean machine:** the codec alone decodes 19% to 33% slower with int6 factors than with
  int8 factors (head size 16: +19% at rank 2, +33% at rank 4; head size 32: +23% and +26%). Encoding costs only +1% to
  +7% more. End to end the cost is no more than about 2%, inside the noise: Big rebuild -1.7%, Small -1.2%;
  one-thread replay ±0.5%; show-time p50 +0.7% (Big) and +1.4% (Small).
- **The pre-registered 15% rule triggers on its own measure.** The previous numbers and the rule come from the
  one-thread replay of a long history (1,000 cards x 24 answers, every card past the rank-4 tier). On the clean
  machine, Small trails int8 rows there by **17.2%** and Tiny by **23.9%** (default format). The next step is
  therefore to batch the codec across several cards during the rebuild; it is not implemented here. On the realistic
  rebuild (mixed answer counts, all threads) the gap is only 3.9% for both sizes. The gap is a long-card, one-thread
  effect, so that is where batching would help.
- **The previous numbers were inflated by load:** the clean one-thread long-history replay is 15% to 25% faster than
  the numbers taken under GPU load, but the ratios are close. Int8 rows: Big 4,571 (was ~3,960), Small 20,684
  (~16,900), Tiny 39,230 (~32,200). T24 with the faster codec: Big 4,994 (4,219), Small 17,194 (13,804), Tiny
  30,262 (25,541).

## Quiet machine

- The CPU read-out (SM164) had already finished when this started (no `sm164_report` process at 03:32).
- A GPU quiet window covered 03:42 to 05:02 ("SM163 clean speed re-measure"). It was removed at 04:32, once the
  timing was done. The queue was empty the whole time.
- Before every step: CPU load was 1% to 10% (median 2% to 3%), and the GPU stayed at 0% and 598 MiB the whole time.
  Only the timed process was running, besides desktop apps at idle.
- Run 03:42 to 04:30. Six configurations (before int8 rows, after int8 rows, before T24, after T24, after default,
  after T24 int6) ran in alternating before/after order (ABAB), reversed on pass 2. There were 3 passes, and every
  number below is the median of the 3. "Spread" is (max - min) / median.

## Method

Runtimes as the app runs them: Big = show-time big model (SM91s2) with the show-time instant-only big model (SM99);
Small = Small two-head (SM118) with Small Instant-Only (SM113); Tiny = six-layer Tiny two-head (SM130) with six-layer
Tiny Instant-Only (SM130). "Alone" = the two-head model by itself.

Formats: `compact_cards_int8` (int8 rows), `compact_lowrank_int8` (T24 int8 factors, the default before),
`compact_lowrank` (the new default by size: Big and Small int6 factors, Tiny int8 factors; the store sizes confirm
it: Tiny's default store equals its T24 int8 store), `compact_lowrank_int6` (int6 on every size). The before build
has no `compact_lowrank` or `compact_lowrank_int6`.

1. **Rebuild, latency, batch sweep:** `sm163_speed.rs`, a temporary untracked example (it was copied into
   `rslib/examples/` for the two builds and removed afterwards). It uses only the public `anki::rwkv` API.
   - The synthetic collection has 14,000 cards (2 per note, 3 decks) over 730 days and 205,785 answers: new cards,
     Learning steps (a tenth of them one-day steps), reviews, and lapses with Relearning. Answer counts run up to
     377, and 78 cards have 128 or more answers. 164 cards are in Learning or Relearning at the end.
   - The rebuild follows the app's path: runtime load, state-only replay in 16,384-review chunks, a full recovery
     checkpoint 8 days before the last review, and a final delta checkpoint into a temporary state store. It uses
     all 16 threads, the app's default.
   - Latency is measured per call on the resident state: the state lookup plus `RwkvRuntime::review`, as the
     desktop calls it. Show-time is the query; answer is a Good answer. It covers 400 Review cards and all 164
     Learning and Relearning cards, 5 calls each. The p50 and p95 are per run, and the tables show the median over
     the 3 runs.
   - The batch sweep runs R-only queue scoring of all 14,000 cards from the resident state
     (`predict_retrievability_many_from_warm_up`) in batches of 32 to 1,024, 3 passes, median.
2. **Replay:** `rwkv_model_size_benchmark` (the SM156 harness, unchanged; 3 samples after a warm-up), at one thread
   on 1,000 x 24 (Big, Small, Tiny) and on 4,000 x 8 (Small, Tiny). The **Deck Options replay** is the same
   benchmark at four threads on 32,000 answers (4,000 x 8), pairs and two-head models alone.
3. **Codec alone:** `rwkv_state_lowrank_codec_micro_benchmark` (after build, one thread, 3 runs).

## Full first-launch rebuild (reviews/s of the replay; pairs, all threads, 205,785 reviews)

| Size | Format | before (spread) | after (spread) | after vs before | after vs int8 rows | whole rebuild, after (incl. load + checkpoints) | state store |
|---|---|---:|---:|---:|---:|---:|---:|
| Big | int8 rows | 10,530 (3.7%) | 10,339 (4.5%) | -1.8% | 0 | 20.38 s | 672.7 MB |
| Big | T24 int8 factors | 14,777 (0.8%) | 14,759 (1.5%) | -0.1% | +42.8% | 14.20 s | 208.9 MB |
| Big | **default (int6)** | 14,777 (before default) | **14,346** (1.4%) | -2.9% | **+38.8%** | 14.56 s | 185.0 MB |
| Big | T24 int6 factors | - | 14,510 (2.5%) | - | +40.3% | 14.40 s | 185.0 MB |
| Small | int8 rows | 46,414 (2.4%) | 47,004 (1.0%) | +1.3% | 0 | 4.56 s | 119.3 MB |
| Small | T24 int8 factors | 45,668 (0.9%) | 45,909 (0.5%) | +0.5% | -2.3% | 4.61 s | 59.1 MB |
| Small | **default (int6)** | 45,668 (before default) | **45,156** (1.3%) | -1.1% | **-3.9%** | 4.68 s | 51.9 MB |
| Small | T24 int6 factors | - | 45,376 (1.0%) | - | -3.5% | 4.65 s | 51.9 MB |
| Tiny | int8 rows | 82,504 (1.9%) | 83,797 (0.8%) | +1.6% | 0 | 2.58 s | 80.3 MB |
| Tiny | T24 int8 factors | 80,115 (0.6%) | 79,949 (0.5%) | -0.2% | -4.6% | 2.68 s | 44.9 MB |
| Tiny | **default (int8 factors)** | 80,115 (before default) | **80,569** (0.4%) | +0.6% | **-3.9%** | 2.66 s | 44.9 MB |
| Tiny | T24 int6 factors | - | 79,278 (0.5%) | - | -5.4% | 2.70 s | 40.2 MB |

"Before default" is T24 int8 factors, the default at fb9f59a18.

## Replay, one thread (the previous numbers' measure)

1,000 cards x 24 answers (24,000 answers; every card reaches the rank-4 tier):

| Size (alone) | Format | before (spread) | after (spread) | after vs before | after vs int8 rows | under load before (D-1124-06) |
|---|---|---:|---:|---:|---:|---:|
| Big | int8 rows | 4,566 (1.7%) | 4,571 (0.7%) | +0.1% | 0 | ~3,960 |
| Big | T24 int8 factors | 5,016 (2.0%) | 4,994 (0.6%) | -0.4% | +9.3% | 4,219 |
| Big | default (int6) | - | 4,965 (1.6%) | -1.0% vs before default | +8.6% | - |
| Big | T24 int6 factors | - | 4,957 (0.4%) | - | +8.4% | - |
| Small | int8 rows | 20,745 (0.4%) | 20,684 (0.8%) | -0.3% | 0 | ~16,900 |
| Small | T24 int8 factors | 17,103 (1.9%) | 17,194 (0.5%) | +0.5% | -16.9% | 13,804 |
| Small | default (int6) | - | 17,119 (1.3%) | +0.1% vs before default | **-17.2%** | - |
| Small | T24 int6 factors | - | 17,166 (1.1%) | - | -17.0% | - |
| Tiny | int8 rows | 39,156 (0.7%) | 39,230 (1.4%) | +0.2% | 0 | ~32,200 |
| Tiny | T24 int8 factors | 30,374 (2.8%) | 30,262 (1.8%) | -0.4% | -22.9% | 25,541 |
| Tiny | default (int8 factors) | - | 29,847 (1.5%) | -1.7% vs before default (same code) | **-23.9%** | - |
| Tiny | T24 int6 factors | - | 30,365 (1.8%) | - | -22.6% | - |

4,000 cards x 8 answers (32,000 answers; cards below 16 answers, rank 1 to 2), after: Small int8 rows 20,396, T24
19,758 (-3.1%), default 19,606 (-3.9%), int6 19,900 (-2.4%); Tiny int8 rows 37,723, T24 36,527 (-3.2%), default
36,638 (-2.9%), int6 36,572 (-3.1%). Before vs after is within ±0.7%. See `replay.csv`.

## Deck Options replay, four threads (32,000 synthetic answers)

| Runtime | int8 rows: before / after | T24 int8 factors: before / after | default, after | T24 int6, after | default vs int8 rows (after) |
|---|---:|---:|---:|---:|---:|
| Big pair | 5,825 / 5,769 | 7,153 / 6,914 | 7,034 | 6,921 | +21.9% |
| Big alone | 11,462 / 11,379 | 13,346 / 13,536 | 13,331 | 13,279 | +17.2% |
| Small pair | 25,170 / 25,103 | 24,561 / 24,693 | 24,515 | 24,439 | -2.3% |
| Small alone | 45,734 / 45,444 | 45,275 / 45,130 | 45,003 | 44,677 | -1.0% |
| Tiny pair | 46,841 / 47,341 | 45,431 / 45,399 | 45,787 | 45,503 | -3.3% |
| Tiny alone | 84,538 / 85,563 | 83,500 / 82,816 | 82,586 | 83,107 | -3.5% |

Replay time saved vs Big at the same format (the Deck Options "replay saving"):

| Format (after) | Small pair | Tiny pair | Small alone | Tiny alone |
|---|---:|---:|---:|---:|
| int8 rows | 77% | 88% | 75% | 87% |
| T24 int8 factors | 72% | 85% | 70% | 84% |
| **default** | **71%** | **85%** | **70%** | **84%** |
| T24 int6 factors | 72% | 85% | 70% | 84% |

The Deck Options table in the fork (paired 72% / 84%, alone 70% / 84%) still matches within 1%; it needs no change.

## Per-review latency (pairs, µs; median over runs of p50 / p95)

| Size | Format | Review show-time | Review answer | Learning show-time | Learning answer |
|---|---|---:|---:|---:|---:|
| Big | int8 rows (before / after) | 3,067 / 3,373 ; 3,070 / 3,386 | 764 / 813 ; 756 / 806 | 3,048 / 3,447 ; 3,080 / 3,378 | 738 / 809 ; 728 / 798 |
| Big | T24 int8 factors (before / after) | 2,998 / 3,307 ; 2,991 / 3,320 | 744 / 815 ; 746 / 807 | 2,808 / 3,078 ; 2,809 / 3,105 | 666 / 756 ; 672 / 750 |
| Big | default, after | 3,037 / 3,426 | 763 / 834 | 2,844 / 3,163 | 686 / 769 |
| Big | T24 int6, after | 3,012 / 3,352 | 761 / 829 | 2,843 / 3,136 | 687 / 768 |
| Small | int8 rows (before / after) | 569 / 682 ; 566 / 700 | 185 / 208 ; 186 / 208 | 567 / 750 ; 563 / 749 | 187 / 204 ; 188 / 206 |
| Small | T24 int8 factors (before / after) | 628 / 771 ; 631 / 773 | 197 / 231 ; 201 / 233 | 563 / 736 ; 566 / 732 | 184 / 205 ; 187 / 207 |
| Small | default, after | 640 / 803 | 203 / 237 | 574 / 642 | 188 / 208 |
| Small | T24 int6, after | 640 / 806 | 202 / 238 | 573 / 728 | 188 / 210 |
| Tiny | int8 rows (before / after) | 295 / 409 ; 290 / 397 | 92 / 125 ; 86 / 124 | 315 / 373 ; 316 / 434 | 108 / 127 ; 106 / 129 |
| Tiny | T24 int8 factors (before / after) | 357 / 464 ; 361 / 453 | 109 / 140 ; 110 / 142 | 312 / 382 ; 319 / 408 | 104 / 128 ; 106 / 129 |
| Tiny | default, after | 359 / 474 | 112 / 141 | 317 / 392 | 106 / 128 |
| Tiny | T24 int6, after | 367 / 471 | 111 / 142 | 318 / 392 | 104 / 129 |

- A show-time call is about 3.0 ms for Big, 0.6 ms for Small and 0.3 ms for Tiny. An answer is 0.75, 0.2 and 0.1 ms.
  All of these are well below what a user can notice.
- On Review cards, the default format's show-time p50 is +13% vs int8 rows for Small and +24% for Tiny, which is
  +74 and +69 µs. Learning cards show no extra cost, because they have few answers (rank 1 to 2). Big is the same
  or faster.
- Show-time costs more than an answer because the query also computes the four buttons' intervals.

## Query batch-size sweep (R-only queue scoring of 14,000 cards, queries/s, after)

| Size | Format | 32 | 64 | 128 | 256 | 512 | 1,024 |
|---|---|---:|---:|---:|---:|---:|---:|
| Big | int8 rows | 29,762 | 33,140 | 35,714 | 37,464 | 38,301 | 39,241 |
| Big | T24 int8 factors | 32,911 | 35,734 | 38,220 | 39,192 | 39,947 | 40,251 |
| Big | default | 33,010 | 35,741 | 37,986 | 38,976 | 39,404 | 39,997 |
| Big | T24 int6 | 33,715 | 35,759 | 37,689 | 38,952 | 39,538 | 39,951 |
| Small | int8 rows | 96,543 | 110,450 | 118,484 | 123,735 | 128,971 | 130,240 |
| Small | T24 int8 factors | 100,298 | 113,464 | 121,412 | 127,426 | 132,420 | 133,615 |
| Small | default | 98,338 | 112,413 | 121,098 | 127,310 | 131,457 | 135,312 |
| Small | T24 int6 | 98,135 | 113,416 | 121,445 | 127,205 | 131,613 | 135,167 |
| Tiny | int8 rows | 162,085 | 192,337 | 218,808 | 237,143 | 246,310 | 256,185 |
| Tiny | T24 int8 factors | 166,947 | 199,160 | 222,095 | 243,344 | 253,460 | 262,437 |
| Tiny | default | 170,642 | 202,860 | 222,149 | 243,293 | 253,542 | 263,788 |
| Tiny | T24 int6 | 162,119 | 203,196 | 223,724 | 236,915 | 254,570 | 251,732 |

- The before build is within ±4% of these numbers at every batch size (`app.csv`).
- Queue scoring is decode-only, and it is about as fast as int8 rows or faster at every batch size (-2% to +11%), because
  the states are smaller.
- From 32 to 512 throughput rises by 29% (Big) to 52% (Tiny). From 512 to 1,024 the gain is only 1% to 4%, so the
  default batch of 512 sits on the plateau.

## Codec alone (after build, one layer of 4 heads, one thread, median of 3 runs)

| Head size (sizes) | rank | encode int8 / int6 | decode int8 / int6 | int6 decode cost |
|---|---:|---:|---:|---:|
| 16 (Small, Tiny) | 2 | 3,008 / 3,038 ns | 278 / 332 ns | +19% |
| 16 | 4 | 9,725 / 9,851 ns | 473 / 629 ns | +33% |
| 32 (Big) | 2 | 5,470 / 5,800 ns | 759 / 935 ns | +23% |
| 32 | 4 | 14,492 / 14,909 ns | 1,169 / 1,473 ns | +26% |

A decode costs at most a tenth of an encode at rank 4, so the extra unpacking work does not show up end to end.

## Follow-up rule (pre-registered in D-1124-06)

- The rule: "if Small or Tiny still trail int8 by more than about 15% on the clean run, batch the codec across
  several cards during the rebuild".
- On the measure behind the previous numbers (one-thread replay, 1,000 x 24), Small trails by 17.2% and Tiny by
  23.9%, so **the rule triggers**.
- For context: on the realistic all-thread first-launch rebuild both sizes trail by only 3.9% (4.7 s and 2.7 s for
  206k reviews), and on short histories by about 3%. The work stays a follow-up. It is not implemented here.

## Files

- `speed-run.log`: the whole run (load probes, replay benchmark lines).
- `app-rebuild-latency-sweep.jsonl`: the 54 rebuild / latency / sweep records.
- `codec-micro-benchmark.log`: the codec micro-benchmark runs.
- `summary.json`, `replay.csv`, `app.csv`: medians, spreads and every run.
- `sm163_speed.rs` (the example), `run.sh` (the driver), `summarize.py` (the tables).
