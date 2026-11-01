# Response-time prediction (SM82) stage 0: answer-time audit

Data: the 723 fit users of the 732 set (`eval-users-732-fit.json`), whole histories, 42,811,256 reviews (36,675,119 of cards already seen). Script: `scripts/sm82_time.py audit` (CPU, seconds); numbers: [stage0-audit.json](stage0-audit.json). Anki stores the answer time in ms; this dataset caps it at 60 s.

## Headline numbers

| what | value |
|---|---|
| time exactly 0 (no time recorded) | 0.09% of reviews; 217 users have some, median user 0%, worst user 12.7% |
| time at the 60 s cap | 5.6% of reviews; per user p10 / median / p90 = 0.8% / 4.8% / 18%, worst 70% |
| users with a lower own cap | 1 user (40 s; 97 reviews at it), handled as censored ("40 s or more") |
| per-user median time, p10 / p50 / p90 | 4.0 s / 7.8 s / 19.4 s (min 0.4 s, max 60 s) |
| per-grade median of the user medians (Again / Hard / Good / Easy) | 11.9 s / 11.0 s / 7.3 s / 5.9 s |
| per-grade share at 60 s (Again / Hard / Good / Easy) | 7.7% / 8.1% / 4.1% / 3.1% |
| share of within-user log-time variance between cards | 38% (median user 44%, p10-p90 25%-60%) |
| correlation of log time with the card's previous time, per user p10 / p50 / p90 | 0.22 / 0.36 / 0.54 |

Bin shares of the 16 log-time bins (under 0.5 s, 14 bins x1.41 up to 60 s, 60 s and more): 1.0, 0.4, 0.7, 2.4, 3.9, 8.4, 11.6, 13.9, 13.1, 11.6, 9.4, 6.8, 5.5, 3.5, 2.5, 5.2 %.

"Fast" is not comparable across users: the median user takes 7.8 s, the 10% fastest under 4 s and the 10% slowest over 19 s. Within a user, 38% of the spread is card-to-card (text length, card type), so the model can only learn "fast for this card" from the card's own earlier times, which it sees as inputs (the card's previous time alone correlates 0.36 with the next one).

## Leak audit

1. The review's own time is an input only after the answer. The instant row that predicts review j has its time and its rating zeroed (`rwkv/data_processing.py`, `reject_columns` on query rows) and sits before review j's own row, which carries the time. CPU proof T3 (`check-cpu-sm82.json`): changing one review's time changes exactly one input value (column 8, `scaled_duration`, on that review's own row, after its instant row), and no output (grade or time) up to and including that instant row changes, bit for bit; later rows do change.
2. But the interval input is answer-to-answer, so it already contains the review's own time (the duration-leak item (SM31)). `elapsed_seconds` of review j = time from the card's previous answer to this answer. Evidence: when the card comes back right away (no other review in between, gap under 1 h: 92,592 reviews), 77% have an interval within 1.5 s of their own answer time. For same-day reviews the interval minus the times of the reviews in between is the review's own time plus idle time, so the model could read the answer time from its inputs.
3. On a later day this is negligible: a time of at most 60 s moves a day-scale interval by under 0.07% (0.0001 standard deviations of the scaled input), and the absolute answer times are not recoverable from the log-scaled inputs.

Decision: the time head is trained and scored only on reviews with a recorded time on a later day than the card's previous review (`elapsed_days >= 1`): 70.4% of the reviews of seen cards (25.8M here; 21.6M benchmark test reviews of the 722 rule users). Same-day (learning / relearning step) times need intervals as Anki knows them when it shows the card (SM31) and are left for later.

## Labels used by stage 1

`scripts/expertium_options.py` `time_labels` (shared by training and scoring): bin = position among the 15 edges 500 x 120^(k/14) ms; no label for time 0, first reviews and same-day reviews; times at a user's lower cap are censored (that bin or later). The grade head's labels and inputs are unchanged.
