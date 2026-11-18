# Scheduling-time vs review-time accuracy (SM81)

724 users of the 732 set (the 723 users of the standard read-out + the 3 long ones when present), 20,751,812 common test reviews (every model scored on exactly the same reviews). LogLoss, lower is better; (% vs FSRS-7 per-user fit at scheduling time, [paired user-bootstrap 95% CI of the LogLoss difference, 2,000 resamples, seed 0]).

| Model | Scheduling time, review-weighted | Scheduling time, per user | Start of day, review-weighted | Start of day, per user | Review time, review-weighted | Review time, per user |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS-7, default parameters (no fit) | 0.3333 (+6.9%, [+0.0184, +0.0249]) | 0.3637 (+7.1%, [+0.0208, +0.0275]) | 0.3333 (+6.9%, [+0.0184, +0.0249]) | 0.3637 (+7.1%, [+0.0208, +0.0275]) | 0.3333 (+6.9%, [+0.0184, +0.0249]) | 0.3637 (+7.1%, [+0.0208, +0.0275]) |
| FSRS-7, per-user fit (reference) | 0.3119 | 0.3397 | 0.3119 | 0.3397 | 0.3119 | 0.3397 |
| FSRS7-Instant (shared start fit, online updates) | 0.3226 (+3.4%, [+0.0086, +0.0130]) | 0.3554 (+4.6%, [+0.0135, +0.0180]) | 0.3088 (-1.0%, [-0.0056, -0.0009]) | 0.3365 (-0.9%, [-0.0051, -0.0012]) | 0.2912 (-6.6%, [-0.0243, -0.0176]) | 0.3103 (-8.6%, [-0.0326, -0.0263]) |
| small model (the leak-free small model (SM72s), cool-down from epoch 25) | 0.2953 (-5.3%, [-0.0181, -0.0152]) | 0.3215 (-5.4%, [-0.0200, -0.0164]) | 0.2908 (-6.8%, [-0.0242, -0.0182]) | 0.3173 (-6.6%, [-0.0249, -0.0199]) | 0.2702 (-13.4%, [-0.0463, -0.0377]) | 0.2813 (-17.2%, [-0.0631, -0.0538]) |
| big model (1 leak-free epoch) | 0.2947 (-5.5%, [-0.0189, -0.0156]) | 0.3210 (-5.5%, [-0.0208, -0.0167]) | 0.2909 (-6.7%, [-0.0241, -0.0181]) | 0.3176 (-6.5%, [-0.0246, -0.0194]) | 0.2696 (-13.6%, [-0.0470, -0.0383]) | 0.2806 (-17.4%, [-0.0638, -0.0545]) |

FSRS-7 has no updates: its one prediction is the same at all three times.

- FSRS7-Instant (shared start fit, online updates): 44% of the review-time gain over its own scheduling-time prediction is already known at the start of the day (review-weighted, CI [38%, 50%]; per user 42%); on the user's first review of a day 100% (199,496 reviews), on the later reviews of the day 44% (20,552,316 reviews).
- small model (the leak-free small model (SM72s), cool-down from epoch 25): 18% of the review-time gain over its own scheduling-time prediction is already known at the start of the day (review-weighted, CI [9%, 26%]; per user 10%); on the user's first review of a day 100% (199,496 reviews), on the later reviews of the day 18% (20,552,316 reviews).
- big model (1 leak-free epoch): 15% of the review-time gain over its own scheduling-time prediction is already known at the start of the day (review-weighted, CI [7%, 23%]; per user 8%); on the user's first review of a day 100% (199,496 reviews), on the later reviews of the day 15% (20,552,316 reviews).

Rows: 20,860,250 FSRS-7 test reviews, 30,906,850 RWKV benchmark reviews, 20,751,812 in both; 108,438 FSRS-7 test reviews outside the RWKV set (its first fold starts later), 10,155,038 RWKV reviews outside the FSRS-7 set (8,739,250 of them same-day repeats, which the published FSRS-7 protocol does not test).
