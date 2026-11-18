# The instant-only small model (SM57) stage 4: the candidate on the 1,500 validation users

Verdict: **go** (instant -0.231 [-0.246, -0.218]: CI below zero and at least the 0.05-% minimum). Unit: % of the release's LogLoss on the same users (1% of it = 1.003 points of the big model (1 leak-free epoch) on the 722 users). Bar: candidate - the release, instant: 95% CI upper < 0 and gain >= 0.05 (% of the release's LogLoss): the SM93 ship bar.

Row check (6 users of the 732 set re-scored with the 1,500-user ticket): {"users": 6, "max_abs_logloss_diff": 0.0}

| users | comparison | head | model | ref | % [95% CI] | users better / worse |
|---|---|---|---|---|---|---|
| 1487 (all) | candidate - the release | instant | 0.27889 | 0.27954 | -0.231 [-0.246, -0.218] | 1358 / 128 |
| 1487 (all) | candidate - stage 1's instant-only arm | instant | 0.27889 | 0.27908 | -0.067 [-0.073, -0.061] | 1150 / 334 |
| 1487 (all) | stage 1 - the release (SM93 reference) | instant | 0.27908 | 0.27954 | -0.164 [-0.179, -0.152] | 1349 / 137 |
| 726 (ids < 5000 (the 732 set)) | candidate - the release | instant | 0.27868 | 0.27935 | -0.242 [-0.269, -0.219] | 667 / 58 |
| 761 (ids >= 5000) | candidate - the release | instant | 0.27907 | 0.27969 | -0.223 [-0.241, -0.206] | 691 / 70 |
