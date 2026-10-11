# R01: retraining optimization loop

## Goal
- Find ways to improve the speed of training the model, expressed in review/seconds.
- Test both Alex state representation (the default one) and Expertium one (the smaller one)

## Procedure
- Do a first baseline of Alex (default) model and Expertium (light) one.
- Reuse each model's original baseline and previous accepted run for later comparisons. Rerun a reference only for a stated comparability problem.
- Profile an execution over 100 users.
- Reuse applicable profiles; run focused smoke checks when changed code needs verification.
- Compute what could be the best improvement in terms of request/second.
    - Consider first ressource usage maximization
    - When the ressources are used at ~80% of their capacity, focus more on reducing the load itself by keeping result 100% similar.
    - When 100% similar results can't be achieved, try to not have worst logloss than 1% of the previous iteration, and if possible also of 2% of the baseline.
    - Apply those limits to measured loss changes; report uncertainty separately.
    - When that logloss accuracy can't be achieved, it's oK to log those speed improvement.
- Implement and try this improvement.
    - If successful (faster and still accurate), commit the new code.
    - If it's accurate, but same speed, but with less ressources usage, try to see if we can't re-adjust ressources usage to re-maximize the it (for ex, if a change reduce by 50% the GPU demand)
    - If partially sucessful (faster but a bit less accurate), maintain that improvement in a separate 'fastmode' implementation. Commit also
    - Keep fastmode separate from the main accepted configuration and its comparison reference.
- Iterate until you can't find 20% request/s improvement after 3 iterations.
- Stop each model before another iteration after three consecutive misses of the 20% improvement target.

## Artifacts
- Fill the results in GPU_OPTIMIZATIONS.md and track the "Optimization tracks" there.
- Add columns to the LOG table only after the user explicitly confirms them.
- In a GPU_OPTIMIZATIONS_LOG.md files, I want you to maintain each run made with a line with as columns (Don't do X/Y for two values inside the same columns) :
    - Timestamp of the start of the training
    - The commit-id of the trial.
    - Model.
    - The differents IDs of improvement (those in GPU_OPTMIZATIONS.md) tried, and also a human readable explanation of what is tried (Concise, 1-3 sentences max). More details can be stored in GPU_OPTIMIZATIONS.md
    - The number of users trained. Define a set of 100 random users that will be used for each run. Define them in GPU_OPTIMIZATIONS.md and re-use if it already exist. Do the same for +150 users to make a 250 set, and +400 to make a 500 set. Each time, use 20% of that set as an hold-oud for testing logloss.
    - The number of reviews trained on.
    - The duration of the training.
    - The review/seconds, and how much percent decrease/increase compared to the previous run and the previous baseline run.
    - The logloss achieved, and how much percent decrease/increase compared to the previous run and the previous baseline run.
        - Use different columns for Instant User, Instant Weighted, Curve User, Curve Weighted
    - Raw result link.
    - Notes.
    - Round percentage to 2 digit after comma.
