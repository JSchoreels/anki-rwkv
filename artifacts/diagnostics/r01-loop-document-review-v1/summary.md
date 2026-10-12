# R01 loop documentation review

Updated the loop wording to make model scope, 80 training / 20 held-out users, previous accepted references, four separate binary losses, provenance, explicit-count progress and separate authorization for larger stages clear. Existing log columns and numerical thresholds were preserved. No training ran.

Pending implementation-policy ambiguity: with non-exact gain 4.9%, timing spread 1.97%, target 5% and unchanged losses, `decide()` returns accepted_main=true while meets_20_percent_target=false and miss streak=3. The current code treats 5% as a miss-counter target rather than a hard promotion gate. [Probe](probe.json) records exact inputs/output and source hash; the unchanged [policy source](gpu_r01_policy.py) is preserved. Recommended next step: make the intended non-exact promotion threshold explicit before changing the decision implementation, and preserve historical protocols/results.
