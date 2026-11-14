# Time to fluency (SM84) and wider simulator search with time-aware scheduling (SM86): handoff

Branch `sm84` (from `sm82s2`), worktree `C:/Users/jonat/workspace/wt/sm84`. CPU only, no GPU.

## Done

- Time to fluency (SM84): `time-to-fluency.md` (takeaways written), `time-to-fluency.json`, `time-to-fluency-users.json`; script `scripts/sm84_fluency.py` (~20-70 s, 150 users). Committed 51c8794c.

## Done: wider simulator search with time-aware scheduling (SM86), finished 11-01 10:36, takeaways written

Wider simulator search with time-aware scheduling (SM86): `scripts/sm86_search.py run --procs 6` (6 processes, below-normal priority), launched Sunday 11-01 10:21 from commit 9c750122 via `launch-sm86.sh`. Expected ~20 min (6-user test: 43 s per user-batch of 6 -> 150 users ~17-20 min), so done by ~10:45.

- Log: `C:/Users/jonat/workspace/wt/sm84/artifacts/small-model/sm84/logs/sm86.log` (last line `done (rc 0) ...`), launcher output `logs/launch-sm86.nohup`.
- Outputs: `simulator-search.md` (with a `TAKEAWAYS` placeholder), `simulator-search.json`, `simulator-search-users.json` (full precision, a few MB).

Relaunch if it died (no resume; it reruns everything, ~20 min; not after 15:40 on Sunday, CPU break from 16:00):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm84/artifacts/small-model/sm84/launch-sm86.sh >> C:/Users/jonat/workspace/wt/sm84/artifacts/small-model/sm84/logs/launch-sm86.nohup 2>&1"'; ProcessStartupInformation=$si}
```

## Follow-up after `done (rc 0)` (Sonnet medium is enough)

```
cd C:/Users/jonat/workspace/wt/sm84 && git fetch && git merge --ff-only
S=artifacts/small-model/sm84; PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
# shrink the per-user file (5 significant digits, compact)
$PY -c "import json;f='$S/simulator-search-users.json';r=lambda x: float(f'{x:.5g}') if isinstance(x,float) else {k:r(v) for k,v in x.items()} if isinstance(x,dict) else [r(v) for v in x] if isinstance(x,list) else x;json.dump(r(json.load(open(f))),open(f,'w'),separators=(',',':'))"
```

1. Check the header of `simulator-search.md`: "Stage 2 reproduced on the overlapping grid: max relative difference" must be 0 (and the simulator-function check 0).
2. Replace `TAKEAWAYS` in `simulator-search.md` with 4-5 plain-language bullets, same style as `artifacts/small-model/sm82/stage2/stage2-result.md` and `time-to-fluency.md` (always "plain name (ID)", never a bare ID; never name outside researchers):
   - Section 1 (unlimited new cards, stage 2's setup): where the optimum sits with 0.30-0.98 (the 6-user test had every user at the 0.30 bound: with an unlimited deck and the FSRS-6 power curve, the sum of recall at day 365 rewards learning new cards over reviewing, so there is still no interior optimum; say so plainly, and that the cost models cannot move a bound optimum).
   - Section 2 (at most 10 new cards a day): the optimum distribution per cost model (median, quartiles, share at the bounds), the shift B / C vs A (head), and the cost of keeping the fixed-cost optimum when B or C is the truth (mean, 90th percentile); also how much desired retention 0.90 loses vs the optimum.
   - Section 3: the time-aware per-card policy (stability gain per predicted second, tuned offset) vs the best single desired retention and vs the same policy with fixed costs (truth B and C): memorized cards at day 365, share of users better; whether the offset hits its bounds (-0.20 / 0.56).
   - What is extrapolation: desired retention below ~0.6 is outside observed scheduling; the cost curve is flat below its lowest bin (median recall ~0.34) and those bins come from stretched-interval queries; the new-card cap is a modelling choice; the policy is a heuristic and both it and the single target are tuned on the evaluated simulation.
3. Commit (author Jonathan Schoreels <jonathan.schoreels@gmail.com>, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`), `git add -f` exactly: `$S/simulator-search.md $S/simulator-search.json $S/simulator-search-users.json $S/logs/sm86.log $S/logs/launch-sm86.nohup`, and this HANDOFF.md updated to "done". Push. Never commit `docs/GPU_OPTIMIZATIONS_LOG.md` or `.claude/`.
4. Report headline numbers of both results (time to fluency (SM84) takeaways are in `time-to-fluency.md`) to the main session, which updates the Factory page.
