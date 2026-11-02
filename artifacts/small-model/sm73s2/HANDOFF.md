# Smaller model, card stream 1 layer (SM73 stage 2) early check: handoff

Branch `sm73s2` (cut from `sm33`, which carries the tiny architecture additions (SM77) warm-start tooling and the data-order re-seed (SM80) code; both option-gated), worktree `C:/Users/jonat/workspace/wt/sm73s2`. Design: `DESIGN-ADDENDUM.md` (this folder) on top of `wt/sm90/artifacts/small-model/sm73/DESIGN.md`.

## What it asks

Does cutting the card stream from 2 layers to 1 (candidate B: 4,554 -> 3,134 B per card + note at int8, 3,018 -> 2,110 at int4) cost more than the pre-registered bar? Arm = the leak-free small model (SM72s) epoch 20 cut to `w64-c1` + epoch 21 (SM72s's protocol) + quarter-epoch 32k Muon cool-down; reference = SM72s epoch 21 + the same cool-down (the tiny architecture additions (SM77) reference, already scored). Rule (`sm73s2_rule.py`): 722 users of the 732 set, paired arm - reference; **stop** if > +0.00020 curve or > +0.00015 instant (point estimate), else **go to finish**.

## What was done before launch (CPU)

- `expertium/variants.py` preset `w64-c1`; `scripts/sm73s2.py` (`cut`, `cpu-check`); `scripts/sm77_warm.py epoch --init` accepts a cut start; `scripts/check_muon_fresh_decay.py` derives the expected Muon / AdamW split for `w64-c1`.
- Cut checkpoint `artifacts/small-model/sm73s2-cut-e20` (not committed; master sha 6f2f9c04...), arm epoch folder `artifacts/small-model/sm73s2-arm-e21` prepared (`sm73s2_pair.py epoch` PASS vs SM72s epoch 21).
- CPU proofs `check-cpu-sm73s2.json`: ALL OK (C0 cut checkpoint, C1 layer steps, C2 TorchScript + eager build, T1 trainer path cut == full with the layer step skipped bitwise eager + scripted, S1 scorer cut == full with the branch zeroed bitwise fp32 + w8+s8, M1 Muon fresh check on a `w64-c1` layout).

## What runs (detached, WMI hidden)

Launcher `launch-sm73s2.sh`, log `gpu/launch-sm73s2.log`, nohup `gpu/launch-sm73s2.nohup`. Launched 2026-11-02 08:52 (WMI pid 77584, commit 28c224df); first log line `enqueue: trainer smoke of the arm epoch (w64-c1, 2 users x 512 reviews) (expected 15 min, ...)`, queue ticket 000886 running at 08:52. Ticket purposes in order, each prefixed "smaller model, card stream 1 layer (SM73 stage 2): ":

1. "trainer smoke of the arm epoch (w64-c1, 2 users x 512 reviews)" (15 min; output `gpu/check.out`)
2. "arm epoch 21 (w64-c1 cut from SM72s epoch 20, SM72s's epoch-21 protocol: ...) (segment to the next window or the end)" (125 min; output `gpu/train.out`, folder `artifacts/small-model/sm73s2-arm-e21`)
3. CPU: cool-down prepare + label-window trim/check + Muon fresh check (`gpu/check-*.json`) + `sm73s2_pair.py decay` vs the reference (log line `[sm73s2-pair] decay OK`)
4. "arm quarter-epoch 32k Muon cool-down from arm epoch 21 (w64-c1, the reference's options and order) (segment to the next window or the end)" (150 min; folder `artifacts/small-model/sm73s2-arm-e21-dec-q32k-muon`)
5. "scoring sm73s2-arm-e21-q32k on the 732 pool minus 9 never-fit (723) (pass 1, 0 done, cap 0.8)" (12 min; a pass without progress is retried once at cap 0.92), then "scoring sm73s2-arm-e21-q32k on the 3 long users of the 732 pool (cap 0.92)" (8 min); rows `gpu/sm73s2-neighbour.jsonl`, log `gpu/score.out`
6. CPU rule -> `sm73s2-decision.{json,md}`; log line `rule exit 0: [sm73s2-rule] ...`, then `done`.

Expected end of the read-out: ~14:15-14:30 on 2026-11-02 if the queue stays free (smoke ~09:07, epoch ~11:15, cool-down ~13:50, scoring ~14:15). If other tickets come first, the epoch only starts when 125 min fit before the 20:00 break; a cool-down cut by the break resumes at 01:00.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm73s2 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm73s2/gpu/launch-sm73s2.log
cat artifacts/small-model/sm73s2/sm73s2-decision.md          # when the log says "done"
# re-run the rule by hand (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm73s2/sm73s2_rule.py decide
```

Report: the verdict line, both heads (delta, points with CI, arm vs big, reference vs big), the projection against the bar, the epoch and cool-down reviews/s and minutes (from the launcher's "complete:" lines), bytes per card + note (3,134 int8 / 2,110 int4). Commit with `git add -f`: `gpu/launch-sm73s2.log`, `gpu/*.out` (if < 20 MB), `gpu/check-*.json`, `gpu/sm73s2-neighbour.jsonl`, `sm73s2-decision.{json,md}`; push. Do not commit checkpoints. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm73s2/artifacts/small-model/sm73s2/launch-sm73s2.sh >> C:/Users/jonat/workspace/wt/sm73s2/artifacts/small-model/sm73s2/gpu/launch-sm73s2.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch / cool-down): `touch C:/Users/jonat/workspace/wt/sm73s2/artifacts/small-model/sm73s2/gpu/STOP` (remove it before relaunching). After a Claude app restart, check the launcher is alive; if its log has been silent for more than 10 minutes outside a ticket, relaunch. Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training).
