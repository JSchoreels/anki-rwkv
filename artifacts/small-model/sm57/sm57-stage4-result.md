# The instant-only small model (SM57) stage 4: result (2026-11-08)

Verdict: **go**. Candidate (instant-only epochs 27-30 + the release's one-epoch 32k Muon cool-down) vs stage 1's instant-only arm, instant head, 722 users: -0.071 [-0.079, -0.064] points (% of the big model (1 leak-free epoch) LogLoss), 553 better / 167 worse, above the 0.05-points minimum.

- Candidate - the release (SM63): -0.233 [-0.252, -0.214] points; candidate - the big model (1 leak-free epoch): +0.051 [+0.029, +0.072] points (smaller-model bar <= +0.4: passes).
- 1,500 validation users (1,487 scored), % of the release's LogLoss: candidate - the release -0.231 [-0.246, -0.218] (1358 / 128), candidate - stage 1 -0.067 [-0.073, -0.061]; row check max diff 0.0. GO on the SM93 bar.
- Constant-LR trajectory (no cool-down, instant, points): e28 - e26 -0.099 [-0.167, -0.040]; e30 - e26 +0.143 [+0.091, +0.196]; e30 - e28 +0.243. Cool-down effect: from e30 -1.067, from e26 -0.852.
- Speeds (wall, reviews/s from 581.94M reviews per 8k epoch): e27 118.8 min 81,975; e28 119.6 min 81,305; e29 122.8 min 79,409; e30 115.7 min (2 segments, break stop) 83.8k wall (87,563 last segment); cool-down 478.1 min 20,313. nvidia-smi max 14.3-15.2 GB (baseline 1.5-2.1 GB); peak allocated ~10.9 GB (epochs), 7.4 GB (cool-down).
- Chain check: e27-e29 True; e30 False because it resumed from its snapshot after the break stop (initial parameters are the snapshot's).
- 10-epoch ETA at ~82k reviews/s: 2k tier ~5.0 h, 8k tier ~19.7 h (training only).

Details: sm57-stage4-decision.md, sm57-stage4-pool1500.md.
