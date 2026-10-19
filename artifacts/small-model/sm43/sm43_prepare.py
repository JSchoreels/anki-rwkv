"""SM43 mini (10-19 evening): data diversity vs epochs at equal reviews seen.

  A: 1,000 gate training users x 1 epoch     (stratified by history length: 100 per length decile)
  B:   100 of those users     x 10 epochs    (10 per decile, picked so B holds 1/10 of A's reviews within 0.5%)

Both: the 2,048-window cached-KD recipe of scale-w64kd-gate-cached-ctx2048 (same options, seed, order seed), constant
LR, from scratch. Run once (CPU): prepares two accuracy-loop experiments, then narrows their run.json user lists.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

WT = Path("C:/Users/jonat/workspace/wt/small-model")
PY = "C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe"
EXP = WT / "artifacts/accuracy/experiments"
REF = json.loads((EXP / "scale-w64kd-gate-cached-ctx2048/experiment.json").read_text())
OUT = WT / "artifacts/small-model/sm43"

tiers = json.loads((WT / "artifacts/accuracy/tiers.json").read_text())
gate = tiers["tiers"]["gate"]["train_users"]
counts = json.loads((EXP / "scale-w64kd-gate-cached-ctx2048/run/run.json").read_text())["review_counts"]
rc = lambda u: counts[str(u)]
by_len = sorted(gate, key=lambda u: (rc(u), u))
deciles = [by_len[i * len(by_len) // 10:(i + 1) * len(by_len) // 10] for i in range(10)]
rng = random.Random(43)
a_by_dec = [sorted(rng.sample(d, 100)) for d in deciles]
users_a = sorted(u for d in a_by_dec for u in d)
ra = sum(map(rc, users_a))
best = None
for seed in range(2000):
    r = random.Random(seed)
    b = sorted(u for d in a_by_dec for u in r.sample(d, 10))
    err = abs(sum(map(rc, b)) * 10 / ra - 1)
    if best is None or err < best[0]:
        best = (err, seed, b)
    if err < 0.005:
        break
err, seed_b, users_b = best
assert err < 0.005, err
rb = sum(map(rc, users_b))
assert set(users_b) <= set(users_a)

overrides = REF["option_overrides"]
arms = {"sm43a-1000u-1ep": (users_a, 1, ra), "sm43b-100u-10ep": (users_b, 10, rb)}
for name, (users, epochs, reviews) in arms.items():
    if not (EXP / name).exists():
        subprocess.run([PY, "-X", "utf8", str(WT / "scripts/accuracy_loop.py"), "prepare", name, "--tier", "gate",
                        "--seed", str(REF["seed"]), "--order-seed", str(REF["order_seed"]), "--epochs", str(epochs),
                        "--options", json.dumps(overrides),
                        "--note", f"SM43 mini: {len(users)} users x {epochs} epochs, equal reviews seen (data diversity vs epochs)"],
                       check=True, cwd=WT)
    run_p = EXP / name / "run/run.json"
    run = json.loads(run_p.read_text())
    run.update(train_users=users, training_reviews_per_epoch=reviews,
               split=run["split"] + f"; SM43 subset of {len(users)} gate users (sm43_prepare.py)")
    run_p.write_text(json.dumps(run, indent=1))
    ex_p = EXP / name / "experiment.json"
    ex = json.loads(ex_p.read_text())
    ex.update(train_users=len(users), sm43={"users": len(users), "epochs": epochs, "reviews_per_epoch": reviews,
                                             "reviews_total": reviews * epochs})
    ex_p.write_text(json.dumps(ex, indent=1))
summary = {"a": {"users": len(users_a), "epochs": 1, "reviews": ra}, "b": {"users": len(users_b), "epochs": 10, "reviews_per_epoch": rb,
           "reviews_total": rb * 10, "pick_seed": seed_b}, "ratio_total_b_over_a": rb * 10 / ra,
           "users_a": users_a, "users_b": users_b, "options_from": "scale-w64kd-gate-cached-ctx2048"}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sm43-users.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("users_")}))
